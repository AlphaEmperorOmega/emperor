from torch import Tensor
from torch.nn import Linear, Sequential
from dataclasses import dataclass, field
from Emperor.base.utils import DataClassBase, Module
from Emperor.experts.experts import MixtureOfExperts
from Emperor.layers.utils.base import LinearBlockStack, LinearBlockStackConfig
from Emperor.layers.utils.routers import RouterModel
from Emperor.layers.utils.samplers import SamplerModel

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


@dataclass
class FeedForwardConfig(DataClassBase):
    weighted_parameters_flag: bool | None = field(
        default=None,
        metadata={
            "help": "When `True` the sepected parameters will be multiplied by their probs."
        },
    )
    num_layers: int | None = field(
        default=None,
        metadata={"help": "Number of layers in the model"},
    )


class FeedForward(Module):
    def __init__(
        self,
        cfg: "FeedForwardConfig | ModelConfig",
        overrides: "FeedForwardConfig | None" = None,
    ) -> None:
        super().__init__()
        config = getattr(cfg, "transformer_feed_forward_config", cfg)
        self.cfg: FeedForwardConfig = self._overwrite_config(config, overrides)

        self.weighted_parameters_flag = self.cfg.weighted_parameters_flag
        self.num_layers = self.cfg.num_layers
        assert (
            self.num_layers is not None
            and self.num_layers >= 2
            and self.num_layers % 2 == 0
        ), "The number of layers should be at least 2"

        self.model = self._create_model(cfg)
        self._store_shape_attributes()

    def _create_model(self, config: "ModelConfig") -> Linear | Sequential:
        return LinearBlockStack(config).build_model()

    def _store_shape_attributes(self):
        self.batch_size = None
        self.sequence_length = None
        self.output_shape = None

    def forward(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        input_batch_matrix, skip_mask = self._ensure_correct_shape(
            input_batch, skip_mask
        )
        output_projection = self.model(input_batch_matrix)
        return self._revert_to_original_shape(output_projection)

    def _ensure_correct_shape(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        self.__resolve_output_shape(input_batch)
        input_batch = input_batch.reshape(self.batch_size * self.sequence_length, -1)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        return input_batch, skip_mask

    def __resolve_output_shape(self, input_batch: Tensor) -> None:
        input_shape = input_batch.shape
        if self.batch_size is not None:
            return
        if input_batch.dim() > 2:
            self.batch_size, self.sequence_length, _ = input_shape
            self.output_shape = [self.batch_size, self.sequence_length, -1]
            return
        self.sequence_length = 1
        self.batch_size, _ = input_shape
        self.output_shape = [self.batch_size, -1]

    def _revert_to_original_shape(self, output_projection: Tensor):
        return output_projection.view(self.output_shape)


@dataclass
class MixtureOfExpertsFeedForwardConfig(DataClassBase):
    weighted_parameters_flag: bool | None = field(
        default=None,
        metadata={
            "help": "When `True` the sepected parameters will be multiplied by their probs."
        },
    )


class MixtureOfExpertsFeedForward(Module):
    def __init__(
        self,
        cfg: "MixtureOfExpertsFeedForwardConfig | ModelConfig",
        overrides: "MixtureOfExpertsFeedForwardConfig | None" = None,
    ) -> None:
        super().__init__()
        config = getattr(cfg, "mixture_of_experts_config", cfg)
        self.cfg: MixtureOfExpertsFeedForwardConfig = self._overwrite_config(
            config, overrides
        )
        self.weighted_parameters_flag = self.cfg.weighted_parameters_flag

        self.router = RouterModel(cfg)
        self.sampler = SamplerModel(cfg)
        self.input_module = MixtureOfExperts(cfg)
        self.output_module = MixtureOfExperts(cfg, is_output_layer_flag=True)

        self.batch_size = None
        self.sequence_length = None
        self.output_shape = None

    def forward(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        input_batch_matrix, skip_mask = self._prepare_inputs(input_batch, skip_mask)
        logits = self.router.compute_logit_scores(input_batch_matrix)
        probabilities, indices, skip_mask, sampler_loss = (
            self.sampler.sample_probabilities_and_indices(logits, skip_mask)
        )
        input_projection, input_loss = self.input_module.compute_expert_outputs(
            input_batch_matrix, indices
        )
        output_projection, output_loss = self.output_module.compute_expert_outputs(
            input_projection, indices, probabilities
        )
        expert_mixture_output = output_projection.view(self.output_shape)
        loss = sampler_loss + input_loss + output_loss
        return expert_mixture_output, skip_mask, loss

    def _prepare_inputs(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        self.__resolve_output_shape(input_batch)
        input_batch = input_batch.reshape(self.batch_size * self.sequence_length, -1)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        return input_batch, skip_mask

    def __resolve_output_shape(self, input_batch: Tensor) -> None:
        input_shape = input_batch.shape
        if self.batch_size is not None:
            return
        if len(input_shape) > 2:
            self.batch_size, self.sequence_length, _ = input_shape
            self.output_shape = [self.batch_size, self.sequence_length, -1]
            return
        self.sequence_length = 1
        self.batch_size, _ = input_shape
        self.output_shape = [self.batch_size, -1]
