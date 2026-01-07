import torch

from torch import Tensor
from torch.nn import Sequential
from dataclasses import dataclass, field
from Emperor.sampler.model import SamplerModel
from Emperor.base.utils import ConfigBase, Module
from Emperor.adaptive.utils.routers import RouterModel
from Emperor.experts.utils.layers import MixtureOfExperts
from Emperor.linears.options import LinearLayerStackOptions
from Emperor.adaptive.options import AdaptiveLayerStackOptions
from Emperor.base.layer import Layer, LayerStackConfig
from Emperor.transformer.utils._validator import FeedForwardValidator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


@dataclass
class FeedForwardConfig(ConfigBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    layer_stack_option: "LinearLayerStackOptions | AdaptiveLayerStackOptions | None" = (
        field(
            default=None,
            metadata={"help": "Number of layers added to the router"},
        )
    )
    num_layers: int | None = field(
        default=None,
        metadata={"help": "Number of layers added to the router"},
    )


class FeedForward(Module):
    def __init__(
        self,
        cfg: "FeedForwardConfig | ModelConfig",
        overrides: "FeedForwardConfig | None" = None,
    ) -> None:
        super().__init__()
        self.cfg: FeedForwardConfig = self._overwrite_config(cfg, overrides)
        self.main_cfg = self._resolve_main_config(self.cfg, cfg)

        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.layer_stack_option = self.cfg.layer_stack_option.value
        self.num_layers = self.cfg.num_layers

        self.validator = FeedForwardValidator(self)
        self.model = self._create_model()
        self._store_shape_attributes()

    def _create_model(self) -> Layer | Sequential:
        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_layers=self.num_layers,
        )
        return self.layer_stack_option(self.main_cfg, overrides).build_model()

    def _store_shape_attributes(self):
        self.batch_size = None
        self.sequence_length = None
        self.output_shape = None

    def forward(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        input_batch, skip_mask = self._ensure_correct_shape(input_batch, skip_mask)
        projected_inputs = self.model(input_batch)
        if isinstance(projected_inputs, tuple):
            if len(projected_inputs) == 2:
                output, loss = projected_inputs
            else:
                output, skip_mask, loss = projected_inputs
            output = self._revert_to_original_shape(output)
            return output, loss
        return self._revert_to_original_shape(projected_inputs), torch.tensor(0.0)

    def _ensure_correct_shape(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if input_batch.dim() == 2:
            return input_batch, skip_mask
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
        if self.output_shape is None:
            return output_projection
        return output_projection.view(self.output_shape)


@dataclass
class MixtureOfExpertsFeedForwardConfig(ConfigBase):
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
