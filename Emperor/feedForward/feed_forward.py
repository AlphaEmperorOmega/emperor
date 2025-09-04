from torch import Tensor
from torch import nn
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
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Input dimension of the first `Linear` layer"},
    )
    hidden_dim: int | None = field(
        default=None,
        metadata={"help": "Dimension of the hidden `Linear` layers"},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Output dimension of the output `Linear` layer"},
    )
    num_layers: int | None = field(
        default=None,
        metadata={"help": "Number of layers in the model"},
    )
    activation: nn.Linear | None = field(
        default=None,
        metadata={"help": "Activation function or layer to use"},
    )
    layer_norm_flag: int | None = field(
        default=None,
        metadata={"help": "Flag indicating whether to apply layer normalization"},
    )
    linear_model: nn.Module | None = field(
        default=None,
        metadata={"help": "Linear model module used for output transformation"},
    )
    weighted_parameters_flag: bool | None = field(
        default=None,
        metadata={
            "help": "When `True` the sepected parameters will be multiplied by their probs."
        },
    )


class FeedForward(Module):
    def __init__(
        self,
        cfg: "FeedForwardConfig | ModelConfig",
        overrides: "FeedForwardConfig | None" = None,
    ) -> None:
        super().__init__()
        config = getattr(cfg, "mixture_of_experts_config", cfg)
        self.cfg: FeedForwardConfig = self._overwrite_config(config, overrides)

        self.input_dim = self.cfg.input_dim
        self.hidden_dim = self.cfg.hidden_dim
        self.output_dim = self.cfg.output_dim
        self.num_layers = self.cfg.num_layers
        self.activation = self.cfg.activation()
        self.layer_norm_flag = self.cfg.layer_norm_flag
        self.linear_model = self.cfg.linear_model
        self.weighted_parameters_flag = self.cfg.weighted_parameters_flag

        self.model = self._create_model(cfg)
        assert self.num_layers is not None and self.num_layers >= 2, (
            "The number of layers should be at least 2"
        )
        self._store_shape_attributes()

    def _store_shape_attributes(self):
        self.batch_size = None
        self.sequence_length = None
        self.output_shape = None

    def _create_multi_layer_model(self) -> Linear | Sequential:
        cfg = LinearBlockStackConfig(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=self.num_layers,
            activation=self.activation,
            layer_norm_flag=self.layer_norm_flag,
            linear_model=self.linear_model,
        )
        return LinearBlockStack(cfg).build_model()

    def forward(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        input_batch_matrix, skip_mask = self._prepare_inputs(input_batch, skip_mask)
        output_projection = self.model(input_batch_matrix)
        return output_projection.view(self.output_shape)

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
        if input_batch.dim() > 2:
            self.batch_size, self.sequence_length, _ = input_shape
            self.output_shape = [self.batch_size, self.sequence_length, -1]
            return
        self.sequence_length = 1
        self.batch_size, _ = input_shape
        self.output_shape = [self.batch_size, -1]


@dataclass
class MixtureOfExpertsFeedForwardConfig(DataClassBase):
    weighted_parameters_flag: bool | None = field(
        default=None,
        metadata={
            "help": "When `True` the sepected parameters will be multiplied by their probs."
        },
    )


class MixtureOfExpertsFeedForward(FeedForward):
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
