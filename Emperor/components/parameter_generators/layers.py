from dataclasses import dataclass, field
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from Emperor.base.decorators import timer
from Emperor.base.utils import Module, DataClassBase
from Emperor.components.parameter_generators.utils.behaviours import (
    DynamicDiagonalParametersBehaviour,
)

from Emperor.components.parameter_generators.utils.samplers import SamplerModel
from Emperor.components.parameter_generators.utils.mixture import (
    GeneratorMixture,
    MatrixMixture,
    VectorMixture,
)
from Emperor.components.parameter_generators.utils.routers import (
    RouterModel,
    VectorRouterModel,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


@dataclass
class ParameterLayerConfig(DataClassBase):
    bias_parameters_flag: bool | None = field(
        default=None,
        metadata={
            "help": "When `True` it will generate bias parameters for each input sample."
        },
    )
    time_tracker_flag: bool | None = field(
        default=None,
        metadata={
            "help": "When `True` it will generate bias parameters for each input sample."
        },
    )


class ParameterLayerBase(Module):
    def __init__(
        self,
        cfg: "ParameterLayerConfig | ModelConfig",
        overrides: "ParameterLayerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "parameter_generator_model_config", cfg)
        self.cfg: "ParameterLayerConfig" = self._overwrite_config(config, overrides)

        self.bias_parameters_flag = self.cfg.bias_parameters_flag
        self.time_tracker_flag = self.cfg.time_tracker_flag

    def forward(
        self,
        input_batch: Tensor,
    ) -> Tensor:
        if self.time_tracker_flag:
            return self._track_layer_output_time(input_batch)
        return self._compute_layer_output(input_batch)

    @timer
    def _track_layer_output_time(
        self,
        input_batch: Tensor,
    ) -> Tensor:
        return self.compute_layer_output(input_batch)

    def _compute_layer_output(self, input_batch: Tensor) -> Tensor:
        generated_weights, generated_biases = self._generate_parameters(input_batch)

        output = self._apply_generated_weights(input_batch, generated_weights)
        output = self._apply_generated_biases(output, generated_biases)

        return output

    def _apply_generated_weights(
        self,
        input_batch: Tensor,
        generated_weights: Tensor,
    ) -> Tensor:
        return torch.einsum("bi,bij->bj", input_batch, generated_weights)

    def _apply_generated_biases(
        self,
        weighted_inputs: Tensor,
        generated_biases: Tensor | None = None,
    ) -> Tensor:
        if self.bias_parameters_flag:
            return weighted_inputs + generated_biases
        return weighted_inputs

    def _generate_parameters(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        weight_probabilities, weight_indices = self._compute_probabilities_and_indices(
            input_batch, False, skip_mask
        )
        bias_probabilities, bias_indices = self._compute_bias_probabilities_and_indices(
            input_batch, skip_mask
        )
        weight_parameters, bias_parameters = self.mixture.compute_mixture(
            weight_probabilities, weight_indices, bias_probabilities, bias_indices
        )

        return weight_parameters, bias_parameters

    def _compute_bias_probabilities_and_indices(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.bias_parameters_flag:
            return self._compute_probabilities_and_indices(
                input_batch, self.bias_parameters_flag, skip_mask
            )
        return None, None

    def _compute_probabilities_and_indices(
        self,
        input_batch: Tensor,
        compute_bias_flag: bool = False,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        logits = self._compute_logits(input_batch, compute_bias_flag)
        probabilities, indices, _ = self.sampler.sample_probabilities_and_indices(
            logits, skip_mask
        )

        return probabilities, indices

    def _compute_logits(
        self,
        input_batch: Tensor,
        compute_bias_flag: bool = False,
    ):
        if compute_bias_flag:
            if self.bias_router is None:
                raise RuntimeError("Bias router is not initialized.")
            return self.bias_router.compute_logit_scores(input_batch)
        return self.weight_router.compute_logit_scores(input_batch)


class DefaultLinearLayer(ParameterLayerBase):
    def __init__(
        self,
        cfg: "ModelConfig",
        overrides: "ParameterLayerConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        self.cfg = cfg
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.output_dim
        self.weight_params, self.bias_params = self.__init_parameter_banks()

    def __init_parameter_banks(self):
        weight_shape = (self.input_dim, self.output_dim)
        bias_shape = (self.output_dim,)
        weight_params = self._init_parameter_bank(weight_shape)
        bias_params = None
        if self.bias_parameters_flag:
            bias_params = self._init_parameter_bank(bias_shape, nn.init.zeros_)
        return weight_params, bias_params

    def _compute_layer_output(self, input_batch: Tensor) -> Tensor:
        return F.linear(input_batch, self.weight_params.T, self.bias_params)


class DynamicDiagonalLinearLayer(DefaultLinearLayer):
    def __init__(
        self,
        cfg: "ModelConfig",
        anti_diagonal_flag: bool = False,
    ):
        super().__init__(cfg)
        self.diagonal_decorator = DynamicDiagonalParametersBehaviour(
            self.weight_params,
            self.bias_params,
            anti_diagonal_flag,
        )

    def _compute_layer_output(self, input_batch: Tensor) -> Tensor:
        return self.diagonal_decorator(input_batch)


class VectorParameterLayer(ParameterLayerBase):
    def __init__(
        self,
        cfg: "ParameterLayerConfig | ModelConfig",
        overrides: "ParameterLayerConfig | None" = None,
    ):
        super().__init__(cfg, overrides)

        self.weight_router: VectorRouterModel = VectorRouterModel(cfg)
        self.bias_router = self._init_bias_router_model(cfg)
        self.sampler: SamplerModel = SamplerModel(cfg)
        self.mixture: VectorMixture = VectorMixture(cfg)

    def _init_bias_router_model(self, cfg: "ModelConfig") -> VectorRouterModel | None:
        if self.bias_parameters_flag:
            return VectorRouterModel(cfg, self.bias_parameters_flag)
        return None

    def _compute_probabilities_and_indices(
        self,
        input_batch: Tensor,
        compute_bias_flag: bool = False,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        logits = self._compute_logits(input_batch, compute_bias_flag)
        input_dim, batch_size, depth_dim = logits.shape
        logits = logits.view(-1, depth_dim)

        probabilities, indices, _ = self.sampler.sample_probabilities_and_indices(
            logits, skip_mask
        )

        probabilities_shape = (input_dim, batch_size)
        indices_shape = (input_dim, batch_size)
        if self.mixture.top_k > 1:
            probabilities_shape = (input_dim, batch_size, -1)
            indices_shape = (input_dim, batch_size, -1)

        probabilities = probabilities.reshape(probabilities_shape)
        if indices is not None:
            indices = indices.reshape(indices_shape)

        return probabilities, indices


class MatrixParameterLayer(ParameterLayerBase):
    def __init__(
        self,
        cfg: "ParameterLayerConfig | ModelConfig",
        overrides: "ParameterLayerConfig | None" = None,
    ):
        super().__init__(cfg, overrides)

        self.weight_router: RouterModel = RouterModel(cfg)
        self.bias_router = self._init_bias_router_model(cfg)
        self.sampler: SamplerModel = SamplerModel(cfg)
        self.mixture: MatrixMixture = MatrixMixture(cfg)

    def _init_bias_router_model(self, cfg: "ModelConfig") -> RouterModel | None:
        if self.bias_parameters_flag:
            return RouterModel(cfg)
        return None


class GeneratorParameterLayer(ParameterLayerBase):
    def __init__(
        self,
        cfg: "ParameterLayerConfig | ModelConfig",
        overrides: "ParameterLayerConfig | None" = None,
    ):
        super().__init__(cfg, overrides)

        self.weight_router: RouterModel = RouterModel(cfg)
        self.bias_router = self._init_bias_router_model(cfg)
        self.sampler: SamplerModel = SamplerModel(cfg)
        self.mixture: GeneratorMixture = GeneratorMixture(cfg)

    def _init_bias_router_model(self, cfg: "ModelConfig") -> RouterModel | None:
        if self.bias_parameters_flag:
            return RouterModel(cfg)
        return None

    def _generate_parameters(
        self,
        input_batch: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        weight_probabilities, weight_indices = self._compute_probabilities_and_indices(
            input_batch
        )
        bias_probabilities, bias_indices = self._compute_bias_probabilities_and_indices(
            input_batch
        )
        weight_parameters, bias_parameters = self.mixture.compute_mixture(
            weight_probabilities,
            weight_indices,
            bias_probabilities,
            bias_indices,
            input_batch,
        )

        return weight_parameters, bias_parameters
