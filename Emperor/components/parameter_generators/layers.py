from dataclasses import dataclass, field
import torch
from torch import Tensor
from Emperor.base.decorators import timer
from Emperor.base.utils import Module, DataClassBase
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
    ) -> tuple[Tensor, Tensor | None]:
        weight_probabilities, weight_indices = self._compute_probabilities_and_indices(
            input_batch
        )
        bias_probabilities, bias_indices = self._compute_bias_probabilities_and_indices(
            input_batch
        )
        weight_parameters, bias_parameters = self.mixture.compute_mixture(
            weight_probabilities, weight_indices, bias_probabilities, bias_indices
        )

        return weight_parameters, bias_parameters

    def _compute_bias_probabilities_and_indices(
        self,
        input_batch: Tensor,
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.bias_parameters_flag:
            return self._compute_probabilities_and_indices(
                input_batch, self.bias_parameters_flag
            )
        return None, None

    def _compute_probabilities_and_indices(
        self,
        input_batch: Tensor,
        compute_bias_flag: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        logits = self._compute_logits(input_batch, compute_bias_flag)
        probabilities, indices, _ = self.sampler.sample_probabilities_and_indices(
            logits
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
    ) -> tuple[Tensor, Tensor | None]:
        logits = self._compute_logits(input_batch, compute_bias_flag)
        input_dim, batch_size, depth_dim = logits.shape
        logits = logits.view(-1, depth_dim)

        probabilities, indices, _ = self.sampler.sample_probabilities_and_indices(
            logits
        )
        probabilities = probabilities.reshape(input_dim, batch_size, -1)
        indices = indices.reshape(input_dim, batch_size, -1)

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
