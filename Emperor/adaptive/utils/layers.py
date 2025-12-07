import torch

from torch import Tensor
from Emperor.adaptive.utils.routers import VectorRouterModel
from Emperor.base.utils import Module
from dataclasses import dataclass, field
from Emperor.base.decorators import timer
from Emperor.sampler.model import SamplerModel
from Emperor.behaviours.utils.enums import DynamicDepthOptions, DynamicDiagonalOptions
from Emperor.adaptive.utils.mixtures.vector import (
    VectorBiasMixture,
    VectorWeightsMixture,
)
from Emperor.behaviours.model import (
    AdaptiveParameterModel,
    AdaptiveParameterModelConfig,
)
from Emperor.adaptive.utils.mixture import (
    GeneratorMixture,
    MatrixMixture,
    VectorMixture,
)
from Emperor.sampler.utils.routers import RouterModel


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig
    from Emperor.linears.options import LinearLayerOptions
    from Emperor.linears.options import LinearLayerStackOptions


@dataclass
class ParameterLayerConfig(AdaptiveParameterModelConfig):
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
    linear_layer_type: "LinearLayerOptions | LinearLayerStackOptions | None" = field(
        default=None,
        metadata={"help": ""},
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

        self.input_dim = cfg.mixture_model_config.input_dim
        self.output_dim = cfg.mixture_model_config.output_dim
        self.bias_parameters_flag = self.cfg.bias_parameters_flag
        self.time_tracker_flag = self.cfg.time_tracker_flag
        self.diagonal_option = self.cfg.diagonal_option
        self.generator_depth = self.cfg.generator_depth
        self.adaptive_behaviour = AdaptiveParameterModel(self.cfg)
        self.__validators()
        self._validate_requiered_child_class_attributes()

    def __validators(self):
        assert self.generator_depth == DynamicDepthOptions.DISABLED, (
            "The generator depth must be set to 'DISABLED' `ParameterLayer` models."
        )
        assert self.diagonal_option != DynamicDiagonalOptions.DISABLED, (
            "The diagonal option must not be set to 'DISABLED' for `ParameterLayer` models."
        )

    def _validate_requiered_child_class_attributes(self) -> None:
        required_attributes = [
            "weight_mixture",
            "bias_mixture",
        ]
        for required_attribute in required_attributes:
            if not hasattr(self, required_attribute):
                raise AttributeError(
                    f"Required attribute is missing in the child class: {required_attribute}."
                )

    def forward(
        self,
        input: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        if self.time_tracker_flag:
            return self._track_layer_output_time(input, skip_mask)
        return self._compute_layer_output(input, skip_mask)

    @timer
    def _track_layer_output_time(
        self,
        input: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        return self.compute_layer_output(input, skip_mask)

    def _compute_layer_output(
        self,
        input: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        weights, biases = self._generate_parameters(input, skip_mask)
        output = self.adaptive_behaviour.compute_adaptive_parameters(
            self._compute_affine_transformation_callback, weights, biases, input
        )

        updated_skip_mask = self.__get_updated_skip_mask()
        total_layer_loss = self.__get_total_layer_loss()

        return output, updated_skip_mask, total_layer_loss

    def _generate_weight_parameters(
        self,
        input: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        weight_probabilities, weight_indices = self._compute_probabilities_and_indices(
            input, skip_mask=skip_mask
        )
        bias_probabilities, bias_indices = self._compute_bias_probabilities_and_indices(
            input, skip_mask
        )
        weight_parameters = self.weight_mixture.compute_mixture(
            weight_probabilities, weight_indices
        )
        bias_parameters = self.bias_mixture.compute_mixture(
            bias_probabilities, bias_indices
        )
        return weight_parameters, bias_parameters

    def _generate_bias_parameters(
        self,
        input: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        weight_probabilities, weight_indices = self._compute_probabilities_and_indices(
            input, skip_mask=skip_mask
        )
        bias_probabilities, bias_indices = self._compute_bias_probabilities_and_indices(
            input, skip_mask
        )
        weight_parameters = self.weight_mixture.compute_mixture(
            weight_probabilities, weight_indices
        )
        bias_parameters = self.bias_mixture.compute_mixture(
            bias_probabilities, bias_indices
        )
        return weight_parameters, bias_parameters

    def _generate_parameters(
        self,
        input: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        weight_probabilities, weight_indices = self._compute_probabilities_and_indices(
            input, skip_mask=skip_mask
        )
        bias_probabilities, bias_indices = self._compute_bias_probabilities_and_indices(
            input, skip_mask
        )
        weight_parameters = self.weight_mixture.compute_mixture(
            weight_probabilities, weight_indices
        )
        bias_parameters = self.bias_mixture.compute_mixture(
            bias_probabilities, bias_indices
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
    ):
        if compute_bias_flag:
            logits = self._compute_logits(input_batch, compute_bias_flag)
            return self._sample_probabilities_and_indices(
                logits, skip_mask, compute_bias_flag
            )
        logits = self._compute_logits(input_batch)
        return self._sample_probabilities_and_indices(logits, skip_mask)

    def _compute_logits(
        self,
        input_batch: Tensor,
        compute_bias_flag: bool = False,
    ) -> Tensor:
        if compute_bias_flag:
            if self.bias_router is None or self.bias_sampler is None:
                raise RuntimeError("Bias router is not initialized.")
            return self.bias_router.compute_logit_scores(input_batch)
        return self.weight_router.compute_logit_scores(input_batch)

    def _sample_probabilities_and_indices(
        self,
        logits: Tensor,
        skip_mask: Tensor | None = None,
        compute_bias_flag: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        if compute_bias_flag:
            if self.bias_sampler is None:
                raise RuntimeError("Bias sampler is not initialized.")
            probabilities, selected_indices, _, _ = (
                self.bias_sampler.sample_probabilities_and_indices(logits, skip_mask)
            )
            return probabilities, selected_indices

        probabilities, selected_indices, _, _ = (
            self.weight_sampler.sample_probabilities_and_indices(logits, skip_mask)
        )
        return probabilities, selected_indices

    def _compute_affine_transformation_callback(
        self, weights: Tensor, bias: Tensor | None, input: Tensor
    ) -> Tensor:
        output = self.__apply_generated_weights(input, weights)
        return self.__apply_generated_biases(output, bias)

    def __apply_generated_weights(
        self,
        input_batch: Tensor,
        generated_weights: Tensor,
    ) -> Tensor:
        return torch.einsum("bi,bij->bj", input_batch, generated_weights)

    def __apply_generated_biases(
        self,
        weighted_inputs: Tensor,
        generated_biases: Tensor | None = None,
    ) -> Tensor:
        if self.bias_parameters_flag:
            return weighted_inputs + generated_biases
        return weighted_inputs

    def __get_updated_skip_mask(self):
        return self.weight_sampler.get_updated_skip_mask()

    def __get_total_layer_loss(self) -> Tensor:
        weight_loss = self.weight_sampler.get_auxiliary_loss()
        bias_loss = self.bias_sampler.get_auxiliary_loss()
        return weight_loss + bias_loss


class VectorParameterLayer(ParameterLayerBase):
    def __init__(
        self,
        cfg: "ParameterLayerConfig | ModelConfig",
        overrides: "ParameterLayerConfig | None" = None,
    ):
        super().__init__(cfg, overrides)

        self.weight_router = VectorRouterModel(cfg)
        self.bias_router = self._init_bias_router_model(cfg)
        self.weight_sampler = SamplerModel(cfg)
        self.bias_sampler = SamplerModel(cfg)
        self.weight_mixture = VectorWeightsMixture(cfg)
        self.bias_mixture = self._init_bias_mixture_model(cfg)

    def _init_bias_router_model(self, cfg: "ModelConfig") -> VectorRouterModel | None:
        if not self.bias_parameters_flag:
            return None
        return VectorRouterModel(
            cfg,
            bias_parameters_flag=self.bias_parameters_flag,
            bias_output_dim=self.mixture.output_dim,
        )

    def _init_bias_mixture_model(self, cfg: "ModelConfig") -> VectorBiasMixture | None:
        if not self.bias_parameters_flag:
            return None
        return VectorBiasMixture(cfg)

    def _compute_probabilities_and_indices(
        self,
        input_batch: Tensor,
        compute_bias_flag: bool = False,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        logits = self._compute_logits(input_batch, compute_bias_flag)
        input_dim, batch_size, depth_dim = logits.shape
        logits = logits.view(-1, depth_dim)

        probabilities, indices = self._sample_probabilities_and_indices(
            logits, skip_mask, compute_bias_flag
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
        self.weight_sampler: SamplerModel = SamplerModel(cfg)
        self.bias_sampler: SamplerModel = SamplerModel(cfg)
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
        self.weight_sampler: SamplerModel = SamplerModel(cfg)
        self.bias_sampler: SamplerModel = SamplerModel(cfg)
        self.mixture: GeneratorMixture = GeneratorMixture(cfg)

    def _init_bias_router_model(self, cfg: "ModelConfig") -> RouterModel | None:
        if self.bias_parameters_flag:
            return RouterModel(cfg)
        return None

    def _generate_parameters(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        weight_probabilities, weight_indices = self._compute_probabilities_and_indices(
            input_batch,
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
