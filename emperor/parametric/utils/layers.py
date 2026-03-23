import torch

from enum import Enum
from torch import Tensor
from dataclasses import dataclass, field
from emperor.base.decorators import timer
from emperor.base.utils import ConfigBase, Module
from emperor.sampler.utils.samplers import SamplerConfig
from emperor.sampler.utils.routers import RouterConfig
from emperor.parametric.utils.mixtures.base import AdaptiveMixtureBase
from emperor.parametric.utils._validator import _AdaptiveParameterLayerValidator
from emperor.parametric.utils.mixtures.selectors import (
    AdaptiveWeightSelector,
    AdaptiveBiasSelector,
)
from emperor.augmentations.adaptive_parameters.model import (
    AdaptiveParameterBehaviour,
    AdaptiveParameterBehaviourConfig,
)
from emperor.parametric.utils.mixtures.options import (
    AdaptiveBiasOptions,
    AdaptiveWeightOptions,
)
from emperor.parametric.utils.handlers import (
    GeneratorParameterHandler,
    MatrixParameterHandler,
    VectorParameterHandler,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class AdaptiveRouterOptions(Enum):
    SHARED_ROUTER = 1
    INDEPENTENT_ROUTER = 2


@dataclass
class AdaptiveParameterLayerConfig(ConfigBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Input dimensionality for weight parameters."},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Output dimensionality for weight parameters."},
    )
    adaptive_weight_option: "AdaptiveWeightOptions | None" = field(
        default=None,
        metadata={
            "help": "Specifies options for generating weight parameters for individual input samples."
        },
    )
    adaptive_bias_option: "AdaptiveBiasOptions | None" = field(
        default=None,
        metadata={
            "help": "Specifies options for generating bias parameters for individual input samples."
        },
    )
    init_sampler_model_option: "AdaptiveRouterOptions | None" = field(
        default=None,
        metadata={
            "help": "When `True` the `RouterModel `and `SamplerModel` will be added to the current layer."
        },
    )
    time_tracker_flag: bool | None = field(
        default=None,
        metadata={
            "help": "When `True` it will generate bias parameters for each input sample."
        },
    )
    router_config: "RouterConfig | None" = field(
        default=None,
        metadata={"help": "Configuration for the router model."},
    )
    sampler_config: "SamplerConfig | None" = field(
        default=None,
        metadata={"help": "Configuration for the sampler model."},
    )
    adaptive_behaviour_config: "AdaptiveParameterBehaviourConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )


class AdaptiveParameterLayer(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterLayerConfig | ModelConfig",
        overrides: "AdaptiveParameterLayerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "parameter_generator_model_config", cfg)
        self.cfg: "AdaptiveParameterLayerConfig" = self._overwrite_config(
            config, overrides
        )
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.adaptive_weight_option = self.cfg.adaptive_weight_option
        self.adaptive_bias_option = self.cfg.adaptive_bias_option
        self.init_sampler_model_option = self.cfg.init_sampler_model_option
        self.time_tracker_flag = self.cfg.time_tracker_flag

        self.adaptive_behaviour_config = self.cfg.adaptive_behaviour_config
        self.router_config = self.cfg.router_config
        self.sampler_config = self.cfg.sampler_config

        self.validator = _AdaptiveParameterLayerValidator(self)
        self.adaptive_behaviour = self.__init_adaptive_behaviour()
        self.weight_parameter_model = self.__init_weight_model()
        self.bias_parameter_model = self.__init_bias_model()

        self.parameter_handler = self.get_parameter_handler()
        self.weights_router, self.bias_router, self.sampler = (
            self.parameter_handler.build_sampler_models()
        )

    def __init_adaptive_behaviour(self):
        if self.adaptive_behaviour_config is None:
            return None
        overrides = AdaptiveParameterBehaviourConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return AdaptiveParameterBehaviour(self.adaptive_behaviour_config, overrides)

    def __init_weight_model(self) -> AdaptiveMixtureBase:
        overrides = AdaptiveParameterLayerConfig(
            input_dim=self.input_dim, output_dim=self.output_dim
        )
        return AdaptiveWeightSelector(self.cfg, overrides).build_model()

    def __init_bias_model(self) -> AdaptiveMixtureBase | None:
        from emperor.parametric.utils.mixtures.options import AdaptiveBiasOptions

        if self.adaptive_bias_option == AdaptiveBiasOptions.DISABLED:
            return None

        overrides = AdaptiveParameterLayerConfig(
            input_dim=self.input_dim, output_dim=self.output_dim
        )
        return AdaptiveBiasSelector(self.cfg, overrides).build_model()

    def get_parameter_handler(
        self,
    ) -> "VectorParameterHandler | MatrixParameterHandler | GeneratorParameterHandler":
        match self.adaptive_weight_option:
            case AdaptiveWeightOptions.VECTOR:
                return VectorParameterHandler(self.cfg)
            case AdaptiveWeightOptions.MATRIX:
                return MatrixParameterHandler(self.cfg)
            case AdaptiveWeightOptions.GENERATOR:
                return GeneratorParameterHandler(self.cfg)
            case _:
                raise ValueError(
                    f"Invalid adaptive_weight_option provided: {self.adaptive_weight_option}. Expected one of `AdaptiveWeightOptions`"
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
        return self._compute_layer_output(input, skip_mask)

    def _compute_layer_output(
        self,
        input: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        weight_parameters, biase_parameters, loss = self._generate_parameters(
            input, skip_mask
        )
        output = self.adaptive_behaviour.compute_adaptive_parameters(
            self._compute_affine_transformation_callback,
            weight_parameters,
            biase_parameters,
            input,
        )

        updated_skip_mask = self.__get_updated_skip_mask()
        total_layer_loss = self.__get_total_layer_loss()
        total_layer_loss = total_layer_loss + loss

        return output, updated_skip_mask, total_layer_loss

    def _generate_parameters(
        self,
        input: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        weight_probabilities, weight_indices, skip_mask, weight_loss = (
            self.__sample_weight_probabilities_and_indices(input, skip_mask=skip_mask)
        )
        weight_parameters, weight_parameters_loss = self.__generate_weight_parameters(
            weight_probabilities, weight_indices, input
        )
        bias_parameters, bias_parameters_loss = self.__generate_bias_parameters(
            input,
            skip_mask,
            weight_probabilities,
            weight_indices,
        )

        loss = weight_loss + weight_parameters_loss + bias_parameters_loss
        return weight_parameters, bias_parameters, loss

    def __sample_weight_probabilities_and_indices(
        self,
        input: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor]:
        if self.weights_router is None:
            return None, None, None, torch.tensor(0.0)
        logits = self.weights_router.compute_logit_scores(input)
        probabilities, indices, skip_mask, loss = (
            self.sampler.sample_probabilities_and_indices(logits, skip_mask)
        )
        return probabilities, indices, skip_mask, loss

    def __generate_weight_parameters(
        self, probabilities: Tensor | None, indices: Tensor | None, input: Tensor
    ) -> tuple[Tensor, Tensor]:
        loss = torch.tensor(0.0)
        weight_parameters = self.weight_parameter_model.compute_mixture(
            probabilities, indices, input
        )
        if isinstance(weight_parameters, tuple):
            weight_parameters, loss = weight_parameters

        return weight_parameters, loss

    def __generate_bias_parameters(
        self,
        input: Tensor,
        skip_mask: Tensor | None,
        probabilities: Tensor | None,
        indices: Tensor | None,
    ) -> tuple[Tensor | None, Tensor]:
        loss = torch.tensor(0.0)
        if self.bias_parameter_model is None:
            return None, loss

        indepentent_option = AdaptiveRouterOptions.INDEPENTENT_ROUTER
        if self.init_sampler_model_option == indepentent_option:
            probabilities, indices, bias_skip_mask, bias_loss = (
                self.__sample_bias_probabilities_and_indices(input, skip_mask=skip_mask)
            )
            loss = loss + bias_loss

        bias_parameters = self.bias_parameter_model.compute_mixture(
            probabilities, indices, input
        )
        if isinstance(bias_parameters, tuple):
            bias_parameters, bias_parameters_loss = bias_parameters
            loss = loss + bias_parameters_loss

        return bias_parameters, loss

    def __sample_bias_probabilities_and_indices(
        self,
        input: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor]:
        if self.bias_router is None:
            return None, None, None, torch.tensor(0.0)
        logits = self.bias_router.compute_logit_scores(input)
        probabilities, indices, skip_mask, loss = (
            self.sampler.sample_probabilities_and_indices(logits, skip_mask)
        )
        return probabilities, indices, skip_mask, loss

    def _compute_affine_transformation_callback(
        self, weights: Tensor, bias: Tensor | None, input: Tensor
    ) -> Tensor:
        output = self.__apply_generated_weights(input, weights)
        return self.__apply_generated_biases(output, bias)

    def __apply_generated_weights(
        self,
        input: Tensor,
        generated_weights: Tensor,
    ) -> Tensor:
        return torch.einsum("bi,bij->bj", input, generated_weights)

    def __apply_generated_biases(
        self,
        inputs: Tensor,
        generated_biases: Tensor | None = None,
    ) -> Tensor:
        if self.adaptive_bias_option == AdaptiveBiasOptions.DISABLED:
            return inputs
        return inputs + generated_biases

    def __get_updated_skip_mask(self):
        return self.sampler.get_updated_skip_mask()

    def __get_total_layer_loss(self) -> Tensor:
        return self.sampler.get_auxiliary_loss()
