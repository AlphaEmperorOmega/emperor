from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.augmentations.adaptive_parameters.config import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.base.module import Module
from emperor.parametric.core._validator import ParametricLayerValidator
from emperor.parametric.core.config import AdaptiveRouterOptions, ParametricLayerConfig
from emperor.parametric.core.handlers import (
    GeneratorParameterHandler,
    MatrixParameterHandler,
    VectorParameterHandler,
)
from emperor.parametric.core.mixtures.config import (
    AdaptiveMixtureConfig,
    GeneratorBiasMixtureConfig,
    GeneratorWeightsMixtureConfig,
    MatrixWeightsMixtureConfig,
    VectorWeightsMixtureConfig,
)

if TYPE_CHECKING:
    from emperor.config import ModelConfig
    from emperor.parametric.core.mixtures.base import AdaptiveMixtureBase


class ParametricLayer(Module):
    VALIDATOR = ParametricLayerValidator

    def __init__(
        self,
        cfg: "ParametricLayerConfig | ModelConfig",
        overrides: "ParametricLayerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "parameter_generator_model_config", cfg)
        self.cfg: ParametricLayerConfig = self._override_config(config, overrides)

        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.weight_mixture_config = self.cfg.weight_mixture_config
        self.bias_mixture_config = self.cfg.bias_mixture_config
        self.routing_initialization_mode = self.cfg.routing_initialization_mode
        self.router_config = self.cfg.router_config
        self.sampler_config = self.cfg.sampler_config
        self.adaptive_augmentation_config = self.cfg.adaptive_augmentation_config

        self.VALIDATOR.validate(self)
        self.adaptive_augmentation_model = self.__init_adaptive_augmentation()
        self.weight_mixture_model = self.__init_weight_model()
        self.bias_mixture_model = self.__init_bias_model()
        self.parameter_handler = self.get_parameter_handler()
        self.weights_router, self.bias_router, self.sampler = (
            self.parameter_handler.build_sampler_models()
        )

    def __init_adaptive_augmentation(self):
        overrides = AdaptiveParameterAugmentationConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return self.adaptive_augmentation_config.build(overrides)

    def __init_weight_model(self) -> "AdaptiveMixtureBase":
        overrides = AdaptiveMixtureConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return self.weight_mixture_config.build(overrides)

    def __init_bias_model(self) -> "AdaptiveMixtureBase | None":
        if self.bias_mixture_config is None:
            return None
        overrides = AdaptiveMixtureConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return self.bias_mixture_config.build(overrides)

    def get_parameter_handler(
        self,
    ) -> "VectorParameterHandler | MatrixParameterHandler | GeneratorParameterHandler":
        if isinstance(self.weight_mixture_config, VectorWeightsMixtureConfig):
            return VectorParameterHandler(self.cfg)
        if isinstance(self.weight_mixture_config, MatrixWeightsMixtureConfig):
            return MatrixParameterHandler(self.cfg)
        if isinstance(self.weight_mixture_config, GeneratorWeightsMixtureConfig):
            return GeneratorParameterHandler(self.cfg)
        raise TypeError(
            "weight_mixture_config must be a supported parametric weight config, "
            f"got {type(self.weight_mixture_config).__name__}."
        )

    def forward(
        self,
        input: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        self.VALIDATOR.validate_forward_inputs(input, self.input_dim)
        return self._compute_layer_output(input, skip_mask)

    def _compute_layer_output(
        self,
        input: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        weight_parameters, bias_parameters, skip_mask, loss = self._generate_parameters(
            input, skip_mask
        )
        output = self.adaptive_augmentation_model(
            self._compute_affine_transformation_callback,
            weight_parameters,
            bias_parameters,
            input,
        )
        return output, skip_mask, loss

    def _generate_parameters(
        self,
        input: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor]:
        weight_probabilities, weight_indices, skip_mask, weight_loss = (
            self.__sample_weight_probabilities_and_indices(input, skip_mask=skip_mask)
        )
        weight_parameters, weight_parameters_loss = self.__generate_weight_parameters(
            weight_probabilities, weight_indices, input
        )
        bias_parameters, skip_mask, bias_parameters_loss = (
            self.__generate_bias_parameters(
                input,
                skip_mask,
                weight_probabilities,
                weight_indices,
            )
        )

        loss = weight_loss + weight_parameters_loss + bias_parameters_loss
        return weight_parameters, bias_parameters, skip_mask, loss

    def __sample_weight_probabilities_and_indices(
        self,
        input: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor]:
        if self.weights_router is None:
            return None, None, skip_mask, input.new_zeros(())
        logits = self.weights_router.compute_logit_scores(input)
        return self.__sample_probabilities_and_indices(logits, input, skip_mask)

    def __sample_bias_probabilities_and_indices(
        self,
        input: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor]:
        if self.bias_router is None:
            return None, None, skip_mask, input.new_zeros(())
        logits = self.bias_router.compute_logit_scores(input)
        return self.__sample_probabilities_and_indices(logits, input, skip_mask)

    def __sample_probabilities_and_indices(
        self,
        logits: Tensor,
        input: Tensor,
        skip_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor]:
        if logits.dim() == 3:
            return self.__sample_vector_probabilities_and_indices(
                logits, input, skip_mask
            )
        probabilities, indices, skip_mask, loss = (
            self.sampler.sample_probabilities_and_indices(logits, skip_mask)
        )
        return probabilities, indices, skip_mask, loss

    def __sample_vector_probabilities_and_indices(
        self,
        logits: Tensor,
        input: Tensor,
        skip_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor]:
        batch_size, vector_dim, num_experts = logits.shape
        flat_logits = logits.reshape(batch_size * vector_dim, num_experts)
        flat_skip_mask = self.__flatten_vector_skip_mask(
            skip_mask, batch_size, vector_dim
        )
        probabilities, indices, flat_skip_mask, loss = (
            self.sampler.sample_probabilities_and_indices(
                flat_logits, flat_skip_mask
            )
        )
        probabilities = self.__reshape_vector_sample(
            probabilities, batch_size, vector_dim
        )
        indices = self.__reshape_vector_sample(indices, batch_size, vector_dim)
        skip_mask = self.__unflatten_vector_skip_mask(
            flat_skip_mask, batch_size, vector_dim
        )
        return probabilities, indices, skip_mask, loss

    def __flatten_vector_skip_mask(
        self,
        skip_mask: Tensor | None,
        batch_size: int,
        vector_dim: int,
    ) -> Tensor | None:
        if skip_mask is None:
            return None
        mask = skip_mask.reshape(batch_size, -1)
        if mask.shape[-1] == 1:
            mask = mask.expand(batch_size, vector_dim)
        return mask.reshape(batch_size * vector_dim, 1)

    def __unflatten_vector_skip_mask(
        self,
        skip_mask: Tensor | None,
        batch_size: int,
        vector_dim: int,
    ) -> Tensor | None:
        if skip_mask is None:
            return None
        mask = skip_mask.reshape(batch_size, vector_dim, -1)
        return mask.amin(dim=1)

    def __reshape_vector_sample(
        self,
        sample: Tensor | None,
        batch_size: int,
        vector_dim: int,
    ) -> Tensor | None:
        if sample is None:
            return None
        if sample.dim() == 1:
            return sample.reshape(batch_size, vector_dim).transpose(1, 0)
        return sample.reshape(batch_size, vector_dim, -1).transpose(1, 0)

    def __generate_weight_parameters(
        self,
        probabilities: Tensor | None,
        indices: Tensor | None,
        input: Tensor,
    ) -> tuple[Tensor, Tensor]:
        loss = input.new_zeros(())
        weight_parameters = self.weight_mixture_model.compute_mixture(
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
    ) -> tuple[Tensor | None, Tensor | None, Tensor]:
        loss = input.new_zeros(())
        if self.bias_mixture_model is None:
            return None, skip_mask, loss

        if (
            self.routing_initialization_mode
            == AdaptiveRouterOptions.INDEPENDENT_ROUTER
        ):
            if self.bias_router is not None:
                probabilities, indices, skip_mask, bias_loss = (
                    self.__sample_bias_probabilities_and_indices(
                        input, skip_mask=skip_mask
                    )
                )
                loss = loss + bias_loss
            elif isinstance(self.bias_mixture_config, GeneratorBiasMixtureConfig):
                probabilities, indices = None, None

        bias_parameters = self.bias_mixture_model.compute_mixture(
            probabilities, indices, input
        )
        if isinstance(bias_parameters, tuple):
            bias_parameters, bias_parameters_loss = bias_parameters
            loss = loss + bias_parameters_loss

        return bias_parameters, skip_mask, loss

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
        if generated_biases is None:
            return inputs
        return inputs + generated_biases
