"""Rank-space modulation with independently owned input and output factors."""

from dataclasses import replace

import torch
from torch import Tensor

from emperor.augmentations.adaptive_parameters._options import (
    LowRankFactorSourceOptions,
)
from emperor.augmentations.adaptive_parameters._weights.base import (
    DynamicWeightAbstract,
)
from emperor.augmentations.adaptive_parameters._weights.config import (
    DiagonallyModulatedLowRankDynamicWeightConfig,
)
from emperor.augmentations.adaptive_parameters._weights.depth_mapping import (
    DepthMappingHandlerConfig,
    DepthMappingLayerStack,
)
from emperor.augmentations.adaptive_parameters._weights.validation import (
    ModulatedLowRankValidator,
)
from emperor.layers import Layer, LayerStack


class DiagonallyModulatedLowRankDynamicWeight(DynamicWeightAbstract):
    VALIDATOR = ModulatedLowRankValidator

    def __init__(
        self,
        cfg: DiagonallyModulatedLowRankDynamicWeightConfig,
        overrides: DiagonallyModulatedLowRankDynamicWeightConfig | None = None,
    ):
        config = self._override_config(cfg, overrides)
        config = self.__resolve_config(config)
        super().__init__(config)
        self.rank = self.generator_depth.value
        self.model_config = self.cfg.model_config
        self.input_factor_source = self.cfg.input_factor_source
        self.output_factor_source = self.cfg.output_factor_source
        self.input_factor_model_config = self.cfg.input_factor_model_config
        self.output_factor_model_config = self.cfg.output_factor_model_config
        self.coefficient_model_config = self.cfg.coefficient_model_config
        self.input_model = self.__init_input_model()
        self.output_model = self.__init_output_model()
        self.__build_input_factor()
        self.__build_output_factor()
        self.coefficient_model = self.__init_coefficient_model()

    def __resolve_config(
        self, cfg: DiagonallyModulatedLowRankDynamicWeightConfig
    ) -> DiagonallyModulatedLowRankDynamicWeightConfig:
        self.VALIDATOR.validate_supported_fields(cfg)
        input_factor_source = cfg.input_factor_source
        if input_factor_source is None:
            input_factor_source = LowRankFactorSourceOptions.GENERATED
        output_factor_source = cfg.output_factor_source
        if output_factor_source is None:
            output_factor_source = LowRankFactorSourceOptions.GENERATED
        coefficient_model_config = cfg.coefficient_model_config
        if coefficient_model_config is None:
            coefficient_model_config = cfg.model_config
        return replace(
            cfg,
            input_factor_source=input_factor_source,
            output_factor_source=output_factor_source,
            coefficient_model_config=coefficient_model_config,
        )

    def __init_input_model(self) -> DepthMappingLayerStack | None:
        if self.input_factor_source is LowRankFactorSourceOptions.SHARED_PARAMETER:
            return None
        model_config = self.input_factor_model_config
        if model_config is None:
            model_config = self.model_config
        input_depth_mapping_config = DepthMappingHandlerConfig(
            input_dim=self.input_dim,
            output_dim=self.input_dim,
            model_config=model_config,
        )
        return self._init_model(input_depth_mapping_config)

    def __init_output_model(self) -> DepthMappingLayerStack | None:
        if self.output_factor_source is LowRankFactorSourceOptions.SHARED_PARAMETER:
            return None
        model_config = self.output_factor_model_config
        if model_config is None:
            model_config = self.model_config
        output_depth_mapping_config = DepthMappingHandlerConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            model_config=model_config,
        )
        return self._init_model(output_depth_mapping_config)

    def __build_input_factor(self) -> None:
        input_factor = None
        if self.input_model is None:
            input_factor_shape = (self.input_dim, self.rank)
            input_factor = self._init_parameter_bank(input_factor_shape)
        self.register_parameter("input_factor", input_factor)

    def __build_output_factor(self) -> None:
        output_factor = None
        if self.output_model is None:
            output_factor_shape = (self.rank, self.output_dim)
            output_factor = self._init_parameter_bank(output_factor_shape)
        self.register_parameter("output_factor", output_factor)

    def __init_coefficient_model(self) -> LayerStack:
        model_config = self.coefficient_model_config
        hidden_dim = model_config.hidden_dim
        if hidden_dim is None:
            hidden_dim = self.input_dim
        model_config = replace(
            model_config,
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.rank,
        )
        return model_config.build()

    def forward(self, weight_params: Tensor, context: Tensor) -> Tensor:
        self.VALIDATOR.validate_input_batch(self, context)
        self.VALIDATOR.validate_batched_weight_params(self, weight_params, context)
        input_vectors = self.__get_input_vectors(context)
        output_vectors = self.__get_output_vectors(context)
        input_components = self._apply_normalization_transform(input_vectors)
        output_components = self._apply_normalization_transform(output_vectors)
        coefficients = self.__get_coefficients(context)
        transposed_input_components = input_components.transpose(-1, -2)
        expanded_coefficients = coefficients.unsqueeze(-2)
        coefficient_scaled_input_components = (
            transposed_input_components * expanded_coefficients
        )
        update = torch.matmul(coefficient_scaled_input_components, output_components)
        return self._maybe_apply_weight_decay(weight_params) + update

    def __get_input_vectors(self, context: Tensor) -> Tensor:
        if self.input_model is not None:
            return self.input_model(context)
        return self.input_factor.T

    def __get_output_vectors(self, context: Tensor) -> Tensor:
        if self.output_model is not None:
            return self.output_model(context)
        return self.output_factor

    def __get_coefficients(self, context: Tensor) -> Tensor:
        return Layer.run_model_from_hidden(self.coefficient_model, context).hidden
