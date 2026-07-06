from emperor.augmentations.adaptive_parameters.config import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.linears.core.config import AdaptiveLinearLayerConfig

import models.bert.linear_adaptive.config as config
import models.linears.linear_adaptive.config as adaptive_defaults
from models.bert.linear.config_builder import BertLinearConfigBuilder
from models.bert.linear_adaptive.experiment_config import ExperimentConfig
from models.linears._builder_adapter import (
    linear_adaptive_builder_kwargs_from_flat,
)
from models.linears.linear_adaptive._builder_options import (
    AdaptiveGeneratorStackOptions,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
)
from models.linears.linear_adaptive._control_config_factory import (
    ControlConfigDependencies,
    ControlConfigFactory,
)


class BertLinearAdaptiveConfigBuilder(BertLinearConfigBuilder):
    def __init__(
        self,
        *args,
        adaptive_generator_stack_options: AdaptiveGeneratorStackOptions | None = None,
        hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions | None = None,
        hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions | None = None,
        hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None = None,
        hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions | None = None,
        **kwargs,
    ) -> None:
        adaptive_defaults_kwargs = linear_adaptive_builder_kwargs_from_flat(
            {},
            adaptive_defaults,
        )
        self.adaptive_generator_stack_options = (
            adaptive_generator_stack_options
            or adaptive_defaults_kwargs["adaptive_generator_stack_options"]
        )
        self.hidden_adaptive_weight_options = (
            hidden_adaptive_weight_options
            or adaptive_defaults_kwargs["hidden_adaptive_weight_options"]
        )
        self.hidden_adaptive_bias_options = (
            hidden_adaptive_bias_options
            or adaptive_defaults_kwargs["hidden_adaptive_bias_options"]
        )
        self.hidden_adaptive_diagonal_options = (
            hidden_adaptive_diagonal_options
            or adaptive_defaults_kwargs["hidden_adaptive_diagonal_options"]
        )
        self.hidden_adaptive_mask_options = (
            hidden_adaptive_mask_options
            or adaptive_defaults_kwargs["hidden_adaptive_mask_options"]
        )
        super().__init__(*args, **kwargs)
        self.experiment_config_type = ExperimentConfig
        self.adaptive_augmentation_config = self._build_adaptive_augmentation_config()

    def _build_linear_layer_config(
        self,
        *,
        bias_flag: bool,
    ) -> AdaptiveLinearLayerConfig:
        return AdaptiveLinearLayerConfig(
            bias_flag=bias_flag,
            adaptive_augmentation_config=self.adaptive_augmentation_config,
        )

    def _build_adaptive_augmentation_config(
        self,
    ) -> AdaptiveParameterAugmentationConfig:
        factory = ControlConfigFactory(
            ControlConfigDependencies(
                stack_options=None,
                submodule_stack_options=None,
                layer_controller_options=None,
                dynamic_memory_options=None,
                recurrent_controller_options=None,
                hidden_adaptive_weight_options=self.hidden_adaptive_weight_options,
                hidden_adaptive_bias_options=self.hidden_adaptive_bias_options,
                hidden_adaptive_diagonal_options=self.hidden_adaptive_diagonal_options,
                hidden_adaptive_mask_options=self.hidden_adaptive_mask_options,
                adaptive_generator_stack_options=(
                    self.adaptive_generator_stack_options
                ),
                output_dim=self.output_dim,
            )
        )
        return factory.adaptive_augmentation_config
