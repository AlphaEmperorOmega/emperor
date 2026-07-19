from dataclasses import dataclass

import models.bert.linear_adaptive.config as config
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AxisMaskConfig,
    DynamicBiasConfig,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
)
from emperor.halting import HaltingConfig
from emperor.layers import (
    GateConfig,
    LayerConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
    ResidualConfig,
)
from emperor.memory import DynamicMemoryConfig
from models.bert.linear_adaptive import _config_defaults as config_defaults
from models.bert.linear_adaptive._adaptive_generator_stack_config_factory import (
    AdaptiveGeneratorStackConfigFactory,
)
from models.bert.linear_adaptive._adaptive_parameter_config_factory import (
    build_bias_config,
    build_diagonal_config,
    build_mask_config,
    build_weight_config,
)
from models.bert.linear_adaptive._gate_config_factory import GateConfigFactory
from models.bert.linear_adaptive._halting_config_factory import HaltingConfigFactory
from models.bert.linear_adaptive._memory_config_factory import MemoryConfigFactory
from models.bert.linear_adaptive._recurrent_config_factory import (
    RecurrentConfigFactory,
)
from models.bert.linear_adaptive.runtime_options import (
    AdaptiveGeneratorStackOptions,
    AdaptiveGeneratorStackSource,
    DynamicMemoryOptions,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
)


@dataclass(frozen=True)
class HiddenModelConfigDependencies:
    hidden_dim: int
    stack_options: MainLayerStackOptions | None
    submodule_stack_options: SubmoduleStackOptions | None
    layer_controller_options: LayerControllerOptions | None
    dynamic_memory_options: DynamicMemoryOptions | None
    recurrent_controller_options: RecurrentControllerOptions | None
    hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions | None
    hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions | None
    hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None
    hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions | None
    adaptive_generator_stack_options: AdaptiveGeneratorStackOptions | None
    output_dim: int


class HiddenModelConfigFactory:
    def __init__(self, dependencies: HiddenModelConfigDependencies) -> None:
        hidden_dim = dependencies.hidden_dim
        stack_options = dependencies.stack_options
        submodule_stack_options = dependencies.submodule_stack_options
        layer_controller_options = dependencies.layer_controller_options
        dynamic_memory_options = dependencies.dynamic_memory_options
        recurrent_controller_options = dependencies.recurrent_controller_options
        hidden_adaptive_weight_options = dependencies.hidden_adaptive_weight_options
        hidden_adaptive_bias_options = dependencies.hidden_adaptive_bias_options
        hidden_adaptive_diagonal_options = dependencies.hidden_adaptive_diagonal_options
        hidden_adaptive_mask_options = dependencies.hidden_adaptive_mask_options
        adaptive_generator_stack_options = dependencies.adaptive_generator_stack_options
        output_dim = dependencies.output_dim

        self._hidden_dim = hidden_dim
        self.stack_options = self.__default_stack_options(stack_options)
        self.submodule_stack_options = self.__default_submodule_stack_options(
            submodule_stack_options
        )
        self.layer_controller_options = self.__default_layer_controller_options(
            layer_controller_options
        )
        self.dynamic_memory_options = self.__default_dynamic_memory_options(
            dynamic_memory_options
        )
        self.recurrent_controller_options = self.__default_recurrent_controller_options(
            recurrent_controller_options
        )
        self.gate_config_factory = GateConfigFactory(
            layer_controller_options=self.layer_controller_options,
            recurrent_controller_options=self.recurrent_controller_options,
            submodule_stack_options=self.submodule_stack_options,
        )
        self.halting_config_factory = HaltingConfigFactory(
            layer_controller_options=self.layer_controller_options,
            recurrent_controller_options=self.recurrent_controller_options,
            submodule_stack_options=self.submodule_stack_options,
            output_dim=output_dim,
        )
        self.memory_config_factory = MemoryConfigFactory(
            hidden_dim=self.hidden_dim,
            stack_options=self.stack_options,
            dynamic_memory_options=self.dynamic_memory_options,
            submodule_stack_options=self.submodule_stack_options,
        )
        self.recurrent_config_factory = RecurrentConfigFactory(
            recurrent_controller_options=self.recurrent_controller_options,
            gate_config_factory=self.gate_config_factory,
            halting_config_factory=self.halting_config_factory,
        )
        self.hidden_adaptive_weight_options = (
            self.__default_hidden_adaptive_weight_options(
                hidden_adaptive_weight_options
            )
        )
        self.hidden_adaptive_bias_options = self.__default_hidden_adaptive_bias_options(
            hidden_adaptive_bias_options
        )
        self.hidden_adaptive_diagonal_options = (
            self.__default_hidden_adaptive_diagonal_options(
                hidden_adaptive_diagonal_options
            )
        )
        self.hidden_adaptive_mask_options = self.__default_hidden_adaptive_mask_options(
            hidden_adaptive_mask_options
        )
        self.adaptive_generator_stack_options = (
            self.__default_adaptive_generator_stack_options(
                adaptive_generator_stack_options
            )
        )
        self.adaptive_generator_stack_config_factory = (
            AdaptiveGeneratorStackConfigFactory(self.adaptive_generator_stack_options)
        )
        self.adaptive_augmentation_config = self.__build_adaptive_augmentation_config()

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def __default_stack_options(
        self,
        stack_options: MainLayerStackOptions | None,
    ) -> MainLayerStackOptions:
        if stack_options is not None:
            return stack_options
        return config_defaults.main_layer_stack_options(config)

    def __default_submodule_stack_options(
        self,
        submodule_stack_options: SubmoduleStackOptions | None,
    ) -> SubmoduleStackOptions:
        if submodule_stack_options is not None:
            return submodule_stack_options
        return config_defaults.linears_submodule_stack_options(
            config,
            "SUBMODULE_STACK",
        )

    def __default_layer_controller_options(
        self,
        layer_controller_options: LayerControllerOptions | None,
    ) -> LayerControllerOptions:
        if layer_controller_options is not None:
            return layer_controller_options
        return config_defaults.linears_layer_controller_options(
            config,
            gate_prefix="GATE",
            gate_stack_prefix="GATE_STACK",
            halting_prefix="HALTING",
            halting_stack_prefix="HALTING_STACK",
        )

    def __default_dynamic_memory_options(
        self,
        dynamic_memory_options: DynamicMemoryOptions | None,
    ) -> DynamicMemoryOptions:
        if dynamic_memory_options is not None:
            return dynamic_memory_options
        return config_defaults.linears_dynamic_memory_options(
            config,
            memory_prefix="MEMORY",
            memory_stack_prefix="MEMORY_STACK",
        )

    def __default_recurrent_controller_options(
        self,
        recurrent_controller_options: RecurrentControllerOptions | None,
    ) -> RecurrentControllerOptions:
        if recurrent_controller_options is not None:
            return recurrent_controller_options
        return config_defaults.linears_recurrent_controller_options(
            config,
            recurrent_prefix="RECURRENT",
            gate_stack_prefix="RECURRENT_GATE_STACK",
            halting_stack_prefix="RECURRENT_HALTING_STACK",
        )

    def __default_adaptive_generator_stack_options(
        self,
        adaptive_generator_stack_options: AdaptiveGeneratorStackOptions | None,
    ) -> AdaptiveGeneratorStackOptions:
        if adaptive_generator_stack_options is not None:
            return adaptive_generator_stack_options
        return config_defaults.adaptive_generator_stack_options(config)

    def __default_hidden_adaptive_weight_options(
        self,
        hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions | None,
    ) -> HiddenAdaptiveWeightOptions:
        if hidden_adaptive_weight_options is not None:
            return hidden_adaptive_weight_options
        return config_defaults.hidden_adaptive_weight_options(config)

    def __default_hidden_adaptive_bias_options(
        self,
        hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions | None,
    ) -> HiddenAdaptiveBiasOptions:
        if hidden_adaptive_bias_options is not None:
            return hidden_adaptive_bias_options
        return config_defaults.hidden_adaptive_bias_options(config)

    def __default_hidden_adaptive_diagonal_options(
        self,
        hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None,
    ) -> HiddenAdaptiveDiagonalOptions:
        if hidden_adaptive_diagonal_options is not None:
            return hidden_adaptive_diagonal_options
        return config_defaults.hidden_adaptive_diagonal_options(config)

    def __default_hidden_adaptive_mask_options(
        self,
        hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions | None,
    ) -> HiddenAdaptiveMaskOptions:
        if hidden_adaptive_mask_options is not None:
            return hidden_adaptive_mask_options
        return config_defaults.hidden_adaptive_mask_options(config)

    def build_hidden_model_config(self) -> LayerStackConfig | RecurrentLayerConfig:
        gate_config = self.gate_config_factory.build_gate_config()
        halting_config = self.halting_config_factory.build_halting_config()
        memory_config = self.memory_config_factory.build_memory_config()
        layer_config = self.__build_layer_config(
            gate_config=gate_config,
            halting_config=halting_config,
        )
        layer_stack_config = self.__build_stack_config(
            memory_config=memory_config,
            layer_config=layer_config,
        )
        return self.recurrent_config_factory.build_config(layer_stack_config)

    def __build_stack_config(
        self,
        *,
        memory_config: DynamicMemoryConfig | None,
        layer_config: LayerConfig,
    ) -> LayerStackConfig:
        return LayerStackConfig(
            hidden_dim=self.hidden_dim,
            num_layers=self.stack_options.num_layers,
            last_layer_bias_option=self.stack_options.last_layer_bias_option,
            apply_output_pipeline_flag=(self.stack_options.apply_output_pipeline_flag),
            shared_gate_config=self.layer_controller_options.shared_gate_config,
            shared_memory_config=memory_config,
            layer_config=layer_config,
        )

    def __build_layer_config(
        self,
        *,
        gate_config: GateConfig | None,
        halting_config: HaltingConfig | None,
    ) -> LayerConfig:
        layer_model_config = self.__build_layer_model_config()
        return LayerConfig(
            activation=self.stack_options.activation,
            layer_norm_position=self.stack_options.layer_norm_position,
            residual_config=None
            if self.stack_options.residual_connection_option is None
            else ResidualConfig(option=self.stack_options.residual_connection_option),
            dropout_probability=self.stack_options.dropout_probability,
            gate_config=gate_config,
            halting_config=halting_config,
            layer_model_config=layer_model_config,
        )

    def __build_layer_model_config(self) -> AdaptiveLinearLayerConfig:
        return AdaptiveLinearLayerConfig(
            bias_flag=self.stack_options.bias_flag,
            adaptive_augmentation_config=self.adaptive_augmentation_config,
        )

    def __build_adaptive_augmentation_config(
        self,
    ) -> AdaptiveParameterAugmentationConfig:
        weight_config = self.__build_weight_config()
        bias_config = self.__build_bias_config()
        diagonal_config = self.__build_diagonal_config()
        mask_config = self.__build_mask_config()
        shared_model_config = self.__build_shared_adaptive_generator_stack_config()
        return AdaptiveParameterAugmentationConfig(
            weight_config=weight_config,
            bias_config=bias_config,
            diagonal_config=diagonal_config,
            mask_config=mask_config,
            model_config=shared_model_config,
        )

    def __resolve_enabled_adaptive_parameter_option(
        self,
        *,
        option_flag: bool,
        option: type | None,
        option_flag_name: str,
        option_name: str,
    ) -> type | None:
        if not option_flag:
            return None
        if option is None:
            raise ValueError(
                f"{option_name} must be set when {option_flag_name} is True."
            )
        return option

    def __build_weight_config(self) -> DynamicWeightConfig | None:
        weight_option = self.__resolve_enabled_adaptive_parameter_option(
            option_flag=self.hidden_adaptive_weight_options.option_flag,
            option=self.hidden_adaptive_weight_options.option,
            option_flag_name="weight_option_flag",
            option_name="weight_option",
        )
        if weight_option is None:
            return None
        model_config = self.__build_adaptive_generator_stack_config(
            self.hidden_adaptive_weight_options.generator_stack_source
        )
        return build_weight_config(
            weight_option,
            generator_depth=self.hidden_adaptive_weight_options.generator_depth,
            decay_schedule=self.hidden_adaptive_weight_options.decay_schedule,
            decay_rate=self.hidden_adaptive_weight_options.decay_rate,
            decay_warmup_batches=(
                self.hidden_adaptive_weight_options.decay_warmup_batches
            ),
            normalization_option=(
                self.hidden_adaptive_weight_options.normalization_option
            ),
            normalization_position_option=(
                self.hidden_adaptive_weight_options.normalization_position_option
            ),
            bank_expansion_factor=(
                self.hidden_adaptive_weight_options.bank_expansion_factor
            ),
            model_config=model_config,
        )

    def __build_bias_config(self) -> DynamicBiasConfig | None:
        bias_option = self.__resolve_enabled_adaptive_parameter_option(
            option_flag=self.hidden_adaptive_bias_options.option_flag,
            option=self.hidden_adaptive_bias_options.option,
            option_flag_name="bias_option_flag",
            option_name="bias_option",
        )
        if bias_option is None:
            return None
        model_config = self.__build_adaptive_generator_stack_config(
            self.hidden_adaptive_bias_options.generator_stack_source
        )
        return build_bias_config(
            bias_option,
            decay_schedule=self.hidden_adaptive_bias_options.decay_schedule,
            decay_rate=self.hidden_adaptive_bias_options.decay_rate,
            decay_warmup_batches=(
                self.hidden_adaptive_bias_options.decay_warmup_batches
            ),
            bank_expansion_factor=(
                self.hidden_adaptive_bias_options.bank_expansion_factor
            ),
            model_config=model_config,
        )

    def __build_diagonal_config(self) -> DynamicDiagonalConfig | None:
        diagonal_option = self.__resolve_enabled_adaptive_parameter_option(
            option_flag=self.hidden_adaptive_diagonal_options.option_flag,
            option=self.hidden_adaptive_diagonal_options.option,
            option_flag_name="diagonal_option_flag",
            option_name="diagonal_option",
        )
        if diagonal_option is None:
            return None
        model_config = self.__build_adaptive_generator_stack_config(
            self.hidden_adaptive_diagonal_options.generator_stack_source
        )
        return build_diagonal_config(
            diagonal_option,
            model_config=model_config,
        )

    def __build_mask_config(self) -> AxisMaskConfig | None:
        row_mask_option = self.__resolve_enabled_adaptive_parameter_option(
            option_flag=self.hidden_adaptive_mask_options.option_flag,
            option=self.hidden_adaptive_mask_options.row_mask_option,
            option_flag_name="mask_option_flag",
            option_name="row_mask_option",
        )
        if row_mask_option is None:
            return None
        model_config = self.__build_adaptive_generator_stack_config(
            self.hidden_adaptive_mask_options.generator_stack_source
        )
        return build_mask_config(
            row_mask_option,
            mask_dimension_option=(
                self.hidden_adaptive_mask_options.mask_dimension_option
            ),
            mask_threshold=self.hidden_adaptive_mask_options.mask_threshold,
            mask_surrogate_scale=(
                self.hidden_adaptive_mask_options.mask_surrogate_scale
            ),
            mask_floor=self.hidden_adaptive_mask_options.mask_floor,
            mask_transition_width=(
                self.hidden_adaptive_mask_options.mask_transition_width
            ),
            model_config=model_config,
        )

    def __build_adaptive_generator_stack_config(
        self,
        source: AdaptiveGeneratorStackSource,
    ) -> LayerStackConfig | None:
        return self.adaptive_generator_stack_config_factory.build_config_from_source(
            source
        )

    def __build_shared_adaptive_generator_stack_config(self) -> LayerStackConfig:
        return self.adaptive_generator_stack_config_factory.build_shared_config()
