from dataclasses import dataclass

from emperor.augmentations.adaptive_parameters.config import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.augmentations.adaptive_parameters.core.bias import DynamicBiasConfig
from emperor.augmentations.adaptive_parameters.core.diagonal import (
    DynamicDiagonalConfig,
)
from emperor.augmentations.adaptive_parameters.core.mask import AxisMaskConfig
from emperor.augmentations.adaptive_parameters.core.weight import DynamicWeightConfig
from emperor.base.layer.config import (
    LayerConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.base.layer.gate import GateConfig
from emperor.halting.config import StickBreakingConfig
from emperor.linears.core.config import AdaptiveLinearLayerConfig
from emperor.memory.config import DynamicMemoryConfig

from models.adaptive_parameter_config_factory import (
    build_bias_config,
    build_diagonal_config,
    build_mask_config,
    build_weight_config,
)
from models.linears._builder_options import (
    DynamicMemoryOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
)
from models.linears._controller_stack import (
    SubmoduleStackOptions,
    SubmoduleStackSource,
)
from models.linears._gate_config_factory import GateConfigFactory
from models.linears._halting_config_factory import HaltingConfigFactory
from models.linears._memory_config_factory import MemoryConfigFactory
from models.linears._recurrent_config_factory import RecurrentConfigFactory
from models.linears.linear_adaptive._adaptive_generator_stack_config_factory import (
    AdaptiveGeneratorStackConfigFactory,
)
from models.linears.linear_adaptive._builder_options import (
    AdaptiveGeneratorStackOptions,
    AdaptiveGeneratorStackSource,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
)

import models.linears.linear_adaptive.config as config


@dataclass(frozen=True)
class ControlConfigDependencies:
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


class ControlConfigFactory:
    def __init__(self, dependencies: ControlConfigDependencies) -> None:
        stack_options = dependencies.stack_options
        submodule_stack_options = dependencies.submodule_stack_options
        layer_controller_options = dependencies.layer_controller_options
        dynamic_memory_options = dependencies.dynamic_memory_options
        recurrent_controller_options = dependencies.recurrent_controller_options
        hidden_adaptive_weight_options = dependencies.hidden_adaptive_weight_options
        hidden_adaptive_bias_options = dependencies.hidden_adaptive_bias_options
        hidden_adaptive_diagonal_options = (
            dependencies.hidden_adaptive_diagonal_options
        )
        hidden_adaptive_mask_options = dependencies.hidden_adaptive_mask_options
        adaptive_generator_stack_options = (
            dependencies.adaptive_generator_stack_options
        )
        output_dim = dependencies.output_dim

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
        self.adaptive_augmentation_config = (
            self.__build_adaptive_augmentation_config()
        )

    def __default_stack_options(
        self,
        stack_options: MainLayerStackOptions | None,
    ) -> MainLayerStackOptions:
        if stack_options is not None:
            return stack_options
        return MainLayerStackOptions(
            hidden_dim=config.STACK_HIDDEN_DIM,
            bias_flag=config.STACK_BIAS_FLAG,
            layer_norm_position=config.STACK_LAYER_NORM_POSITION,
            num_layers=config.STACK_NUM_LAYERS,
            activation=config.STACK_ACTIVATION,
            residual_connection_option=config.STACK_RESIDUAL_CONNECTION_OPTION,
            dropout_probability=config.STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=config.STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        )

    def __default_submodule_stack_options(
        self,
        submodule_stack_options: SubmoduleStackOptions | None,
    ) -> SubmoduleStackOptions:
        if submodule_stack_options is not None:
            return submodule_stack_options
        return SubmoduleStackOptions(
            hidden_dim=config.SUBMODULE_STACK_HIDDEN_DIM,
            num_layers=config.SUBMODULE_STACK_NUM_LAYERS,
            last_layer_bias_option=config.SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=(
                config.SUBMODULE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.SUBMODULE_STACK_ACTIVATION,
            layer_norm_position=config.SUBMODULE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            dropout_probability=config.SUBMODULE_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.SUBMODULE_STACK_BIAS_FLAG,
        )

    def __default_layer_controller_options(
        self,
        layer_controller_options: LayerControllerOptions | None,
    ) -> LayerControllerOptions:
        if layer_controller_options is not None:
            return layer_controller_options
        gate_stack_source = self.__default_controller_stack_source("GATE_STACK")
        halting_stack_source = self.__default_controller_stack_source("HALTING_STACK")
        return LayerControllerOptions(
            stack_gate_flag=config.GATE_FLAG,
            gate_option=config.GATE_OPTION,
            gate_activation=config.GATE_ACTIVATION,
            gate_stack_source=gate_stack_source,
            stack_halting_flag=config.HALTING_FLAG,
            halting_threshold=config.HALTING_THRESHOLD,
            halting_dropout=config.HALTING_DROPOUT,
            halting_hidden_state_mode=config.HALTING_HIDDEN_STATE_MODE,
            halting_stack_source=halting_stack_source,
        )

    def __default_dynamic_memory_options(
        self,
        dynamic_memory_options: DynamicMemoryOptions | None,
    ) -> DynamicMemoryOptions:
        if dynamic_memory_options is not None:
            return dynamic_memory_options
        memory_stack_source = self.__default_controller_stack_source("MEMORY_STACK")
        return DynamicMemoryOptions(
            memory_flag=config.MEMORY_FLAG,
            memory_option=config.MEMORY_OPTION,
            memory_position_option=config.MEMORY_POSITION_OPTION,
            memory_test_time_training_learning_rate=(
                config.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
            ),
            memory_test_time_training_num_inner_steps=(
                config.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
            ),
            memory_stack_source=memory_stack_source,
        )

    def __default_recurrent_controller_options(
        self,
        recurrent_controller_options: RecurrentControllerOptions | None,
    ) -> RecurrentControllerOptions:
        if recurrent_controller_options is not None:
            return recurrent_controller_options
        recurrent_gate_stack_source = self.__default_controller_stack_source(
            "RECURRENT_GATE_STACK"
        )
        recurrent_halting_stack_source = self.__default_controller_stack_source(
            "RECURRENT_HALTING_STACK"
        )
        return RecurrentControllerOptions(
            recurrent_flag=config.RECURRENT_FLAG,
            recurrent_max_steps=config.RECURRENT_MAX_STEPS,
            recurrent_layer_norm_position=config.RECURRENT_LAYER_NORM_POSITION,
            recurrent_gate_flag=config.RECURRENT_GATE_FLAG,
            recurrent_gate_option=config.RECURRENT_GATE_OPTION,
            recurrent_gate_activation=config.RECURRENT_GATE_ACTIVATION,
            recurrent_gate_stack_source=recurrent_gate_stack_source,
            recurrent_halting_flag=config.RECURRENT_HALTING_FLAG,
            recurrent_halting_threshold=config.RECURRENT_HALTING_THRESHOLD,
            recurrent_halting_dropout=config.RECURRENT_HALTING_DROPOUT,
            recurrent_halting_hidden_state_mode=(
                config.RECURRENT_HALTING_HIDDEN_STATE_MODE
            ),
            recurrent_halting_stack_source=recurrent_halting_stack_source,
        )

    def __default_adaptive_generator_stack_options(
        self,
        adaptive_generator_stack_options: AdaptiveGeneratorStackOptions | None,
    ) -> AdaptiveGeneratorStackOptions:
        if adaptive_generator_stack_options is not None:
            return adaptive_generator_stack_options
        return AdaptiveGeneratorStackOptions(
            hidden_dim=config.ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=config.ADAPTIVE_GENERATOR_STACK_LAYER_NORM_POSITION,
            num_layers=config.ADAPTIVE_GENERATOR_STACK_NUM_LAYERS,
            activation=config.ADAPTIVE_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=(
                config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            dropout_probability=config.ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=(
                config.ADAPTIVE_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_pipeline_flag=(
                config.ADAPTIVE_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            bias_flag=config.ADAPTIVE_GENERATOR_STACK_BIAS_FLAG,
        )

    def __default_controller_stack_source(
        self,
        prefix: str,
    ) -> SubmoduleStackSource:
        independent_flag = getattr(config, f"{prefix}_INDEPENDENT_FLAG")
        hidden_dim = getattr(config, f"{prefix}_HIDDEN_DIM")
        num_layers = getattr(config, f"{prefix}_NUM_LAYERS")
        last_layer_bias_option = getattr(config, f"{prefix}_LAST_LAYER_BIAS_OPTION")
        apply_output_pipeline_flag = getattr(
            config,
            f"{prefix}_APPLY_OUTPUT_PIPELINE_FLAG",
        )
        activation = getattr(config, f"{prefix}_ACTIVATION")
        layer_norm_position = getattr(config, f"{prefix}_LAYER_NORM_POSITION")
        residual_connection_option = getattr(
            config,
            f"{prefix}_RESIDUAL_CONNECTION_OPTION",
        )
        dropout_probability = getattr(config, f"{prefix}_DROPOUT_PROBABILITY")
        bias_flag = getattr(config, f"{prefix}_BIAS_FLAG")

        return SubmoduleStackSource(
            independent_flag=independent_flag,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            last_layer_bias_option=last_layer_bias_option,
            apply_output_pipeline_flag=apply_output_pipeline_flag,
            activation=activation,
            layer_norm_position=layer_norm_position,
            residual_connection_option=residual_connection_option,
            dropout_probability=dropout_probability,
            bias_flag=bias_flag,
        )

    def __default_hidden_adaptive_weight_options(
        self,
        hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions | None,
    ) -> HiddenAdaptiveWeightOptions:
        if hidden_adaptive_weight_options is not None:
            return hidden_adaptive_weight_options
        generator_stack_source = self.__default_adaptive_generator_stack_source(
            "WEIGHT_GENERATOR_STACK"
        )
        return HiddenAdaptiveWeightOptions(
            generator_depth=config.WEIGHT_GENERATOR_DEPTH,
            option_flag=config.WEIGHT_OPTION_FLAG,
            option=config.WEIGHT_OPTION,
            normalization_option=config.WEIGHT_NORMALIZATION_OPTION,
            normalization_position_option=config.WEIGHT_NORMALIZATION_POSITION_OPTION,
            decay_schedule=config.WEIGHT_DECAY_SCHEDULE,
            decay_rate=config.WEIGHT_DECAY_RATE,
            decay_warmup_batches=config.WEIGHT_DECAY_WARMUP_BATCHES,
            bank_expansion_factor=config.WEIGHT_BANK_EXPANSION_FACTOR,
            generator_stack_source=generator_stack_source,
        )

    def __default_hidden_adaptive_bias_options(
        self,
        hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions | None,
    ) -> HiddenAdaptiveBiasOptions:
        if hidden_adaptive_bias_options is not None:
            return hidden_adaptive_bias_options
        generator_stack_source = self.__default_adaptive_generator_stack_source(
            "BIAS_GENERATOR_STACK"
        )
        return HiddenAdaptiveBiasOptions(
            option_flag=config.BIAS_OPTION_FLAG,
            option=config.BIAS_OPTION,
            decay_schedule=config.BIAS_DECAY_SCHEDULE,
            decay_rate=config.BIAS_DECAY_RATE,
            decay_warmup_batches=config.BIAS_DECAY_WARMUP_BATCHES,
            bank_expansion_factor=config.BIAS_BANK_EXPANSION_FACTOR,
            generator_stack_source=generator_stack_source,
        )

    def __default_hidden_adaptive_diagonal_options(
        self,
        hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None,
    ) -> HiddenAdaptiveDiagonalOptions:
        if hidden_adaptive_diagonal_options is not None:
            return hidden_adaptive_diagonal_options
        generator_stack_source = self.__default_adaptive_generator_stack_source(
            "DIAGONAL_GENERATOR_STACK"
        )
        return HiddenAdaptiveDiagonalOptions(
            option_flag=config.DIAGONAL_OPTION_FLAG,
            option=config.DIAGONAL_OPTION,
            generator_stack_source=generator_stack_source,
        )

    def __default_hidden_adaptive_mask_options(
        self,
        hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions | None,
    ) -> HiddenAdaptiveMaskOptions:
        if hidden_adaptive_mask_options is not None:
            return hidden_adaptive_mask_options
        generator_stack_source = self.__default_adaptive_generator_stack_source(
            "MASK_GENERATOR_STACK"
        )
        return HiddenAdaptiveMaskOptions(
            option_flag=config.MASK_OPTION_FLAG,
            row_mask_option=config.ROW_MASK_OPTION,
            mask_dimension_option=config.MASK_DIMENSION_OPTION,
            mask_threshold=config.MASK_THRESHOLD,
            mask_surrogate_scale=config.MASK_SURROGATE_SCALE,
            mask_floor=config.MASK_FLOOR,
            mask_transition_width=config.MASK_TRANSITION_WIDTH,
            generator_stack_source=generator_stack_source,
        )

    def __default_adaptive_generator_stack_source(
        self,
        prefix: str,
    ) -> AdaptiveGeneratorStackSource:
        independent_flag = getattr(config, f"{prefix}_INDEPENDENT_FLAG")
        hidden_dim = getattr(config, f"{prefix}_HIDDEN_DIM")
        layer_norm_position = getattr(config, f"{prefix}_LAYER_NORM_POSITION")
        num_layers = getattr(config, f"{prefix}_NUM_LAYERS")
        activation = getattr(config, f"{prefix}_ACTIVATION")
        residual_connection_option = getattr(
            config,
            f"{prefix}_RESIDUAL_CONNECTION_OPTION",
        )
        dropout_probability = getattr(config, f"{prefix}_DROPOUT_PROBABILITY")
        last_layer_bias_option = getattr(config, f"{prefix}_LAST_LAYER_BIAS_OPTION")
        apply_output_pipeline_flag = getattr(
            config,
            f"{prefix}_APPLY_OUTPUT_PIPELINE_FLAG",
        )
        bias_flag = getattr(config, f"{prefix}_BIAS_FLAG")

        return AdaptiveGeneratorStackSource(
            independent_flag=independent_flag,
            hidden_dim=hidden_dim,
            layer_norm_position=layer_norm_position,
            num_layers=num_layers,
            activation=activation,
            residual_connection_option=residual_connection_option,
            dropout_probability=dropout_probability,
            last_layer_bias_option=last_layer_bias_option,
            apply_output_pipeline_flag=apply_output_pipeline_flag,
            bias_flag=bias_flag,
        )

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
            hidden_dim=self.stack_options.hidden_dim,
            num_layers=self.stack_options.num_layers,
            last_layer_bias_option=self.stack_options.last_layer_bias_option,
            apply_output_pipeline_flag=(
                self.stack_options.apply_output_pipeline_flag
            ),
            shared_gate_config=self.layer_controller_options.shared_gate_config,
            shared_memory_config=memory_config,
            layer_config=layer_config,
        )

    def __build_layer_config(
        self,
        *,
        gate_config: GateConfig | None,
        halting_config: StickBreakingConfig | None,
    ) -> LayerConfig:
        layer_model_config = self.__build_layer_model_config()
        return LayerConfig(
            activation=self.stack_options.activation,
            layer_norm_position=self.stack_options.layer_norm_position,
            residual_connection_option=self.stack_options.residual_connection_option,
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
