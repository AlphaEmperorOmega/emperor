from typing import TYPE_CHECKING

from emperor.augmentations.adaptive_parameters.config import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.augmentations.adaptive_parameters.core.bias import DynamicBiasConfig
from emperor.augmentations.adaptive_parameters.core.diagonal import (
    DynamicDiagonalConfig,
)
from emperor.augmentations.adaptive_parameters.core.mask import AxisMaskConfig
from emperor.augmentations.adaptive_parameters.core.weight import DynamicWeightConfig
from emperor.augmentations.adaptive_parameters.options import (
    BankExpansionFactorOptions,
    DynamicDepthOptions,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.base.layer.config import LayerConfig, LayerStackConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.linears.core.config import LinearLayerConfig
from models.adaptive_parameter_config_factory import (
    build_bias_config,
    build_diagonal_config,
    build_mask_config,
    build_weight_config,
)
from models.linears._builder_options import (
    DynamicMemoryOptions,
    LayerControllerOptions,
    LinearStackOptions,
    RecurrentControllerOptions,
)
from models.linears._builder_adapter import (
    default_adaptive_generator_stack_options,
    default_adaptive_generator_stack_source,
    default_boundary_options,
    default_dynamic_memory_options,
    default_layer_controller_options,
    default_linear_stack_options,
    default_recurrent_controller_options,
    default_submodule_stack_options,
)
from models.linears._controller_stack import ControllerStackOptions
from models.linears.linear_adaptive._boundary_config_factory import (
    AdaptiveBoundaryProjectionOptions,
    BoundaryConfigDependencies,
    BoundaryConfigFactory,
)
from models.linears.linear_adaptive import _builder_options as adaptive_options
from models.linears.linear_adaptive._control_config_factory import (
    ControlConfigDependencies,
    ControlConfigFactory,
)
from models.linears.linear_adaptive.experiment_config import ExperimentConfig

import models.linears.linear_adaptive.config as config

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class LinearAdaptiveConfigBuilder:
    def __init__(
        self,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        output_dim: int = config.OUTPUT_DIM,
        generator_depth: DynamicDepthOptions = config.WEIGHT_GENERATOR_DEPTH,
        diagonal_option_flag: bool = config.DIAGONAL_OPTION_FLAG,
        diagonal_option: type[DynamicDiagonalConfig] | None = config.DIAGONAL_OPTION,
        bias_option_flag: bool = config.BIAS_OPTION_FLAG,
        bias_option: type[DynamicBiasConfig] | None = config.BIAS_OPTION,
        weight_option_flag: bool = config.WEIGHT_OPTION_FLAG,
        weight_option: type[DynamicWeightConfig] | None = config.WEIGHT_OPTION,
        weight_normalization_option: WeightNormalizationOptions = (
            config.WEIGHT_NORMALIZATION_OPTION
        ),
        weight_normalization_position_option: WeightNormalizationPositionOptions = (
            config.WEIGHT_NORMALIZATION_POSITION_OPTION
        ),
        weight_decay_schedule: WeightDecayScheduleOptions = (
            config.WEIGHT_DECAY_SCHEDULE
        ),
        weight_decay_rate: float = config.WEIGHT_DECAY_RATE,
        weight_decay_warmup_batches: int = config.WEIGHT_DECAY_WARMUP_BATCHES,
        weight_bank_expansion_factor: BankExpansionFactorOptions = (
            config.WEIGHT_BANK_EXPANSION_FACTOR
        ),
        bias_decay_schedule: WeightDecayScheduleOptions = config.BIAS_DECAY_SCHEDULE,
        bias_decay_rate: float = config.BIAS_DECAY_RATE,
        bias_decay_warmup_batches: int = config.BIAS_DECAY_WARMUP_BATCHES,
        bias_bank_expansion_factor: BankExpansionFactorOptions = (
            config.BIAS_BANK_EXPANSION_FACTOR
        ),
        mask_option_flag: bool = config.MASK_OPTION_FLAG,
        row_mask_option: type[AxisMaskConfig] | None = config.ROW_MASK_OPTION,
        mask_dimension_option: MaskDimensionOptions = config.MASK_DIMENSION_OPTION,
        mask_threshold: float = config.MASK_THRESHOLD,
        mask_surrogate_scale: float = config.MASK_SURROGATE_SCALE,
        mask_floor: float = config.MASK_FLOOR,
        mask_transition_width: float = config.MASK_TRANSITION_WIDTH,
        stack_options: LinearStackOptions | None = None,
        submodule_stack_options: ControllerStackOptions | None = None,
        layer_controller_options: LayerControllerOptions | None = None,
        dynamic_memory_options: DynamicMemoryOptions | None = None,
        recurrent_controller_options: RecurrentControllerOptions | None = None,
        adaptive_generator_stack_options: (
            adaptive_options.AdaptiveGeneratorStackOptions | None
        ) = None,
        weight_generator_stack_source: (
            adaptive_options.AdaptiveGeneratorStackSource | None
        ) = None,
        bias_generator_stack_source: (
            adaptive_options.AdaptiveGeneratorStackSource | None
        ) = None,
        diagonal_generator_stack_source: (
            adaptive_options.AdaptiveGeneratorStackSource | None
        ) = None,
        mask_generator_stack_source: (
            adaptive_options.AdaptiveGeneratorStackSource | None
        ) = None,
        input_boundary_options: AdaptiveBoundaryProjectionOptions | None = None,
        output_boundary_options: AdaptiveBoundaryProjectionOptions | None = None,
    ) -> None:
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
        self.adaptive_generator_stack_options = (
            self.__default_adaptive_generator_stack_options(
                adaptive_generator_stack_options
            )
        )
        self.weight_generator_stack_source = self.__default_adaptive_generator_stack_source(
            weight_generator_stack_source,
            "weight_generator_stack",
        )
        self.bias_generator_stack_source = self.__default_adaptive_generator_stack_source(
            bias_generator_stack_source,
            "bias_generator_stack",
        )
        self.diagonal_generator_stack_source = (
            self.__default_adaptive_generator_stack_source(
                diagonal_generator_stack_source,
                "diagonal_generator_stack",
            )
        )
        self.mask_generator_stack_source = self.__default_adaptive_generator_stack_source(
            mask_generator_stack_source,
            "mask_generator_stack",
        )
        self.input_boundary_options = self.__default_boundary_options(
            input_boundary_options,
            "input_layer",
        )
        self.output_boundary_options = self.__default_boundary_options(
            output_boundary_options,
            "output_layer",
        )

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.generator_depth = generator_depth
        self.diagonal_option_flag = diagonal_option_flag
        self.diagonal_option = diagonal_option
        self.bias_option_flag = bias_option_flag
        self.bias_option = bias_option
        self.weight_option_flag = weight_option_flag
        self.weight_option = weight_option
        self.weight_normalization_option = weight_normalization_option
        self.weight_normalization_position_option = weight_normalization_position_option
        self.weight_decay_schedule = weight_decay_schedule
        self.weight_decay_rate = weight_decay_rate
        self.weight_decay_warmup_batches = weight_decay_warmup_batches
        self.weight_bank_expansion_factor = weight_bank_expansion_factor
        self.bias_decay_schedule = bias_decay_schedule
        self.bias_decay_rate = bias_decay_rate
        self.bias_decay_warmup_batches = bias_decay_warmup_batches
        self.bias_bank_expansion_factor = bias_bank_expansion_factor
        self.mask_option_flag = mask_option_flag
        self.row_mask_option = row_mask_option
        self.mask_dimension_option = mask_dimension_option
        self.mask_threshold = mask_threshold
        self.mask_surrogate_scale = mask_surrogate_scale
        self.mask_floor = mask_floor
        self.mask_transition_width = mask_transition_width

    def build(self) -> "ModelConfig":
        from emperor.config import ModelConfig

        boundary_dependencies = self.__boundary_config_dependencies()
        control_dependencies = self.__control_config_dependencies()
        boundary_factory = BoundaryConfigFactory(boundary_dependencies)
        control_factory = ControlConfigFactory(control_dependencies)

        return ModelConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            hidden_dim=self.stack_options.hidden_dim,
            output_dim=self.output_dim,
            experiment_config=ExperimentConfig(
                input_model_config=boundary_factory.build_input_model_config(),
                model_config=control_factory.build(),
                output_model_config=boundary_factory.build_output_model_config(),
            ),
        )

    def __boundary_config_dependencies(self) -> BoundaryConfigDependencies:
        return BoundaryConfigDependencies(
            stack_options=self.stack_options,
            input_boundary_options=self.input_boundary_options,
            output_boundary_options=self.output_boundary_options,
            model_config=self._build_model_config(),
        )

    def __control_config_dependencies(self) -> ControlConfigDependencies:
        return ControlConfigDependencies(
            stack_options=self.stack_options,
            submodule_stack_options=self.submodule_stack_options,
            layer_controller_options=self.layer_controller_options,
            dynamic_memory_options=self.dynamic_memory_options,
            recurrent_controller_options=self.recurrent_controller_options,
            adaptive_augmentation_config=self.__adaptive_augmentation_config(),
            output_dim=self.output_dim,
        )

    def __adaptive_augmentation_config(
        self,
    ) -> AdaptiveParameterAugmentationConfig:
        return AdaptiveParameterAugmentationConfig(
            weight_config=self._build_weight_config(),
            bias_config=self._build_bias_config(),
            diagonal_config=self._build_diagonal_config(),
            mask_config=self._build_mask_config(),
            model_config=self._build_model_config(),
        )

    @staticmethod
    def _enabled_component_option(
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

    def _build_weight_config(self) -> DynamicWeightConfig | None:
        weight_option = self._enabled_component_option(
            option_flag=self.weight_option_flag,
            option=self.weight_option,
            option_flag_name="weight_option_flag",
            option_name="weight_option",
        )
        if weight_option is None:
            return None
        return self._build_weight_config_from_options(
            weight_option=weight_option,
            generator_depth=self.generator_depth,
            decay_schedule=self.weight_decay_schedule,
            decay_rate=self.weight_decay_rate,
            decay_warmup_batches=self.weight_decay_warmup_batches,
            normalization_option=self.weight_normalization_option,
            normalization_position_option=self.weight_normalization_position_option,
            bank_expansion_factor=self.weight_bank_expansion_factor,
            model_config=self._build_weight_generator_stack_config(),
        )

    def _build_weight_config_from_options(
        self,
        weight_option: type[DynamicWeightConfig] | None,
        generator_depth: DynamicDepthOptions,
        decay_schedule: WeightDecayScheduleOptions,
        decay_rate: float,
        decay_warmup_batches: int,
        normalization_option: WeightNormalizationOptions,
        normalization_position_option: WeightNormalizationPositionOptions,
        bank_expansion_factor: BankExpansionFactorOptions,
        model_config: LayerStackConfig | None = None,
    ) -> DynamicWeightConfig | None:
        return build_weight_config(
            weight_option,
            generator_depth=generator_depth,
            decay_schedule=decay_schedule,
            decay_rate=decay_rate,
            decay_warmup_batches=decay_warmup_batches,
            normalization_option=normalization_option,
            normalization_position_option=normalization_position_option,
            bank_expansion_factor=bank_expansion_factor,
            model_config=model_config,
        )

    def _build_bias_config(self) -> DynamicBiasConfig | None:
        bias_option = self._enabled_component_option(
            option_flag=self.bias_option_flag,
            option=self.bias_option,
            option_flag_name="bias_option_flag",
            option_name="bias_option",
        )
        if bias_option is None:
            return None
        return self._build_bias_config_from_options(
            bias_option=bias_option,
            decay_schedule=self.bias_decay_schedule,
            decay_rate=self.bias_decay_rate,
            decay_warmup_batches=self.bias_decay_warmup_batches,
            bank_expansion_factor=self.bias_bank_expansion_factor,
            model_config=self._build_bias_generator_stack_config(),
        )

    def _build_bias_config_from_options(
        self,
        bias_option: type[DynamicBiasConfig] | None,
        decay_schedule: WeightDecayScheduleOptions,
        decay_rate: float,
        decay_warmup_batches: int,
        bank_expansion_factor: BankExpansionFactorOptions,
        model_config: LayerStackConfig | None = None,
    ) -> DynamicBiasConfig | None:
        return build_bias_config(
            bias_option,
            decay_schedule=decay_schedule,
            decay_rate=decay_rate,
            decay_warmup_batches=decay_warmup_batches,
            bank_expansion_factor=bank_expansion_factor,
            model_config=model_config,
        )

    def _build_diagonal_config(self) -> DynamicDiagonalConfig | None:
        diagonal_option = self._enabled_component_option(
            option_flag=self.diagonal_option_flag,
            option=self.diagonal_option,
            option_flag_name="diagonal_option_flag",
            option_name="diagonal_option",
        )
        if diagonal_option is None:
            return None
        return self._build_diagonal_config_from_option(
            diagonal_option,
            model_config=self._build_diagonal_generator_stack_config(),
        )

    def _build_diagonal_config_from_option(
        self,
        diagonal_option: type[DynamicDiagonalConfig] | None,
        model_config: LayerStackConfig | None = None,
    ) -> DynamicDiagonalConfig | None:
        return build_diagonal_config(
            diagonal_option,
            model_config=model_config,
        )

    def _build_mask_config(self) -> AxisMaskConfig | None:
        row_mask_option = self._enabled_component_option(
            option_flag=self.mask_option_flag,
            option=self.row_mask_option,
            option_flag_name="mask_option_flag",
            option_name="row_mask_option",
        )
        if row_mask_option is None:
            return None
        return self._build_mask_config_from_options(
            row_mask_option=row_mask_option,
            mask_dimension_option=self.mask_dimension_option,
            mask_threshold=self.mask_threshold,
            mask_surrogate_scale=self.mask_surrogate_scale,
            mask_floor=self.mask_floor,
            mask_transition_width=self.mask_transition_width,
            model_config=self._build_mask_generator_stack_config(),
        )

    def _build_mask_config_from_options(
        self,
        row_mask_option: type[AxisMaskConfig] | None,
        mask_dimension_option: MaskDimensionOptions,
        mask_threshold: float,
        mask_surrogate_scale: float,
        mask_floor: float,
        mask_transition_width: float,
        model_config: LayerStackConfig | None = None,
    ) -> AxisMaskConfig | None:
        return build_mask_config(
            row_mask_option,
            mask_dimension_option=mask_dimension_option,
            mask_threshold=mask_threshold,
            mask_surrogate_scale=mask_surrogate_scale,
            mask_floor=mask_floor,
            mask_transition_width=mask_transition_width,
            model_config=model_config,
        )

    def _build_weight_generator_stack_config(self) -> LayerStackConfig | None:
        return self._build_adaptive_generator_stack_config_from_source(
            self._weight_generator_stack_source()
        )

    def _build_bias_generator_stack_config(self) -> LayerStackConfig | None:
        return self._build_adaptive_generator_stack_config_from_source(
            self._bias_generator_stack_source()
        )

    def _build_diagonal_generator_stack_config(self) -> LayerStackConfig | None:
        return self._build_adaptive_generator_stack_config_from_source(
            self._diagonal_generator_stack_source()
        )

    def _build_mask_generator_stack_config(self) -> LayerStackConfig | None:
        return self._build_adaptive_generator_stack_config_from_source(
            self._mask_generator_stack_source()
        )

    def _build_adaptive_generator_stack_config_from_source(
        self,
        source: adaptive_options.AdaptiveGeneratorStackSource,
    ) -> LayerStackConfig | None:
        options = self._resolve_adaptive_generator_stack_options(source)
        if options is None:
            return None
        return self._build_model_config_from_options(
            hidden_dim=options.hidden_dim,
            num_layers=options.num_layers,
            activation=options.activation,
            residual_connection_option=options.residual_connection_option,
            dropout_probability=options.dropout_probability,
            layer_norm_position=options.layer_norm_position,
            last_layer_bias_option=options.last_layer_bias_option,
            apply_output_pipeline_flag=options.apply_output_pipeline_flag,
            bias_flag=options.bias_flag,
        )

    def _resolve_adaptive_generator_stack_options(
        self,
        source: adaptive_options.AdaptiveGeneratorStackSource,
    ) -> adaptive_options.AdaptiveGeneratorStackOptions | None:
        if not source.independent_flag:
            return None
        defaults = self._shared_adaptive_generator_stack_options()
        return adaptive_options.AdaptiveGeneratorStackOptions(
            hidden_dim=self._resolve_adaptive_generator_stack_option(
                source.hidden_dim,
                defaults.hidden_dim,
            ),
            layer_norm_position=self._resolve_adaptive_generator_stack_option(
                source.layer_norm_position,
                defaults.layer_norm_position,
            ),
            num_layers=self._resolve_adaptive_generator_stack_option(
                source.num_layers,
                defaults.num_layers,
            ),
            activation=self._resolve_adaptive_generator_stack_option(
                source.activation,
                defaults.activation,
            ),
            residual_connection_option=self._resolve_adaptive_generator_stack_option(
                source.residual_connection_option,
                defaults.residual_connection_option,
            ),
            dropout_probability=self._resolve_adaptive_generator_stack_option(
                source.dropout_probability,
                defaults.dropout_probability,
            ),
            last_layer_bias_option=self._resolve_adaptive_generator_stack_option(
                source.last_layer_bias_option,
                defaults.last_layer_bias_option,
            ),
            apply_output_pipeline_flag=self._resolve_adaptive_generator_stack_option(
                source.apply_output_pipeline_flag,
                defaults.apply_output_pipeline_flag,
            ),
            bias_flag=self._resolve_adaptive_generator_stack_option(
                source.bias_flag,
                defaults.bias_flag,
            ),
        )

    @staticmethod
    def _resolve_adaptive_generator_stack_option(override, shared_default):
        return shared_default if override is None else override

    def _shared_adaptive_generator_stack_options(
        self,
    ) -> adaptive_options.AdaptiveGeneratorStackOptions:
        return self.adaptive_generator_stack_options

    def _weight_generator_stack_source(
        self,
    ) -> adaptive_options.AdaptiveGeneratorStackSource:
        return self.weight_generator_stack_source

    def _bias_generator_stack_source(
        self,
    ) -> adaptive_options.AdaptiveGeneratorStackSource:
        return self.bias_generator_stack_source

    def _diagonal_generator_stack_source(
        self,
    ) -> adaptive_options.AdaptiveGeneratorStackSource:
        return self.diagonal_generator_stack_source

    def _mask_generator_stack_source(
        self,
    ) -> adaptive_options.AdaptiveGeneratorStackSource:
        return self.mask_generator_stack_source

    def _build_model_config(self) -> LayerStackConfig:
        options = self.adaptive_generator_stack_options
        return self._build_model_config_from_options(
            hidden_dim=options.hidden_dim,
            num_layers=options.num_layers,
            activation=options.activation,
            residual_connection_option=options.residual_connection_option,
            dropout_probability=options.dropout_probability,
            layer_norm_position=options.layer_norm_position,
            last_layer_bias_option=options.last_layer_bias_option,
            apply_output_pipeline_flag=options.apply_output_pipeline_flag,
            bias_flag=options.bias_flag,
        )

    def _build_model_config_from_options(
        self,
        hidden_dim: int,
        num_layers: int,
        activation: ActivationOptions,
        residual_connection_option: ResidualConnectionOptions,
        dropout_probability: float,
        layer_norm_position: LayerNormPositionOptions,
        last_layer_bias_option: LastLayerBiasOptions,
        apply_output_pipeline_flag: bool,
        bias_flag: bool,
    ) -> LayerStackConfig:
        return LayerStackConfig(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            last_layer_bias_option=last_layer_bias_option,
            apply_output_pipeline_flag=apply_output_pipeline_flag,
            layer_config=LayerConfig(
                activation=activation,
                layer_norm_position=layer_norm_position,
                residual_connection_option=residual_connection_option,
                dropout_probability=dropout_probability,
                gate_config=None,
                halting_config=None,
                layer_model_config=LinearLayerConfig(
                    bias_flag=bias_flag,
                ),
            ),
        )

    @staticmethod
    def __default_stack_options(
        stack_options: LinearStackOptions | None,
    ) -> LinearStackOptions:
        return stack_options or default_linear_stack_options(config)

    @staticmethod
    def __default_submodule_stack_options(
        submodule_stack_options: ControllerStackOptions | None,
    ) -> ControllerStackOptions:
        return submodule_stack_options or default_submodule_stack_options(config)

    @staticmethod
    def __default_layer_controller_options(
        layer_controller_options: LayerControllerOptions | None,
    ) -> LayerControllerOptions:
        return layer_controller_options or default_layer_controller_options(config)

    @staticmethod
    def __default_dynamic_memory_options(
        dynamic_memory_options: DynamicMemoryOptions | None,
    ) -> DynamicMemoryOptions:
        return dynamic_memory_options or default_dynamic_memory_options(config)

    @staticmethod
    def __default_recurrent_controller_options(
        recurrent_controller_options: RecurrentControllerOptions | None,
    ) -> RecurrentControllerOptions:
        return recurrent_controller_options or default_recurrent_controller_options(
            config
        )

    @staticmethod
    def __default_adaptive_generator_stack_options(
        adaptive_generator_stack_options: (
            adaptive_options.AdaptiveGeneratorStackOptions | None
        ),
    ) -> (
        adaptive_options.AdaptiveGeneratorStackOptions
    ):
        return adaptive_generator_stack_options or default_adaptive_generator_stack_options(
            config
        )

    @staticmethod
    def __default_adaptive_generator_stack_source(
        adaptive_generator_stack_source: (
            adaptive_options.AdaptiveGeneratorStackSource | None
        ),
        prefix: str,
    ) -> adaptive_options.AdaptiveGeneratorStackSource:
        return adaptive_generator_stack_source or default_adaptive_generator_stack_source(
            config,
            prefix,
        )

    @staticmethod
    def __default_boundary_options(
        boundary_options: AdaptiveBoundaryProjectionOptions | None,
        prefix: str,
    ) -> AdaptiveBoundaryProjectionOptions:
        return boundary_options or default_boundary_options(config, prefix)
