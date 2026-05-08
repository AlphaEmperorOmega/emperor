import models.linear_adaptive.config as config

from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.augmentations.adaptive_parameters.core.bias import (
    AdditiveDynamicBiasConfig,
    AffineTransformDynamicBiasConfig,
    GeneratorDynamicBiasConfig,
    MultiplicativeDynamicBiasConfig,
    SigmoidGatedDynamicBiasConfig,
    TanhGatedDynamicBiasConfig,
    WeightedBankDynamicBiasConfig,
)
from emperor.augmentations.adaptive_parameters.core.diagonal import (
    AntiDynamicDiagonalConfig,
    CombinedDynamicDiagonalConfig,
    StandardDynamicDiagonalConfig,
)
from emperor.augmentations.adaptive_parameters.core.mask import (
    DiagonalAxisMaskConfig,
    OuterProductMaskConfig,
    PerAxisScoreMaskConfig,
    TopSliceAxisMaskConfig,
    WeightInformedScoreAxisMaskConfig,
)
from emperor.augmentations.adaptive_parameters.core.weight import (
    DualModelDynamicWeightConfig,
)
from emperor.augmentations.adaptive_parameters.options import (
    BankExpansionFactorOptions,
    DynamicDepthOptions,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.augmentations.adaptive_parameters.core.bias import DynamicBiasConfig
from emperor.augmentations.adaptive_parameters.core.diagonal import (
    DynamicDiagonalConfig,
)
from emperor.augmentations.adaptive_parameters.core.mask import AxisMaskConfig
from emperor.base.enums import (
    ActivationOptions,
    BaseOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.base.layer import LayerConfig, LayerStackConfig
from emperor.datasets.image.classification.mnist import Mnist
from emperor.experiments.base import (
    ExperimentBase,
    ExperimentPresetsBase,
    SearchMode,
    create_search_space,
)
from emperor.halting.config import StickBreakingConfig
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.linears.core.config import AdaptiveLinearLayerConfig, LinearLayerConfig
from models.linear_adaptive.experiment_config import ExperimentConfig
from models.linear_adaptive.model import Model

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig
    from emperor.augmentations.adaptive_parameters.core.bias import DynamicBiasConfig
    from emperor.augmentations.adaptive_parameters.core.diagonal import (
        DynamicDiagonalConfig,
    )
    from emperor.augmentations.adaptive_parameters.core.mask import AxisMaskConfig
    from emperor.augmentations.adaptive_parameters.core.weight import (
        DynamicWeightConfig,
    )


class ExperimentOptions(BaseOptions):
    PRESET = 0
    CONFIG = 1
    GENERATOR_DEPTH = 2
    DIAGONAL = 3
    BIAS = 4
    COMBINED = 5
    WEIGHT = 6
    WEIGHT_NORMALIZATION = 7
    ROW_MASK = 8


class ExperimentPresets(ExperimentPresetsBase):
    def get_config(
        self,
        model_config_options: ExperimentOptions = ExperimentOptions.PRESET,
        dataset: type = Mnist,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
    ) -> list["ModelConfig"]:
        match model_config_options:
            case ExperimentOptions.PRESET:
                return self._create_default_preset_configs(dataset)
            case ExperimentOptions.CONFIG:
                return self._create_preset_search_space_configs(
                    dataset, search_mode
                )
            case ExperimentOptions.GENERATOR_DEPTH:
                return self._search_configs(
                    dataset,
                    search_mode,
                    log_folder,
                    generator_depth=[
                        DynamicDepthOptions.DISABLED,
                        DynamicDepthOptions.DEPTH_OF_ONE,
                        DynamicDepthOptions.DEPTH_OF_TWO,
                        DynamicDepthOptions.DEPTH_OF_THREE,
                    ],
                )
            case ExperimentOptions.DIAGONAL:
                return self._search_configs(
                    dataset,
                    search_mode,
                    log_folder,
                    diagonal_option=[
                        None,
                        StandardDynamicDiagonalConfig,
                        AntiDynamicDiagonalConfig,
                        CombinedDynamicDiagonalConfig,
                    ],
                )
            case ExperimentOptions.BIAS:
                return self._search_configs(
                    dataset,
                    search_mode,
                    log_folder,
                    bias_option=[
                        None,
                        AffineTransformDynamicBiasConfig,
                        AdditiveDynamicBiasConfig,
                        GeneratorDynamicBiasConfig,
                        MultiplicativeDynamicBiasConfig,
                        SigmoidGatedDynamicBiasConfig,
                        TanhGatedDynamicBiasConfig,
                        WeightedBankDynamicBiasConfig,
                    ],
                )
            case ExperimentOptions.COMBINED:
                return self._search_configs(
                    dataset,
                    search_mode,
                    log_folder,
                    generator_depth=[
                        DynamicDepthOptions.DEPTH_OF_ONE,
                        DynamicDepthOptions.DEPTH_OF_TWO,
                    ],
                    diagonal_option=[
                        None,
                        StandardDynamicDiagonalConfig,
                    ],
                    bias_option=[
                        None,
                        GeneratorDynamicBiasConfig,
                    ],
                    weight_flag=[False, True],
                    row_mask_option=[
                        None,
                        WeightInformedScoreAxisMaskConfig,
                    ],
                )
            case ExperimentOptions.WEIGHT:
                return self._search_configs(
                    dataset,
                    search_mode,
                    log_folder,
                    weight_flag=[False, True],
                )
            case ExperimentOptions.WEIGHT_NORMALIZATION:
                return self._search_configs(
                    dataset,
                    search_mode,
                    log_folder,
                    weight_flag=[True],
                    weight_normalization=[
                        WeightNormalizationOptions.DISABLED,
                        WeightNormalizationOptions.CLAMP,
                        WeightNormalizationOptions.L2_SCALE,
                        WeightNormalizationOptions.SOFT_CLAMP,
                        WeightNormalizationOptions.RMS,
                        WeightNormalizationOptions.SIGMOID_SCALE,
                    ],
                )
            case ExperimentOptions.ROW_MASK:
                return self._search_configs(
                    dataset,
                    search_mode,
                    log_folder,
                    row_mask_option=[
                        None,
                        WeightInformedScoreAxisMaskConfig,
                        PerAxisScoreMaskConfig,
                        TopSliceAxisMaskConfig,
                    ],
                )
            case _:
                raise ValueError(
                    "The specified option is not supported. Please choose a valid "
                    "`ExperimentOptions`."
                )

    def _search_configs(
        self,
        dataset: type,
        search_mode: SearchMode,
        log_folder: str | None,
        **search_space_overrides,
    ) -> list["ModelConfig"]:
        base_config = {
            **self._dataset_config(dataset),
            **self._best_params(dataset, log_folder),
        }
        search_space = {
            **self._extract_search_space_from_config(search_mode),
            **search_space_overrides,
        }
        return create_search_space(self._preset, base_config, search_space, search_mode)

    def _preset(
        self,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        output_dim: int = config.OUTPUT_DIM,
        bias_flag: bool = config.BIAS_FLAG,
        layer_norm_position: LayerNormPositionOptions = config.LAYER_NORM_POSITION,
        stack_layer_norm_position: LayerNormPositionOptions | None = None,
        generator_depth: DynamicDepthOptions = config.WEIGHT_GENERATOR_DEPTH,
        diagonal_option: type[DynamicDiagonalConfig] | None = config.DIAGONAL_OPTION,
        bias_option: type[DynamicBiasConfig] | None = config.BIAS_OPTION,
        weight_flag: bool = config.WEIGHT_FLAG,
        weight_normalization: WeightNormalizationOptions = config.WEIGHT_NORMALIZATION,
        weight_normalization_position: WeightNormalizationPositionOptions = config.WEIGHT_NORMALIZATION_POSITION,
        weight_decay_schedule: WeightDecayScheduleOptions = config.WEIGHT_DECAY_SCHEDULE,
        weight_decay_rate: float = config.WEIGHT_DECAY_RATE,
        weight_decay_warmup_batches: int = config.WEIGHT_DECAY_WARMUP_BATCHES,
        bias_decay_schedule: WeightDecayScheduleOptions = config.BIAS_DECAY_SCHEDULE,
        bias_decay_rate: float = config.BIAS_DECAY_RATE,
        bias_decay_warmup_batches: int = config.BIAS_DECAY_WARMUP_BATCHES,
        bias_bank_expansion_factor: BankExpansionFactorOptions = config.BIAS_BANK_EXPANSION_FACTOR,
        row_mask_option: type[AxisMaskConfig] | None = config.ROW_MASK_OPTION,
        mask_dimension_option: MaskDimensionOptions = config.MASK_DIMENSION_OPTION,
        mask_threshold: float = config.MASK_THRESHOLD,
        mask_surrogate_scale: float = config.MASK_SURROGATE_SCALE,
        mask_floor: float = config.MASK_FLOOR,
        mask_transition_width: float = config.MASK_TRANSITION_WIDTH,
        stack_num_layers: int = config.STACK_NUM_LAYERS,
        stack_activation: ActivationOptions = config.STACK_ACTIVATION,
        stack_residual_flag: bool = config.STACK_RESIDUAL_FLAG,
        stack_dropout_probability: float = config.STACK_DROPOUT_PROBABILITY,
        stack_last_layer_bias_option: LastLayerBiasOptions = config.STACK_LAST_LAYER_BIAS_OPTION,
        stack_apply_output_pipeline_flag: bool = config.STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        stack_gate_flag: bool = config.GATE_FLAG,
        gate_hidden_dim: int = config.GATE_HIDDEN_DIM,
        gate_layer_norm_position: LayerNormPositionOptions = config.GATE_LAYER_NORM_POSITION,
        gate_stack_num_layers: int = config.GATE_STACK_NUM_LAYERS,
        gate_stack_activation: ActivationOptions = config.GATE_STACK_ACTIVATION,
        gate_stack_residual_flag: bool = config.GATE_STACK_RESIDUAL_FLAG,
        gate_stack_dropout_probability: float = config.GATE_STACK_DROPOUT_PROBABILITY,
        gate_stack_last_layer_bias_option: LastLayerBiasOptions = config.GATE_STACK_LAST_LAYER_BIAS_OPTION,
        gate_stack_apply_output_pipeline_flag: bool = config.GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        gate_bias_flag: bool = config.GATE_BIAS_FLAG,
        stack_halting_flag: bool = config.HALTING_FLAG,
        halting_threshold: float = config.HALTING_THRESHOLD,
        halting_dropout: float = config.HALTING_DROPOUT,
        halting_hidden_state_mode: HaltingHiddenStateModeOptions = config.HALTING_HIDDEN_STATE_MODE,
        halting_gate_hidden_dim: int = config.HALTING_GATE_HIDDEN_DIM,
        halting_gate_output_dim: int = config.HALTING_GATE_OUTPUT_DIM,
        halting_gate_layer_norm_position: LayerNormPositionOptions = config.HALTING_GATE_LAYER_NORM_POSITION,
        halting_gate_stack_num_layers: int = config.HALTING_GATE_STACK_NUM_LAYERS,
        halting_gate_stack_activation: ActivationOptions = config.HALTING_GATE_STACK_ACTIVATION,
        halting_gate_stack_residual_flag: bool = config.HALTING_GATE_STACK_RESIDUAL_FLAG,
        halting_gate_stack_dropout_probability: float = config.HALTING_GATE_STACK_DROPOUT_PROBABILITY,
        halting_gate_stack_last_layer_bias_option: LastLayerBiasOptions = config.HALTING_GATE_STACK_LAST_LAYER_BIAS_OPTION,
        halting_gate_stack_apply_output_pipeline_flag: bool = config.HALTING_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        halting_gate_bias_flag: bool = config.HALTING_GATE_BIAS_FLAG,
        adaptive_generator_stack_num_layers: int = config.AUGMENTATION_GENERATOR_STACK_NUM_LAYERS,
        adaptive_generator_stack_hidden_dim: int = config.AUGMENTATION_GENERATOR_STACK_HIDDEN_DIM,
        adaptive_generator_stack_activation: ActivationOptions = config.AUGMENTATION_GENERATOR_STACK_ACTIVATION,
        adaptive_generator_stack_residual_flag: bool = config.AUGMENTATION_GENERATOR_STACK_RESIDUAL_FLAG,
        adaptive_generator_stack_dropout_probability: float = config.AUGMENTATION_GENERATOR_STACK_DROPOUT_PROBABILITY,
        adaptive_generator_stack_layer_norm_position: LayerNormPositionOptions = config.AUGMENTATION_GENERATOR_STACK_LAYER_NORM_POSITION,
        adaptive_generator_stack_last_layer_bias_option: LastLayerBiasOptions = config.AUGMENTATION_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
        adaptive_generator_stack_apply_output_pipeline_flag: bool = config.AUGMENTATION_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
    ) -> "ModelConfig":
        from models.linear_adaptive.config_builder import LinearAdaptiveConfigBuilder

        return LinearAdaptiveConfigBuilder(
            batch_size=batch_size,
            learning_rate=learning_rate,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            bias_flag=bias_flag,
            layer_norm_position=layer_norm_position,
            stack_layer_norm_position=stack_layer_norm_position,
            generator_depth=generator_depth,
            diagonal_option=diagonal_option,
            bias_option=bias_option,
            weight_flag=weight_flag,
            weight_normalization=weight_normalization,
            weight_normalization_position=weight_normalization_position,
            weight_decay_schedule=weight_decay_schedule,
            weight_decay_rate=weight_decay_rate,
            weight_decay_warmup_batches=weight_decay_warmup_batches,
            bias_decay_schedule=bias_decay_schedule,
            bias_decay_rate=bias_decay_rate,
            bias_decay_warmup_batches=bias_decay_warmup_batches,
            bias_bank_expansion_factor=bias_bank_expansion_factor,
            row_mask_option=row_mask_option,
            mask_dimension_option=mask_dimension_option,
            mask_threshold=mask_threshold,
            mask_surrogate_scale=mask_surrogate_scale,
            mask_floor=mask_floor,
            mask_transition_width=mask_transition_width,
            stack_num_layers=stack_num_layers,
            stack_activation=stack_activation,
            stack_residual_flag=stack_residual_flag,
            stack_dropout_probability=stack_dropout_probability,
            stack_last_layer_bias_option=stack_last_layer_bias_option,
            stack_apply_output_pipeline_flag=stack_apply_output_pipeline_flag,
            stack_gate_flag=stack_gate_flag,
            gate_hidden_dim=gate_hidden_dim,
            gate_layer_norm_position=gate_layer_norm_position,
            gate_stack_num_layers=gate_stack_num_layers,
            gate_stack_activation=gate_stack_activation,
            gate_stack_residual_flag=gate_stack_residual_flag,
            gate_stack_dropout_probability=gate_stack_dropout_probability,
            gate_stack_last_layer_bias_option=gate_stack_last_layer_bias_option,
            gate_stack_apply_output_pipeline_flag=gate_stack_apply_output_pipeline_flag,
            gate_bias_flag=gate_bias_flag,
            stack_halting_flag=stack_halting_flag,
            halting_threshold=halting_threshold,
            halting_dropout=halting_dropout,
            halting_hidden_state_mode=halting_hidden_state_mode,
            halting_gate_hidden_dim=halting_gate_hidden_dim,
            halting_gate_output_dim=halting_gate_output_dim,
            halting_gate_layer_norm_position=halting_gate_layer_norm_position,
            halting_gate_stack_num_layers=halting_gate_stack_num_layers,
            halting_gate_stack_activation=halting_gate_stack_activation,
            halting_gate_stack_residual_flag=halting_gate_stack_residual_flag,
            halting_gate_stack_dropout_probability=halting_gate_stack_dropout_probability,
            halting_gate_stack_last_layer_bias_option=halting_gate_stack_last_layer_bias_option,
            halting_gate_stack_apply_output_pipeline_flag=halting_gate_stack_apply_output_pipeline_flag,
            halting_gate_bias_flag=halting_gate_bias_flag,
            adaptive_generator_stack_num_layers=adaptive_generator_stack_num_layers,
            adaptive_generator_stack_hidden_dim=adaptive_generator_stack_hidden_dim,
            adaptive_generator_stack_activation=adaptive_generator_stack_activation,
            adaptive_generator_stack_residual_flag=adaptive_generator_stack_residual_flag,
            adaptive_generator_stack_dropout_probability=adaptive_generator_stack_dropout_probability,
            adaptive_generator_stack_layer_norm_position=adaptive_generator_stack_layer_norm_position,
            adaptive_generator_stack_last_layer_bias_option=adaptive_generator_stack_last_layer_bias_option,
            adaptive_generator_stack_apply_output_pipeline_flag=adaptive_generator_stack_apply_output_pipeline_flag,
        ).build()



class Experiment(ExperimentBase):
    def __init__(
        self,
        experiment_option: ExperimentOptions | None = None,
    ) -> None:
        super().__init__(experiment_option)

    def _num_epochs(self) -> int:
        return config.NUM_EPOCHS

    def _dataset_options(self) -> list:
        return config.DATASET_OPTIONS

    def _model_type(self) -> type:
        return Model

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        return ExperimentPresets()

    def _experiment_enumeration(self) -> type[BaseOptions]:
        return ExperimentOptions
