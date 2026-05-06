import models.linear_adaptive.config as config

from emperor.halting.config import StickBreakingConfig
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.base.layer.config import LayerConfig, LayerStackConfig
from models.linear_adaptive.experiment_config import ExperimentConfig
from emperor.linears.core.config import AdaptiveLinearLayerConfig, LinearLayerConfig
from emperor.augmentations.adaptive_parameters.config import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.augmentations.adaptive_parameters.core.bias import (
    GeneratorDynamicBiasConfig,
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
    AxisMaskOptions,
    BankExpansionFactorOptions,
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.base.enums import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class LinearAdaptiveConfigBuilder:
    def __init__(
        self,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        output_dim: int = config.OUTPUT_DIM,
        bias_flag: bool = config.BIAS_FLAG,
        layer_norm_position: LayerNormPositionOptions = config.LAYER_NORM_POSITION,
        stack_layer_norm_position: LayerNormPositionOptions | None = None,
        generator_depth: DynamicDepthOptions = config.GENERATOR_DEPTH,
        diagonal_option: DynamicDiagonalOptions = config.DIAGONAL_OPTION,
        bias_option: DynamicBiasOptions = config.BIAS_OPTION,
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
        row_mask_option: AxisMaskOptions = config.ROW_MASK_OPTION,
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
        halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
            config.HALTING_HIDDEN_STATE_MODE
        ),
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
        adaptive_generator_stack_num_layers: int = config.ADAPTIVE_GENERATOR_STACK_NUM_LAYERS,
        adaptive_generator_stack_hidden_dim: int = config.ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM,
        adaptive_generator_stack_activation: ActivationOptions = config.ADAPTIVE_GENERATOR_STACK_ACTIVATION,
        adaptive_generator_stack_residual_flag: bool = config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_FLAG,
        adaptive_generator_stack_dropout_probability: float = config.ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY,
        adaptive_generator_stack_layer_norm_position: LayerNormPositionOptions = config.ADAPTIVE_GENERATOR_STACK_LAYER_NORM_POSITION,
        adaptive_generator_stack_last_layer_bias_option: LastLayerBiasOptions = config.ADAPTIVE_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
        adaptive_generator_stack_apply_output_pipeline_flag: bool = config.ADAPTIVE_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
    ) -> None:
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bias_flag = bias_flag
        self.layer_norm_position = layer_norm_position
        self.stack_layer_norm_position = stack_layer_norm_position
        self.generator_depth = generator_depth
        self.diagonal_option = diagonal_option
        self.bias_option = bias_option
        self.weight_flag = weight_flag
        self.weight_normalization = weight_normalization
        self.weight_normalization_position = weight_normalization_position
        self.weight_decay_schedule = weight_decay_schedule
        self.weight_decay_rate = weight_decay_rate
        self.weight_decay_warmup_batches = weight_decay_warmup_batches
        self.bias_decay_schedule = bias_decay_schedule
        self.bias_decay_rate = bias_decay_rate
        self.bias_decay_warmup_batches = bias_decay_warmup_batches
        self.bias_bank_expansion_factor = bias_bank_expansion_factor
        self.row_mask_option = row_mask_option
        self.mask_dimension_option = mask_dimension_option
        self.mask_threshold = mask_threshold
        self.mask_surrogate_scale = mask_surrogate_scale
        self.mask_floor = mask_floor
        self.mask_transition_width = mask_transition_width
        self.stack_num_layers = stack_num_layers
        self.stack_activation = stack_activation
        self.stack_residual_flag = stack_residual_flag
        self.stack_dropout_probability = stack_dropout_probability
        self.stack_last_layer_bias_option = stack_last_layer_bias_option
        self.stack_apply_output_pipeline_flag = stack_apply_output_pipeline_flag
        self.stack_gate_flag = stack_gate_flag
        self.gate_hidden_dim = gate_hidden_dim
        self.gate_layer_norm_position = gate_layer_norm_position
        self.gate_stack_num_layers = gate_stack_num_layers
        self.gate_stack_activation = gate_stack_activation
        self.gate_stack_residual_flag = gate_stack_residual_flag
        self.gate_stack_dropout_probability = gate_stack_dropout_probability
        self.gate_stack_last_layer_bias_option = gate_stack_last_layer_bias_option
        self.gate_stack_apply_output_pipeline_flag = (
            gate_stack_apply_output_pipeline_flag
        )
        self.gate_bias_flag = gate_bias_flag
        self.stack_halting_flag = stack_halting_flag
        self.halting_threshold = halting_threshold
        self.halting_dropout = halting_dropout
        self.halting_hidden_state_mode = halting_hidden_state_mode
        self.halting_gate_hidden_dim = halting_gate_hidden_dim
        self.halting_gate_output_dim = halting_gate_output_dim
        self.halting_gate_layer_norm_position = halting_gate_layer_norm_position
        self.halting_gate_stack_num_layers = halting_gate_stack_num_layers
        self.halting_gate_stack_activation = halting_gate_stack_activation
        self.halting_gate_stack_residual_flag = halting_gate_stack_residual_flag
        self.halting_gate_stack_dropout_probability = (
            halting_gate_stack_dropout_probability
        )
        self.halting_gate_stack_last_layer_bias_option = (
            halting_gate_stack_last_layer_bias_option
        )
        self.halting_gate_stack_apply_output_pipeline_flag = (
            halting_gate_stack_apply_output_pipeline_flag
        )
        self.halting_gate_bias_flag = halting_gate_bias_flag
        self.adaptive_generator_stack_num_layers = adaptive_generator_stack_num_layers
        self.adaptive_generator_stack_hidden_dim = adaptive_generator_stack_hidden_dim
        self.adaptive_generator_stack_activation = adaptive_generator_stack_activation
        self.adaptive_generator_stack_residual_flag = (
            adaptive_generator_stack_residual_flag
        )
        self.adaptive_generator_stack_dropout_probability = (
            adaptive_generator_stack_dropout_probability
        )
        self.adaptive_generator_stack_layer_norm_position = (
            adaptive_generator_stack_layer_norm_position
        )
        self.adaptive_generator_stack_last_layer_bias_option = (
            adaptive_generator_stack_last_layer_bias_option
        )
        self.adaptive_generator_stack_apply_output_pipeline_flag = (
            adaptive_generator_stack_apply_output_pipeline_flag
        )

    def build(self) -> "ModelConfig":
        from emperor.config import ModelConfig

        gate_config = self._build_gate_config()
        halting_config = self._build_halting_config()
        adaptive_weight_config = self._build_weight_config()
        adaptive_bias_config = self._build_bias_config()
        adaptive_diagonal_config = self._build_diagonal_config()
        adaptive_mask_config = self._build_mask_config()
        adaptive_model_config = self._build_model_config()

        input_model_config = LayerConfig(
            activation=self.stack_activation,
            layer_norm_position=self.layer_norm_position,
            residual_flag=False,
            dropout_probability=self.stack_dropout_probability,
            gate_config=None,
            halting_config=None,
            shared_halting_flag=False,
            layer_model_config=LinearLayerConfig(
                bias_flag=self.bias_flag,
            ),
        )

        model_config = LayerStackConfig(
            hidden_dim=self.hidden_dim,
            num_layers=self.stack_num_layers,
            last_layer_bias_option=self.stack_last_layer_bias_option,
            apply_output_pipeline_flag=self.stack_apply_output_pipeline_flag,
            layer_config=LayerConfig(
                activation=self.stack_activation,
                layer_norm_position=self.layer_norm_position,
                residual_flag=self.stack_residual_flag,
                dropout_probability=self.stack_dropout_probability,
                gate_config=gate_config,
                halting_config=halting_config,
                shared_halting_flag=False,
                layer_model_config=AdaptiveLinearLayerConfig(
                    bias_flag=self.bias_flag,
                    adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                        weight_config=adaptive_weight_config,
                        bias_config=adaptive_bias_config,
                        diagonal_config=adaptive_diagonal_config,
                        mask_config=adaptive_mask_config,
                        model_config=adaptive_model_config,
                    ),
                ),
            ),
        )

        output_model_config = LayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_flag=False,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            shared_halting_flag=False,
            layer_model_config=LinearLayerConfig(
                bias_flag=self.bias_flag,
            ),
        )

        return ModelConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            experiment_config=ExperimentConfig(
                input_model_config=input_model_config,
                model_config=model_config,
                output_model_config=output_model_config,
            ),
        )

    def _build_gate_config(self) -> LayerStackConfig | None:
        if not self.stack_gate_flag:
            return None
        return LayerStackConfig(
            hidden_dim=self.gate_hidden_dim,
            num_layers=self.gate_stack_num_layers,
            last_layer_bias_option=self.gate_stack_last_layer_bias_option,
            apply_output_pipeline_flag=self.gate_stack_apply_output_pipeline_flag,
            layer_config=LayerConfig(
                activation=self.gate_stack_activation,
                layer_norm_position=self.gate_layer_norm_position,
                residual_flag=self.gate_stack_residual_flag,
                dropout_probability=self.gate_stack_dropout_probability,
                halting_config=None,
                shared_halting_flag=False,
                gate_config=None,
                layer_model_config=LinearLayerConfig(
                    bias_flag=self.gate_bias_flag,
                ),
            ),
        )

    def _build_halting_config(self) -> StickBreakingConfig | None:
        if not self.stack_halting_flag:
            return None
        return StickBreakingConfig(
            threshold=self.halting_threshold,
            halting_dropout=self.halting_dropout,
            hidden_state_mode=self.halting_hidden_state_mode,
            halting_gate_config=LayerStackConfig(
                hidden_dim=self.halting_gate_hidden_dim or self.output_dim,
                output_dim=self.halting_gate_output_dim,
                num_layers=self.halting_gate_stack_num_layers,
                last_layer_bias_option=self.halting_gate_stack_last_layer_bias_option,
                apply_output_pipeline_flag=self.halting_gate_stack_apply_output_pipeline_flag,
                layer_config=LayerConfig(
                    activation=self.halting_gate_stack_activation,
                    layer_norm_position=self.halting_gate_layer_norm_position,
                    residual_flag=self.halting_gate_stack_residual_flag,
                    dropout_probability=self.halting_gate_stack_dropout_probability,
                    halting_config=None,
                    shared_halting_flag=False,
                    gate_config=None,
                    layer_model_config=LinearLayerConfig(
                        bias_flag=self.halting_gate_bias_flag,
                    ),
                ),
            ),
        )

    def _build_weight_config(self) -> DualModelDynamicWeightConfig | None:
        if not self.weight_flag:
            return None
        return DualModelDynamicWeightConfig(
            generator_depth=self.generator_depth,
            normalization_option=self.weight_normalization,
            normalization_position_option=self.weight_normalization_position,
            decay_schedule=self.weight_decay_schedule,
            decay_rate=self.weight_decay_rate,
            decay_warmup_batches=self.weight_decay_warmup_batches,
        )

    def _build_bias_config(self) -> GeneratorDynamicBiasConfig | None:
        if self.bias_option != DynamicBiasOptions.DYNAMIC_PARAMETERS:
            return None
        return GeneratorDynamicBiasConfig(
            bias_flag=self.bias_flag,
            decay_schedule=self.bias_decay_schedule,
            decay_rate=self.bias_decay_rate,
            decay_warmup_batches=self.bias_decay_warmup_batches,
        )

    def _build_diagonal_config(self) -> StandardDynamicDiagonalConfig | None:
        config_cls = {
            DynamicDiagonalOptions.DISABLED: None,
            DynamicDiagonalOptions.DIAGONAL: StandardDynamicDiagonalConfig,
            DynamicDiagonalOptions.ANTI_DIAGONAL: AntiDynamicDiagonalConfig,
            DynamicDiagonalOptions.DIAGONAL_AND_ANTI_DIAGONAL: CombinedDynamicDiagonalConfig,
        }[self.diagonal_option]
        if config_cls is None:
            return None
        return config_cls()

    def _build_mask_config(self) -> TopSliceAxisMaskConfig | None:
        config_cls = {
            AxisMaskOptions.DISABLED: None,
            AxisMaskOptions.WEIGHT_INFORMED_SCORE: WeightInformedScoreAxisMaskConfig,
            AxisMaskOptions.PER_AXIS_SCORE: PerAxisScoreMaskConfig,
            AxisMaskOptions.TOP_SLICE: TopSliceAxisMaskConfig,
            AxisMaskOptions.OUTER_PRODUCT: OuterProductMaskConfig,
            AxisMaskOptions.DIAGONAL: DiagonalAxisMaskConfig,
        }[self.row_mask_option]
        if config_cls is None:
            return None
        kwargs = dict(
            mask_threshold=self.mask_threshold,
            mask_surrogate_scale=self.mask_surrogate_scale,
            mask_floor=self.mask_floor,
        )
        if config_cls in {
            WeightInformedScoreAxisMaskConfig,
            PerAxisScoreMaskConfig,
            TopSliceAxisMaskConfig,
        }:
            kwargs["mask_dimension_option"] = self.mask_dimension_option
        if config_cls in {TopSliceAxisMaskConfig, DiagonalAxisMaskConfig}:
            kwargs["mask_transition_width"] = self.mask_transition_width
        return config_cls(**kwargs)

    def _build_model_config(self) -> LayerStackConfig:
        return LayerStackConfig(
            hidden_dim=self.adaptive_generator_stack_hidden_dim,
            num_layers=self.adaptive_generator_stack_num_layers,
            last_layer_bias_option=self.adaptive_generator_stack_last_layer_bias_option,
            apply_output_pipeline_flag=self.adaptive_generator_stack_apply_output_pipeline_flag,
            layer_config=LayerConfig(
                activation=self.adaptive_generator_stack_activation,
                layer_norm_position=self.adaptive_generator_stack_layer_norm_position,
                residual_flag=self.adaptive_generator_stack_residual_flag,
                dropout_probability=self.adaptive_generator_stack_dropout_probability,
                gate_config=None,
                halting_config=None,
                shared_halting_flag=False,
                layer_model_config=LinearLayerConfig(
                    bias_flag=self.bias_flag,
                ),
            ),
        )
