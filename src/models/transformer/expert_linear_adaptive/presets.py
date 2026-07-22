from emperor.augmentations.adaptive_parameters import (
    AdditiveDynamicBiasConfig,
    AffineTransformDynamicBiasConfig,
    AntiDynamicDiagonalConfig,
    CombinedDynamicDiagonalConfig,
    DiagonalAxisMaskConfig,
    DualModelDynamicWeightConfig,
    GeneratorDynamicBiasConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    MultiplicativeDynamicBiasConfig,
    OuterProductMaskConfig,
    PerAxisScoreMaskConfig,
    SigmoidGatedDynamicBiasConfig,
    SingleModelDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
    StandardDynamicDiagonalConfig,
    TanhGatedDynamicBiasConfig,
    TopSliceAxisMaskConfig,
    WeightedBankDynamicBiasConfig,
    WeightInformedScoreAxisMaskConfig,
)
from emperor.config import BaseOptions
from emperor.datasets.text.translation import Multi30kDeEn
from emperor.embedding.absolute import TextLearnedPositionalEmbeddingConfig
from emperor.layers import LayerNormPositionOptions, ResidualConnectionOptions
from model_runtime.packages import (
    BuilderBackedExperimentPresetsBase,
    ExperimentPresetsBase,
    PresetDefinition,
)
from model_runtime.runs import ExperimentBase

from . import config, dataset_options
from .config_builder import TransformerExpertLinearAdaptiveConfigBuilder
from .model import Model
from .runtime_defaults import runtime_from_flat


class ExperimentPreset(BaseOptions):
    BASELINE = 1
    PRE_NORM = 2
    POST_NORM = 3
    LEARNED_POSITIONAL = 4
    ATTENTION_BIAS = 5
    GATING = 6
    HALTING = 7
    GATING_HALTING = 8
    MEMORY = 9
    GATING_MEMORY = 10
    HALTING_MEMORY = 11
    GATING_HALTING_MEMORY = 12
    RECURRENT = 13
    RECURRENT_GATING = 14
    RECURRENT_HALTING = 15
    RECURRENT_MEMORY = 16
    RECURRENT_GATING_HALTING = 17
    RECURRENT_GATING_MEMORY = 18
    RECURRENT_HALTING_MEMORY = 19
    RECURRENT_GATING_HALTING_MEMORY = 20
    RESIDUAL = 21
    RESIDUAL_POST_NORM = 22
    RESIDUAL_GATING = 23
    RESIDUAL_HALTING = 24
    RESIDUAL_MEMORY = 25
    RECURRENT_RESIDUAL = 26
    RECURRENT_POST_NORM = 27
    SINGLE_MODEL_WEIGHT = 28
    DUAL_MODEL_WEIGHT = 29
    LOW_RANK_WEIGHT = 30
    HYPERNETWORK_WEIGHT = 31
    LAYERED_WEIGHTED_BANK_WEIGHT = 32
    SOFT_WEIGHTED_BANK_WEIGHT = 33
    AFFINE_TRANSFORM_BIAS = 34
    ADDITIVE_BIAS = 35
    GENERATOR_BIAS = 36
    MULTIPLICATIVE_BIAS = 37
    SIGMOID_GATED_BIAS = 38
    TANH_GATED_BIAS = 39
    WEIGHTED_BANK_BIAS = 40
    STANDARD_DIAGONAL = 41
    ANTI_DIAGONAL = 42
    COMBINED_DIAGONAL = 43
    DIAGONAL_AXIS_MASK = 44
    OUTER_PRODUCT_MASK = 45
    PER_AXIS_SCORE_MASK = 46
    TOP_SLICE_AXIS_MASK = 47
    WEIGHT_INFORMED_SCORE_MASK = 48
    TOP1_SWITCH_AUX = 49
    LOW_RANK_EXPERT_WEIGHT = 50


_COMMON_OVERRIDES = {
    "BASELINE": {},
    "PRE_NORM": {
        "encoder_layer_norm_position": LayerNormPositionOptions.BEFORE,
        "decoder_layer_norm_position": LayerNormPositionOptions.BEFORE,
    },
    "POST_NORM": {
        "encoder_layer_norm_position": LayerNormPositionOptions.AFTER,
        "decoder_layer_norm_position": LayerNormPositionOptions.AFTER,
    },
    "LEARNED_POSITIONAL": {
        "positional_embedding_option": TextLearnedPositionalEmbeddingConfig
    },
    "ATTENTION_BIAS": {
        "attn_bias_flag": True,
        "attn_add_key_value_bias_flag": True,
    },
    "GATING": {"stack_gate_flag": True},
    "HALTING": {"stack_halting_flag": True},
    "GATING_HALTING": {
        "stack_gate_flag": True,
        "stack_halting_flag": True,
    },
    "MEMORY": {"memory_flag": True},
    "GATING_MEMORY": {"stack_gate_flag": True, "memory_flag": True},
    "HALTING_MEMORY": {"stack_halting_flag": True, "memory_flag": True},
    "GATING_HALTING_MEMORY": {
        "stack_gate_flag": True,
        "stack_halting_flag": True,
        "memory_flag": True,
    },
    "RECURRENT": {"recurrent_flag": True},
    "RECURRENT_GATING": {
        "recurrent_flag": True,
        "recurrent_stack_gate_flag": True,
    },
    "RECURRENT_HALTING": {
        "recurrent_flag": True,
        "recurrent_stack_halting_flag": True,
    },
    "RECURRENT_MEMORY": {"recurrent_flag": True, "memory_flag": True},
    "RECURRENT_GATING_HALTING": {
        "recurrent_flag": True,
        "recurrent_stack_gate_flag": True,
        "recurrent_stack_halting_flag": True,
    },
    "RECURRENT_GATING_MEMORY": {
        "recurrent_flag": True,
        "recurrent_stack_gate_flag": True,
        "memory_flag": True,
    },
    "RECURRENT_HALTING_MEMORY": {
        "recurrent_flag": True,
        "recurrent_stack_halting_flag": True,
        "memory_flag": True,
    },
    "RECURRENT_GATING_HALTING_MEMORY": {
        "recurrent_flag": True,
        "recurrent_stack_gate_flag": True,
        "recurrent_stack_halting_flag": True,
        "memory_flag": True,
    },
    "RESIDUAL": {
        "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL
    },
    "RESIDUAL_POST_NORM": {
        "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
        "encoder_layer_norm_position": LayerNormPositionOptions.AFTER,
        "decoder_layer_norm_position": LayerNormPositionOptions.AFTER,
    },
    "RESIDUAL_GATING": {
        "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
        "stack_gate_flag": True,
    },
    "RESIDUAL_HALTING": {
        "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
        "stack_halting_flag": True,
    },
    "RESIDUAL_MEMORY": {
        "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
        "memory_flag": True,
    },
    "RECURRENT_RESIDUAL": {
        "recurrent_flag": True,
        "recurrent_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
    },
    "RECURRENT_POST_NORM": {
        "recurrent_flag": True,
        "encoder_layer_norm_position": LayerNormPositionOptions.AFTER,
        "decoder_layer_norm_position": LayerNormPositionOptions.AFTER,
    },
}

_ADAPTIVE_OVERRIDES = {
    "SINGLE_MODEL_WEIGHT": {
        "attention_projection_adaptive_weight_option": (SingleModelDynamicWeightConfig)
    },
    "DUAL_MODEL_WEIGHT": {
        "attention_projection_adaptive_weight_option": DualModelDynamicWeightConfig
    },
    "LOW_RANK_WEIGHT": {
        "attention_projection_adaptive_weight_option": LowRankDynamicWeightConfig
    },
    "HYPERNETWORK_WEIGHT": {
        "attention_projection_adaptive_weight_option": (HypernetworkDynamicWeightConfig)
    },
    "LAYERED_WEIGHTED_BANK_WEIGHT": {
        "attention_projection_adaptive_weight_option": (
            LayeredWeightedBankDynamicWeightConfig
        )
    },
    "SOFT_WEIGHTED_BANK_WEIGHT": {
        "attention_projection_adaptive_weight_option": (
            SoftWeightedBankDynamicWeightConfig
        )
    },
}


def _all_adaptive_roles(field: str, value: type) -> dict[str, type]:
    return {
        f"attention_projection_adaptive_{field}": value,
        f"attention_expert_adaptive_{field}": value,
        f"router_adaptive_{field}": value,
        f"feed_forward_adaptive_{field}": value,
    }


_ADAPTIVE_OVERRIDES.update(
    {
        "AFFINE_TRANSFORM_BIAS": _all_adaptive_roles(
            "bias_option", AffineTransformDynamicBiasConfig
        ),
        "ADDITIVE_BIAS": _all_adaptive_roles("bias_option", AdditiveDynamicBiasConfig),
        "GENERATOR_BIAS": _all_adaptive_roles(
            "bias_option", GeneratorDynamicBiasConfig
        ),
        "MULTIPLICATIVE_BIAS": _all_adaptive_roles(
            "bias_option", MultiplicativeDynamicBiasConfig
        ),
        "SIGMOID_GATED_BIAS": _all_adaptive_roles(
            "bias_option", SigmoidGatedDynamicBiasConfig
        ),
        "TANH_GATED_BIAS": _all_adaptive_roles(
            "bias_option", TanhGatedDynamicBiasConfig
        ),
        "WEIGHTED_BANK_BIAS": _all_adaptive_roles(
            "bias_option", WeightedBankDynamicBiasConfig
        ),
        "STANDARD_DIAGONAL": _all_adaptive_roles(
            "diagonal_option", StandardDynamicDiagonalConfig
        ),
        "ANTI_DIAGONAL": _all_adaptive_roles(
            "diagonal_option", AntiDynamicDiagonalConfig
        ),
        "COMBINED_DIAGONAL": _all_adaptive_roles(
            "diagonal_option", CombinedDynamicDiagonalConfig
        ),
        "DIAGONAL_AXIS_MASK": _all_adaptive_roles(
            "row_mask_option", DiagonalAxisMaskConfig
        ),
        "OUTER_PRODUCT_MASK": _all_adaptive_roles(
            "row_mask_option", OuterProductMaskConfig
        ),
        "PER_AXIS_SCORE_MASK": _all_adaptive_roles(
            "row_mask_option", PerAxisScoreMaskConfig
        ),
        "TOP_SLICE_AXIS_MASK": _all_adaptive_roles(
            "row_mask_option", TopSliceAxisMaskConfig
        ),
        "WEIGHT_INFORMED_SCORE_MASK": _all_adaptive_roles(
            "row_mask_option", WeightInformedScoreAxisMaskConfig
        ),
    }
)

_EXPERT_OVERRIDES = {
    "TOP1_SWITCH_AUX": {
        "top_k": 1,
        "normalize_probabilities_flag": False,
        "switch_loss_weight": 0.1,
    },
    "LOW_RANK_EXPERT_WEIGHT": {
        "attention_expert_adaptive_weight_option": LowRankDynamicWeightConfig,
        "feed_forward_adaptive_weight_option": LowRankDynamicWeightConfig,
    },
}

_PRESET_DEFINITIONS = {
    ExperimentPreset[name]: PresetDefinition(
        preset_values=values,
        description=(
            f"Adaptive expert Transformer preset: {name.replace('_', ' ').lower()}."
        ),
    )
    for name, values in {
        **_COMMON_OVERRIDES,
        **_ADAPTIVE_OVERRIDES,
        **_EXPERT_OVERRIDES,
    }.items()
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=TransformerExpertLinearAdaptiveConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
            default_dataset=Multi30kDeEn,
        )

    def _dataset_config(self, dataset: type) -> dict:
        return {
            "vocab_size": dataset.vocab_size,
            "source_sequence_length": dataset.source_sequence_length,
            "target_sequence_length": dataset.target_sequence_length,
        }

    def _preset(self, **kwargs):
        return self._builder_type(runtime=runtime_from_flat(kwargs)).build()


class Experiment(ExperimentBase):
    def __init__(
        self,
        experiment_preset: ExperimentPreset | None = None,
        experiment_task=None,
        *,
        model_package,
        run_artifacts=None,
    ) -> None:
        super().__init__(
            experiment_preset,
            experiment_task=experiment_task,
            model_package=model_package,
            run_artifacts=run_artifacts,
        )

    def _num_epochs(self) -> int:
        return config.NUM_EPOCHS

    def _dataset_options(self) -> list:
        return dataset_options.DATASET_OPTIONS_BY_TASK[
            dataset_options.DEFAULT_EXPERIMENT_TASK
        ]

    def _model_type(self) -> type:
        return Model

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        return ExperimentPresets()

    def _experiment_preset_enum(self) -> type[BaseOptions]:
        return ExperimentPreset

    def _dataset_constructor_kwargs(self, training_run) -> dict:
        experiment_config = training_run.config.experiment_config
        return {
            "batch_size": training_run.config.batch_size,
            "source_sequence_length": experiment_config.source_sequence_length,
            "target_sequence_length": experiment_config.target_sequence_length,
        }
