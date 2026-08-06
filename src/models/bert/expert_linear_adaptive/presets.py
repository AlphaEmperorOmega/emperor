import models.bert.expert_linear_adaptive.config as config
import models.bert.expert_linear_adaptive.dataset_options as dataset_options
from emperor.augmentations.adaptive_parameters import (
    CombinedDynamicDiagonalConfig,
    DynamicDepthOptions,
    GeneratorDynamicBiasConfig,
    SingleModelDynamicWeightConfig,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.config import BaseOptions
from emperor.datasets.text.bert_pretraining import PennTreebankBertPretraining
from emperor.embedding.absolute import (
    TextSinusoidalPositionalEmbeddingConfig,
)
from emperor.layers import LayerNormPositionOptions
from model_runtime.packages import (
    BuilderBackedExperimentPresetsBase,
    ExperimentPresetsBase,
    PresetDefinition,
)
from model_runtime.runs import ExperimentBase
from models.bert.expert_linear_adaptive.config_builder import (
    BertExpertLinearAdaptiveConfigBuilder,
)
from models.bert.expert_linear_adaptive.model import Model
from models.bert.expert_linear_adaptive.runtime_defaults import (
    runtime_from_flat,
)


class ExperimentPreset(BaseOptions):
    BASELINE = 1
    PRE_NORM = 2
    POST_NORM = 3
    SINUSOIDAL = 4
    CAUSAL = 5
    ATTENTION_BIAS = 6
    GATING = 7
    HALTING = 8
    GATING_HALTING = 9
    MEMORY = 10
    GATING_MEMORY = 11
    HALTING_MEMORY = 12
    GATING_HALTING_MEMORY = 13
    RECURRENT = 14
    RECURRENT_GATING = 15
    RECURRENT_HALTING = 16
    RECURRENT_MEMORY = 17
    RECURRENT_GATING_HALTING = 18
    RECURRENT_GATING_MEMORY = 19
    RECURRENT_HALTING_MEMORY = 20
    RECURRENT_GATING_HALTING_MEMORY = 21
    TOP1_SWITCH_AUX = 22
    LOW_RANK_EXPERT_WEIGHT = 23
    SINGLE_LAYER_RECURRENT_ATTENTION_RESIDUAL = 24


_PRESET_DEFINITIONS = {
    ExperimentPreset.BASELINE: PresetDefinition(
        preset_values={},
        description=(
            "Default config: a BERT-style pretraining encoder with an adaptive "
            "Mixture of Attention Heads, Mixture-of-Experts feed-forward "
            "sub-stacks, learned positional embeddings, and bidirectional attention."
        ),
    ),
    ExperimentPreset.PRE_NORM: PresetDefinition(
        preset_values={
            "layer_norm_position": LayerNormPositionOptions.BEFORE,
        },
        description=(
            "Default config with layer normalization applied before each encoder "
            "sub-block."
        ),
    ),
    ExperimentPreset.POST_NORM: PresetDefinition(
        preset_values={
            "layer_norm_position": LayerNormPositionOptions.AFTER,
        },
        description=(
            "Default config with layer normalization applied after each encoder "
            "sub-block."
        ),
    ),
    ExperimentPreset.SINUSOIDAL: PresetDefinition(
        preset_values={
            "positional_embedding_option": TextSinusoidalPositionalEmbeddingConfig,
        },
        description="Default config with fixed sinusoidal positional embeddings.",
    ),
    ExperimentPreset.CAUSAL: PresetDefinition(
        preset_values={
            "causal_attention_mask_flag": True,
        },
        description="Default config with causal attention masking enabled for "
        "autoregressive modeling.",
    ),
    ExperimentPreset.ATTENTION_BIAS: PresetDefinition(
        preset_values={
            "attn_bias_flag": True,
            "attn_add_key_value_bias_flag": True,
        },
        description="Default config with attention projection bias and key/value bias "
        "enabled.",
    ),
    ExperimentPreset.GATING: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
        },
        description="Default config with per-encoder-block gating enabled.",
    ),
    ExperimentPreset.HALTING: PresetDefinition(
        preset_values={
            "stack_halting_flag": True,
        },
        description="Default config with encoder-block stack halting enabled.",
    ),
    ExperimentPreset.GATING_HALTING: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
            "stack_halting_flag": True,
        },
        description=(
            "Default config with both encoder-block gating and halting enabled."
        ),
    ),
    ExperimentPreset.MEMORY: PresetDefinition(
        preset_values={
            "memory_flag": True,
        },
        description="Default config with shared encoder-stack memory enabled.",
    ),
    ExperimentPreset.GATING_MEMORY: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
            "memory_flag": True,
        },
        description=(
            "Default config with encoder-block gating and shared memory enabled."
        ),
    ),
    ExperimentPreset.HALTING_MEMORY: PresetDefinition(
        preset_values={
            "stack_halting_flag": True,
            "memory_flag": True,
        },
        description=(
            "Default config with encoder-block halting and shared memory enabled."
        ),
    ),
    ExperimentPreset.GATING_HALTING_MEMORY: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
            "stack_halting_flag": True,
            "memory_flag": True,
        },
        description=(
            "Default config with encoder-block gating, halting, and shared memory."
        ),
    ),
    ExperimentPreset.RECURRENT: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
        },
        description="Default encoder stack wrapped in fixed-step recurrence.",
    ),
    ExperimentPreset.RECURRENT_GATING: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_stack_gate_flag": True,
        },
        description="Default recurrent encoder with step-level gating enabled.",
    ),
    ExperimentPreset.RECURRENT_HALTING: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_stack_halting_flag": True,
        },
        description="Default recurrent encoder with recurrent halting enabled.",
    ),
    ExperimentPreset.RECURRENT_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "memory_flag": True,
        },
        description="Default recurrent encoder whose reused stack has shared memory.",
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_stack_gate_flag": True,
            "recurrent_stack_halting_flag": True,
        },
        description="Default recurrent encoder with step-level gating and halting.",
    ),
    ExperimentPreset.RECURRENT_GATING_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_stack_gate_flag": True,
            "memory_flag": True,
        },
        description=(
            "Default recurrent encoder with step-level gating and shared memory."
        ),
    ),
    ExperimentPreset.RECURRENT_HALTING_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_stack_halting_flag": True,
            "memory_flag": True,
        },
        description=(
            "Default recurrent encoder with recurrent halting and shared memory."
        ),
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_stack_gate_flag": True,
            "recurrent_stack_halting_flag": True,
            "memory_flag": True,
        },
        description=(
            "Default recurrent encoder with step-level gating, recurrent halting, "
            "and shared memory."
        ),
    ),
}

_PRESET_DEFINITIONS[ExperimentPreset.TOP1_SWITCH_AUX] = PresetDefinition(
    preset_values={
        "top_k": 1,
        "sampler_normalize_probabilities_flag": False,
        "sampler_switch_loss_weight": 0.1,
    },
    description="Default config with top-1 expert routing and switch auxiliary loss.",
)
_PRESET_DEFINITIONS[ExperimentPreset.LOW_RANK_EXPERT_WEIGHT] = PresetDefinition(
    preset_values={
        "weight_option_flag": True,
        "weight_option": config.LowRankDynamicWeightConfig,
    },
    description="Default config with adaptive low-rank dynamic weights inside expert "
    "feed-forward internals.",
)
_PRESET_DEFINITIONS[ExperimentPreset.SINGLE_LAYER_RECURRENT_ATTENTION_RESIDUAL] = (
    PresetDefinition(
        preset_values={
            "hidden_dim": 32,
            "sequence_length": 35,
            "stack_num_layers": 1,
            "stack_dropout_probability": 0.0,
            "attn_num_heads": 4,
            "ff_stack_hidden_dim": 32,
            "num_experts": 12,
            "top_k": 2,
            "capacity_factor": 0.0,
            "expert_attention_use_kv_expert_models_flag": False,
            "sampler_switch_loss_weight": 0.01,
            "sampler_zero_centred_loss_weight": 0.001,
            "recurrent_flag": True,
            "recurrent_max_steps": 10,
            "recurrent_residual_connection_option": config.AttentionResidualConfig,
            "recurrent_stack_halting_flag": True,
            "recurrent_halting_option": config.StickBreakingConfig,
            "recurrent_halting_threshold": 0.99,
            "recurrent_halting_hidden_state_mode": (
                config.HaltingHiddenStateModeOptions.RAW
            ),
            "expert_bias_flag": True,
            "weight_option_flag": True,
            "weight_option": SingleModelDynamicWeightConfig,
            "generator_depth": DynamicDepthOptions.DEPTH_OF_FIVE,
            "weight_decay_schedule": WeightDecayScheduleOptions.EXPONENTIAL,
            "weight_decay_rate": 1e-4,
            "weight_decay_warmup_batches": 5000,
            "weight_normalization_option": WeightNormalizationOptions.L2_SCALE,
            "weight_normalization_position_option": (
                WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT
            ),
            "bias_option_flag": True,
            "bias_option": GeneratorDynamicBiasConfig,
            "diagonal_option_flag": True,
            "diagonal_option": CombinedDynamicDiagonalConfig,
            "mask_option_flag": False,
        },
        description=(
            "One dropout-free 32-dimensional encoder layer over 35-token sequences, "
            "with four attention heads and a 32-dimensional feed-forward stack. "
            "Top-two routing selects from twelve adaptive experts without a capacity "
            "limit, while key/value paths remain dense. The bidirectional layer is "
            "reused for up to ten recurrent steps with an outer attention residual "
            "and raw-state stick-breaking halting at 0.99. Expert weights use a "
            "depth-five single-model generator, post-outer-product L2 normalization, "
            "and delayed exponential base-weight decay; generated biases and combined "
            "diagonals are enabled without weight masking."
        ),
    )
)


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=BertExpertLinearAdaptiveConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
            default_dataset=PennTreebankBertPretraining,
        )

    def _dataset_config(self, dataset: type) -> dict:
        return {
            **super()._dataset_config(dataset),
            "sequence_length": dataset.sequence_length,
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
