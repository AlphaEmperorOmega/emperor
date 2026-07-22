import models.gpt.expert_linear_adaptive.config as config
import models.gpt.expert_linear_adaptive.dataset_options as dataset_options
from emperor.config import BaseOptions
from emperor.datasets.text.language_modeling import WikiText2
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
from models.gpt.expert_linear_adaptive.config_builder import (
    GptExpertLinearAdaptiveConfigBuilder,
)
from models.gpt.expert_linear_adaptive.model import Model
from models.gpt.expert_linear_adaptive.runtime_defaults import (
    runtime_from_flat,
)


class ExperimentPreset(BaseOptions):
    BASELINE = 1
    PRE_NORM = 2
    POST_NORM = 3
    SINUSOIDAL = 4
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
    TOP1_SWITCH_AUX = 21
    LOW_RANK_EXPERT_WEIGHT = 22


_PRESET_DEFINITIONS = {
    ExperimentPreset.BASELINE: PresetDefinition(
        preset_values={},
        description=(
            "Default config: a GPT-style causal language-modeling decoder with "
            "an adaptive Mixture of Attention Heads, Mixture-of-Experts "
            "feed-forward sub-stacks, learned positional embeddings, and causal "
            "attention."
        ),
    ),
    ExperimentPreset.PRE_NORM: PresetDefinition(
        preset_values={
            "layer_norm_position": LayerNormPositionOptions.BEFORE,
        },
        description=(
            "Default config with layer normalization applied before each decoder "
            "sub-block."
        ),
    ),
    ExperimentPreset.POST_NORM: PresetDefinition(
        preset_values={
            "layer_norm_position": LayerNormPositionOptions.AFTER,
        },
        description=(
            "Default config with layer normalization applied after each decoder "
            "sub-block."
        ),
    ),
    ExperimentPreset.SINUSOIDAL: PresetDefinition(
        preset_values={
            "positional_embedding_option": TextSinusoidalPositionalEmbeddingConfig,
        },
        description="Default config with fixed sinusoidal positional embeddings.",
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
        description="Default config with per-decoder-block gating enabled.",
    ),
    ExperimentPreset.HALTING: PresetDefinition(
        preset_values={
            "stack_halting_flag": True,
        },
        description="Default config with decoder-block stack halting enabled.",
    ),
    ExperimentPreset.GATING_HALTING: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
            "stack_halting_flag": True,
        },
        description=(
            "Default config with both decoder-block gating and halting enabled."
        ),
    ),
    ExperimentPreset.MEMORY: PresetDefinition(
        preset_values={
            "memory_flag": True,
        },
        description="Default config with shared decoder-stack memory enabled.",
    ),
    ExperimentPreset.GATING_MEMORY: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
            "memory_flag": True,
        },
        description=(
            "Default config with decoder-block gating and shared memory enabled."
        ),
    ),
    ExperimentPreset.HALTING_MEMORY: PresetDefinition(
        preset_values={
            "stack_halting_flag": True,
            "memory_flag": True,
        },
        description=(
            "Default config with decoder-block halting and shared memory enabled."
        ),
    ),
    ExperimentPreset.GATING_HALTING_MEMORY: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
            "stack_halting_flag": True,
            "memory_flag": True,
        },
        description=(
            "Default config with decoder-block gating, halting, and shared memory."
        ),
    ),
    ExperimentPreset.RECURRENT: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
        },
        description="Default decoder stack wrapped in fixed-step recurrence.",
    ),
    ExperimentPreset.RECURRENT_GATING: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_stack_gate_flag": True,
        },
        description="Default recurrent decoder with step-level gating enabled.",
    ),
    ExperimentPreset.RECURRENT_HALTING: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_stack_halting_flag": True,
        },
        description="Default recurrent decoder with recurrent halting enabled.",
    ),
    ExperimentPreset.RECURRENT_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "memory_flag": True,
        },
        description="Default recurrent decoder whose reused stack has shared memory.",
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_stack_gate_flag": True,
            "recurrent_stack_halting_flag": True,
        },
        description="Default recurrent decoder with step-level gating and halting.",
    ),
    ExperimentPreset.RECURRENT_GATING_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_stack_gate_flag": True,
            "memory_flag": True,
        },
        description=(
            "Default recurrent decoder with step-level gating and shared memory."
        ),
    ),
    ExperimentPreset.RECURRENT_HALTING_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_stack_halting_flag": True,
            "memory_flag": True,
        },
        description=(
            "Default recurrent decoder with recurrent halting and shared memory."
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
            "Default recurrent decoder with step-level gating, recurrent halting, "
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


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=GptExpertLinearAdaptiveConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
            default_dataset=WikiText2,
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
