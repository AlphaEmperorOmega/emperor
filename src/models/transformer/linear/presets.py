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
from .config_builder import TransformerLinearConfigBuilder
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


def _definition(values: dict, description: str) -> PresetDefinition:
    return PresetDefinition(preset_values=values, description=description)


_PRESET_DEFINITIONS = {
    ExperimentPreset.BASELINE: _definition(
        {}, "Canonical De-to-En Transformer baseline."
    ),
    ExperimentPreset.PRE_NORM: _definition(
        {
            "encoder_layer_norm_position": LayerNormPositionOptions.BEFORE,
            "decoder_layer_norm_position": LayerNormPositionOptions.BEFORE,
        },
        "Use pre-normalization in both encoder and decoder blocks.",
    ),
    ExperimentPreset.POST_NORM: _definition(
        {
            "encoder_layer_norm_position": LayerNormPositionOptions.AFTER,
            "decoder_layer_norm_position": LayerNormPositionOptions.AFTER,
        },
        "Use post-normalization in both encoder and decoder blocks.",
    ),
    ExperimentPreset.LEARNED_POSITIONAL: _definition(
        {"positional_embedding_option": TextLearnedPositionalEmbeddingConfig},
        "Replace the sinusoidal baseline with separate learned positions.",
    ),
    ExperimentPreset.ATTENTION_BIAS: _definition(
        {"attn_bias_flag": True, "attn_add_key_value_bias_flag": True},
        "Enable projection and learned key/value bias in every attention path.",
    ),
    ExperimentPreset.GATING: _definition(
        {"stack_gate_flag": True}, "Gate encoder and decoder block outputs."
    ),
    ExperimentPreset.HALTING: _definition(
        {"stack_halting_flag": True}, "Enable block-level adaptive halting."
    ),
    ExperimentPreset.GATING_HALTING: _definition(
        {"stack_gate_flag": True, "stack_halting_flag": True},
        "Combine block gating and halting.",
    ),
    ExperimentPreset.MEMORY: _definition(
        {"memory_flag": True}, "Enable shared dynamic memory in both stacks."
    ),
    ExperimentPreset.GATING_MEMORY: _definition(
        {"stack_gate_flag": True, "memory_flag": True},
        "Combine block gating and dynamic memory.",
    ),
    ExperimentPreset.HALTING_MEMORY: _definition(
        {"stack_halting_flag": True, "memory_flag": True},
        "Combine block halting and dynamic memory.",
    ),
    ExperimentPreset.GATING_HALTING_MEMORY: _definition(
        {
            "stack_gate_flag": True,
            "stack_halting_flag": True,
            "memory_flag": True,
        },
        "Combine block gating, halting, and memory.",
    ),
    ExperimentPreset.RECURRENT: _definition(
        {"recurrent_flag": True}, "Reuse both Transformer stacks recurrently."
    ),
    ExperimentPreset.RECURRENT_GATING: _definition(
        {"recurrent_flag": True, "recurrent_stack_gate_flag": True},
        "Use recurrent stacks with step gates.",
    ),
    ExperimentPreset.RECURRENT_HALTING: _definition(
        {"recurrent_flag": True, "recurrent_stack_halting_flag": True},
        "Use recurrent stacks with adaptive step halting.",
    ),
    ExperimentPreset.RECURRENT_MEMORY: _definition(
        {"recurrent_flag": True, "memory_flag": True},
        "Use recurrent stacks with shared block memory.",
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING: _definition(
        {
            "recurrent_flag": True,
            "recurrent_stack_gate_flag": True,
            "recurrent_stack_halting_flag": True,
        },
        "Combine recurrence, step gating, and step halting.",
    ),
    ExperimentPreset.RECURRENT_GATING_MEMORY: _definition(
        {
            "recurrent_flag": True,
            "recurrent_stack_gate_flag": True,
            "memory_flag": True,
        },
        "Combine recurrence, step gating, and memory.",
    ),
    ExperimentPreset.RECURRENT_HALTING_MEMORY: _definition(
        {
            "recurrent_flag": True,
            "recurrent_stack_halting_flag": True,
            "memory_flag": True,
        },
        "Combine recurrence, step halting, and memory.",
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: _definition(
        {
            "recurrent_flag": True,
            "recurrent_stack_gate_flag": True,
            "recurrent_stack_halting_flag": True,
            "memory_flag": True,
        },
        "Enable all recurrent stack controllers.",
    ),
    ExperimentPreset.RESIDUAL: _definition(
        {"stack_residual_connection_option": (ResidualConnectionOptions.RESIDUAL)},
        "Add residual joins between full Transformer blocks.",
    ),
    ExperimentPreset.RESIDUAL_POST_NORM: _definition(
        {
            "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
            "encoder_layer_norm_position": LayerNormPositionOptions.AFTER,
            "decoder_layer_norm_position": LayerNormPositionOptions.AFTER,
        },
        "Combine block residuals with internal post-normalization.",
    ),
    ExperimentPreset.RESIDUAL_GATING: _definition(
        {
            "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
            "stack_gate_flag": True,
        },
        "Combine block residuals and block gates.",
    ),
    ExperimentPreset.RESIDUAL_HALTING: _definition(
        {
            "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
            "stack_halting_flag": True,
        },
        "Combine block residuals and block halting.",
    ),
    ExperimentPreset.RESIDUAL_MEMORY: _definition(
        {
            "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
            "memory_flag": True,
        },
        "Combine block residuals and dynamic memory.",
    ),
    ExperimentPreset.RECURRENT_RESIDUAL: _definition(
        {
            "recurrent_flag": True,
            "recurrent_residual_connection_option": (
                ResidualConnectionOptions.RESIDUAL
            ),
        },
        "Add residual joins between recurrent applications.",
    ),
    ExperimentPreset.RECURRENT_POST_NORM: _definition(
        {
            "recurrent_flag": True,
            "encoder_layer_norm_position": LayerNormPositionOptions.AFTER,
            "decoder_layer_norm_position": LayerNormPositionOptions.AFTER,
        },
        "Use recurrent Transformer stacks with post-normalization.",
    ),
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=TransformerLinearConfigBuilder,
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
