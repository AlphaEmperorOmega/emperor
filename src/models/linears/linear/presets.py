import models.linears.linear.config as config
import models.linears.linear.dataset_options as dataset_options
from emperor.config import BaseOptions
from emperor.layers import LayerNormPositionOptions, ResidualConnectionOptions
from model_runtime.packages import (
    BuilderBackedExperimentPresetsBase,
    ExperimentPresetsBase,
    PresetDefinition,
)
from model_runtime.runs import ExperimentBase
from models.linears.linear.config_builder import LinearConfigBuilder
from models.linears.linear.model import Model
from models.linears.linear.runtime_defaults import runtime_from_flat


class ExperimentPreset(BaseOptions):
    BASELINE = 1
    GATING = 2
    HALTING = 3
    MEMORY = 4
    GATING_HALTING = 5
    GATING_MEMORY = 6
    HALTING_MEMORY = 7
    GATING_HALTING_MEMORY = 8
    RESIDUAL = 9
    POST_NORM = 10
    RESIDUAL_POST_NORM = 11
    RESIDUAL_GATING = 12
    RESIDUAL_HALTING = 13
    RESIDUAL_MEMORY = 14
    RECURRENT = 15
    RECURRENT_GATING = 16
    RECURRENT_HALTING = 17
    RECURRENT_MEMORY = 18
    RECURRENT_GATING_HALTING = 19
    RECURRENT_GATING_MEMORY = 20
    RECURRENT_HALTING_MEMORY = 21
    RECURRENT_GATING_HALTING_MEMORY = 22
    RECURRENT_RESIDUAL = 23
    RECURRENT_POST_NORM = 24


_PRESET_DEFINITIONS = {
    ExperimentPreset.BASELINE: PresetDefinition(
        preset_values={},
        description="Default config: a GELU hidden linear stack with pre-layer norm and "
        "dropout.",
    ),
    ExperimentPreset.GATING: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
        },
        description="Default config with per-layer gating enabled, so each hidden layer "
        "output is modulated by a learned sigmoid gate.",
    ),
    ExperimentPreset.HALTING: PresetDefinition(
        preset_values={
            "stack_halting_flag": True,
        },
        description="Default config with stack halting enabled, so examples can stop early "
        "as they move through the hidden stack.",
    ),
    ExperimentPreset.MEMORY: PresetDefinition(
        preset_values={
            "memory_flag": True,
        },
        description="Default config with shared stack memory enabled across the hidden "
        "layers.",
    ),
    ExperimentPreset.GATING_HALTING: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
            "stack_halting_flag": True,
        },
        description="Default config with both per-layer gating and stack halting enabled.",
    ),
    ExperimentPreset.GATING_MEMORY: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
            "memory_flag": True,
        },
        description="Default config with both per-layer gating and shared stack memory "
        "enabled.",
    ),
    ExperimentPreset.HALTING_MEMORY: PresetDefinition(
        preset_values={
            "stack_halting_flag": True,
            "memory_flag": True,
        },
        description="Default config with both stack halting and shared stack memory "
        "enabled.",
    ),
    ExperimentPreset.GATING_HALTING_MEMORY: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
            "stack_halting_flag": True,
            "memory_flag": True,
        },
        description="Default config with per-layer gating, stack halting, and shared stack "
        "memory enabled.",
    ),
    ExperimentPreset.RESIDUAL: PresetDefinition(
        preset_values={
            "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
        },
        description="Default config with residual skip connections enabled between "
        "same-width hidden layers.",
    ),
    ExperimentPreset.POST_NORM: PresetDefinition(
        preset_values={
            "layer_norm_position": LayerNormPositionOptions.AFTER,
        },
        description="Default config with layer norm applied after each layer instead of "
        "before it.",
    ),
    ExperimentPreset.RESIDUAL_POST_NORM: PresetDefinition(
        preset_values={
            "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
            "layer_norm_position": LayerNormPositionOptions.AFTER,
        },
        description="Default config with residual skip connections and post-layer "
        "normalization enabled.",
    ),
    ExperimentPreset.RESIDUAL_GATING: PresetDefinition(
        preset_values={
            "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
            "stack_gate_flag": True,
        },
        description="Default config with residual skip connections and per-layer gating "
        "enabled.",
    ),
    ExperimentPreset.RESIDUAL_HALTING: PresetDefinition(
        preset_values={
            "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
            "stack_halting_flag": True,
        },
        description="Default config with residual skip connections and stack halting "
        "enabled.",
    ),
    ExperimentPreset.RESIDUAL_MEMORY: PresetDefinition(
        preset_values={
            "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
            "memory_flag": True,
        },
        description="Default config with residual skip connections and shared stack memory "
        "enabled.",
    ),
    ExperimentPreset.RECURRENT: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
        },
        description="Default config wrapped in fixed-step recurrence, reusing the hidden "
        "stack for each recurrent step.",
    ),
    ExperimentPreset.RECURRENT_GATING: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_gate_flag": True,
        },
        description="Default recurrent config with step-level gating enabled after each "
        "recurrent update.",
    ),
    ExperimentPreset.RECURRENT_HALTING: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_halting_flag": True,
        },
        description="Default recurrent config with recurrent halting enabled, allowing "
        "early stopping before the max step count.",
    ),
    ExperimentPreset.RECURRENT_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "memory_flag": True,
        },
        description="Default recurrent config whose reused hidden stack has shared memory "
        "enabled.",
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_gate_flag": True,
            "recurrent_halting_flag": True,
        },
        description="Default recurrent config with both step-level gating and recurrent "
        "halting enabled.",
    ),
    ExperimentPreset.RECURRENT_GATING_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_gate_flag": True,
            "memory_flag": True,
        },
        description="Default recurrent config with step-level gating and shared memory in "
        "the reused hidden stack.",
    ),
    ExperimentPreset.RECURRENT_HALTING_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_halting_flag": True,
            "memory_flag": True,
        },
        description="Default recurrent config with recurrent halting and shared memory in "
        "the reused hidden stack.",
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_gate_flag": True,
            "recurrent_halting_flag": True,
            "memory_flag": True,
        },
        description="Default recurrent config with step-level gating, recurrent halting, "
        "and shared memory in the reused hidden stack.",
    ),
    ExperimentPreset.RECURRENT_RESIDUAL: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
        },
        description="Default recurrent config using a residual hidden stack at each "
        "recurrent step.",
    ),
    ExperimentPreset.RECURRENT_POST_NORM: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "layer_norm_position": LayerNormPositionOptions.AFTER,
        },
        description="Default recurrent config using a post-normalized hidden stack at each "
        "recurrent step.",
    ),
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=LinearConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
        )

    def _preset(self, **kwargs):
        runtime = runtime_from_flat(kwargs)
        return self._builder_type(runtime=runtime).build()


class Experiment(ExperimentBase):
    def __init__(
        self,
        experiment_preset: ExperimentPreset | None = None,
        experiment_task=None,
        *,
        model_package=None,
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
