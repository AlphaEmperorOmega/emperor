from emperor.base.options import BaseOptions
from emperor.experiments.base import (
    BuilderBackedExperimentPresetsBase,
    ExperimentBase,
    ExperimentPresetsBase,
    PresetDefinition,
)

import models.neuron.neuron_linear.config as config
from models.neuron.neuron_linear.config_builder import NeuronLinearConfigBuilder
from models.neuron.neuron_linear.model import Model
from models.neuron.neuron_linear._source_linear_adapter import (
    source_locks_for_preset,
    source_values_for_preset,
)


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
        preset_values=source_values_for_preset(ExperimentPreset.BASELINE),
        description="Default config: a GELU hidden linear stack with pre-layer norm and "
        "dropout.",
    ),
    ExperimentPreset.GATING: PresetDefinition(
        preset_values=source_values_for_preset(ExperimentPreset.GATING),
        description="Default config with per-layer gating enabled, so each hidden layer "
        "output is modulated by a learned sigmoid gate.",
    ),
    ExperimentPreset.HALTING: PresetDefinition(
        preset_values=source_values_for_preset(ExperimentPreset.HALTING),
        description="Default config with stack halting enabled, so examples can stop early "
        "as they move through the hidden stack.",
    ),
    ExperimentPreset.MEMORY: PresetDefinition(
        preset_values=source_values_for_preset(ExperimentPreset.MEMORY),
        description="Default config with shared stack memory enabled across the hidden "
        "layers.",
    ),
    ExperimentPreset.GATING_HALTING: PresetDefinition(
        preset_values=source_values_for_preset(ExperimentPreset.GATING_HALTING),
        description="Default config with both per-layer gating and stack halting enabled.",
    ),
    ExperimentPreset.GATING_MEMORY: PresetDefinition(
        preset_values=source_values_for_preset(ExperimentPreset.GATING_MEMORY),
        description="Default config with both per-layer gating and shared stack memory "
        "enabled.",
    ),
    ExperimentPreset.HALTING_MEMORY: PresetDefinition(
        preset_values=source_values_for_preset(ExperimentPreset.HALTING_MEMORY),
        description="Default config with both stack halting and shared stack memory "
        "enabled.",
    ),
    ExperimentPreset.GATING_HALTING_MEMORY: PresetDefinition(
        preset_values=source_values_for_preset(ExperimentPreset.GATING_HALTING_MEMORY),
        description="Default config with per-layer gating, stack halting, and shared stack "
        "memory enabled.",
    ),
    ExperimentPreset.RESIDUAL: PresetDefinition(
        preset_values=source_values_for_preset(ExperimentPreset.RESIDUAL),
        description="Default config with residual skip connections enabled between "
        "same-width hidden layers.",
    ),
    ExperimentPreset.POST_NORM: PresetDefinition(
        preset_values=source_values_for_preset(ExperimentPreset.POST_NORM),
        description="Default config with layer norm applied after each layer instead of "
        "before it.",
    ),
    ExperimentPreset.RESIDUAL_POST_NORM: PresetDefinition(
        preset_values=source_values_for_preset(ExperimentPreset.RESIDUAL_POST_NORM),
        description="Default config with residual skip connections and post-layer "
        "normalization enabled.",
    ),
    ExperimentPreset.RESIDUAL_GATING: PresetDefinition(
        preset_values=source_values_for_preset(ExperimentPreset.RESIDUAL_GATING),
        description="Default config with residual skip connections and per-layer gating "
        "enabled.",
    ),
    ExperimentPreset.RESIDUAL_HALTING: PresetDefinition(
        preset_values=source_values_for_preset(ExperimentPreset.RESIDUAL_HALTING),
        description="Default config with residual skip connections and stack halting "
        "enabled.",
    ),
    ExperimentPreset.RESIDUAL_MEMORY: PresetDefinition(
        preset_values=source_values_for_preset(ExperimentPreset.RESIDUAL_MEMORY),
        description="Default config with residual skip connections and shared stack memory "
        "enabled.",
    ),
    ExperimentPreset.RECURRENT: PresetDefinition(
        preset_values=source_values_for_preset(ExperimentPreset.RECURRENT),
        description="Default config wrapped in fixed-step recurrence, reusing the hidden "
        "stack for each recurrent step.",
    ),
    ExperimentPreset.RECURRENT_GATING: PresetDefinition(
        preset_values=source_values_for_preset(ExperimentPreset.RECURRENT_GATING),
        description="Default recurrent config with step-level gating enabled after each "
        "recurrent update.",
    ),
    ExperimentPreset.RECURRENT_HALTING: PresetDefinition(
        preset_values=source_values_for_preset(ExperimentPreset.RECURRENT_HALTING),
        description="Default recurrent config with recurrent halting enabled, allowing "
        "early stopping before the max step count.",
    ),
    ExperimentPreset.RECURRENT_MEMORY: PresetDefinition(
        preset_values=source_values_for_preset(ExperimentPreset.RECURRENT_MEMORY),
        description="Default recurrent config whose reused hidden stack has shared memory "
        "enabled.",
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING: PresetDefinition(
        preset_values=source_values_for_preset(ExperimentPreset.RECURRENT_GATING_HALTING),
        description="Default recurrent config with both step-level gating and recurrent "
        "halting enabled.",
    ),
    ExperimentPreset.RECURRENT_GATING_MEMORY: PresetDefinition(
        preset_values=source_values_for_preset(ExperimentPreset.RECURRENT_GATING_MEMORY),
        description="Default recurrent config with step-level gating and shared memory in "
        "the reused hidden stack.",
    ),
    ExperimentPreset.RECURRENT_HALTING_MEMORY: PresetDefinition(
        preset_values=source_values_for_preset(ExperimentPreset.RECURRENT_HALTING_MEMORY),
        description="Default recurrent config with recurrent halting and shared memory in "
        "the reused hidden stack.",
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: PresetDefinition(
        preset_values=source_values_for_preset(
            ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY
        ),
        description="Default recurrent config with step-level gating, recurrent halting, "
        "and shared memory in the reused hidden stack.",
    ),
    ExperimentPreset.RECURRENT_RESIDUAL: PresetDefinition(
        preset_values=source_values_for_preset(ExperimentPreset.RECURRENT_RESIDUAL),
        description="Default recurrent config using a residual hidden stack at each "
        "recurrent step.",
    ),
    ExperimentPreset.RECURRENT_POST_NORM: PresetDefinition(
        preset_values=source_values_for_preset(ExperimentPreset.RECURRENT_POST_NORM),
        description="Default recurrent config using a post-normalized hidden stack at each "
        "recurrent step.",
    ),
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=NeuronLinearConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
            default_dataset=config.DATASET_OPTIONS[0],
        )

    def _normalize_model_config_preset(self, preset):
        return ExperimentPreset.BASELINE if preset is None else preset

    def locks_for_preset(self, preset: ExperimentPreset):
        return source_locks_for_preset(preset)


class Experiment(ExperimentBase):
    def __init__(
        self,
        experiment_preset: ExperimentPreset | None = None,
    ) -> None:
        super().__init__(experiment_preset)

    def _num_epochs(self) -> int:
        return config.NUM_EPOCHS

    def _dataset_options(self) -> list:
        return config.DATASET_OPTIONS

    def _model_type(self) -> type:
        return Model

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        return ExperimentPresets()

    def _experiment_preset_enum(self) -> type[BaseOptions]:
        return ExperimentPreset
