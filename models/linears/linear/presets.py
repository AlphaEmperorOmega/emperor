import models.linears.linear.config as config

from models.linears.linear.config_builder import LinearConfigBuilder
from models.linears.linear.model import Model
from emperor.experiments.base import SearchMode
from emperor.datasets.image.classification.mnist import Mnist
from emperor.experiments.base import ExperimentBase, ExperimentPresetsBase, PresetLock
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import BaseOptions, LayerNormPositionOptions

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


def _lock(preset, value, behavior: str) -> PresetLock:
    return PresetLock(
        value=value,
        reason=(
            f"Locked by the {preset.name} preset because this preset enables "
            f"{behavior}."
        ),
    )


def _preset_locks(
    preset_overrides: dict["ExperimentPreset", dict[str, object]],
) -> dict["ExperimentPreset", dict[str, PresetLock]]:
    return {
        preset: {
            field: _lock(preset, value, _PRESET_LOCK_BEHAVIORS[field])
            for field, value in overrides.items()
        }
        for preset, overrides in preset_overrides.items()
        if overrides
    }


class ExperimentPreset(BaseOptions):
    BASELINE = (
        "Default config: a GELU hidden linear stack with pre-layer norm and dropout."
    )
    GATING = (
        "Default config with per-layer gating enabled, so each hidden layer "
        "output is modulated by a learned sigmoid gate."
    )
    HALTING = (
        "Default config with stack halting enabled, so examples can stop early "
        "as they move through the hidden stack."
    )
    MEMORY = (
        "Default config with shared stack memory enabled across the hidden " "layers."
    )
    GATING_HALTING = (
        "Default config with both per-layer gating and stack halting enabled."
    )
    GATING_MEMORY = (
        "Default config with both per-layer gating and shared stack memory " "enabled."
    )
    HALTING_MEMORY = (
        "Default config with both stack halting and shared stack memory enabled."
    )
    GATING_HALTING_MEMORY = (
        "Default config with per-layer gating, stack halting, and shared stack "
        "memory enabled."
    )
    RESIDUAL = (
        "Default config with residual skip connections enabled between same-width "
        "hidden layers."
    )
    POST_NORM = (
        "Default config with layer norm applied after each layer instead of "
        "before it."
    )
    RESIDUAL_POST_NORM = (
        "Default config with residual skip connections and post-layer "
        "normalization enabled."
    )
    RESIDUAL_GATING = (
        "Default config with residual skip connections and per-layer gating " "enabled."
    )
    RESIDUAL_HALTING = (
        "Default config with residual skip connections and stack halting enabled."
    )
    RESIDUAL_MEMORY = (
        "Default config with residual skip connections and shared stack memory "
        "enabled."
    )
    RECURRENT = (
        "Default config wrapped in fixed-step recurrence, reusing the hidden "
        "stack for each recurrent step."
    )
    RECURRENT_GATING = (
        "Default recurrent config with step-level gating enabled after each "
        "recurrent update."
    )
    RECURRENT_HALTING = (
        "Default recurrent config with recurrent halting enabled, allowing early "
        "stopping before the max step count."
    )
    RECURRENT_MEMORY = (
        "Default recurrent config whose reused hidden stack has shared memory "
        "enabled."
    )
    RECURRENT_GATING_HALTING = (
        "Default recurrent config with both step-level gating and recurrent "
        "halting enabled."
    )
    RECURRENT_GATING_MEMORY = (
        "Default recurrent config with step-level gating and shared memory in "
        "the reused hidden stack."
    )
    RECURRENT_HALTING_MEMORY = (
        "Default recurrent config with recurrent halting and shared memory in "
        "the reused hidden stack."
    )
    RECURRENT_GATING_HALTING_MEMORY = (
        "Default recurrent config with step-level gating, recurrent halting, "
        "and shared memory in the reused hidden stack."
    )
    RECURRENT_RESIDUAL = (
        "Default recurrent config using a residual hidden stack at each recurrent "
        "step."
    )
    RECURRENT_POST_NORM = (
        "Default recurrent config using a post-normalized hidden stack at each "
        "recurrent step."
    )


_PRESET_LOCK_BEHAVIORS = {
    "stack_gate_flag": "stack gating",
    "stack_halting_flag": "adaptive stack halting",
    "memory_flag": "shared stack memory",
    "stack_residual_connection_option": "stack residuals",
    "layer_norm_position": "post-layer normalization",
    "recurrent_flag": "recurrent execution",
    "recurrent_gate_flag": "recurrent gating",
    "recurrent_halting_flag": "adaptive recurrent halting",
}

_PRESET_OVERRIDES = {
    ExperimentPreset.BASELINE: {},
    ExperimentPreset.GATING: {
        "stack_gate_flag": True,
    },
    ExperimentPreset.HALTING: {
        "stack_halting_flag": True,
    },
    ExperimentPreset.MEMORY: {
        "memory_flag": True,
    },
    ExperimentPreset.GATING_HALTING: {
        "stack_gate_flag": True,
        "stack_halting_flag": True,
    },
    ExperimentPreset.GATING_MEMORY: {
        "stack_gate_flag": True,
        "memory_flag": True,
    },
    ExperimentPreset.HALTING_MEMORY: {
        "stack_halting_flag": True,
        "memory_flag": True,
    },
    ExperimentPreset.GATING_HALTING_MEMORY: {
        "stack_gate_flag": True,
        "stack_halting_flag": True,
        "memory_flag": True,
    },
    ExperimentPreset.RESIDUAL: {
        "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
    },
    ExperimentPreset.POST_NORM: {
        "layer_norm_position": LayerNormPositionOptions.AFTER,
    },
    ExperimentPreset.RESIDUAL_POST_NORM: {
        "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
        "layer_norm_position": LayerNormPositionOptions.AFTER,
    },
    ExperimentPreset.RESIDUAL_GATING: {
        "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
        "stack_gate_flag": True,
    },
    ExperimentPreset.RESIDUAL_HALTING: {
        "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
        "stack_halting_flag": True,
    },
    ExperimentPreset.RESIDUAL_MEMORY: {
        "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
        "memory_flag": True,
    },
    ExperimentPreset.RECURRENT: {
        "recurrent_flag": True,
    },
    ExperimentPreset.RECURRENT_GATING: {
        "recurrent_flag": True,
        "recurrent_gate_flag": True,
    },
    ExperimentPreset.RECURRENT_HALTING: {
        "recurrent_flag": True,
        "recurrent_halting_flag": True,
    },
    ExperimentPreset.RECURRENT_MEMORY: {
        "recurrent_flag": True,
        "memory_flag": True,
    },
    ExperimentPreset.RECURRENT_GATING_HALTING: {
        "recurrent_flag": True,
        "recurrent_gate_flag": True,
        "recurrent_halting_flag": True,
    },
    ExperimentPreset.RECURRENT_GATING_MEMORY: {
        "recurrent_flag": True,
        "recurrent_gate_flag": True,
        "memory_flag": True,
    },
    ExperimentPreset.RECURRENT_HALTING_MEMORY: {
        "recurrent_flag": True,
        "recurrent_halting_flag": True,
        "memory_flag": True,
    },
    ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: {
        "recurrent_flag": True,
        "recurrent_gate_flag": True,
        "recurrent_halting_flag": True,
        "memory_flag": True,
    },
    ExperimentPreset.RECURRENT_RESIDUAL: {
        "recurrent_flag": True,
        "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
    },
    ExperimentPreset.RECURRENT_POST_NORM: {
        "recurrent_flag": True,
        "layer_norm_position": LayerNormPositionOptions.AFTER,
    },
}


class ExperimentPresets(ExperimentPresetsBase):
    PRESET_OVERRIDES = _PRESET_OVERRIDES
    PRESET_LOCKS = _preset_locks(PRESET_OVERRIDES)

    def __init__(self) -> None:
        super().__init__()

    def get_config(
        self,
        model_config_preset: ExperimentPreset = ExperimentPreset.BASELINE,
        dataset: type = Mnist,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
        search_keys: list[str] | None = None,
        config_overrides: dict | None = None,
        search_overrides: dict | None = None,
    ) -> list["ModelConfig"]:
        preset_callback = self._preset_callback_for_preset(model_config_preset)
        return self._create_preset_search_space_configs(
            dataset,
            search_mode,
            preset_callback,
            search_keys,
            config_overrides=config_overrides,
            search_overrides=search_overrides,
            model_config_preset=model_config_preset,
        )

    def _preset_callback_for_preset(self, preset: ExperimentPreset):
        if preset not in self.PRESET_OVERRIDES:
            raise ValueError(
                "The specified preset is not supported. Please choose a valid `ExperimentPreset`."
            )
        return lambda **kwargs: self._preset_for_preset(preset, **kwargs)

    def _preset_for_preset(
        self,
        preset: ExperimentPreset,
        **kwargs,
    ) -> "ModelConfig":
        preset_overrides = self.PRESET_OVERRIDES[preset]
        return self._preset(**{**kwargs, **preset_overrides})

    def _preset(self, **kwargs) -> "ModelConfig":
        return LinearConfigBuilder(**kwargs).build()


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
