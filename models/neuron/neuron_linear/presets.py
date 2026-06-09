import models.neuron.neuron_linear.config as config

from models.linears.linear.presets import (
    ExperimentOptions as SourceExperimentOptions,
    ExperimentPresets as SourceExperimentPresets,
)
from models.neuron.neuron_linear.config_builder import NeuronLinearConfigBuilder
from models.neuron.neuron_linear.model import Model
from emperor.base.options import BaseOptions
from emperor.experiments.base import (
    ExperimentBase,
    ExperimentPresetsBase,
    PresetLock,
    SearchMode,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExperimentOptions(BaseOptions):
    BASELINE = 'Baseline linear stack preset; supports search-space flags.'
    GATING = 'Linear stack with a learned gate applied to hidden-layer outputs.'
    HALTING = 'Linear stack with adaptive computation halting enabled.'
    GATING_HALTING = 'Linear stack with both learned gating and adaptive computation halting.'
    RESIDUAL = 'Linear stack with residual (skip) connections on hidden layers.'
    POST_NORM = 'Linear stack with layer norm applied after each layer (post-norm).'
    RESIDUAL_POST_NORM = 'Linear stack with residual connections and post-layer normalization.'
    RESIDUAL_GATING = 'Linear stack with residual connections and learned gating.'
    RESIDUAL_HALTING = 'Linear stack with residual connections and adaptive computation halting.'
    RECURRENT = 'Linear stack applied recurrently for a fixed number of steps.'
    RECURRENT_GATING = 'Linear stack applied recurrently with a learned recurrent gate.'
    RECURRENT_HALTING = 'Linear stack applied recurrently with adaptive recurrent halting.'
    RECURRENT_GATING_HALTING = 'Linear stack applied recurrently with both learned recurrent gating and adaptive recurrent halting.'
    RECURRENT_RESIDUAL = 'Residual linear stack applied recurrently.'
    RECURRENT_POST_NORM = 'Post-normalized linear stack applied recurrently.'


def _mirrored_source_locks() -> dict[ExperimentOptions, dict[str, PresetLock | object]]:
    mirrored_locks = {}
    for option in ExperimentOptions:
        source_option = SourceExperimentOptions[option.name]
        mirrored_locks[option] = SourceExperimentPresets.PRESET_LOCKS.get(
            source_option,
            {},
        )
    return mirrored_locks


class ExperimentPresets(ExperimentPresetsBase):
    PRESET_LOCKS = _mirrored_source_locks()

    def get_config(
        self,
        model_config_options: ExperimentOptions = ExperimentOptions.BASELINE,
        dataset: type = config.DATASET_OPTIONS[0],
        search_mode: SearchMode = None,
        log_folder: str | None = None,
        search_keys: list[str] | None = None,
        config_overrides: dict | None = None,
        search_overrides: dict | None = None,
    ) -> list["ModelConfig"]:
        if model_config_options is None:
            model_config_options = ExperimentOptions.BASELINE

        def preset_callback(**kwargs) -> "ModelConfig":
            return self._preset_for_option(model_config_options, **kwargs)

        return self._create_preset_search_space_configs(
            dataset,
            search_mode,
            preset_callback,
            search_keys,
            config_overrides=config_overrides,
            search_overrides=search_overrides,
        )

    def _preset_for_option(
        self,
        option: ExperimentOptions,
        **kwargs,
    ) -> "ModelConfig":
        locked_values = {
            key: lock.value
            for key, lock in self.locked_fields(option).items()
        }
        return self._preset(**{**locked_values, **kwargs})

    def _preset(self, **kwargs) -> "ModelConfig":
        return NeuronLinearConfigBuilder(**kwargs).build()


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
