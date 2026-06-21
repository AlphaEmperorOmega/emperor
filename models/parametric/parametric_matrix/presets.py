from typing import TYPE_CHECKING

import models.parametric.parametric_matrix.config as config

from emperor.base.options import BaseOptions
from emperor.datasets.image.classification.mnist import Mnist
from emperor.experiments.base import (
    ExperimentBase,
    ExperimentPresetsBase,
    PresetLock,
    SearchMode,
)
from models.parametric.parametric_matrix.config_builder import (
    ParametricMatrixConfigBuilder,
)
from models.parametric.parametric_matrix.model import Model

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExperimentPreset(BaseOptions):
    PRESET = (
        "Default config: a parametric matrix classifier with a GELU linear stack "
        "and top-1 adaptive mixture."
    )
    CONFIG = (
        "Config/search preset for overriding parametric matrix classifier "
        "settings."
    )


def _lock(preset, value, field: str) -> PresetLock:
    label = field.replace("_", " ")
    return PresetLock(
        value=value,
        reason=(
            f"Locked by the {preset.name} preset because this preset sets {label}."
        ),
    )


def _preset_locks(
    preset_overrides: dict["ExperimentPreset", dict[str, object]],
) -> dict["ExperimentPreset", dict[str, PresetLock]]:
    return {
        preset: {
            field: _lock(preset, value, field) for field, value in overrides.items()
        }
        for preset, overrides in preset_overrides.items()
        if overrides
    }


_PRESET_OVERRIDES = {
    ExperimentPreset.PRESET: {},
    ExperimentPreset.CONFIG: {},
}


class ExperimentPresets(ExperimentPresetsBase):
    PRESET_OVERRIDES = _PRESET_OVERRIDES
    PRESET_LOCKS = _preset_locks(PRESET_OVERRIDES)

    def get_config(
        self,
        model_config_preset: ExperimentPreset = ExperimentPreset.PRESET,
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
                "The specified preset is not supported. Please choose a valid "
                "`ExperimentPreset`."
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
        return ParametricMatrixConfigBuilder(**kwargs).build()


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
