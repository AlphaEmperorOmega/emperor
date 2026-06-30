from emperor.base.options import BaseOptions
from emperor.experiments.base import (
    BuilderBackedExperimentPresetsBase,
    ExperimentBase,
    ExperimentPresetsBase,
    PresetDefinition,
)

import models.parametric.parametric_vector.config as config
from models.parametric.parametric_vector.config_builder import (
    ParametricVectorConfigBuilder,
)
from models.parametric.parametric_vector.model import Model


class ExperimentPreset(BaseOptions):
    PRESET = 1
    CONFIG = 2


_PRESET_DEFINITIONS = {
    ExperimentPreset.PRESET: PresetDefinition(
        preset_values={},
        description="Default config: a parametric vector classifier with a GELU linear "
        "stack and top-1 adaptive mixture.",
    ),
    ExperimentPreset.CONFIG: PresetDefinition(
        preset_values={},
        description="Config/search preset for overriding parametric vector classifier "
        "settings.",
    ),
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=ParametricVectorConfigBuilder,
            default_preset=ExperimentPreset.PRESET,
        )


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
