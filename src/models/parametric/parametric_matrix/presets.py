import models.parametric.parametric_matrix.config as config
import models.parametric.parametric_matrix.dataset_options as dataset_options
from emperor.config import BaseOptions
from model_runtime.packages import (
    BuilderBackedExperimentPresetsBase,
    ExperimentPresetsBase,
    PresetDefinition,
)
from model_runtime.runs import ExperimentBase
from models.parametric.parametric_matrix.config_builder import (
    ParametricMatrixConfigBuilder,
)
from models.parametric.parametric_matrix.model import Model


class ExperimentPreset(BaseOptions):
    PRESET = 1
    CONFIG = 2


_PRESET_DEFINITIONS = {
    ExperimentPreset.PRESET: PresetDefinition(
        preset_values={},
        description="Default config: a parametric matrix classifier with a GELU linear "
        "stack and top-1 adaptive mixture.",
    ),
    ExperimentPreset.CONFIG: PresetDefinition(
        preset_values={},
        description="Config/search preset for overriding parametric matrix classifier "
        "settings.",
    ),
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=ParametricMatrixConfigBuilder,
            default_preset=ExperimentPreset.PRESET,
        )


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
