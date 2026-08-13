import models.parametric.parametric_vector.config as config
from emperor.config import BaseOptions
from model_runtime.packages import (
    BuilderBackedExperimentPresetsBase,
    PresetDefinition,
)
from model_runtime.runs import ExperimentBase
from models.parametric.parametric_vector.config_builder import (
    ParametricVectorConfigBuilder,
)
from models.parametric.parametric_vector.runtime_defaults import runtime_from_flat


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
            runtime_factory=runtime_from_flat,
            default_preset=ExperimentPreset.PRESET,
        )


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
