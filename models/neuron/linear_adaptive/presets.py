from emperor.base.options import BaseOptions
from emperor.experiments.base import (
    BuilderBackedExperimentPresetsBase,
    ExperimentBase,
    ExperimentPresetsBase,
)

import models.neuron.linear_adaptive.config as config
from models.linears.linear_adaptive.presets import (
    ExperimentPreset as SourceExperimentPreset,
)
from models.neuron._presets import (
    create_experiment_preset,
    create_preset_definitions,
)
from models.neuron.linear_adaptive._source_linear_adaptive_adapter import (
    SOURCE_ADAPTER,
)
from models.neuron.linear_adaptive.config_builder import (
    NeuronLinearAdaptiveConfigBuilder,
)
from models.neuron.linear_adaptive.model import Model


import models.neuron.linear_adaptive.dataset_options as dataset_options
ExperimentPreset = create_experiment_preset(SourceExperimentPreset, __name__)
_PRESET_DEFINITIONS = create_preset_definitions(ExperimentPreset, SOURCE_ADAPTER)


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=NeuronLinearAdaptiveConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
            default_dataset=dataset_options.DATASET_OPTIONS_BY_TASK[dataset_options.DEFAULT_EXPERIMENT_TASK][0],
        )

    def _normalize_model_config_preset(self, preset):
        return ExperimentPreset.BASELINE if preset is None else preset

    def locks_for_preset(self, preset: ExperimentPreset):
        return SOURCE_ADAPTER.source_locks_for_preset(preset)


class Experiment(ExperimentBase):
    def __init__(
        self,
        experiment_preset: ExperimentPreset | None = None,
        experiment_task=None,
    ) -> None:
        super().__init__(experiment_preset, experiment_task=experiment_task)

    def _num_epochs(self) -> int:
        return config.NUM_EPOCHS

    def _dataset_options(self) -> list:
        return dataset_options.DATASET_OPTIONS_BY_TASK[dataset_options.DEFAULT_EXPERIMENT_TASK]

    def _model_type(self) -> type:
        return Model

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        return ExperimentPresets()

    def _experiment_preset_enum(self) -> type[BaseOptions]:
        return ExperimentPreset
