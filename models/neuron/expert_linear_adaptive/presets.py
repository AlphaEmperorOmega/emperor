from emperor.base.options import BaseOptions
from emperor.experiments.base import (
    BuilderBackedExperimentPresetsBase,
    ExperimentBase,
    ExperimentPresetsBase,
)

import models.neuron.expert_linear_adaptive.config as config
from models.experts.linear_adaptive.presets import (
    ExperimentPreset as SourceExperimentPreset,
)
from models.neuron._presets import (
    create_experiment_preset,
    create_preset_definitions,
)
from models.neuron.expert_linear_adaptive._source_expert_linear_adaptive_adapter import (
    SOURCE_ADAPTER,
)
from models.neuron.expert_linear_adaptive.config_builder import (
    NeuronExpertLinearAdaptiveConfigBuilder,
)
from models.neuron.expert_linear_adaptive.model import Model


ExperimentPreset = create_experiment_preset(SourceExperimentPreset, __name__)
_PRESET_DEFINITIONS = create_preset_definitions(ExperimentPreset, SOURCE_ADAPTER)


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=NeuronExpertLinearAdaptiveConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
            default_dataset=config.DATASET_OPTIONS[0],
        )

    def _normalize_model_config_preset(self, preset):
        return ExperimentPreset.BASELINE if preset is None else preset

    def locks_for_preset(self, preset: ExperimentPreset):
        return SOURCE_ADAPTER.source_locks_for_preset(preset)


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
