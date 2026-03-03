from emperor.base.enums import BaseOptions
from emperor.experiments.base import ExperimentBase, ExperimentPresetsBase
from models.experts.config import ExperimentConfig
from models.experts.model import Model
from models.experts.presets import ExperimentOptions, ExperimentPresets

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class Experiment(ExperimentBase):
    def __init__(
        self,
        experiment_option: ExperimentOptions | None = None,
    ) -> None:
        super().__init__(experiment_option)

    def _model_type(self) -> type:
        return Model

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        return ExperimentPresets()

    def _experiment_enumeration(self) -> type[BaseOptions]:
        return ExperimentOptions


__all__ = [
    "Experiment",
    "ExperimentOptions",
    "ExperimentPresets",
    "ExperimentConfig",
    "Model",
]
