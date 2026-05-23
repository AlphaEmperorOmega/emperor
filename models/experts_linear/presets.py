import models.experts_linear.config as config

from models.experts_linear.config_builder import ExpertsLinearConfigBuilder
from models.experts_linear.model import Model
from emperor.experiments.base import SearchMode
from emperor.datasets.image.classification.mnist import Mnist
from emperor.experiments.base import ExperimentBase, ExperimentPresetsBase
from emperor.base.options import BaseOptions

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExperimentOptions(BaseOptions):
    BASELINE = "Baseline mixture-of-experts with linear expert and sampler stacks."
    GATING = "Mixture-of-experts stack with per-layer gating enabled."
    HALTING = "Mixture-of-experts stack with per-layer halting enabled."
    GATING_HALTING = "Mixture-of-experts stack with both per-layer gating and halting enabled."


class ExperimentPresets(ExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__()

    def get_config(
        self,
        model_config_options: ExperimentOptions = ExperimentOptions.BASELINE,
        dataset: type = Mnist,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
        search_keys: list[str] | None = None,
        config_overrides: dict | None = None,
        search_overrides: dict | None = None,
    ) -> list["ModelConfig"]:
        preset_callback = self._preset_callback_for_option(model_config_options)
        return self._create_preset_search_space_configs(
            dataset,
            search_mode,
            preset_callback,
            search_keys,
            config_overrides=config_overrides,
            search_overrides=search_overrides,
        )

    def _preset_callback_for_option(self, option: ExperimentOptions):
        callbacks = {
            ExperimentOptions.BASELINE: self._baseline_preset,
            ExperimentOptions.GATING: self._gating_preset,
            ExperimentOptions.HALTING: self._halting_preset,
            ExperimentOptions.GATING_HALTING: self._gating_halting_preset,
        }
        if option not in callbacks:
            raise ValueError(
                "The specified option is not supported. Please choose a valid `ExperimentOptions`."
            )
        return callbacks[option]

    def _baseline_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**kwargs)

    def _gating_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"stack_gate_flag": True, **kwargs})

    def _halting_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"stack_halting_flag": True, **kwargs})

    def _gating_halting_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{"stack_gate_flag": True, "stack_halting_flag": True, **kwargs}
        )

    def _preset(self, **kwargs) -> "ModelConfig":
        return ExpertsLinearConfigBuilder(**kwargs).build()


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
