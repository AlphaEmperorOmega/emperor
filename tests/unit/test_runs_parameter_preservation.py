from __future__ import annotations

import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from lightning.pytorch.callbacks import Callback

from emperor.config import BaseOptions
from emperor.experiments import ExperimentTask
from model_runtime.packages import ModelIdentity, ModelMetadata, ModelPackage
from model_runtime.runs import execution, experiment
from model_runtime.runs.experiment import ExperimentBase
from model_runtime.runs.records import RunParameter, RunPlan, RunSpec


class SyntheticDataset:
    flattened_input_dim = 4
    num_classes = 2

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size


class _Preset(BaseOptions):
    BASELINE = "baseline"


class _Config:
    batch_size = 2


class _PresetGenerator:
    @staticmethod
    def get_config(*_args, **_kwargs) -> list[_Config]:
        return [_Config()]


class _ParameterPackageAdapter:
    def __init__(self) -> None:
        identity = ModelIdentity("fixtures", "parameters")
        runtime_defaults = ModuleType("tests.parameter_runtime_defaults")
        runtime_defaults.DATA_NUM_WORKERS = None
        runtime_defaults.RUN_TEST_AFTER_FIT = False
        runtime_defaults.SEED = None
        dataset_options = ModuleType("tests.parameter_dataset_options")
        dataset_options.DEFAULT_EXPERIMENT_TASK = ExperimentTask.IMAGE_CLASSIFICATION
        dataset_options.DATASET_OPTIONS_BY_TASK = {
            ExperimentTask.IMAGE_CLASSIFICATION: [SyntheticDataset]
        }
        monitor_options = ModuleType("tests.parameter_monitor_options")
        monitor_options.MONITOR_OPTIONS = []
        search_space = ModuleType("tests.parameter_search_space")
        self.metadata = ModelMetadata(
            identity=identity,
            runtime_defaults=runtime_defaults,
            dataset_options=dataset_options,
            monitor_options_source=monitor_options,
            search_space=search_space,
        )

    def load_metadata(self):
        return self.metadata

    def load_runtime_options_type(self):
        return object

    def bind_runtime_defaults(self, _values):
        return object()

    def load_preset_type(self):
        return _Preset

    def load_presets(self):
        return _PresetGenerator()

    def build_configurations(self, presets, preset, dataset, **kwargs):
        return presets.get_config(preset, dataset, **kwargs)

    def build_model(self, configuration):
        return _Model(configuration)

    def build_experiment(
        self,
        preset,
        *,
        experiment_task,
        model_package,
        run_artifacts,
    ):
        return _ParameterExperiment(
            preset,
            experiment_task=experiment_task,
            model_package=model_package,
            run_artifacts=run_artifacts,
        )


def _parameter_model_package() -> ModelPackage:
    adapter = _ParameterPackageAdapter()
    return ModelPackage(adapter.metadata.identity, adapter)


class _Model:
    def __init__(self, config: _Config) -> None:
        self.config = config


class _Metric:
    @staticmethod
    def item() -> float:
        return 0.75


class _Logger:
    def __init__(self, save_dir: str, name: str) -> None:
        self.log_dir = str(Path(save_dir) / name)


class _Trainer:
    def __init__(self, *_args, **_kwargs) -> None:
        self.callback_metrics = {"validation_accuracy": _Metric()}
        self.current_epoch = 0
        self.global_step = 1

    def fit(self, _model, *, datamodule) -> None:
        self.datamodule = datamodule


class _Progress(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict] = []

    def write_event(self, event: dict) -> None:
        self.events.append(dict(event))


class _ParameterExperiment(ExperimentBase):
    def _num_epochs(self) -> int:
        return 1

    def _load_trainer_config(self, config_overrides=None) -> dict:
        return {"trainer_args": {}, "callbacks": []}

    def _load_runtime_config(self, config_overrides=None) -> dict:
        return {
            "data_num_workers": None,
            "run_test_after_fit": False,
            "seed": None,
        }

    def _write_training_result(self, _log_dir: str, _result: dict) -> None:
        pass

    def _update_best_results(
        self,
        _result: dict,
        _top5: dict,
        _log_folder: str | None = None,
    ) -> None:
        pass


class RunsParameterPreservationTests(unittest.TestCase):
    def test_materialized_run_retains_requested_parameter_names(self) -> None:
        package = _parameter_model_package()
        run = RunSpec(
            id="run-0001",
            experiment_task="image-classification",
            preset="baseline",
            dataset="SyntheticDataset",
            parameters=(
                RunParameter(
                    key="NUM_EPOCHS",
                    value=3,
                    source="override",
                ),
            ),
        )
        plan = RunPlan(
            identity=package.identity,
            presets=("baseline",),
            experiment_task="image-classification",
            datasets=("SyntheticDataset",),
            overrides={"NUM_EPOCHS": 3},
            search=None,
            runs=(run,),
        )

        with (
            patch.object(
                execution,
                "parse_overrides",
                return_value=SimpleNamespace(values={"num_epochs": 3}),
            ),
            patch.object(execution, "_reject_conflicting_locks"),
        ):
            _task, _presets, materialized = execution._validated_materialized_runs(
                package,
                plan,
            )

        self.assertEqual(materialized[0]["parameters"], {"NUM_EPOCHS": 3})
        self.assertEqual(materialized[0]["config_overrides"], {"num_epochs": 3})

    def test_requested_parameters_drive_artifacts_and_progress(self) -> None:
        package = _parameter_model_package()
        runtime = _ParameterExperiment(
            experiment_task=ExperimentTask.IMAGE_CLASSIFICATION,
            model_package=package,
        )
        training_run = runtime.materialize_training_runs(
            [
                {
                    "id": "run-0001",
                    "preset": _Preset.BASELINE,
                    "dataset_type": SyntheticDataset,
                    "parameters": {"NUM_EPOCHS": 3},
                    "config_overrides": {"num_epochs": 3},
                }
            ],
            "runs",
        )[0]
        progress = _Progress()

        with (
            patch.object(experiment, "Trainer", _Trainer),
            patch.object(experiment, "TensorBoardLogger", _Logger),
        ):
            result, log_dir = runtime.execute_training_run(
                training_run,
                log_folder="runs",
                callbacks=[progress],
                best_results={},
            )

        self.assertEqual(training_run.parameters, {"NUM_EPOCHS": 3})
        self.assertEqual(result["params"], {"NUM_EPOCHS": 3})
        self.assertEqual(progress.events[0]["params"], {"NUM_EPOCHS": 3})
        self.assertNotIn("/default_", log_dir)


if __name__ == "__main__":
    unittest.main()
