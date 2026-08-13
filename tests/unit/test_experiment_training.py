import json
import os
import tempfile
import unittest
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import cast
from unittest.mock import patch

from lightning.pytorch.callbacks import Callback

import model_runtime.runs.experiment as experiments_base
from emperor.experiments import ExperimentTask
from emperor.monitoring import MonitorOption
from model_runtime.packages import (
    BuilderBackedExperimentPresetsBase,
    ExperimentPresetsBase,
    ModelIdentity,
    ModelMetadata,
    ModelPackage,
    PresetDefinition,
)
from model_runtime.runs import ExperimentBase, JsonlRunProgress
from model_runtime.runs._handoff import TrainingRunRequest
from model_runtime.runs._lightning_progress import lightning_progress_adapter
from model_runtime.runs.artifacts import (
    DEFAULT_RESULT_METRIC_KEY_LIMIT,
    FilesystemRunArtifacts,
)
from model_runtime.runs.progress import ContextualRunProgress, RunProgressContext


def _progress_callback(
    path: Path,
    *,
    step_interval: int = 1,
    **writer_options,
) -> Callback:
    writer = JsonlRunProgress(path, **writer_options)
    progress = ContextualRunProgress(
        writer,
        RunProgressContext(
            experiment_task="image-classification",
            dataset="FakeDatasetB",
            preset="baseline",
            preset_key="BASELINE",
            log_dir="logs/test",
            run_id="run-0001",
            run_index=1,
            run_total=1,
            total_epochs=1,
        ),
    )
    return lightning_progress_adapter(progress, step_interval=step_interval)


class FakeMetric:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeConfig:
    batch_size = 2


class FakeDatasetA:
    flattened_input_dim = 4
    num_classes = 2

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.num_workers = 4


class FakeDatasetB:
    flattened_input_dim = 8
    num_classes = 3

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.num_workers = 4


class _SnapshotPreset(Enum):
    BASELINE = "baseline"
    UNKNOWN = "unknown"


class _SnapshotConfigBuilder:
    def __init__(self, **values: object) -> None:
        self.values = values

    def build(self) -> dict[str, object]:
        return dict(self.values)


class _DefinitionOverridingPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self, projected_value: object) -> None:
        super().__init__(
            {
                _SnapshotPreset.BASELINE: PresetDefinition(
                    preset_values={"enabled": True},
                    description="Stored description.",
                )
            },
            builder_type=_SnapshotConfigBuilder,
            default_preset=_SnapshotPreset.BASELINE,
            default_dataset=FakeDatasetA,
        )
        self.projected_value = projected_value

    def definition_for_preset(
        self,
        model_config_preset: object,
    ) -> PresetDefinition:
        definition = super().definition_for_preset(model_config_preset)
        return PresetDefinition(
            preset_values={
                **definition.preset_values,
                "projected_value": self.projected_value,
            },
            description="Projected description.",
        )


class PresetDefinitionSnapshotTests(unittest.TestCase):
    def test_builder_backed_provider_projects_values_through_runtime_factory(
        self,
    ) -> None:
        runtime = object()
        received_values: list[dict[str, object]] = []

        def runtime_factory(values: dict[str, object]) -> object:
            received_values.append(dict(values))
            return runtime

        presets = BuilderBackedExperimentPresetsBase(
            {
                _SnapshotPreset.BASELINE: PresetDefinition(
                    preset_values={"enabled": True},
                    description="Runtime-backed baseline.",
                )
            },
            builder_type=_SnapshotConfigBuilder,
            default_preset=_SnapshotPreset.BASELINE,
            default_dataset=FakeDatasetA,
            runtime_factory=runtime_factory,
        )

        configuration = cast(
            dict[str, object],
            presets.get_config(
                config_overrides={"batch_size": 4, "trainer_devices": 2}
            )[0],
        )

        self.assertEqual(
            received_values,
            [
                {
                    "input_dim": 4,
                    "output_dim": 2,
                    "batch_size": 4,
                    "enabled": True,
                }
            ],
        )
        self.assertEqual(configuration, {"runtime": runtime})

    def test_provider_owns_definitions_and_returns_fresh_public_maps(self) -> None:
        opaque_value = object()
        source_values: dict[str, object] = {
            "enabled": True,
            "opaque_value": opaque_value,
        }
        source_definition = PresetDefinition(
            preset_values=source_values,
            description="Snapshot baseline.",
        )
        source_definitions = {_SnapshotPreset.BASELINE: source_definition}
        presets = BuilderBackedExperimentPresetsBase(
            source_definitions,
            builder_type=_SnapshotConfigBuilder,
            default_preset=_SnapshotPreset.BASELINE,
            default_dataset=FakeDatasetA,
        )

        first_definition = presets.definition_for_preset(_SnapshotPreset.BASELINE)
        first_overrides = presets.overrides_for_preset(_SnapshotPreset.BASELINE)
        first_locks = presets.locks_for_preset(_SnapshotPreset.BASELINE)
        first_configuration = cast(
            dict[str, object],
            presets.get_config(_SnapshotPreset.BASELINE)[0],
        )
        expected_definition = PresetDefinition(
            preset_values={"enabled": True, "opaque_value": opaque_value},
            description="Snapshot baseline.",
        )

        self.assertEqual(first_definition, expected_definition)
        self.assertEqual(repr(first_definition), repr(expected_definition))
        self.assertIsInstance(first_definition.preset_values, dict)
        self.assertIs(first_definition.preset_values["opaque_value"], opaque_value)
        self.assertIs(first_overrides["opaque_value"], opaque_value)
        self.assertIs(first_locks["opaque_value"].value, opaque_value)
        self.assertIs(first_configuration["opaque_value"], opaque_value)

        source_definitions[_SnapshotPreset.BASELINE] = PresetDefinition(
            preset_values={"enabled": False},
            description="Replacement.",
        )
        source_values["enabled"] = False
        source_values["added"] = "source mutation"
        cast(dict[str, object], first_definition.preset_values)["enabled"] = False
        first_overrides["enabled"] = False

        second_definition = presets.definition_for_preset(_SnapshotPreset.BASELINE)
        second_overrides = presets.overrides_for_preset(_SnapshotPreset.BASELINE)
        second_locks = presets.locks_for_preset(_SnapshotPreset.BASELINE)
        second_configuration = cast(
            dict[str, object],
            presets.get_config(_SnapshotPreset.BASELINE)[0],
        )
        self.assertEqual(second_definition, expected_definition)
        self.assertEqual(second_overrides, expected_definition.preset_values)
        self.assertTupleEqual(tuple(second_locks), ("enabled", "opaque_value"))
        self.assertEqual(
            second_locks["enabled"].reason,
            "Locked by the BASELINE preset because this preset locks `enabled`.",
        )
        self.assertIsNot(first_definition, second_definition)
        self.assertIsNot(
            first_definition.preset_values, second_definition.preset_values
        )
        self.assertIsNot(first_overrides, second_overrides)
        self.assertIs(second_definition.preset_values["opaque_value"], opaque_value)
        self.assertIs(second_overrides["opaque_value"], opaque_value)
        self.assertIs(second_locks["opaque_value"].value, opaque_value)
        self.assertIs(second_configuration["opaque_value"], opaque_value)

        with self.assertRaisesRegex(
            ValueError,
            "The specified preset is not supported. Please choose a valid "
            "`ExperimentPreset`.",
        ) as raised:
            presets.definition_for_preset(_SnapshotPreset.UNKNOWN)
        self.assertIsInstance(raised.exception.__cause__, KeyError)

    def test_definition_override_propagates_through_every_consumer(self) -> None:
        projected_value = object()
        presets = _DefinitionOverridingPresets(projected_value)

        overrides = presets.overrides_for_preset(_SnapshotPreset.BASELINE)
        locks = presets.locks_for_preset(_SnapshotPreset.BASELINE)
        configuration = cast(
            dict[str, object],
            presets.get_config(_SnapshotPreset.BASELINE)[0],
        )

        self.assertIs(overrides["projected_value"], projected_value)
        self.assertEqual(
            presets.description_for_preset(_SnapshotPreset.BASELINE),
            "Projected description.",
        )
        self.assertIs(locks["projected_value"].value, projected_value)
        self.assertIs(configuration["projected_value"], projected_value)


class FakeOption(Enum):
    BASELINE = "baseline"
    GATING = "gating"
    HALTING = "halting"


class FakePresetGenerator(ExperimentPresetsBase):
    def __init__(self):
        super().__init__(
            {
                preset: PresetDefinition(
                    preset_values={},
                    description=preset.name.lower(),
                )
                for preset in FakeOption
            }
        )
        self.seen_datasets = []
        self.seen_presets = []

    def get_config(
        self,
        model_config_preset,
        dataset,
        *,
        config_overrides=None,
    ):
        self.seen_presets.append(model_config_preset.name)
        self.seen_datasets.append(dataset.__name__)
        return [FakeConfig()]


class FakeModel:
    def __init__(self, config):
        self.config = config


class FakeLogger:
    def __init__(self, save_dir, name, version=None):
        selected_version = 0 if version is None else version
        self.log_dir = str(Path(save_dir) / name / f"version_{selected_version}")


class FakeTrainer:
    instances = []

    def __init__(self, max_epochs, logger, callbacks, **kwargs):
        self.max_epochs = max_epochs
        self.logger = logger
        self.callbacks = callbacks
        self.callback_metrics = {"validation_accuracy": FakeMetric(0.75)}
        self.current_epoch = 1
        self.global_step = 2
        type(self).instances.append(self)

    def fit(self, model, datamodule):
        self.model = model
        self.fit_datamodule = datamodule

    def test(self, model, datamodule):
        self.test_datamodule = datamodule


class FakeMonitorCallback(Callback):
    pass


class CaptureRunProgress:
    def __init__(self):
        self.events = []

    def write_event(self, event):
        self.events.append(dict(event))


class FakePackageAdapter:
    def __init__(self) -> None:
        identity = ModelIdentity("test", "fake")
        runtime_defaults = ModuleType("tests.fake_runtime_defaults")
        runtime_defaults.DATA_NUM_WORKERS = 4
        runtime_defaults.NUM_EPOCHS = 1
        runtime_defaults.RUN_TEST_AFTER_FIT = True
        runtime_defaults.SEED = None
        dataset_metadata = ModuleType("tests.fake_dataset_metadata")
        dataset_metadata.DEFAULT_EXPERIMENT_TASK = ExperimentTask.IMAGE_CLASSIFICATION
        dataset_metadata.DATASET_OPTIONS_BY_TASK = {
            ExperimentTask.IMAGE_CLASSIFICATION: [FakeDatasetA, FakeDatasetB]
        }
        monitor_metadata = ModuleType("tests.fake_monitor_metadata")
        monitor_metadata.MONITOR_OPTIONS = []
        search_metadata = ModuleType("tests.fake_search_metadata")
        self.metadata = ModelMetadata(
            identity=identity,
            runtime_defaults=runtime_defaults,
            dataset_options=dataset_metadata,
            monitor_options_source=monitor_metadata,
            search_space=search_metadata,
        )

    def load_metadata(self):
        return self.metadata

    def load_runtime_options_type(self):
        return object

    def bind_runtime_defaults(self, values):
        return object()

    def load_preset_type(self):
        return FakeOption

    def load_presets(self):
        return FakePresetGenerator()

    def build_configuration(self, presets, preset, dataset, **kwargs):
        return presets.get_config(preset, dataset, **kwargs)[0]

    def build_model(self, configuration):
        return FakeModel(configuration)

    def build_experiment(self, preset, **kwargs):
        raise AssertionError("The fake package does not construct experiments.")


def fake_model_package() -> ModelPackage:
    adapter = FakePackageAdapter()
    return ModelPackage(adapter.metadata.identity, adapter)


class FakeExperiment(ExperimentBase):
    def _load_trainer_config(self, config_overrides=None):
        return {"trainer_args": {}, "callbacks": []}


class TestExperimentTraining(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.original_cwd = os.getcwd()
        os.chdir(self.tempdir.name)
        FakeTrainer.instances.clear()
        self.original_trainer = experiments_base.Trainer
        self.original_logger = experiments_base.TensorBoardLogger
        experiments_base.Trainer = FakeTrainer
        experiments_base.TensorBoardLogger = FakeLogger
        self.model_package = fake_model_package()

    def tearDown(self):
        experiments_base.Trainer = self.original_trainer
        experiments_base.TensorBoardLogger = self.original_logger
        os.chdir(self.original_cwd)
        self.tempdir.cleanup()

    def _execute_run(
        self,
        experiment,
        *,
        dataset_type=FakeDatasetA,
        preset=FakeOption.BASELINE,
        config_overrides=None,
        parameters=None,
        callbacks=None,
        progress=None,
        run_id="run-0001",
        run_index=1,
        run_total=1,
    ):
        training_run = experiment.materialize_training_runs(
            [
                TrainingRunRequest(
                    run_id=run_id,
                    run_index=run_index,
                    run_total=run_total,
                    preset=preset,
                    dataset_type=dataset_type,
                    parameters=parameters or {},
                    config_overrides=config_overrides or {},
                )
            ]
        )[0]
        return experiment.execute_training_run(
            training_run,
            callbacks=callbacks or [],
            progress=progress,
        )

    def test_constructor_preserves_preset_keyword_and_rejects_ambiguous_alias(self):
        experiment = ExperimentBase(
            preset=FakeOption.BASELINE,
            model_package=self.model_package,
        )

        self.assertIs(experiment.preset, FakeOption.BASELINE)
        with self.assertRaisesRegex(
            TypeError,
            "Pass only 'preset' or 'experiment_preset'",
        ):
            ExperimentBase(
                FakeOption.BASELINE,
                experiment_preset=FakeOption.GATING,
                model_package=self.model_package,
            )

    def test_data_num_workers_override_updates_datamodule(self):
        experiment = FakeExperiment(
            FakeOption.BASELINE, model_package=self.model_package
        )

        self._execute_run(
            experiment,
            config_overrides={"data_num_workers": 0},
        )

        self.assertEqual(FakeTrainer.instances[0].fit_datamodule.num_workers, 0)

    def test_run_test_after_fit_override_skips_test_phase(self):
        experiment = FakeExperiment(
            FakeOption.BASELINE, model_package=self.model_package
        )

        self._execute_run(
            experiment,
            config_overrides={"run_test_after_fit": False},
        )

        self.assertFalse(hasattr(FakeTrainer.instances[0], "test_datamodule"))

    def test_run_test_after_fit_defaults_to_enabled(self):
        experiment = FakeExperiment(
            FakeOption.BASELINE, model_package=self.model_package
        )

        self._execute_run(experiment)

        self.assertIs(
            FakeTrainer.instances[0].test_datamodule,
            FakeTrainer.instances[0].fit_datamodule,
        )

    def test_run_execution_instantiates_model_with_materialized_config(self):
        experiment = FakeExperiment(
            FakeOption.BASELINE, model_package=self.model_package
        )

        self._execute_run(experiment)

        self.assertIsInstance(FakeTrainer.instances[0].model.config, FakeConfig)

    def test_real_lightning_logger_consumes_the_reserved_directory(self):
        experiment = FakeExperiment(
            FakeOption.BASELINE,
            model_package=self.model_package,
        )

        with patch.object(
            experiments_base,
            "TensorBoardLogger",
            self.original_logger,
        ):
            result, log_dir = self._execute_run(experiment)

        resolved_log_dir = Path(log_dir).resolve()
        self.assertEqual(resolved_log_dir.name, "version_0")
        self.assertTrue(resolved_log_dir.is_dir())
        self.assertEqual(
            json.loads((resolved_log_dir / "result.json").read_text(encoding="utf-8")),
            result,
        )

    def test_run_result_keeps_core_metrics_after_monitor_metric_pressure(self):
        class MonitorHeavyTrainer(FakeTrainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.callback_metrics = {
                    **{
                        f"monitor/module_{index:04d}/mean": FakeMetric(index)
                        for index in range(DEFAULT_RESULT_METRIC_KEY_LIMIT + 200)
                    },
                    "validation/accuracy": FakeMetric(0.75),
                    "validation/loss": FakeMetric(0.5),
                }

        experiments_base.Trainer = MonitorHeavyTrainer
        experiment = FakeExperiment(
            FakeOption.BASELINE,
            model_package=self.model_package,
        )

        result, log_dir = self._execute_run(experiment)

        self.assertEqual(result["metrics"]["validation/accuracy"], 0.75)
        self.assertEqual(result["metrics"]["validation/loss"], 0.5)
        self.assertEqual(
            json.loads(Path(log_dir, "result.json").read_text(encoding="utf-8")),
            result,
        )

    def test_materialized_run_sets_progress_context_and_events(self):
        experiment = FakeExperiment(model_package=self.model_package)
        progress = CaptureRunProgress()

        self._execute_run(
            experiment,
            progress=progress,
            dataset_type=FakeDatasetB,
            preset=FakeOption.HALTING,
            parameters={"NUM_EPOCHS": 3},
            config_overrides={"num_epochs": 3},
            run_id="run-from-plan",
            run_index=7,
            run_total=9,
        )

        self.assertEqual(experiment.preset_generator.seen_presets, ["HALTING"])
        self.assertEqual(experiment.preset_generator.seen_datasets, ["FakeDatasetB"])
        self.assertEqual(len(progress.events), 2)
        started, completed = progress.events
        self.assertEqual(
            {
                key: started[key]
                for key in (
                    "dataset",
                    "logDir",
                    "preset",
                    "presetKey",
                    "runId",
                    "runIndex",
                    "runTotal",
                    "totalEpochs",
                )
            },
            {
                "dataset": "FakeDatasetB",
                "logDir": started["logDir"],
                "preset": "halting",
                "presetKey": "HALTING",
                "runId": "run-from-plan",
                "runIndex": 7,
                "runTotal": 9,
                "totalEpochs": 3,
            },
        )
        self.assertTrue(
            started["logDir"].startswith(
                f"logs/{experiment.model_package.catalog_key}/HALTING/FakeDatasetB/"
            )
        )
        self.assertNotIn("/default_", started["logDir"])
        self.assertEqual(
            [event["type"] for event in progress.events],
            ["dataset_started", "dataset_completed"],
        )
        self.assertEqual(started["status"], "running")
        self.assertEqual(started["dataset"], "FakeDatasetB")
        self.assertEqual(started["preset"], "halting")
        self.assertEqual(started["presetKey"], "HALTING")
        self.assertEqual(started["runId"], "run-from-plan")
        self.assertEqual(started["runIndex"], 7)
        self.assertEqual(started["runTotal"], 9)
        self.assertEqual(started["totalEpochs"], 3)
        self.assertEqual(started["params"], {"NUM_EPOCHS": 3})
        self.assertEqual(completed["metrics"], {"validation_accuracy": 0.75})

    def test_training_lifecycle_order_is_stable(self):
        events = []
        configured_callback = FakeMonitorCallback()
        explicit_callback = FakeMonitorCallback()
        progress_callback = FakeMonitorCallback()

        class TracingArtifacts:
            root = Path("logs")

            def run_name(self, identity, preset_key, dataset, parameters):
                events.append("run_name")
                return "trace/model/BASELINE/FakeDatasetA/run"

            def result_metrics_payload(self, metrics):
                return {
                    "metrics": {key: value.item() for key, value in metrics.items()}
                }

            def write_result(self, log_dir, result):
                events.append("write_result")
                return Path(log_dir) / "result.json"

            def update_best_results(self, identity, experiment_task, result):
                events.append("update_best_results")
                return {}

        class TracingPackageAdapter(FakePackageAdapter):
            def build_model(self, configuration):
                events.append("model")
                return super().build_model(configuration)

        adapter = TracingPackageAdapter()
        package = ModelPackage(adapter.metadata.identity, adapter)

        class TracingExperiment(FakeExperiment):
            def _run_progress_context(self, training_run):
                events.append("progress_context")
                return super()._run_progress_context(training_run)

            def _load_trainer_config(self, config_overrides=None):
                events.append("trainer_config")
                return {
                    "trainer_args": {"fixture_option": "kept"},
                    "callbacks": [configured_callback],
                }

            def _load_runtime_config(self, config_overrides=None):
                events.append("runtime_config")
                return {
                    "data_num_workers": 2,
                    "run_test_after_fit": True,
                    "seed": 17,
                }

            def _build_dataset(self, training_run):
                events.append("dataset")
                return super()._build_dataset(training_run)

            def _configure_dataset(self, dataset, runtime_config):
                events.append("configure_dataset")
                super()._configure_dataset(dataset, runtime_config)

            def _training_result(self, training_run, trainer, *, resumed_from=None):
                events.append("training_result")
                return super()._training_result(
                    training_run,
                    trainer,
                    resumed_from=resumed_from,
                )

            def _emit_dataset_started(
                self,
                training_run,
                progress,
                *,
                resumed_from=None,
            ):
                events.append("dataset_started")
                super()._emit_dataset_started(
                    training_run,
                    progress,
                    resumed_from=resumed_from,
                )

            def _emit_dataset_completed(
                self,
                result,
                progress,
                *,
                resumed_from=None,
            ):
                events.append("dataset_completed")
                super()._emit_dataset_completed(
                    result,
                    progress,
                    resumed_from=resumed_from,
                )

        class TracingLogger:
            def __init__(self, save_dir, name, version=None):
                events.append("logger")
                self.version = version
                self.log_dir = str(Path(save_dir) / name)

        class TracingTrainer:
            instances = []

            def __init__(self, max_epochs, logger, callbacks, **kwargs):
                events.append("trainer")
                self.max_epochs = max_epochs
                self.logger = logger
                self.callbacks = callbacks
                self.kwargs = kwargs
                self.callback_metrics = {"validation_accuracy": FakeMetric(0.75)}
                type(self).instances.append(self)

            def fit(self, model, datamodule, **kwargs):
                events.append("fit")
                self.model = model
                self.fit_datamodule = datamodule
                self.fit_kwargs = kwargs

            def test(self, model, datamodule):
                events.append("test")
                self.test_datamodule = datamodule

        experiment = TracingExperiment(
            FakeOption.BASELINE,
            model_package=package,
            run_artifacts=TracingArtifacts(),
        )
        training_run = experiment.materialize_training_runs(
            [
                TrainingRunRequest(
                    run_id="trace-run",
                    run_index=1,
                    run_total=1,
                    preset=FakeOption.BASELINE,
                    dataset_type=FakeDatasetA,
                    parameters={"SEED": 17},
                    config_overrides={},
                )
            ]
        )[0]
        progress = CaptureRunProgress()
        resumed_from = {"checkpoint": "safe-name.ckpt"}
        original_contextual_progress = experiments_base.contextual_run_progress

        def contextual_progress(writer, context):
            events.append("contextual_progress")
            return original_contextual_progress(writer, context)

        def validate_model(model):
            events.append("validate_model")

        with (
            patch.object(experiments_base, "Trainer", TracingTrainer),
            patch.object(experiments_base, "TensorBoardLogger", TracingLogger),
            patch.object(
                experiments_base,
                "seed_everything",
                side_effect=lambda seed, *, workers: events.append("seed"),
            ) as seed_everything_mock,
            patch.object(
                experiments_base,
                "contextual_run_progress",
                side_effect=contextual_progress,
            ),
            patch.object(
                experiments_base,
                "lightning_progress_adapter",
                side_effect=lambda selected_progress, *, step_interval: (
                    events.append("progress_callback") or progress_callback
                ),
            ) as progress_adapter,
        ):
            result, log_dir = experiment.execute_training_run(
                training_run,
                callbacks=[explicit_callback],
                progress=progress,
                progress_step_interval=7,
                ckpt_path=Path("resume.ckpt"),
                model_validator=validate_model,
                resumed_from=resumed_from,
            )

        self.assertEqual(
            events,
            [
                "progress_context",
                "contextual_progress",
                "trainer_config",
                "runtime_config",
                "seed",
                "dataset",
                "configure_dataset",
                "model",
                "validate_model",
                "run_name",
                "logger",
                "dataset_started",
                "progress_callback",
                "trainer",
                "fit",
                "test",
                "training_result",
                "write_result",
                "update_best_results",
                "dataset_completed",
            ],
        )
        seed_everything_mock.assert_called_once_with(17, workers=True)
        progress_adapter.assert_called_once()
        self.assertEqual(progress_adapter.call_args.kwargs, {"step_interval": 7})
        trainer = TracingTrainer.instances[0]
        self.assertEqual(
            trainer.callbacks,
            [configured_callback, explicit_callback, progress_callback],
        )
        self.assertEqual(trainer.kwargs, {"fixture_option": "kept"})
        self.assertEqual(
            trainer.fit_kwargs,
            {"ckpt_path": Path("resume.ckpt"), "weights_only": True},
        )
        self.assertEqual(log_dir, "logs/trace/model/BASELINE/FakeDatasetA/run")
        self.assertIsNone(trainer.logger.version)
        self.assertEqual(result["resumedFrom"], resumed_from)
        self.assertEqual(
            [event["type"] for event in progress.events],
            ["dataset_started", "dataset_completed"],
        )

    def test_progress_context_failure_is_outside_training_error_boundary(self):
        experiment = FakeExperiment(
            FakeOption.BASELINE,
            model_package=self.model_package,
        )
        training_run = experiment.materialize_training_runs(
            [
                TrainingRunRequest(
                    run_id="run-0001",
                    run_index=1,
                    run_total=1,
                    preset=FakeOption.BASELINE,
                    dataset_type=FakeDatasetA,
                    parameters={},
                    config_overrides={},
                )
            ]
        )[0]

        with (
            patch.object(
                experiment,
                "_run_progress_context",
                side_effect=RuntimeError("progress context exploded"),
            ),
            patch.object(experiment, "_emit_training_error") as emit_error,
            self.assertRaisesRegex(RuntimeError, "progress context exploded"),
        ):
            experiment.execute_training_run(training_run, callbacks=[])

        emit_error.assert_not_called()

    def test_training_failure_remains_primary_when_error_event_sink_fails(self):
        class FailingPreparationExperiment(FakeExperiment):
            def _prepare_training_runtime(self, state):
                raise RuntimeError("primary training failure")

        class FailingProgress:
            def write_event(self, event):
                raise OSError("progress sink failure")

        experiment = FailingPreparationExperiment(
            FakeOption.BASELINE,
            model_package=self.model_package,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "primary training failure",
        ) as raised:
            self._execute_run(experiment, progress=FailingProgress())

        self.assertTrue(
            any(
                "OSError: progress sink failure" in note
                for note in raised.exception.__notes__
            )
        )

    def test_dataset_started_sink_failure_is_the_primary_start_failure(self):
        class FailingStartedProgress:
            def __init__(self):
                self.events = []

            def write_event(self, event):
                payload = dict(event)
                if payload["type"] == "dataset_started":
                    raise OSError("dataset-started sink failure")
                self.events.append(payload)

        experiment = FakeExperiment(
            FakeOption.BASELINE,
            model_package=self.model_package,
        )
        progress = FailingStartedProgress()

        with self.assertRaisesRegex(
            OSError,
            "dataset-started sink failure",
        ):
            self._execute_run(experiment, progress=progress)

        self.assertEqual(FakeTrainer.instances, [])
        self.assertEqual([event["type"] for event in progress.events], ["error"])
        self.assertEqual(
            progress.events[0]["error"],
            "dataset-started sink failure",
        )

    def test_healthy_sink_receives_training_error_and_traceback(self):
        class FailingPreparationExperiment(FakeExperiment):
            def _prepare_training_runtime(self, state):
                raise RuntimeError("primary training failure")

        experiment = FailingPreparationExperiment(
            FakeOption.BASELINE,
            model_package=self.model_package,
        )
        progress = CaptureRunProgress()

        with self.assertRaisesRegex(RuntimeError, "primary training failure"):
            self._execute_run(experiment, progress=progress)

        self.assertEqual(len(progress.events), 1)
        error_event = progress.events[0]
        self.assertEqual(error_event["type"], "error")
        self.assertEqual(error_event["status"], "failed")
        self.assertEqual(error_event["error"], "primary training failure")
        self.assertIn("Traceback (most recent call last)", error_event["traceback"])
        self.assertIn(
            "RuntimeError: primary training failure",
            error_event["traceback"],
        )

    def test_dataset_completed_sink_failure_is_the_primary_completion_failure(self):
        class FailingCompletedProgress:
            def __init__(self):
                self.events = []

            def write_event(self, event):
                payload = dict(event)
                if payload["type"] == "dataset_completed":
                    raise OSError("dataset-completed sink failure")
                self.events.append(payload)

        experiment = FakeExperiment(
            FakeOption.BASELINE,
            model_package=self.model_package,
        )
        progress = FailingCompletedProgress()

        with self.assertRaisesRegex(
            OSError,
            "dataset-completed sink failure",
        ):
            self._execute_run(experiment, progress=progress)

        self.assertEqual(len(FakeTrainer.instances), 1)
        self.assertEqual(
            [event["type"] for event in progress.events],
            ["dataset_started", "error"],
        )
        self.assertEqual(
            progress.events[-1]["error"],
            "dataset-completed sink failure",
        )

    def test_run_execution_rejects_path_like_log_folder(self):
        with self.assertRaises(ValueError):
            FakeExperiment(
                FakeOption.BASELINE,
                model_package=self.model_package,
                run_artifacts=FilesystemRunArtifacts(namespace="../escape"),
            )

    def test_causal_language_model_dataset_receives_configured_sequence_length(self):
        experiment = FakeExperiment(
            FakeOption.BASELINE, model_package=self.model_package
        )
        training_run = type(
            "TrainingRun",
            (),
            {
                "experiment_task": ExperimentTask.CAUSAL_LANGUAGE_MODELING,
                "config": type(
                    "Config",
                    (),
                    {"batch_size": 7, "sequence_length": 19},
                )(),
            },
        )()

        self.assertEqual(
            experiment._dataset_constructor_kwargs(training_run),
            {"batch_size": 7, "sequence_length": 19},
        )

    def test_progress_callback_emits_jsonl_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            callback = _progress_callback(path)
            trainer = FakeTrainer(
                max_epochs=1,
                logger=FakeLogger("logs", "test"),
                callbacks=[],
            )
            callback.on_train_batch_end(trainer, None, None, None, 3)

            event = json.loads(path.read_text().splitlines()[0])
            self.assertEqual(event["status"], "running")
            self.assertEqual(event["dataset"], "FakeDatasetB")
            self.assertEqual(event["preset"], "baseline")
            self.assertEqual(event["presetKey"], "BASELINE")
            self.assertEqual(event["epoch"], 1)
            self.assertEqual(event["step"], 2)
            self.assertEqual(event["metrics"]["validation_accuracy"], 0.75)

    def test_progress_callback_can_throttle_step_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            callback = _progress_callback(path, step_interval=5)
            trainer = FakeTrainer(
                max_epochs=1,
                logger=FakeLogger("logs", "test"),
                callbacks=[],
            )
            callback.on_train_batch_end(trainer, None, None, None, 3)
            self.assertFalse(path.exists())

            trainer.global_step = 5
            callback.on_train_batch_end(trainer, None, None, None, 4)

            events = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["type"], "step")
            self.assertEqual(events[0]["step"], 5)

    def test_progress_callback_filters_high_cardinality_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            callback = _progress_callback(
                path,
                metric_key_limit=3,
            )
            trainer = FakeTrainer(
                max_epochs=1,
                logger=FakeLogger("logs", "test"),
                callbacks=[],
            )
            trainer.callback_metrics = {
                "validation_accuracy": FakeMetric(0.75),
                "train/confusion_matrix/0/0": FakeMetric(12),
                "validation/per_class/0/accuracy": FakeMetric(0.5),
                "train/loss": FakeMetric(0.25),
                "grad_norm": FakeMetric(1.2),
                "extra_metric": FakeMetric(99),
            }

            callback.on_train_batch_end(trainer, None, None, None, 3)

            event = json.loads(path.read_text().splitlines()[0])
            self.assertEqual(
                event["metrics"],
                {
                    "validation_accuracy": 0.75,
                    "train/loss": 0.25,
                    "grad_norm": 1.2,
                },
            )
            self.assertEqual(event["metricsOriginalCount"], 6)
            self.assertEqual(event["metricsDroppedCount"], 3)

    def test_result_metrics_filter_high_cardinality_metrics(self):
        payload = FilesystemRunArtifacts().result_metrics_payload(
            {
                "validation_accuracy": FakeMetric(0.75),
                "train/confusion_matrix/0/0": FakeMetric(12),
                "validation/per_class/0/accuracy": FakeMetric(0.5),
                "train/loss": FakeMetric(0.25),
            }
        )

        self.assertEqual(
            payload,
            {
                "metrics": {
                    "validation_accuracy": 0.75,
                    "train/loss": 0.25,
                },
                "metricsOriginalCount": 4,
                "metricsDroppedCount": 2,
            },
        )

    def test_progress_callback_enforces_event_byte_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            callback = _progress_callback(
                path,
                metric_key_limit=100,
                event_byte_limit=800,
            )
            trainer = FakeTrainer(
                max_epochs=1,
                logger=FakeLogger("logs", "test"),
                callbacks=[],
            )
            trainer.callback_metrics = {
                "validation_accuracy": FakeMetric(0.75),
                **{f"metric_{index}": "x" * 120 for index in range(20)},
            }

            callback.on_train_batch_end(trainer, None, None, None, 3)

            raw_line = path.read_text().splitlines()[0]
            event = json.loads(raw_line)
            self.assertLessEqual(len(raw_line.encode("utf-8")), 800)
            self.assertEqual(event["metrics"]["validation_accuracy"], 0.75)
            self.assertGreater(event["metricsDroppedCount"], 0)

    def test_monitor_option_build_callback_returns_fresh_instances(self):
        option = MonitorOption(
            name="fake",
            label="Fake monitor",
            description="Test monitor.",
            kinds=["scalar"],
            callback_factory=lambda _settings: FakeMonitorCallback(),
        )

        first = option.build_callback()
        second = option.build_callback()

        self.assertIsInstance(first, FakeMonitorCallback)
        self.assertIsInstance(second, FakeMonitorCallback)
        self.assertIsNot(first, second)
