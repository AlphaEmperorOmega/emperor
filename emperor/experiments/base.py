import fcntl
import hashlib
import importlib
import itertools
import json
import os
import random
import tempfile
import traceback
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from models.catalog import model_identity_payload_from_id, public_id_for_module
from models.config_overrides import canonical_config_key, config_key_to_model_param

from emperor.base.options import BaseOptions
from emperor.config import ModelConfig
from emperor.datasets.image.classification.cifar_10 import Cifar10
from emperor.datasets.image.classification.cifar_100 import Cifar100
from emperor.datasets.image.classification.fashion_mnist import FashionMNIST
from emperor.datasets.image.classification.mnist import Mnist
from emperor.experiments.progress import sanitize_metric_payload
from emperor.experiments.tasks import (
    ExperimentTask,
    experiment_task_name,
    resolve_experiment_task,
)

DEFAULT_RESULT_METRIC_KEY_LIMIT = 512
DEFAULT_RESULT_STRING_VALUE_LIMIT = 20_000


@dataclass
class GridSearch:
    pass


@dataclass
class RandomSearch:
    num_samples: int


@dataclass(frozen=True)
class PresetLock:
    value: object
    reason: str


@dataclass(frozen=True)
class PresetDefinition:
    preset_values: Mapping[str, object]
    description: str


@dataclass(frozen=True)
class _EffectivePresetLock:
    field: str
    lock: PresetLock


@dataclass(frozen=True)
class _PresetLockConflict:
    field: str
    source: str
    attempted_key: str
    attempted_value: object
    lock: PresetLock


SearchMode = GridSearch | RandomSearch | None
_DEFAULT_MODEL_CONFIG_PRESET = object()
_DEFAULT_DATASET = object()


@dataclass
class _TrainingRun:
    experiment_task: ExperimentTask | None
    preset: BaseOptions
    dataset_type: type
    config: ModelConfig
    config_overrides: dict
    num_epochs: int
    run_id: str | None = None
    run_index: int | None = None
    run_total: int | None = None


def _public_model_id_from_package(package: str) -> str:
    public_id = public_id_for_module(package)
    if public_id is not None:
        return public_id
    return package.rsplit(".", 1)[-1]


def _validate_log_folder(log_folder: str | None) -> str | None:
    if log_folder is None:
        return None
    folder = str(log_folder)
    path = Path(folder)
    if (
        not folder
        or folder in {".", ".."}
        or "\\" in folder
        or path.is_absolute()
        or len(path.parts) != 1
    ):
        raise ValueError(
            "log_folder must be a single relative folder name without path separators"
        )
    return folder


def _random_access_search_axes(parameter_value_options: list) -> list:
    axes = []
    for options in parameter_value_options:
        if isinstance(options, Sequence):
            axes.append(options)
        else:
            axes.append(tuple(options))
    return axes


def _search_space_combination_count(parameter_value_options: list) -> int:
    combination_count = 1
    for options in parameter_value_options:
        combination_count *= len(options)
    return combination_count


def _sample_unique_combination_indices(
    total_combinations: int,
    num_samples: int,
) -> list[int]:
    sample_count = min(num_samples, total_combinations)
    if sample_count < 0:
        raise ValueError("Sample larger than population or is negative")

    try:
        return random.sample(range(total_combinations), sample_count)
    except OverflowError:
        pass

    selected_indices = []
    selected_index_set = set()
    start = total_combinations - sample_count
    for offset in range(sample_count):
        upper_bound = start + offset
        candidate = random.randrange(upper_bound + 1)
        if candidate in selected_index_set:
            candidate = upper_bound
        selected_index_set.add(candidate)
        selected_indices.append(candidate)
    return selected_indices


def _combination_at_index(parameter_value_options: list, index: int) -> tuple:
    parameter_values = []
    for options in reversed(parameter_value_options):
        index, option_index = divmod(index, len(options))
        parameter_values.append(options[option_index])
    return tuple(reversed(parameter_values))


def _result_metrics_payload(metrics: dict) -> dict:
    sanitized, original_count, dropped_count = sanitize_metric_payload(
        metrics,
        metric_key_limit=DEFAULT_RESULT_METRIC_KEY_LIMIT,
        string_value_limit=DEFAULT_RESULT_STRING_VALUE_LIMIT,
    )
    payload = {"metrics": sanitized}
    if dropped_count > 0:
        payload["metricsOriginalCount"] = original_count
        payload["metricsDroppedCount"] = dropped_count
    return payload


@contextmanager
def _best_results_lock(summary_path: Path):
    lock_path = summary_path.with_suffix(summary_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _read_best_results_path(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _write_json_atomic(summary_path: Path, payload: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=summary_path.parent,
            encoding="utf-8",
            prefix=f".{summary_path.name}.",
            suffix=".tmp",
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(payload, temp_file, indent=2, default=str)
            temp_file.write("\n")
        os.replace(temp_path, summary_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def create_search_space(
    base_preset_callback: Callable,
    base_config: dict,
    search_space: dict | None = None,
    search_mode: SearchMode = None,
) -> list["ModelConfig"]:
    search_space = search_space or {}
    if search_space == {}:
        return [base_preset_callback(**base_config)]

    experiments = []
    parameter_names = list(search_space.keys())
    parameter_value_options = list(search_space.values())

    if isinstance(search_mode, RandomSearch):
        parameter_value_options = _random_access_search_axes(parameter_value_options)
        total_combinations = _search_space_combination_count(parameter_value_options)
        all_combinations = (
            _combination_at_index(parameter_value_options, combination_index)
            for combination_index in _sample_unique_combination_indices(
                total_combinations,
                search_mode.num_samples,
            )
        )
    else:
        all_combinations = itertools.product(*parameter_value_options)

    for parameter_values in all_combinations:
        updated_params = {**base_config}
        for param_name, param_value in zip(
            parameter_names,
            parameter_values,
            strict=True,
        ):
            updated_params[param_name] = param_value
        preset = base_preset_callback(**updated_params)
        experiments.append(preset)

    return experiments


class ExperimentPresetsBase:
    def __init__(
        self,
        preset_definitions: Mapping[object, PresetDefinition],
    ) -> None:
        self._preset_definitions = dict(preset_definitions)

    def get_config(
        self,
        model_config_preset,
        dataset,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
        search_keys: list[str] | None = None,
        config_overrides: dict | None = None,
        search_overrides: dict | None = None,
    ) -> list["ModelConfig"]:
        raise NotImplementedError(
            "The method 'train_model' must be implemented in the subclass."
        )

    def _preset(self, *args, **kwargs) -> "ModelConfig":
        raise NotImplementedError(
            "The method '_preset' must be implemented in the subclass."
        )

    def definition_for_preset(self, model_config_preset) -> PresetDefinition:
        try:
            return self._preset_definitions[model_config_preset]
        except KeyError as exc:
            raise ValueError(
                "The specified preset is not supported. Please choose a valid "
                "`ExperimentPreset`."
            ) from exc

    def overrides_for_preset(self, model_config_preset) -> dict[str, object]:
        return dict(self.definition_for_preset(model_config_preset).preset_values)

    def description_for_preset(self, model_config_preset) -> str:
        return self.definition_for_preset(model_config_preset).description

    def locks_for_preset(self, model_config_preset) -> dict[str, PresetLock]:
        return {
            field: PresetLock(
                value=value,
                reason=self._preset_lock_reason(model_config_preset, field),
            )
            for field, value in self.overrides_for_preset(model_config_preset).items()
        }

    def _preset_lock_reason(self, model_config_preset, field: str) -> str:
        return (
            f"Locked by the {model_config_preset.name} preset because this preset "
            f"locks `{field}`."
        )

    def locked_fields(self, model_config_preset) -> dict[str, PresetLock]:
        return dict(self.locks_for_preset(model_config_preset))

    def _create_default_preset_configs(
        self,
        dataset: type = Mnist,
        search_mode: SearchMode = None,
        search_keys: list[str] | None = None,
        config_overrides: dict | None = None,
        search_overrides: dict | None = None,
        model_config_preset=None,
    ) -> list["ModelConfig"]:
        return self._create_preset_search_space_configs(
            dataset,
            search_mode,
            self._preset,
            search_keys,
            config_overrides=config_overrides,
            search_overrides=search_overrides,
            model_config_preset=model_config_preset,
        )

    def _create_preset_search_space_configs(
        self,
        dataset: type = Mnist,
        search_mode: SearchMode = None,
        preset_callback: Callable | None = None,
        search_keys: list[str] | None = None,
        config_overrides: dict | None = None,
        search_overrides: dict | None = None,
        model_config_preset=None,
    ) -> list["ModelConfig"]:
        model_config_overrides = self._model_config_overrides(config_overrides)
        self._validate_preset_config_overrides(
            model_config_preset,
            model_config_overrides,
        )
        self._validate_preset_search_overrides(
            model_config_preset,
            search_overrides or {},
        )
        base_config = {
            **self._dataset_config(dataset),
            **model_config_overrides,
        }
        if search_overrides and search_keys is None:
            search_space = self._search_space_for_model_params(search_overrides)
        else:
            search_space = self._extract_search_space_from_config(
                search_mode,
                search_keys,
                model_config_preset=model_config_preset,
            )
            search_space.update(
                self._search_space_for_model_params(search_overrides or {})
            )
        return create_search_space(
            preset_callback or self._preset,
            base_config,
            search_space,
            search_mode,
        )

    def _dataset_config(self, dataset: type) -> dict:
        return {
            "input_dim": dataset.flattened_input_dim,
            "output_dim": dataset.num_classes,
        }

    def _model_config_overrides(self, config_overrides: dict | None = None) -> dict:
        ignored_prefixes = ("trainer_", "callback_", "data_", "run_")
        ignored_keys = {"num_epochs"}
        return {
            key: value
            for key, value in (config_overrides or {}).items()
            if key not in ignored_keys
            and not any(key.startswith(prefix) for prefix in ignored_prefixes)
        }

    def _best_params(self, dataset: type, log_folder: str | None = None) -> dict:
        package = type(self).__module__.rsplit(".", 1)[0]
        model_id = _public_model_id_from_package(package)
        log_folder = _validate_log_folder(log_folder)
        folder = Path(log_folder) / model_id if log_folder else Path(model_id)
        path = Path("logs") / folder / "best_results.json"
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
        runs = data.get(dataset.__name__, [])
        if not runs:
            return {}
        best = min(runs, key=lambda r: r.get("rank", 999))
        return {
            k: v
            for k, v in best.get("params", {}).items()
            if type(v) in (int, float, bool)
        }

    def _extract_search_space_from_config(
        self,
        search_mode: SearchMode = None,
        search_keys: list[str] | None = None,
        model_config_preset=None,
    ) -> dict:
        if search_mode is None:
            return {}
        package = type(self).__module__.rsplit(".", 1)[0]
        search_space_module = importlib.import_module(f"{package}.search_space")
        prefix = "SEARCH_SPACE_"
        full_space = {
            key[len(prefix) :].lower(): value
            for key, value in vars(search_space_module).items()
            if key.startswith(prefix)
        }
        if search_keys is not None:
            unknown_keys = set(search_keys) - set(full_space)
            if unknown_keys:
                raise ValueError(
                    f"Unknown --search-keys: {sorted(unknown_keys)}. "
                    f"Valid keys: {sorted(full_space)}"
                )
            self._validate_preset_search_keys(
                model_config_preset,
                search_keys,
                full_space,
            )
            return self._search_space_for_model_params(
                {key: full_space[key] for key in search_keys}
            )

        full_space = self._dedupe_search_space_aliases(full_space)

        locked_fields = self._effective_locked_fields(model_config_preset)
        if not locked_fields:
            return self._search_space_for_model_params(full_space)
        return self._search_space_for_model_params(
            {
                key: value
                for key, value in full_space.items()
                if self._effective_model_param_name(key) not in locked_fields
            }
        )

    def _search_space_for_model_params(self, search_space: dict) -> dict:
        return {
            self._effective_model_param_name(key): value
            for key, value in search_space.items()
        }

    def _effective_model_param_name(self, key: str) -> str:
        return config_key_to_model_param(key)

    def _dedupe_search_space_aliases(self, full_space: dict) -> dict:
        selected: dict[str, object] = {}
        selected_by_param: dict[str, tuple[str, int]] = {}
        for key, value in full_space.items():
            model_param = self._effective_model_param_name(key)
            canonical_key = canonical_config_key(key).lower()
            preference = 1 if canonical_key == key else 0
            existing = selected_by_param.get(model_param)
            if existing is None:
                selected[key] = value
                selected_by_param[model_param] = (key, preference)
                continue
            existing_key, existing_preference = existing
            if preference <= existing_preference:
                continue
            del selected[existing_key]
            selected[key] = value
            selected_by_param[model_param] = (key, preference)
        return selected

    def _effective_locked_fields(
        self,
        model_config_preset,
    ) -> dict[str, _EffectivePresetLock]:
        return {
            self._effective_model_param_name(field): _EffectivePresetLock(field, lock)
            for field, lock in self.locked_fields(model_config_preset).items()
        }

    def _validate_preset_config_overrides(
        self,
        model_config_preset,
        config_overrides: dict,
    ) -> None:
        locked_fields = self._effective_locked_fields(model_config_preset)
        conflicts = []
        for key, value in config_overrides.items():
            locked = locked_fields.get(self._effective_model_param_name(key))
            if locked is None or value == locked.lock.value:
                continue
            conflicts.append(
                _PresetLockConflict(
                    field=locked.field,
                    source="config override",
                    attempted_key=key,
                    attempted_value=value,
                    lock=locked.lock,
                )
            )
        self._raise_preset_lock_conflicts(model_config_preset, conflicts)

    def _validate_preset_search_overrides(
        self,
        model_config_preset,
        search_overrides: dict,
    ) -> None:
        locked_fields = self._effective_locked_fields(model_config_preset)
        conflicts = []
        for key, values in search_overrides.items():
            locked = locked_fields.get(self._effective_model_param_name(key))
            if locked is None:
                continue
            attempted_values = self._search_values(values)
            if all(value == locked.lock.value for value in attempted_values):
                continue
            conflicts.append(
                _PresetLockConflict(
                    field=locked.field,
                    source="search override",
                    attempted_key=key,
                    attempted_value=attempted_values,
                    lock=locked.lock,
                )
            )
        self._raise_preset_lock_conflicts(model_config_preset, conflicts)

    def _validate_preset_search_keys(
        self,
        model_config_preset,
        search_keys: list[str],
        full_space: dict,
    ) -> None:
        locked_fields = self._effective_locked_fields(model_config_preset)
        conflicts = []
        for key in search_keys:
            locked = locked_fields.get(self._effective_model_param_name(key))
            if locked is None:
                continue
            conflicts.append(
                _PresetLockConflict(
                    field=locked.field,
                    source="search key",
                    attempted_key=key,
                    attempted_value=full_space[key],
                    lock=locked.lock,
                )
            )
        self._raise_preset_lock_conflicts(model_config_preset, conflicts)

    def _search_values(self, values) -> list:
        if isinstance(values, (list, tuple, set)):
            return list(values)
        return [values]

    def _raise_preset_lock_conflicts(
        self,
        model_config_preset,
        conflicts: list[_PresetLockConflict],
    ) -> None:
        if not conflicts:
            return
        preset_name = (
            model_config_preset.name
            if hasattr(model_config_preset, "name")
            else str(model_config_preset)
        )
        messages = []
        for conflict in conflicts:
            messages.append(
                f"{preset_name} locks {conflict.field}="
                f"{self._format_preset_lock_value(conflict.lock.value)}. "
                f"Cannot override it with {conflict.source} "
                f"{conflict.attempted_key}="
                f"{self._format_preset_lock_value(conflict.attempted_value)}. "
                f"{conflict.lock.reason} "
                "Remove that config/search override or choose a preset that owns "
                "different behavior."
            )
        raise ValueError(" ".join(messages))

    def _format_preset_lock_value(self, value: object) -> str:
        if isinstance(value, list):
            return (
                "[" + ", ".join(self._format_preset_lock_value(v) for v in value) + "]"
            )
        if isinstance(value, tuple):
            return (
                "(" + ", ".join(self._format_preset_lock_value(v) for v in value) + ")"
            )
        if isinstance(value, set):
            return (
                "{"
                + ", ".join(sorted(self._format_preset_lock_value(v) for v in value))
                + "}"
            )
        if isinstance(value, Enum):
            return value.name
        if isinstance(value, type):
            return value.__name__
        return repr(value)


class BuilderBackedExperimentPresetsBase(ExperimentPresetsBase):
    def __init__(
        self,
        preset_definitions: Mapping[object, PresetDefinition],
        *,
        builder_type: type,
        default_preset: object,
        default_dataset: type = Mnist,
    ) -> None:
        super().__init__(preset_definitions)
        self._builder_type = builder_type
        self._default_preset = default_preset
        self._default_dataset = default_dataset

    def get_config(
        self,
        model_config_preset=_DEFAULT_MODEL_CONFIG_PRESET,
        dataset: type = _DEFAULT_DATASET,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
        search_keys: list[str] | None = None,
        config_overrides: dict | None = None,
        search_overrides: dict | None = None,
    ) -> list["ModelConfig"]:
        if model_config_preset is _DEFAULT_MODEL_CONFIG_PRESET:
            model_config_preset = self._default_preset
        model_config_preset = self._normalize_model_config_preset(model_config_preset)
        if dataset is _DEFAULT_DATASET:
            dataset = self._default_dataset
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

    def _normalize_model_config_preset(self, model_config_preset):
        return model_config_preset

    def _preset_callback_for_preset(self, preset):
        self.definition_for_preset(preset)
        return lambda **kwargs: self._preset_for_preset(preset, **kwargs)

    def _preset_for_preset(
        self,
        preset,
        **kwargs,
    ) -> "ModelConfig":
        return self._preset(**{**kwargs, **self.overrides_for_preset(preset)})

    def _preset(self, **kwargs) -> "ModelConfig":
        return self._builder_type(**kwargs).build()


class ExperimentBase:
    def __init__(
        self,
        preset: BaseOptions | None = None,
        experiment_task: ExperimentTask | str | None = None,
    ) -> None:
        self.preset = preset
        self.num_epochs = self._num_epochs()
        self.experiment_task = self._resolve_experiment_task(experiment_task)
        self.dataset_options = self._dataset_options_for_task(self.experiment_task)
        self.model_type = self._model_type()
        self.preset_generator = self._preset_generator_instance()
        self.preset_enum = self._experiment_preset_enum()

    def _num_epochs(self) -> int:
        return 10

    def _dataset_options(self) -> list:
        return [Mnist, FashionMNIST, Cifar10, Cifar100]

    def _resolve_experiment_task(
        self,
        experiment_task: ExperimentTask | str | None,
    ) -> ExperimentTask | None:
        metadata = self._model_metadata()
        if metadata is None:
            return resolve_experiment_task(experiment_task)
        if experiment_task is None:
            return metadata.default_experiment_task
        return resolve_experiment_task(experiment_task)

    def _dataset_options_for_task(
        self,
        experiment_task: ExperimentTask | None,
    ) -> list:
        metadata = self._model_metadata()
        if metadata is None:
            return self._dataset_options()
        return metadata.dataset_options_for_task(experiment_task)

    def _model_metadata(self):
        from models.model_metadata import load_model_metadata_from_module_path

        package = type(self).__module__.rsplit(".", 1)[0]
        try:
            return load_model_metadata_from_module_path(package)
        except ModuleNotFoundError as exc:
            expected_metadata_modules = {
                f"{package}.config",
                f"{package}.dataset_options",
                f"{package}.monitor_options",
                f"{package}.search_space",
            }
            if (
                public_id_for_module(package) is not None
                or exc.name not in expected_metadata_modules
            ):
                raise
            return None

    def _model_type(self) -> type:
        raise NotImplementedError(
            "The method '_model_type' must be implemented in the subclass."
        )

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        raise NotImplementedError(
            "The method '_preset_generator_instance' must be implemented in the "
            "subclass."
        )

    def _experiment_preset_enum(self) -> type[BaseOptions]:
        raise NotImplementedError(
            "The method '_experiment_preset_enum' must be implemented in the subclass."
        )

    def _load_trainer_config(self, config_overrides: dict | None = None) -> dict:
        package = type(self.preset_generator).__module__.rsplit(".", 1)[0]
        config = importlib.import_module(f"{package}.config")
        config_overrides = config_overrides or {}
        return {
            "trainer_args": self._trainer_args(config, config_overrides),
            "callbacks": self._trainer_callbacks(config, config_overrides),
        }

    def _trainer_callbacks(self, config, config_overrides: dict) -> list[Callback]:
        callbacks = []
        early_stopping = self._early_stopping_callback(config, config_overrides)
        if early_stopping is not None:
            callbacks.append(early_stopping)
        checkpoint = self._checkpoint_callback(config, config_overrides)
        if checkpoint is not None:
            callbacks.append(checkpoint)

        for key, value in vars(config).items():
            if key.startswith("CALLBACK_") and isinstance(value, Callback):
                callbacks.append(value)
        return callbacks

    def _early_stopping_callback(
        self,
        config,
        config_overrides: dict,
    ) -> EarlyStopping | None:
        early_stopping_patience = self._trainer_config_value(
            config,
            config_overrides,
            "CALLBACK_EARLY_STOPPING_PATIENCE",
            0,
        )
        if early_stopping_patience <= 0:
            return None
        early_stopping_metric = self._trainer_config_value(
            config,
            config_overrides,
            "CALLBACK_EARLY_STOPPING_METRIC",
            "validation/loss",
        )
        return EarlyStopping(
            monitor=early_stopping_metric,
            patience=early_stopping_patience,
            min_delta=self._trainer_config_value(
                config,
                config_overrides,
                "CALLBACK_EARLY_STOPPING_MIN_DELTA",
                0.0,
            ),
            strict=self._trainer_config_value(
                config,
                config_overrides,
                "CALLBACK_EARLY_STOPPING_STRICT",
                True,
            ),
            check_finite=self._trainer_config_value(
                config,
                config_overrides,
                "CALLBACK_EARLY_STOPPING_CHECK_FINITE",
                True,
            ),
            mode="min" if "loss" in early_stopping_metric else "max",
        )

    def _checkpoint_callback(
        self,
        config,
        config_overrides: dict,
    ) -> ModelCheckpoint | None:
        checkpoint_flag = self._trainer_config_value(
            config,
            config_overrides,
            "CALLBACK_CHECKPOINT_FLAG",
            False,
        )
        if not checkpoint_flag:
            return None
        early_stopping_metric = self._trainer_config_value(
            config,
            config_overrides,
            "CALLBACK_EARLY_STOPPING_METRIC",
            "validation/loss",
        )
        return ModelCheckpoint(
            monitor=early_stopping_metric,
            save_top_k=1,
            mode="min" if "loss" in early_stopping_metric else "max",
        )

    def _trainer_config_value(
        self,
        config,
        config_overrides: dict,
        key: str,
        default=None,
    ):
        return config_overrides.get(key.lower(), getattr(config, key, default))

    def _load_runtime_config(self, config_overrides: dict | None = None) -> dict:
        package = type(self.preset_generator).__module__.rsplit(".", 1)[0]
        try:
            config = importlib.import_module(f"{package}.config")
        except ModuleNotFoundError as exc:
            if exc.name != f"{package}.config":
                raise
            config = None
        config_overrides = config_overrides or {}

        def runtime_value(key: str, default):
            if config is None:
                return config_overrides.get(key.lower(), default)
            return self._trainer_config_value(config, config_overrides, key, default)

        return {
            "data_num_workers": runtime_value("DATA_NUM_WORKERS", None),
            "run_test_after_fit": runtime_value("RUN_TEST_AFTER_FIT", True),
            "seed": runtime_value("SEED", None),
        }

    def _configure_dataset(self, dataset, runtime_config: dict) -> None:
        data_num_workers = runtime_config.get("data_num_workers")
        if data_num_workers is None or not hasattr(dataset, "num_workers"):
            pass
        else:
            dataset.num_workers = int(data_num_workers)
        seed = runtime_config.get("seed")
        if seed is not None and hasattr(dataset, "seed"):
            dataset.seed = int(seed)

    def _dataset_constructor_kwargs(self, training_run: _TrainingRun) -> dict:
        """Return package-specific keyword arguments for a data module."""

        kwargs = {"batch_size": training_run.config.batch_size}
        if training_run.experiment_task == ExperimentTask.CAUSAL_LANGUAGE_MODELING:
            kwargs["sequence_length"] = training_run.config.sequence_length
        return kwargs

    def _build_dataset(self, training_run: _TrainingRun):
        return training_run.dataset_type(
            **self._dataset_constructor_kwargs(training_run)
        )

    def _trainer_args(self, config, config_overrides: dict) -> dict:
        trainer_args = {}
        for key, value in vars(config).items():
            if not key.startswith("TRAINER_"):
                continue
            if value is None:
                continue
            clean_key = key[len("TRAINER_") :].lower()
            trainer_args[clean_key] = config_overrides.get(key.lower(), value)

        for key, value in config_overrides.items():
            if not key.startswith("trainer_"):
                continue
            clean_key = key[len("trainer_") :]
            if value is not None:
                trainer_args[clean_key] = value
        return trainer_args

    def train_model(
        self,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
        search_keys: list[str] | None = None,
        config_overrides: dict | None = None,
        search_overrides: dict | None = None,
        selected_datasets: list[type] | None = None,
        selected_presets: list[BaseOptions] | None = None,
        callbacks: list[Callback] | None = None,
        materialized_runs: list[dict] | None = None,
    ) -> None:
        log_folder = _validate_log_folder(log_folder)
        config_overrides = config_overrides or {}
        search_overrides = search_overrides or {}
        callbacks = callbacks or []
        best_results = self._load_best_results(log_folder)
        training_run_plan = self._build_training_run_plan(
            search_mode=search_mode,
            log_folder=log_folder,
            search_keys=search_keys,
            config_overrides=config_overrides,
            search_overrides=search_overrides,
            selected_datasets=selected_datasets,
            selected_presets=selected_presets,
            materialized_runs=materialized_runs,
        )
        for training_run in training_run_plan:
            self._execute_training_run(
                training_run,
                log_folder=log_folder,
                callbacks=callbacks,
                best_results=best_results,
            )

    def _build_training_run_plan(
        self,
        *,
        search_mode: SearchMode,
        log_folder: str | None,
        search_keys: list[str] | None,
        config_overrides: dict,
        search_overrides: dict,
        selected_datasets: list[type] | None,
        selected_presets: list[BaseOptions] | None,
        materialized_runs: list[dict] | None,
    ) -> list[_TrainingRun]:
        presets = (
            selected_presets
            if selected_presets is not None
            else [self.preset]
            if self.preset
            else self.preset_enum
        )
        dataset_options = selected_datasets or self.dataset_options
        if materialized_runs is None:
            return self._planned_training_runs(
                presets=presets,
                dataset_options=dataset_options,
                search_mode=search_mode,
                log_folder=log_folder,
                search_keys=search_keys,
                config_overrides=config_overrides,
                search_overrides=search_overrides,
            )
        return self._materialized_training_runs(materialized_runs, log_folder)

    def _planned_training_runs(
        self,
        *,
        presets,
        dataset_options: list[type],
        search_mode: SearchMode,
        log_folder: str | None,
        search_keys: list[str] | None,
        config_overrides: dict,
        search_overrides: dict,
    ) -> list[_TrainingRun]:
        training_runs = []
        for preset in presets:
            for dataset_type in dataset_options:
                run_overrides = config_overrides
                run_epochs = run_overrides.get("num_epochs", self.num_epochs)
                for config in self.preset_generator.get_config(
                    preset,
                    dataset_type,
                    search_mode,
                    log_folder,
                    search_keys,
                    config_overrides=run_overrides,
                    search_overrides=search_overrides,
                ):
                    training_runs.append(
                        _TrainingRun(
                            experiment_task=self.experiment_task,
                            preset=preset,
                            dataset_type=dataset_type,
                            config=config,
                            config_overrides=run_overrides,
                            num_epochs=run_epochs,
                        )
                    )
        return training_runs

    def _materialized_training_runs(
        self,
        materialized_runs: list[dict],
        log_folder: str | None,
    ) -> list[_TrainingRun]:
        training_runs = []
        run_total = len(materialized_runs)
        for run_index, run in enumerate(materialized_runs, start=1):
            preset = run["preset"]
            dataset_type = run["dataset_type"]
            run_overrides = run.get("config_overrides") or {}
            run_epochs = run_overrides.get("num_epochs", self.num_epochs)
            for config in self.preset_generator.get_config(
                preset,
                dataset_type,
                None,
                log_folder,
                None,
                config_overrides=run_overrides,
                search_overrides={},
            ):
                training_runs.append(
                    _TrainingRun(
                        experiment_task=self.experiment_task,
                        preset=preset,
                        dataset_type=dataset_type,
                        config=config,
                        config_overrides=run_overrides,
                        num_epochs=run_epochs,
                        run_id=run.get("id"),
                        run_index=run.get("index", run_index),
                        run_total=run.get("run_total", run_total),
                    )
                )
        return training_runs

    def _execute_training_run(
        self,
        training_run: _TrainingRun,
        *,
        log_folder: str | None,
        callbacks: list[Callback],
        best_results: dict,
    ) -> None:
        trainer_config = self._load_trainer_config(training_run.config_overrides)
        runtime_config = self._load_runtime_config(training_run.config_overrides)
        if runtime_config["seed"] is not None:
            seed_everything(int(runtime_config["seed"]), workers=True)
        dataset = self._build_dataset(training_run)
        self._configure_dataset(dataset, runtime_config)
        model = self.model_type(training_run.config)
        logger = TensorBoardLogger(
            save_dir="logs",
            name=self._build_log_path(
                training_run.preset,
                training_run.dataset_type,
                training_run.config,
                log_folder,
            ),
        )
        self._set_training_run_context(training_run, logger, callbacks)
        self._emit_dataset_started(training_run, logger, callbacks)
        trainer = Trainer(
            max_epochs=training_run.num_epochs,
            logger=logger,
            callbacks=[*trainer_config["callbacks"], *callbacks],
            **trainer_config["trainer_args"],
        )
        try:
            trainer.fit(model, datamodule=dataset)
            if runtime_config["run_test_after_fit"]:
                trainer.test(model, datamodule=dataset)
        except Exception as exc:
            self._emit_training_error(training_run, exc, callbacks)
            raise

        result = self._training_result(training_run, trainer)
        self._write_training_result(logger.log_dir, result)
        self._update_best_results(result, best_results, log_folder)
        self._emit_dataset_completed(training_run, logger, result, callbacks)

    def _set_training_run_context(
        self,
        training_run: _TrainingRun,
        logger,
        callbacks: list[Callback],
    ) -> None:
        for callback in callbacks:
            set_run_context = getattr(callback, "set_run_context", None)
            if callable(set_run_context):
                set_run_context(
                    training_run.dataset_type.__name__,
                    logger.log_dir,
                    self._preset_cli_name(training_run.preset),
                    training_run.preset.name,
                    run_id=training_run.run_id,
                    run_index=training_run.run_index,
                    run_total=training_run.run_total,
                    total_epochs=training_run.num_epochs,
                )

    def _emit_dataset_started(
        self,
        training_run: _TrainingRun,
        logger,
        callbacks: list[Callback],
    ) -> None:
        self._write_progress_event(
            callbacks,
            {
                **self._training_run_event_fields(training_run),
                "type": "dataset_started",
                "status": "running",
                "logDir": logger.log_dir,
                "params": training_run.config.get_custom_parameters(),
            },
        )

    def _emit_training_error(
        self,
        training_run: _TrainingRun,
        exc: Exception,
        callbacks: list[Callback],
    ) -> None:
        self._write_progress_event(
            callbacks,
            {
                **self._training_run_event_fields(training_run),
                "type": "error",
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )

    def _emit_dataset_completed(
        self,
        training_run: _TrainingRun,
        logger,
        result: dict,
        callbacks: list[Callback],
    ) -> None:
        self._write_progress_event(
            callbacks,
            {
                **self._training_run_event_fields(training_run),
                "type": "dataset_completed",
                "status": "running",
                "metrics": result["metrics"],
                "logDir": logger.log_dir,
            },
        )

    def _training_run_event_fields(self, training_run: _TrainingRun) -> dict:
        experiment_task = (
            experiment_task_name(training_run.experiment_task)
            if training_run.experiment_task is not None
            else None
        )
        return {
            "experimentTask": experiment_task,
            "dataset": training_run.dataset_type.__name__,
            "preset": self._preset_cli_name(training_run.preset),
            "presetKey": training_run.preset.name,
            "runId": training_run.run_id,
            "runIndex": training_run.run_index,
            "runTotal": training_run.run_total,
            "totalEpochs": training_run.num_epochs,
        }

    def _write_progress_event(
        self,
        callbacks: list[Callback],
        event: dict,
    ) -> None:
        for callback in callbacks:
            write_event = getattr(callback, "write_event", None)
            if callable(write_event):
                write_event(event)

    def _training_result(self, training_run: _TrainingRun, trainer) -> dict:
        experiment_task = (
            experiment_task_name(training_run.experiment_task)
            if training_run.experiment_task is not None
            else None
        )
        return {
            **self._public_model_identity_payload(),
            "experimentTask": experiment_task,
            "dataset": training_run.dataset_type.__name__,
            "preset": self._preset_cli_name(training_run.preset),
            "presetKey": training_run.preset.name,
            "params": training_run.config.get_custom_parameters(),
            **_result_metrics_payload(trainer.callback_metrics),
        }

    def _write_training_result(self, log_dir: str, result: dict) -> None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        (Path(log_dir) / "result.json").write_text(
            json.dumps(result, indent=2, default=str)
        )

    def _preset_cli_name(self, preset: BaseOptions) -> str:
        cli_name = getattr(type(preset), "cli_name", None)
        if callable(cli_name):
            return cli_name(preset.name)
        return preset.name.lower().replace("_", "-")

    def _load_best_results(self, log_folder: str | None = None) -> dict:
        return _read_best_results_path(self._best_results_path(log_folder))

    def _update_best_results(
        self, result: dict, top5: dict, log_folder: str | None = None
    ) -> None:
        summary_path = self._best_results_path(log_folder)
        with _best_results_lock(summary_path):
            merged_top5 = _read_best_results_path(summary_path)
            dataset = result["dataset"]
            runs = list(merged_top5.get(dataset, []))
            new_acc = result["metrics"].get("validation_accuracy", 0)
            worst_acc = min(
                (r["metrics"].get("validation_accuracy", 0) for r in runs),
                default=-1,
            )

            if len(runs) < 5 or new_acc > worst_acc:
                runs.append(result)
                merged_top5[dataset] = [
                    {**run, "rank": i + 1}
                    for i, run in enumerate(
                        sorted(
                            runs,
                            key=lambda r: r["metrics"].get("validation_accuracy", 0),
                            reverse=True,
                        )[:5]
                    )
                ]
                _write_json_atomic(summary_path, merged_top5)

            top5.clear()
            top5.update(merged_top5)

    def _best_results_path(self, log_folder: str | None = None) -> Path:
        log_folder = _validate_log_folder(log_folder)
        model_id = self._public_model_id()
        folder = (
            Path(log_folder) / model_id if log_folder is not None else Path(model_id)
        )
        return Path("logs") / folder / "best_results.json"

    def _build_log_path(
        self,
        preset: BaseOptions,
        dataset_type: type,
        config: "ModelConfig",
        log_folder: str | None = None,
    ) -> str:
        params = config.get_custom_parameters()
        param_str = "_".join(f"{k}={v}" for k, v in params.items())
        param_id = (
            hashlib.md5(param_str.encode()).hexdigest()[:8] if param_str else "default"
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = self._public_model_id()
        log_folder = _validate_log_folder(log_folder)
        folder = f"{log_folder}/{model_id}" if log_folder is not None else model_id
        return f"{folder}/{preset.name}/{dataset_type.__name__}/{param_id}_{timestamp}"

    def _public_model_id(self) -> str:
        package = type(self).__module__.rsplit(".", 1)[0]
        return _public_model_id_from_package(package)

    def _public_model_identity_payload(self) -> dict[str, str]:
        model_id = self._public_model_id()
        try:
            return model_identity_payload_from_id(model_id)
        except ValueError:
            return {"modelType": "models", "model": model_id}
