from __future__ import annotations

import itertools
import json
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from emperor.config import ModelConfig
from emperor.datasets.image.classification import Mnist
from model_runtime.packages.configuration import config_key_to_model_param
from model_runtime.packages.definition import ModelPackage
from model_runtime.runs.artifacts import validate_artifact_namespace
from model_runtime.runs.search import (
    _combination_at_index as _runs_combination_at_index,
)
from model_runtime.runs.search import _combination_count as _runs_combination_count
from model_runtime.runs.search import (
    _sample_unique_combination_indices as _runs_sample_unique_indices,
)


def _validate_log_folder(log_folder: str | None) -> str | None:
    return validate_artifact_namespace(log_folder)


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


def _random_access_search_axes(parameter_value_options: list) -> list:
    axes = []
    for options in parameter_value_options:
        if isinstance(options, Sequence):
            axes.append(options)
        else:
            axes.append(tuple(options))
    return axes


def _search_space_combination_count(parameter_value_options: list) -> int:
    return _runs_combination_count(parameter_value_options)


def _sample_unique_combination_indices(
    total_combinations: int,
    num_samples: int,
) -> list[int]:
    return _runs_sample_unique_indices(
        total_combinations,
        num_samples,
        random,
    )


def _combination_at_index(parameter_value_options: list, index: int) -> tuple:
    return _runs_combination_at_index(parameter_value_options, index)


def create_search_space(
    base_preset_callback: Callable[..., ModelConfig],
    base_config: Mapping[str, object],
    search_space: Mapping[str, Iterable[object]] | None = None,
    search_mode: SearchMode = None,
) -> list[ModelConfig]:
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
        self._model_package: ModelPackage | None = None

    @property
    def default_preset(self):
        raise NotImplementedError(
            "Preset providers must declare one canonical default preset."
        )

    def bind_model_package(self, package: ModelPackage) -> None:
        if not isinstance(package, ModelPackage):
            raise TypeError("Preset binding requires a ModelPackage.")
        if self._model_package is not None and self._model_package != package:
            raise ValueError("Preset instances cannot be rebound to another package.")
        self._model_package = package

    def _bound_model_package(self) -> ModelPackage:
        if self._model_package is None:
            raise RuntimeError(
                "Preset operations that access metadata require a selected "
                "ModelPackage. Use MODEL_PACKAGE.presets."
            )
        return self._model_package

    def get_config(
        self,
        model_config_preset,
        dataset,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
        search_keys: list[str] | None = None,
        config_overrides: dict | None = None,
        search_overrides: dict | None = None,
    ) -> list[ModelConfig]:
        raise NotImplementedError(
            "The method 'train_model' must be implemented in the subclass."
        )

    def _preset(self, *args, **kwargs) -> ModelConfig:
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
    ) -> list[ModelConfig]:
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
    ) -> list[ModelConfig]:
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
        model_id = self._bound_model_package().identity.catalog_key
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
        search_space_module = self._bound_model_package().metadata.search_space
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

    @property
    def default_preset(self):
        return self._default_preset

    def get_config(
        self,
        model_config_preset=_DEFAULT_MODEL_CONFIG_PRESET,
        dataset: type = _DEFAULT_DATASET,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
        search_keys: list[str] | None = None,
        config_overrides: dict | None = None,
        search_overrides: dict | None = None,
    ) -> list[ModelConfig]:
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
    ) -> ModelConfig:
        return self._preset(**{**kwargs, **self.overrides_for_preset(preset)})

    def _preset(self, **kwargs) -> ModelConfig:
        return self._builder_type(**kwargs).build()


__all__ = [
    "BuilderBackedExperimentPresetsBase",
    "ExperimentPresetsBase",
    "GridSearch",
    "PresetDefinition",
    "PresetLock",
    "RandomSearch",
    "SearchMode",
    "create_search_space",
]
