from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from emperor.config import ModelConfig
from emperor.datasets.image.classification import Mnist
from model_runtime.packages.configuration import config_key_to_model_param


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


_DEFAULT_MODEL_CONFIG_PRESET = object()
_DEFAULT_DATASET = object()


class ExperimentPresetsBase:
    def __init__(
        self,
        preset_definitions: Mapping[object, PresetDefinition],
    ) -> None:
        self._preset_definitions = dict(preset_definitions)

    @property
    def default_preset(self):
        raise NotImplementedError(
            "Preset providers must declare one canonical default preset."
        )

    def get_config(
        self,
        model_config_preset,
        dataset,
        *,
        config_overrides: dict | None = None,
    ) -> list[ModelConfig]:
        raise NotImplementedError(
            "Preset providers must implement configuration materialization."
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
        *,
        config_overrides: dict | None = None,
    ) -> list[ModelConfig]:
        if model_config_preset is _DEFAULT_MODEL_CONFIG_PRESET:
            model_config_preset = self._default_preset
        model_config_preset = self._normalize_model_config_preset(model_config_preset)
        if dataset is _DEFAULT_DATASET:
            dataset = self._default_dataset
        preset_callback = self._preset_callback_for_preset(model_config_preset)
        model_config_overrides = self._model_config_overrides(config_overrides)
        self._validate_preset_config_overrides(
            model_config_preset,
            model_config_overrides,
        )
        base_config = {
            **self._dataset_config(dataset),
            **model_config_overrides,
        }
        return [preset_callback(**base_config)]

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
    "PresetDefinition",
    "PresetLock",
]
