from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from types import ModuleType, NoneType
from typing import TYPE_CHECKING, Any, cast, get_args

from model_runtime.packages.configuration import (
    abstract_config_class_error,
    config_key_to_model_param,
    normalize_key,
    parse_config_value,
    serialize_config_value,
)

if TYPE_CHECKING:
    from model_runtime.packages.definition import ModelPackage
    from model_runtime.packages.inspection_limits import InspectionConstructionLimits


class RuntimeDefaultsError(Exception):
    """A selected Model Package has invalid Runtime Defaults semantics."""


def _package_error(package: ModelPackage, exc: Exception) -> RuntimeDefaultsError:
    return RuntimeDefaultsError(
        f"Failed to import model package '{package.catalog_key}': {exc}"
    )


def _annotation_accepts_none(annotation: Any) -> bool:
    if annotation is None:
        return False
    if annotation is NoneType:
        return True
    if isinstance(annotation, str):
        return "None" in annotation or "Optional" in annotation
    if NoneType in get_args(annotation):
        return True
    return any(_annotation_accepts_none(arg) for arg in get_args(annotation))


@dataclass(frozen=True, slots=True)
class RuntimeDefaultsSpec:
    """Immutable interpretation of one selected package's Runtime Defaults."""

    package: ModelPackage
    _config_module: ModuleType = dataclass_field(repr=False)
    _search_space_module: ModuleType = dataclass_field(repr=False)
    supported_keys: tuple[str, ...]
    keys_by_alias: Mapping[str, str]
    annotations: Mapping[str, Any]
    search_annotations: Mapping[str, Any]
    configuration_metadata: Mapping[str, Mapping[str, Any]]
    search_metadata: Mapping[str, Mapping[str, Any]]
    search_values: Mapping[str, tuple[Any, ...]]
    skipped_schema_keys: frozenset[str]
    inspection_limits: InspectionConstructionLimits

    def resolve_key(self, key: str) -> str | None:
        return self.keys_by_alias.get(normalize_key(key))

    def model_parameter(self, config_key: str) -> str:
        return config_key_to_model_param(config_key)

    def current_value(self, config_key: str) -> Any:
        return getattr(self._config_module, config_key, None)

    def current_value_or(self, config_key: str, default: Any) -> Any:
        return getattr(self._config_module, config_key, default)

    def has_current_value(self, config_key: str) -> bool:
        return hasattr(self._config_module, config_key)

    def default_items(self) -> tuple[tuple[str, Any], ...]:
        return tuple(
            (key, value)
            for key, value in vars(self._config_module).items()
            if key in self.supported_keys
        )

    def items_with_prefix(self, prefix: str) -> Iterator[tuple[str, Any]]:
        for key, value in vars(self._config_module).items():
            if key.startswith(prefix):
                yield key, value

    def configuration_values(self) -> Iterator[Any]:
        yield from vars(self._config_module).values()

    def search_source_value(self, search_key: str) -> Any:
        return getattr(self._search_space_module, search_key, None)

    def has_search_value(self, search_key: str) -> bool:
        return hasattr(self._search_space_module, search_key)

    def configuration_applicability(self) -> object:
        return getattr(self._config_module, "CONFIG_FIELD_APPLICABILITY", {})

    def accepts_none(self, config_key: str) -> bool:
        current_value = self.current_value(config_key)
        if current_value is None:
            return True
        if isinstance(current_value, list) and any(
            value is None for value in cast(list[Any], current_value)
        ):
            return True
        return _annotation_accepts_none(self.annotations.get(config_key))

    def parse_value(self, config_key: str, raw_value: Any) -> Any:
        if raw_value is None:
            value = "None" if self.accepts_none(config_key) else ""
        else:
            value = str(raw_value)
            if value == "" and self.accepts_none(config_key):
                value = "None"
        return parse_config_value(self._config_module, config_key, value)

    def parse_search_value(
        self,
        config_key: str,
        raw_value: Any,
        *,
        search_key: str | None = None,
    ) -> Any:
        if raw_value is None:
            return None
        module = (
            self._search_space_module if search_key is not None else self._config_module
        )
        parse_key = search_key or config_key
        return parse_config_value(
            module,
            parse_key,
            str(self.serialize_value(raw_value)),
        )

    def parse_search_key_value(self, search_key: str, raw_value: Any) -> Any:
        return parse_config_value(
            self._search_space_module,
            search_key,
            str(raw_value),
        )

    def serialize_value(self, value: Any) -> Any:
        return serialize_config_value(value)

    def maximum_for(self, config_key: str) -> int | float | None:
        return self.inspection_limits.maximum_for(config_key)

    def resolve_preset_locks(
        self,
        preset_name: str | None,
    ) -> tuple[Any | None, dict[str, Any]]:
        if preset_name is None:
            return None, {}
        try:
            preset = self.package.resolve_preset(preset_name)
        except ValueError as exc:
            raise RuntimeDefaultsError(str(exc)) from exc
        except Exception as exc:
            raise _package_error(self.package, exc) from exc
        return preset, self.locks_for_preset(preset, label=preset_name)

    def locks_for_preset(
        self,
        preset: Any,
        *,
        label: str | None = None,
    ) -> dict[str, Any]:
        preset_label = label or getattr(preset, "name", str(preset))
        try:
            locks = self.package.preset_locks(preset)
        except Exception as exc:
            raise _package_error(self.package, exc) from exc

        canonical: dict[str, Any] = {}
        source_fields: dict[str, str] = {}
        for field, lock in locks.items():
            model_param = self.model_parameter(field)
            previous = canonical.get(model_param)
            if previous is not None:
                previous_value = self.serialize_value(getattr(previous, "value", None))
                value = self.serialize_value(getattr(lock, "value", None))
                if previous_value != value:
                    raise RuntimeDefaultsError(
                        f"Preset '{preset_label}' for model "
                        f"'{self.package.catalog_key}' defines conflicting locks for "
                        f"Runtime Defaults parameter '{model_param}' through "
                        f"'{source_fields[model_param]}' and '{field}'."
                    )
                continue
            canonical[model_param] = lock
            source_fields[model_param] = field
        return canonical

    def preset_locks(self, preset_name: str | None) -> dict[str, Any]:
        return self.resolve_preset_locks(preset_name)[1]

    def canonicalize_overrides(
        self,
        overrides: Mapping[str, Any] | None,
        *,
        ignore_unknown: bool = False,
    ) -> dict[str, Any]:
        if not overrides:
            return {}
        canonical: dict[str, Any] = {}
        for raw_key, raw_value in overrides.items():
            config_key = self.resolve_key(raw_key)
            if config_key is None:
                if ignore_unknown:
                    continue
                raise RuntimeDefaultsError(f"Unknown override '{raw_key}'.")
            canonical[config_key] = raw_value
        return canonical

    def parse_overrides(
        self,
        overrides: Mapping[str, Any] | None,
        *,
        preset: str | None = None,
        ignore_unknown: bool = False,
    ) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        if overrides:
            for raw_key, raw_value in overrides.items():
                config_key = self.resolve_key(raw_key)
                if config_key is None:
                    if ignore_unknown:
                        continue
                    raise RuntimeDefaultsError(f"Unknown override '{raw_key}'.")
                try:
                    parsed_value = self.parse_value(config_key, raw_value)
                    if isinstance(parsed_value, type):
                        abstract_error = abstract_config_class_error(parsed_value)
                        if abstract_error is not None:
                            raise ValueError(abstract_error)
                    parsed[self.model_parameter(config_key)] = parsed_value
                except RuntimeDefaultsError:
                    raise
                except Exception as exc:
                    raise RuntimeDefaultsError(
                        f"Invalid value for override '{raw_key}': {raw_value!r}. {exc}"
                    ) from exc
        if preset is not None:
            self.reject_locked_overrides(preset, parsed)
        return parsed

    def serialize_overrides(
        self,
        overrides: Mapping[str, Any] | None,
        *,
        ignore_unknown: bool = False,
    ) -> dict[str, Any]:
        canonical = self.canonicalize_overrides(
            overrides,
            ignore_unknown=ignore_unknown,
        )
        serialized: dict[str, Any] = {}
        for config_key, raw_value in canonical.items():
            try:
                parsed = self.parse_value(config_key, raw_value)
            except RuntimeDefaultsError:
                raise
            except Exception as exc:
                raise RuntimeDefaultsError(
                    f"Invalid value for override '{config_key}': {raw_value!r}. {exc}"
                ) from exc
            serialized[config_key] = self.serialize_value(parsed)
        return serialized

    def reject_locked_overrides(
        self,
        preset_name: str,
        parsed_overrides: Mapping[str, Any] | None,
    ) -> None:
        locks = self.preset_locks(preset_name)
        locked_keys = sorted(set(parsed_overrides or {}) & set(locks))
        if not locked_keys:
            return
        details = ", ".join(
            f"{key} ({getattr(locks[key], 'reason', '')})" for key in locked_keys
        )
        raise RuntimeDefaultsError(
            f"Preset '{preset_name}' does not allow overriding locked fields: {details}"
        )

    def reject_conflicting_locked_overrides(
        self,
        preset_name: str,
        parsed_overrides: Mapping[str, Any],
    ) -> None:
        locks = self.preset_locks(preset_name)
        conflicts = sorted(
            key
            for key, value in parsed_overrides.items()
            if key in locks and value != getattr(locks[key], "value", None)
        )
        if not conflicts:
            return
        details = ", ".join(
            f"{key} ({getattr(locks[key], 'reason', '')})" for key in conflicts
        )
        raise RuntimeDefaultsError(
            f"Preset '{preset_name}' does not allow overriding locked fields: {details}"
        )


def runtime_defaults_spec_for_package(package: ModelPackage) -> RuntimeDefaultsSpec:
    try:
        metadata = package.metadata
    except ValueError as exc:
        raise RuntimeDefaultsError(str(exc)) from exc
    except Exception as exc:
        raise _package_error(package, exc) from exc
    return metadata.compile_runtime_defaults_spec(
        package,
        package.inspection_construction_limits,
    )


__all__ = [
    "RuntimeDefaultsError",
    "RuntimeDefaultsSpec",
]
