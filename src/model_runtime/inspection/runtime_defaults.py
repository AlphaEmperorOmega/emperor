from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType, ModuleType, NoneType
from typing import Any, get_args

from model_runtime.inspection.errors import InspectionError, _model_package_failure
from model_runtime.packages import (
    InspectionConstructionLimits,
    ModelPackage,
    config_key_to_model_param,
    iter_supported_config_keys,
    normalize_key,
    parse_config_value,
    serialize_config_value,
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
    package: ModelPackage
    config_module: ModuleType
    search_space_module: ModuleType
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
        return getattr(self.config_module, config_key, None)

    def accepts_none(self, config_key: str) -> bool:
        current_value = self.current_value(config_key)
        if current_value is None:
            return True
        if isinstance(current_value, list) and any(
            value is None for value in current_value
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
        return parse_config_value(self.config_module, config_key, value)

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
            self.search_space_module if search_key is not None else self.config_module
        )
        parse_key = search_key or config_key
        return parse_config_value(
            module,
            parse_key,
            str(self.serialize_value(raw_value)),
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
            raise InspectionError(str(exc)) from exc
        except Exception as exc:
            raise _model_package_failure(self.package.catalog_key, exc) from exc
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
            raise _model_package_failure(self.package.catalog_key, exc) from exc

        canonical: dict[str, Any] = {}
        source_fields: dict[str, str] = {}
        for field, lock in locks.items():
            model_param = self.model_parameter(field)
            previous = canonical.get(model_param)
            if previous is not None:
                previous_value = self.serialize_value(getattr(previous, "value", None))
                value = self.serialize_value(getattr(lock, "value", None))
                if previous_value != value:
                    raise InspectionError(
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


class _PackageCacheKey:
    __slots__ = ("package",)

    def __init__(self, package: ModelPackage) -> None:
        self.package = package

    def __hash__(self) -> int:
        return id(self.package)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _PackageCacheKey) and self.package is other.package


@lru_cache(maxsize=128)
def _cached_runtime_defaults_spec(
    cache_key: _PackageCacheKey,
) -> RuntimeDefaultsSpec:
    package = cache_key.package
    try:
        config_module = package.runtime_defaults
        search_space_module = package.metadata.search_space
        supported_keys = tuple(iter_supported_config_keys(config_module))
        keys_by_alias: dict[str, str] = {}
        for config_key in supported_keys:
            keys_by_alias[normalize_key(config_key)] = config_key
            keys_by_alias[normalize_key(config_key_to_model_param(config_key))] = (
                config_key
            )
        configuration_metadata = package.configuration_field_metadata()
        search_metadata = package.configuration_field_metadata(
            include_search_space=True
        )
        search_values = {
            key: tuple(values) for key, values in package.search_metadata.items()
        }
        skipped_schema_keys = frozenset(
            key
            for key in getattr(config_module, "CONFIG_SCHEMA_SKIP_KEYS", ())
            if isinstance(key, str)
        )
    except ValueError as exc:
        raise InspectionError(str(exc)) from exc
    except Exception as exc:
        raise _model_package_failure(package.catalog_key, exc) from exc

    return RuntimeDefaultsSpec(
        package=package,
        config_module=config_module,
        search_space_module=search_space_module,
        supported_keys=supported_keys,
        keys_by_alias=MappingProxyType(keys_by_alias),
        annotations=MappingProxyType(
            dict(getattr(config_module, "__annotations__", {}))
        ),
        search_annotations=MappingProxyType(
            dict(getattr(search_space_module, "__annotations__", {}))
        ),
        configuration_metadata=MappingProxyType(
            {
                key: MappingProxyType(dict(value))
                for key, value in configuration_metadata.items()
            }
        ),
        search_metadata=MappingProxyType(
            {
                key: MappingProxyType(dict(value))
                for key, value in search_metadata.items()
            }
        ),
        search_values=MappingProxyType(search_values),
        skipped_schema_keys=skipped_schema_keys,
        inspection_limits=package.inspection_construction_limits,
    )


def runtime_defaults_spec(package: ModelPackage) -> RuntimeDefaultsSpec:
    if not isinstance(package, ModelPackage):
        raise TypeError("Runtime Defaults require a selected ModelPackage.")
    # ModelPackage equality intentionally ignores its Adapter. Include object
    # identity in the cache key so distinct selected packages never share an
    # interpretation merely because their public identities compare equal.
    return _cached_runtime_defaults_spec(_PackageCacheKey(package))


__all__ = ["RuntimeDefaultsSpec", "runtime_defaults_spec"]
