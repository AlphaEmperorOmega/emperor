from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Never

from model_runtime.inspection.errors import InspectionError
from model_runtime.inspection.records import ParsedOverrides
from model_runtime.inspection.runtime_defaults import runtime_defaults_spec
from model_runtime.packages import (
    ModelPackage,
    RuntimeDefaultsError,
    config_key_to_model_param,
    normalize_key,
)


def _raise_inspection_error(exc: RuntimeDefaultsError) -> Never:
    raise InspectionError(str(exc)) from (exc.__cause__ or exc)


def supported_config_keys(package: ModelPackage) -> dict[str, str]:
    spec = runtime_defaults_spec(package)
    return {normalize_key(config_key): config_key for config_key in spec.supported_keys}


def resolve_override_key(
    normalized_key: str,
    supported: Mapping[str, str],
) -> str | None:
    config_key = supported.get(normalized_key)
    if config_key is not None:
        return config_key
    for config_key in supported.values():
        if config_key_to_model_param(config_key) == normalized_key:
            return config_key
    return None


def reject_locked_overrides(
    package: ModelPackage,
    preset_name: str,
    parsed_overrides: Mapping[str, Any] | None,
) -> None:
    try:
        runtime_defaults_spec(package).reject_locked_overrides(
            preset_name,
            parsed_overrides,
        )
    except RuntimeDefaultsError as exc:
        _raise_inspection_error(exc)


def reject_conflicting_locked_overrides(
    package: ModelPackage,
    preset_name: str,
    parsed_overrides: Mapping[str, Any],
) -> None:
    try:
        runtime_defaults_spec(package).reject_conflicting_locked_overrides(
            preset_name,
            parsed_overrides,
        )
    except RuntimeDefaultsError as exc:
        _raise_inspection_error(exc)


def parse_overrides(
    package: ModelPackage,
    overrides: Mapping[str, Any] | None,
    *,
    preset: str | None = None,
    ignore_unknown: bool = False,
) -> ParsedOverrides:
    try:
        parsed = runtime_defaults_spec(package).parse_overrides(
            overrides,
            preset=preset,
            ignore_unknown=ignore_unknown,
        )
    except RuntimeDefaultsError as exc:
        _raise_inspection_error(exc)
    return ParsedOverrides(parsed)


def canonicalize_overrides(
    package: ModelPackage,
    overrides: Mapping[str, Any] | None,
    *,
    ignore_unknown: bool = False,
) -> dict[str, Any]:
    if not overrides:
        return {}
    try:
        return runtime_defaults_spec(package).canonicalize_overrides(
            overrides,
            ignore_unknown=ignore_unknown,
        )
    except RuntimeDefaultsError as exc:
        _raise_inspection_error(exc)
    raise AssertionError("unreachable")


def serialize_overrides(
    package: ModelPackage,
    overrides: Mapping[str, Any] | None,
    *,
    ignore_unknown: bool = False,
) -> dict[str, Any]:
    try:
        return runtime_defaults_spec(package).serialize_overrides(
            overrides,
            ignore_unknown=ignore_unknown,
        )
    except RuntimeDefaultsError as exc:
        _raise_inspection_error(exc)
    raise AssertionError("unreachable")


__all__ = [
    "canonicalize_overrides",
    "parse_overrides",
    "reject_conflicting_locked_overrides",
    "reject_locked_overrides",
    "resolve_override_key",
    "serialize_overrides",
    "supported_config_keys",
]
