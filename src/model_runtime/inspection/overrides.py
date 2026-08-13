from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from model_runtime.inspection.errors import InspectionError
from model_runtime.inspection.records import (
    ParsedOverrides,
    parsed_overrides_for_identity,
    parsed_overrides_identity,
)
from model_runtime.inspection.runtime_defaults import apply_runtime_defaults
from model_runtime.packages import (
    ModelPackage,
    config_key_to_model_param,
    normalize_key,
)


def supported_config_keys(package: ModelPackage) -> dict[str, str]:
    return apply_runtime_defaults(
        package,
        lambda spec: {
            normalize_key(config_key): config_key for config_key in spec.supported_keys
        },
    )


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
    apply_runtime_defaults(
        package,
        lambda spec: spec.reject_locked_overrides(
            preset_name,
            parsed_overrides,
        ),
    )


def reject_conflicting_locked_overrides(
    package: ModelPackage,
    preset_name: str,
    parsed_overrides: Mapping[str, Any],
) -> None:
    apply_runtime_defaults(
        package,
        lambda spec: spec.reject_conflicting_locked_overrides(
            preset_name,
            parsed_overrides,
        ),
    )


def parse_overrides(
    package: ModelPackage,
    overrides: Mapping[str, Any] | None,
    *,
    preset: str | None = None,
    ignore_unknown: bool = False,
) -> ParsedOverrides:
    parsed = apply_runtime_defaults(
        package,
        lambda spec: spec.parse_overrides(
            overrides,
            preset=preset,
            ignore_unknown=ignore_unknown,
        ),
    )
    return parsed_overrides_for_identity(parsed, package.identity)


def validate_typed_overrides(
    package: ModelPackage,
    overrides: Mapping[str, Any] | None,
    *,
    preset: str | None = None,
) -> ParsedOverrides:
    validated = apply_runtime_defaults(
        package,
        lambda spec: spec.validate_typed_overrides(overrides, preset=preset),
    )
    return parsed_overrides_for_identity(validated, package.identity)


def validated_overrides_for_materialization(
    package: ModelPackage,
    overrides: Mapping[str, Any] | ParsedOverrides,
    preset: str,
) -> ParsedOverrides:
    if not isinstance(overrides, ParsedOverrides):
        return parse_overrides(package, overrides, preset=preset)
    validated_identity = parsed_overrides_identity(overrides)
    if validated_identity is not None and validated_identity != package.identity:
        raise InspectionError(
            f"Parsed overrides were validated for model "
            f"'{validated_identity.catalog_key}', not selected model "
            f"'{package.catalog_key}'."
        )
    return validate_typed_overrides(package, overrides.values, preset=preset)


def canonicalize_overrides(
    package: ModelPackage,
    overrides: Mapping[str, Any] | None,
    *,
    ignore_unknown: bool = False,
) -> dict[str, Any]:
    if not overrides:
        return {}
    return apply_runtime_defaults(
        package,
        lambda spec: spec.canonicalize_overrides(
            overrides,
            ignore_unknown=ignore_unknown,
        ),
    )


def serialize_overrides(
    package: ModelPackage,
    overrides: Mapping[str, Any] | None,
    *,
    ignore_unknown: bool = False,
) -> dict[str, Any]:
    return apply_runtime_defaults(
        package,
        lambda spec: spec.serialize_overrides(
            overrides,
            ignore_unknown=ignore_unknown,
        ),
    )


__all__ = [
    "canonicalize_overrides",
    "parse_overrides",
    "reject_conflicting_locked_overrides",
    "reject_locked_overrides",
    "resolve_override_key",
    "serialize_overrides",
    "supported_config_keys",
]
