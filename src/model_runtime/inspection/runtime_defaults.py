from __future__ import annotations

from collections.abc import Callable
from typing import Never, TypeVar, cast

from model_runtime.inspection.errors import InspectionError
from model_runtime.packages import (
    ModelPackage,
    RuntimeDefaultsError,
    RuntimeDefaultsSpec,
)

_RuntimeDefaultsResult = TypeVar("_RuntimeDefaultsResult")


def raise_runtime_defaults_inspection_error(exc: RuntimeDefaultsError) -> Never:
    """Translate a package Runtime Defaults failure at the Inspection edge."""
    raise InspectionError(str(exc)) from (exc.__cause__ or exc)


def runtime_defaults_spec(package: ModelPackage) -> RuntimeDefaultsSpec:
    if not isinstance(cast(object, package), ModelPackage):
        raise TypeError("Runtime Defaults require a selected ModelPackage.")
    try:
        return package.runtime_defaults_spec
    except RuntimeDefaultsError as exc:
        raise_runtime_defaults_inspection_error(exc)


def apply_runtime_defaults(
    package: ModelPackage,
    operation: Callable[[RuntimeDefaultsSpec], _RuntimeDefaultsResult],
) -> _RuntimeDefaultsResult:
    """Apply one eager Inspection operation through selected Runtime Defaults."""
    spec = runtime_defaults_spec(package)
    try:
        return operation(spec)
    except RuntimeDefaultsError as exc:
        raise_runtime_defaults_inspection_error(exc)


__all__ = [
    "RuntimeDefaultsSpec",
    "apply_runtime_defaults",
    "raise_runtime_defaults_inspection_error",
    "runtime_defaults_spec",
]
