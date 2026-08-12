from __future__ import annotations

from typing import cast

from model_runtime.inspection.errors import InspectionError
from model_runtime.packages import (
    ModelPackage,
    RuntimeDefaultsError,
    RuntimeDefaultsSpec,
)


def _inspection_error(exc: RuntimeDefaultsError) -> InspectionError:
    return InspectionError(str(exc))


def _runtime_defaults_error_cause(exc: RuntimeDefaultsError) -> BaseException:
    return exc.__cause__ or exc


def runtime_defaults_spec(package: ModelPackage) -> RuntimeDefaultsSpec:
    if not isinstance(cast(object, package), ModelPackage):
        raise TypeError("Runtime Defaults require a selected ModelPackage.")
    try:
        return package.runtime_defaults_spec
    except RuntimeDefaultsError as exc:
        raise _inspection_error(exc) from _runtime_defaults_error_cause(exc)


__all__ = ["RuntimeDefaultsSpec", "runtime_defaults_spec"]
