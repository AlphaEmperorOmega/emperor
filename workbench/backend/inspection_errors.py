"""Explicit mapping from Emperor Inspection failures to Workbench failures."""

from __future__ import annotations

from collections.abc import Callable
from typing import ParamSpec, TypeVar

from emperor.inspection import InspectionError
from emperor.model_packages import ModelPackage

from workbench.backend.inspector.errors import InspectorError

P = ParamSpec("P")
T = TypeVar("T")


def call_inspection(call: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    try:
        return call(*args, **kwargs)
    except InspectionError as exc:
        raise InspectorError(str(exc)) from exc


def call_model_package(
    package: ModelPackage,
    call: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Map selected-package metadata failures at the Workbench boundary."""

    try:
        return call(*args, **kwargs)
    except InspectionError as exc:
        raise InspectorError(str(exc)) from exc
    except ValueError as exc:
        raise InspectorError(str(exc)) from exc
    except Exception as exc:
        raise InspectorError(
            f"Failed to import model package '{package.catalog_key}': {exc}"
        ) from exc


__all__ = ["call_inspection", "call_model_package"]
