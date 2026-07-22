from __future__ import annotations


class InspectionError(Exception):
    """A selected Model Package cannot satisfy an Inspection request."""


def model_package_failure(model_id: str, exc: Exception) -> InspectionError:
    return InspectionError(f"Failed to import model package '{model_id}': {exc}")


__all__ = ["InspectionError", "model_package_failure"]
