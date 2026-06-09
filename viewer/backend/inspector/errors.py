from __future__ import annotations

from viewer.backend.core.errors import ApiError


class InspectorError(ApiError):
    """Raised when a model cannot be inspected from user-facing input."""
