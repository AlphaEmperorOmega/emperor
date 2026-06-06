"""Shared API schema base classes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

ConfigValue = bool | int | float | str | None


class ApiResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
