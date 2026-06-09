"""Shared API schema base classes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypeAliasType

JsonValue = TypeAliasType(
    "JsonValue",
    bool | int | float | str | None | list["JsonValue"] | dict[str, "JsonValue"],
)
JsonObject = TypeAliasType("JsonObject", dict[str, JsonValue])

ConfigValue = bool | int | float | str | None
ConfigOverrides = dict[str, ConfigValue]


class ApiResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
