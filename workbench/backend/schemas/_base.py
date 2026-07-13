"""Shared API schema base classes."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StringConstraints
from typing_extensions import TypeAliasType

JsonValue = TypeAliasType(
    "JsonValue",
    bool | int | FiniteFloat | str | None | list["JsonValue"] | dict[str, "JsonValue"],
)
JsonObject = TypeAliasType("JsonObject", dict[str, JsonValue])

BoundedConfigString = Annotated[str, StringConstraints(max_length=20_000)]
BoundedIdentifier = Annotated[str, StringConstraints(max_length=256)]
BoundedRequestString = Annotated[str, StringConstraints(max_length=20_000)]
ConfigKey = Annotated[str, StringConstraints(max_length=256)]
ConfigValue = bool | int | FiniteFloat | BoundedConfigString | None
ConfigOverrides = Annotated[dict[ConfigKey, ConfigValue], Field(max_length=512)]


class ApiResponseModel(BaseModel):
    # Keep FastAPI's native Pydantic-core JSON-byte path while preserving strict JSON.
    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="null")
