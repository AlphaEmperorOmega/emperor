from __future__ import annotations

from typing import Literal

from pydantic import Field

from workbench.backend.schemas._base import (
    ApiResponseModel,
    BoundedIdentifier,
    ConfigOverrides,
    JsonObject,
    JsonValue,
)


class GraphConfigFieldResponse(ApiResponseModel):
    key: str
    value: JsonValue
    description: str | None = None


class GraphConfigResponse(ApiResponseModel):
    typeName: str
    fields: list[GraphConfigFieldResponse]


class GraphNodeResponse(ApiResponseModel):
    id: str
    label: str
    typeName: str
    description: str | None = None
    path: str
    graphRole: Literal["architecture", "internal", "runtime"]
    parameterCount: int
    parameterSizeBytes: int
    details: JsonObject
    config: GraphConfigResponse | None


class GraphEdgeResponse(ApiResponseModel):
    id: str
    source: str
    target: str


class InspectRequest(ApiResponseModel):
    modelType: BoundedIdentifier
    model: BoundedIdentifier
    preset: BoundedIdentifier
    overrides: ConfigOverrides = Field(default_factory=dict)
    experimentTask: BoundedIdentifier | None = None
    dataset: BoundedIdentifier | None = None
    logRunId: BoundedIdentifier | None = None


class InspectResponse(ApiResponseModel):
    modelType: str
    model: str
    preset: str
    parameterCount: int
    parameterSizeBytes: int
    nodes: list[GraphNodeResponse]
    edges: list[GraphEdgeResponse]
