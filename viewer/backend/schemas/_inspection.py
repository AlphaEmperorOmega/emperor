"""Model inspection schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from viewer.backend.schemas._base import (
    ApiResponseModel,
    ConfigOverrides,
    JsonObject,
    JsonValue,
)


class GraphConfigFieldResponse(ApiResponseModel):
    key: str
    value: JsonValue


class GraphConfigResponse(ApiResponseModel):
    typeName: str
    fields: list[GraphConfigFieldResponse]


class GraphNodeResponse(ApiResponseModel):
    id: str
    label: str
    typeName: str
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
    modelType: str
    model: str
    preset: str
    overrides: ConfigOverrides = Field(default_factory=dict)
    dataset: str | None = None


class InspectResponse(ApiResponseModel):
    modelType: str
    model: str
    preset: str
    parameterCount: int
    parameterSizeBytes: int
    nodes: list[GraphNodeResponse]
    edges: list[GraphEdgeResponse]


class OperationGraphNodeResponse(ApiResponseModel):
    id: str
    label: str
    opKind: Literal[
        "placeholder",
        "call_function",
        "call_module",
        "call_method",
        "get_attr",
        "output",
    ]
    target: str
    modulePath: str | None = None
    groupId: str | None = None
    details: JsonObject = Field(default_factory=dict)


class OperationGraphEdgeResponse(ApiResponseModel):
    id: str
    source: str
    target: str


class OperationGraphResponse(ApiResponseModel):
    modelType: str
    model: str
    preset: str
    source: Literal["torch-export"]
    status: Literal["ok", "unsupported"]
    nodes: list[OperationGraphNodeResponse]
    edges: list[OperationGraphEdgeResponse]
    warnings: list[str] = Field(default_factory=list)
