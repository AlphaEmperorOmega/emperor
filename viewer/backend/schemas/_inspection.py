"""Model inspection schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from viewer.backend.schemas._base import ApiResponseModel


class GraphConfigFieldResponse(ApiResponseModel):
    key: str
    value: Any


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
    details: dict[str, Any]
    config: GraphConfigResponse | None


class GraphEdgeResponse(ApiResponseModel):
    id: str
    source: str
    target: str


class InspectRequest(ApiResponseModel):
    model: str
    preset: str
    overrides: dict[str, Any] = Field(default_factory=dict)
    dataset: str | None = None


class InspectResponse(ApiResponseModel):
    model: str
    preset: str
    parameterCount: int
    nodes: list[GraphNodeResponse]
    edges: list[GraphEdgeResponse]
