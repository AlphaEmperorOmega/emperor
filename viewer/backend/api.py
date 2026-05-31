from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from viewer.backend.inspector.discovery import discover_models, list_model_presets
from viewer.backend.inspector.errors import InspectorError
from viewer.backend.inspector.schema import config_schema
from viewer.backend.inspector.service import inspect_model

LOCAL_FRONTEND_ORIGINS = [
    "http://localhost:9000",
    "http://127.0.0.1:9000",
]


class InspectRequest(BaseModel):
    model: str
    preset: str
    overrides: dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="Emperor Model Viewer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=LOCAL_FRONTEND_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
async def models() -> dict[str, list[str]]:
    return {"models": discover_models()}


@app.get("/models/{model}/presets")
async def presets(model: str) -> dict[str, Any]:
    try:
        return {"model": model, "presets": list_model_presets(model)}
    except InspectorError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/models/{model}/config-schema")
async def schema(model: str) -> dict[str, Any]:
    try:
        return config_schema(model)
    except InspectorError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/inspect")
async def inspect(request: InspectRequest) -> dict[str, Any]:
    try:
        return inspect_model(
            request.model,
            request.preset,
            request.overrides,
        )
    except InspectorError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
