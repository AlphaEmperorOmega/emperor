"""Model discovery and configuration-schema endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from workbench.backend.blocking import run_blocking_io
from workbench.backend.core.security import require_bearer_auth
from workbench.backend.dependencies import get_model_catalog_service
from workbench.backend.schemas import (
    ConfigSchemaResponse,
    DatasetsResponse,
    ModelsResponse,
    MonitorsResponse,
    PresetsResponse,
    SearchSpaceResponse,
)
from workbench.backend.services.models import ModelCatalogService

router = APIRouter(
    prefix="/models",
    tags=["models"],
    dependencies=[Depends(require_bearer_auth)],
)


@router.get(
    "",
    response_model=ModelsResponse,
    summary="List models",
    response_description="Available model package names.",
)
async def models(
    service: Annotated[ModelCatalogService, Depends(get_model_catalog_service)],
) -> ModelsResponse:
    return ModelsResponse(models=await run_blocking_io(service.list_models))


@router.get(
    "/{modelType}/{model}/presets",
    response_model=PresetsResponse,
    summary="List model presets",
    response_description="Available presets for the selected model.",
)
async def presets(
    modelType: str,
    model: str,
    service: Annotated[ModelCatalogService, Depends(get_model_catalog_service)],
) -> PresetsResponse:
    return PresetsResponse(
        modelType=modelType,
        model=model,
        presets=await run_blocking_io(service.list_presets, modelType, model),
    )


@router.get(
    "/{modelType}/{model}/datasets",
    response_model=DatasetsResponse,
    summary="List model datasets",
    response_description="Supported datasets for the selected model.",
)
async def datasets(
    modelType: str,
    model: str,
    service: Annotated[ModelCatalogService, Depends(get_model_catalog_service)],
) -> DatasetsResponse:
    dataset_payload = await run_blocking_io(service.list_datasets, modelType, model)
    return DatasetsResponse(
        modelType=modelType,
        model=model,
        **dataset_payload,
    )


@router.get(
    "/{modelType}/{model}/monitors",
    response_model=MonitorsResponse,
    summary="List model monitors",
    response_description="Monitor options supported by the selected model.",
)
async def monitors(
    modelType: str,
    model: str,
    service: Annotated[ModelCatalogService, Depends(get_model_catalog_service)],
) -> MonitorsResponse:
    return MonitorsResponse(
        modelType=modelType,
        model=model,
        monitors=await run_blocking_io(service.list_monitors, modelType, model),
    )


@router.get(
    "/{modelType}/{model}/config-schema",
    response_model=ConfigSchemaResponse,
    summary="Get model config schema",
    response_description="Config fields for the selected model and optional preset.",
)
async def schema(
    modelType: str,
    model: str,
    service: Annotated[ModelCatalogService, Depends(get_model_catalog_service)],
    preset: str | None = None,
) -> ConfigSchemaResponse:
    return ConfigSchemaResponse.model_validate(
        await run_blocking_io(service.config_schema, modelType, model, preset)
    )


@router.get(
    "/{modelType}/{model}/search-space",
    response_model=SearchSpaceResponse,
    summary="Get model search space",
    response_description="Search axes for the selected model and optional preset.",
)
async def search_space(
    modelType: str,
    model: str,
    service: Annotated[ModelCatalogService, Depends(get_model_catalog_service)],
    preset: str | None = None,
    presets: str | None = None,
) -> SearchSpaceResponse:
    selected_presets = (
        [item.strip() for item in presets.split(",") if item.strip()]
        if presets
        else None
    )
    return SearchSpaceResponse.model_validate(
        await run_blocking_io(
            service.search_space_schema,
            modelType,
            model,
            preset,
            selected_presets,
        )
    )
