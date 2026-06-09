"""Model discovery and configuration-schema endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from viewer.backend.core.security import require_bearer_auth
from viewer.backend.dependencies import get_model_catalog_service
from viewer.backend.schemas import (
    ConfigSchemaResponse,
    DatasetsResponse,
    ModelsResponse,
    MonitorsResponse,
    PresetsResponse,
    SearchSpaceResponse,
)
from viewer.backend.services.models import ModelCatalogService

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
    return ModelsResponse(models=service.list_models())


@router.get(
    "/{model:path}/presets",
    response_model=PresetsResponse,
    summary="List model presets",
    response_description="Available presets for the selected model.",
)
async def presets(
    model: str,
    service: Annotated[ModelCatalogService, Depends(get_model_catalog_service)],
) -> PresetsResponse:
    return PresetsResponse(model=model, presets=service.list_presets(model))


@router.get(
    "/{model:path}/datasets",
    response_model=DatasetsResponse,
    summary="List model datasets",
    response_description="Supported datasets for the selected model.",
)
async def datasets(
    model: str,
    service: Annotated[ModelCatalogService, Depends(get_model_catalog_service)],
) -> DatasetsResponse:
    return DatasetsResponse(model=model, datasets=service.list_datasets(model))


@router.get(
    "/{model:path}/monitors",
    response_model=MonitorsResponse,
    summary="List model monitors",
    response_description="Monitor options supported by the selected model.",
)
async def monitors(
    model: str,
    service: Annotated[ModelCatalogService, Depends(get_model_catalog_service)],
) -> MonitorsResponse:
    return MonitorsResponse(model=model, monitors=service.list_monitors(model))


@router.get(
    "/{model:path}/config-schema",
    response_model=ConfigSchemaResponse,
    summary="Get model config schema",
    response_description="Config fields for the selected model and optional preset.",
)
async def schema(
    model: str,
    service: Annotated[ModelCatalogService, Depends(get_model_catalog_service)],
    preset: str | None = None,
) -> ConfigSchemaResponse:
    return ConfigSchemaResponse.model_validate(service.config_schema(model, preset))


@router.get(
    "/{model:path}/search-space",
    response_model=SearchSpaceResponse,
    summary="Get model search space",
    response_description="Search axes for the selected model and optional preset.",
)
async def search_space(
    model: str,
    service: Annotated[ModelCatalogService, Depends(get_model_catalog_service)],
    preset: str | None = None,
) -> SearchSpaceResponse:
    return SearchSpaceResponse.model_validate(
        service.search_space_schema(model, preset)
    )
