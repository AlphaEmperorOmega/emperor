from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from model_runtime.inspection import ConfigurationSchema, SearchSpace

from emperor_workbench.api._blocking import run_blocking_io
from emperor_workbench.api._dependencies import get_project_adapter_client
from emperor_workbench.api._security import require_bearer_auth
from emperor_workbench.api.v1.model_packages._contracts import (
    ConfigSchemaResponse,
    DatasetsResponse,
    ModelsResponse,
    MonitorsResponse,
    PresetsResponse,
    SearchSpaceResponse,
)
from emperor_workbench.api.v1.model_packages._mapping import (
    config_schema_response,
    datasets_response,
    models_response,
    monitors_response,
    presets_response,
    search_space_response,
)
from emperor_workbench.model_packages import (
    ModelMetadata,
    ModelPackageCatalog,
    ModelPackageIdentity,
)
from emperor_workbench.project_adapter import ProjectAdapterClient

router = APIRouter(
    prefix="/models",
    tags=["models"],
    dependencies=[Depends(require_bearer_auth)],
)


async def _catalog(
    project_adapter: Annotated[
        ProjectAdapterClient,
        Depends(get_project_adapter_client),
    ],
) -> ModelPackageCatalog:
    return ModelPackageCatalog(project_adapter)


def _list_models(
    catalog: ModelPackageCatalog,
) -> tuple[ModelPackageIdentity, ...]:
    return catalog.identities()


def _list_metadata(
    catalog: ModelPackageCatalog,
    model_type: str,
    model: str,
) -> tuple[ModelPackageIdentity, ModelMetadata]:
    selected = catalog.select_parts(model_type, model)
    return selected.identity, selected.metadata()


def _config_schema(
    catalog: ModelPackageCatalog,
    model_type: str,
    model: str,
    preset: str | None,
) -> ConfigurationSchema:
    return catalog.select_parts(model_type, model).configuration(preset)


def _search_space(
    catalog: ModelPackageCatalog,
    model_type: str,
    model: str,
    preset: str | None,
    presets: list[str] | None,
) -> SearchSpace:
    return catalog.select_parts(model_type, model).search_space(preset, presets)


@router.get(
    "",
    response_model=ModelsResponse,
    summary="List models",
    response_description="Available model package names.",
)
async def models(
    catalog: Annotated[ModelPackageCatalog, Depends(_catalog)],
) -> ModelsResponse:
    return models_response(await run_blocking_io(_list_models, catalog))


@router.get(
    "/{modelType}/{model}/presets",
    response_model=PresetsResponse,
    summary="List model presets",
    response_description="Available presets for the selected model.",
)
async def presets(
    modelType: str,
    model: str,
    catalog: Annotated[ModelPackageCatalog, Depends(_catalog)],
) -> PresetsResponse:
    identity, metadata = await run_blocking_io(
        _list_metadata,
        catalog,
        modelType,
        model,
    )
    return presets_response(identity, metadata)


@router.get(
    "/{modelType}/{model}/datasets",
    response_model=DatasetsResponse,
    summary="List model datasets",
    response_description="Supported datasets for the selected model.",
)
async def datasets(
    modelType: str,
    model: str,
    catalog: Annotated[ModelPackageCatalog, Depends(_catalog)],
) -> DatasetsResponse:
    identity, metadata = await run_blocking_io(
        _list_metadata,
        catalog,
        modelType,
        model,
    )
    return datasets_response(identity, metadata)


@router.get(
    "/{modelType}/{model}/monitors",
    response_model=MonitorsResponse,
    summary="List model monitors",
    response_description="Monitor options supported by the selected model.",
)
async def monitors(
    modelType: str,
    model: str,
    catalog: Annotated[ModelPackageCatalog, Depends(_catalog)],
) -> MonitorsResponse:
    identity, metadata = await run_blocking_io(
        _list_metadata,
        catalog,
        modelType,
        model,
    )
    return monitors_response(identity, metadata)


@router.get(
    "/{modelType}/{model}/config-schema",
    response_model=ConfigSchemaResponse,
    summary="Get model config schema",
    response_description="Config fields for the selected model and optional preset.",
)
async def schema(
    modelType: str,
    model: str,
    catalog: Annotated[ModelPackageCatalog, Depends(_catalog)],
    preset: str | None = None,
) -> ConfigSchemaResponse:
    configuration = await run_blocking_io(
        _config_schema,
        catalog,
        modelType,
        model,
        preset,
    )
    return config_schema_response(configuration)


@router.get(
    "/{modelType}/{model}/search-space",
    response_model=SearchSpaceResponse,
    summary="Get model search space",
    response_description="Search axes for the selected model and optional preset.",
)
async def search_space(
    modelType: str,
    model: str,
    catalog: Annotated[ModelPackageCatalog, Depends(_catalog)],
    preset: str | None = None,
    presets: str | None = None,
) -> SearchSpaceResponse:
    selected_presets = (
        [item.strip() for item in presets.split(",") if item.strip()]
        if presets
        else None
    )
    semantic_search = await run_blocking_io(
        _search_space,
        catalog,
        modelType,
        model,
        preset,
        selected_presets,
    )
    return search_space_response(semantic_search)


__all__ = ["router"]
