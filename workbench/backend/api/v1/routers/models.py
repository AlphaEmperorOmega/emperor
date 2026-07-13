from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from workbench.backend.blocking import run_blocking_io
from workbench.backend.core.security import require_bearer_auth
from workbench.backend.schemas import (
    ConfigSchemaResponse,
    DatasetsResponse,
    ModelsResponse,
    MonitorsResponse,
    PresetsResponse,
    SearchSpaceResponse,
)

router = APIRouter(
    prefix="/models",
    tags=["models"],
    dependencies=[Depends(require_bearer_auth)],
)


def _list_models() -> list[dict[str, str]]:
    from workbench.backend.inspection_adapter import WorkbenchInspectionAdapter

    return WorkbenchInspectionAdapter.catalog_payload()


def _list_presets(model_type: str, model: str) -> list[dict[str, Any]]:
    from workbench.backend.inspection_adapter import WorkbenchInspectionAdapter

    return WorkbenchInspectionAdapter.select_parts(model_type, model).presets_payload()


def _list_datasets(model_type: str, model: str) -> dict[str, Any]:
    from workbench.backend.inspection_adapter import WorkbenchInspectionAdapter

    return WorkbenchInspectionAdapter.select_parts(model_type, model).datasets_payload()


def _list_monitors(model_type: str, model: str) -> list[dict[str, Any]]:
    from workbench.backend.inspection_adapter import WorkbenchInspectionAdapter

    return WorkbenchInspectionAdapter.select_parts(model_type, model).monitors_payload()


def _config_schema(
    model_type: str,
    model: str,
    preset: str | None,
) -> dict[str, Any]:
    from workbench.backend.inspection_adapter import WorkbenchInspectionAdapter

    return WorkbenchInspectionAdapter.select_parts(
        model_type,
        model,
    ).configuration_payload(preset)


def _search_space(
    model_type: str,
    model: str,
    preset: str | None,
    presets: list[str] | None,
) -> dict[str, Any]:
    from workbench.backend.inspection_adapter import WorkbenchInspectionAdapter

    return WorkbenchInspectionAdapter.select_parts(
        model_type,
        model,
    ).search_space_payload(
        preset,
        presets,
    )


@router.get(
    "",
    response_model=ModelsResponse,
    summary="List models",
    response_description="Available model package names.",
)
async def models() -> ModelsResponse:
    return ModelsResponse(models=await run_blocking_io(_list_models))


@router.get(
    "/{modelType}/{model}/presets",
    response_model=PresetsResponse,
    summary="List model presets",
    response_description="Available presets for the selected model.",
)
async def presets(
    modelType: str,
    model: str,
) -> PresetsResponse:
    return PresetsResponse(
        modelType=modelType,
        model=model,
        presets=await run_blocking_io(_list_presets, modelType, model),
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
) -> DatasetsResponse:
    dataset_payload = await run_blocking_io(
        _list_datasets,
        modelType,
        model,
    )
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
) -> MonitorsResponse:
    return MonitorsResponse(
        modelType=modelType,
        model=model,
        monitors=await run_blocking_io(
            _list_monitors,
            modelType,
            model,
        ),
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
    preset: str | None = None,
) -> ConfigSchemaResponse:
    return ConfigSchemaResponse.model_validate(
        await run_blocking_io(
            _config_schema,
            modelType,
            model,
            preset,
        )
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
            _search_space,
            modelType,
            model,
            preset,
            selected_presets,
        )
    )
