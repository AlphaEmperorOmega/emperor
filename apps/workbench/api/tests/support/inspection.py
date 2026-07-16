from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from model_runtime.inspection import (
    GraphConfiguration,
    GraphEdge,
    GraphNode,
    InspectionRequest,
    InspectionResult,
    ParsedOverrides,
    inspect_model_graph,
)
from model_runtime.packages import is_safe_model_identity

from emperor_workbench.api.v1.inspection import (
    GraphConfigFieldResponse,
    GraphConfigResponse,
    GraphEdgeResponse,
    GraphNodeResponse,
    InspectResponse,
)
from emperor_workbench.api.v1.model_packages import (
    ConfigFieldResponse,
    ConfigSchemaResponse,
    DatasetGroupResponse,
    DatasetResponse,
    DatasetsResponse,
    MonitorOptionResponse,
    MonitorsResponse,
    PresetResponse,
    PresetsResponse,
    SearchAxisResponse,
    SearchSpaceResponse,
)
from emperor_workbench.inspection import (
    InProcessInspectionExecutor,
    InspectionFailure,
    InspectionService,
)
from emperor_workbench.model_packages import (
    ModelPackageCatalog,
    ModelPackageFailure,
    SelectedModelPackage,
)
from tests.support.model_packages import project_adapter_client


def _camel_case(key: str) -> str:
    pieces = key.split("_")
    return pieces[0] + "".join(piece[:1].upper() + piece[1:] for piece in pieces[1:])


def _http_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {_camel_case(str(key)): _http_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_http_value(item) for item in value]
    return value


def _graph_configuration_response(
    configuration: GraphConfiguration | None,
) -> GraphConfigResponse | None:
    if configuration is None:
        return None
    return GraphConfigResponse(
        typeName=configuration.type_name,
        fields=[
            GraphConfigFieldResponse(
                key=field.key,
                value=field.value,
                description=field.description,
            )
            for field in configuration.fields
        ],
    )


def _graph_node_response(node: GraphNode) -> GraphNodeResponse:
    return GraphNodeResponse(
        id=node.id,
        label=node.type_name,
        typeName=node.type_name,
        description=node.description,
        path=node.path,
        graphRole=node.graph_role,
        parameterCount=node.parameter_count,
        parameterSizeBytes=node.parameter_size_bytes,
        details=_http_value(node.details),
        config=_graph_configuration_response(node.configuration),
    )


def _graph_edge_response(edge: GraphEdge) -> GraphEdgeResponse:
    return GraphEdgeResponse(
        id=edge.id,
        source=edge.source,
        target=edge.target,
    )


def _graph_node_payload(node: GraphNode) -> dict[str, Any]:
    payload = _graph_node_response(node).model_dump(mode="json")
    if node.description is None:
        payload.pop("description")
    configuration = payload["config"]
    if configuration is not None and node.configuration is not None:
        for field_payload, field in zip(
            configuration["fields"],
            node.configuration.fields,
            strict=True,
        ):
            if field.description is None:
                field_payload.pop("description")
    return payload


def inspection_response(result: InspectionResult) -> InspectResponse:
    return InspectResponse(
        **result.identity.to_payload(),
        preset=result.preset,
        parameterCount=result.parameter_count,
        parameterSizeBytes=result.parameter_size_bytes,
        nodes=[_graph_node_response(node) for node in result.nodes],
        edges=[_graph_edge_response(edge) for edge in result.edges],
    )


def discover_models() -> list[str]:
    return [
        identity.catalog_key
        for identity in ModelPackageCatalog(project_adapter_client()).identities()
    ]


def _selected(model_name: str) -> SelectedModelPackage:
    parts = model_name.split("/")
    if len(parts) != 2 or not is_safe_model_identity(parts[0], parts[1]):
        raise InspectionFailure(f"Invalid model name: {model_name!r}")
    try:
        return ModelPackageCatalog(project_adapter_client()).select(model_name)
    except ModelPackageFailure as exc:
        raise InspectionFailure(exc.detail, kind=exc.kind) from exc


def config_schema(model_name: str, preset_name: str | None = None) -> dict[str, Any]:
    schema = _selected(model_name).configuration(preset_name)
    return ConfigSchemaResponse(
        modelType=schema.identity.model_type,
        model=schema.identity.model,
        fields=[
            ConfigFieldResponse(
                key=field.key,
                configKey=field.key,
                flag=field.flag,
                label=field.key.lower().replace("_", " "),
                section=field.section_path[-1],
                sectionPath=list(field.section_path),
                description=field.description,
                type=field.value_type,
                default=field.default,
                nullable=field.nullable,
                choices=list(field.choices),
                maximum=field.maximum,
                locked=field.locked,
                lockedValue=field.locked_value,
                lockedReason=field.locked_reason,
            )
            for field in schema.fields
        ],
    ).model_dump()


def search_space_schema(
    model_name: str,
    preset_name: str | None = None,
    preset_names: list[str] | None = None,
) -> dict[str, Any]:
    search_space = _selected(model_name).search_space(
        preset_name,
        preset_names,
    )
    return SearchSpaceResponse(
        modelType=search_space.identity.model_type,
        model=search_space.identity.model,
        preset=search_space.preset,
        axes=[
            SearchAxisResponse(
                key=axis.key,
                configKey=axis.key,
                searchKey=axis.search_key,
                label=axis.key.lower().replace("_", " "),
                section=axis.section,
                type=axis.value_type,
                values=list(axis.values),
                locked=axis.locked,
                lockedValue=axis.locked_value,
                lockedReason=axis.locked_reason,
                lockedByPresets=list(axis.locked_by_presets),
                lockReasons=list(axis.lock_reasons),
            )
            for axis in search_space.axes
        ],
    ).model_dump()


def list_model_presets(model_name: str) -> list[dict[str, str]]:
    selected = _selected(model_name)
    metadata = selected.metadata()
    return PresetsResponse(
        modelType=selected.identity.model_type,
        model=selected.identity.model,
        presets=[
            PresetResponse(
                name=preset.name,
                label=preset.label,
                description=preset.description,
            )
            for preset in metadata.presets
        ],
    ).model_dump()["presets"]


def list_model_datasets(model_name: str) -> dict[str, Any]:
    selected = _selected(model_name)
    metadata = selected.metadata()
    payload = DatasetsResponse(
        modelType=selected.identity.model_type,
        model=selected.identity.model,
        defaultExperimentTask=metadata.default_experiment_task,
        datasetGroups=[
            DatasetGroupResponse(
                experimentTask=group.experiment_task,
                label=group.label,
                datasets=[
                    DatasetResponse(
                        name=dataset.name,
                        label=dataset.label,
                        inputDim=dataset.input_dim,
                        outputDim=dataset.output_dim,
                    )
                    for dataset in group.datasets
                ],
            )
            for group in metadata.dataset_groups
        ],
    ).model_dump()
    return {
        "defaultExperimentTask": payload["defaultExperimentTask"],
        "datasetGroups": payload["datasetGroups"],
    }


def list_model_monitors(model_name: str) -> list[dict[str, Any]]:
    selected = _selected(model_name)
    metadata = selected.metadata()
    return MonitorsResponse(
        modelType=selected.identity.model_type,
        model=selected.identity.model,
        monitors=[
            MonitorOptionResponse(
                name=monitor.name,
                label=monitor.label,
                description=monitor.description,
                kinds=list(monitor.kinds),
                defaultEnabled=monitor.default_enabled,
            )
            for monitor in metadata.monitors
        ],
    ).model_dump()["monitors"]


def inspect_model(
    model_name: str,
    preset_name: str,
    overrides: Mapping[str, Any] | None = None,
    dataset: str | None = None,
    experiment_task: str | None = None,
    *,
    parsed_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = _selected(model_name)
    executor = InProcessInspectionExecutor()
    if parsed_overrides is not None:
        result = executor.inspect(
            selected,
            InspectionRequest(
                preset=preset_name,
                overrides=ParsedOverrides(parsed_overrides),
                dataset=dataset,
                experiment_task=experiment_task,
            ),
        )
    else:
        result = InspectionService(executor).inspect(
            selected,
            preset=preset_name,
            overrides=overrides or {},
            dataset=dataset,
            experiment_task=experiment_task,
        )
    return inspection_response(result).model_dump(mode="json")


def serialize_graph(
    module: object,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    graph = inspect_model_graph(module)
    return (
        [_graph_node_payload(node) for node in graph.nodes],
        [_graph_edge_response(edge).model_dump(mode="json") for edge in graph.edges],
    )


__all__ = [
    "config_schema",
    "discover_models",
    "inspect_model",
    "inspection_response",
    "list_model_datasets",
    "list_model_monitors",
    "list_model_presets",
    "search_space_schema",
    "serialize_graph",
]
