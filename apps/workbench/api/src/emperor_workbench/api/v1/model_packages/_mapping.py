from __future__ import annotations

from model_runtime.inspection import ConfigurationSchema, SearchSpace

from emperor_workbench.api.v1.model_packages._contracts import (
    ConfigFieldResponse,
    ConfigSchemaResponse,
    DatasetGroupResponse,
    DatasetResponse,
    DatasetsResponse,
    ModelIdentityResponse,
    ModelsResponse,
    MonitorOptionResponse,
    MonitorsResponse,
    PresetResponse,
    PresetsResponse,
    SearchAxisResponse,
    SearchSpaceResponse,
)
from emperor_workbench.model_packages import (
    ModelMetadata,
    ModelPackageIdentity,
)


def models_response(
    identities: tuple[ModelPackageIdentity, ...],
) -> ModelsResponse:
    return ModelsResponse(
        models=[
            ModelIdentityResponse(
                modelType=identity.model_type,
                model=identity.model,
            )
            for identity in identities
        ]
    )


def presets_response(
    identity: ModelPackageIdentity,
    metadata: ModelMetadata,
) -> PresetsResponse:
    return PresetsResponse(
        modelType=identity.model_type,
        model=identity.model,
        presets=[
            PresetResponse(
                name=preset.name,
                label=preset.label,
                description=preset.description,
            )
            for preset in metadata.presets
        ],
    )


def datasets_response(
    identity: ModelPackageIdentity,
    metadata: ModelMetadata,
) -> DatasetsResponse:
    return DatasetsResponse(
        modelType=identity.model_type,
        model=identity.model,
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
    )


def monitors_response(
    identity: ModelPackageIdentity,
    metadata: ModelMetadata,
) -> MonitorsResponse:
    return MonitorsResponse(
        modelType=identity.model_type,
        model=identity.model,
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
    )


def config_schema_response(schema: ConfigurationSchema) -> ConfigSchemaResponse:
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
    )


def search_space_response(search_space: SearchSpace) -> SearchSpaceResponse:
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
    )


__all__ = [
    "config_schema_response",
    "datasets_response",
    "models_response",
    "monitors_response",
    "presets_response",
    "search_space_response",
]
