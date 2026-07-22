from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import NamedTuple, get_type_hints

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from fastapi.routing import APIRoute, RouteContext, iter_route_contexts

from emperor_workbench.api import _probes as probe_contracts
from emperor_workbench.api import app
from emperor_workbench.api.v1 import _base_contracts as base_contracts
from emperor_workbench.api.v1 import _capabilities as capability_contracts
from emperor_workbench.api.v1 import _monitoring_contracts as monitoring_contracts
from emperor_workbench.api.v1 import (
    config_snapshots as config_snapshot_contracts,
)
from emperor_workbench.api.v1 import inspection as inspection_contracts
from emperor_workbench.api.v1 import model_packages as model_package_contracts
from emperor_workbench.api.v1 import run_history as run_history_contracts
from emperor_workbench.api.v1 import run_plans as run_plan_contracts
from emperor_workbench.api.v1 import training_jobs as training_job_contracts
from emperor_workbench.api.v1.training_jobs import (
    _contracts as training_job_private_contracts,
)


class _ContractNamespace:
    def __getattr__(self, name: str):
        for contracts in (
            base_contracts,
            capability_contracts,
            config_snapshot_contracts,
            inspection_contracts,
            model_package_contracts,
            monitoring_contracts,
            probe_contracts,
            run_history_contracts,
            run_plan_contracts,
            training_job_contracts,
            training_job_private_contracts,
        ):
            if hasattr(contracts, name):
                return getattr(contracts, name)
        raise AttributeError(name)


schemas = _ContractNamespace()

RouteKey = tuple[tuple[str, ...], str]


class EndpointSchemaMapping(NamedTuple):
    backend_body_request_schemas: tuple[type[schemas.ApiResponseModel], ...]
    backend_response_schema: type[schemas.ApiResponseModel]
    frontend_api_function: str
    frontend_response_schema: str


EXPECTED_BUSINESS_ROUTES = [
    (("DELETE",), "/config-snapshots/{snapshot_id}"),
    (("DELETE",), "/logs/experiments/{experiment}"),
    (("GET",), "/capabilities"),
    (("GET",), "/config-snapshots"),
    (("GET",), "/config-snapshots/library"),
    (("GET",), "/health"),
    (("GET",), "/logs/experiments"),
    (("GET",), "/logs/runs"),
    (("GET",), "/logs/runs/{run_id}/artifacts"),
    (("GET",), "/logs/runs/{run_id}/monitor-data"),
    (("GET",), "/models"),
    (("GET",), "/models/{modelType}/{model}/config-schema"),
    (("GET",), "/models/{modelType}/{model}/datasets"),
    (("GET",), "/models/{modelType}/{model}/monitors"),
    (("GET",), "/models/{modelType}/{model}/presets"),
    (("GET",), "/models/{modelType}/{model}/search-space"),
    (("GET",), "/training/jobs/{job_id}"),
    (("GET",), "/training/jobs/{job_id}/events"),
    (("GET",), "/training/jobs/{job_id}/monitor-data"),
    (("GET",), "/training/jobs/{job_id}/monitor-parameter-status"),
    (("PATCH",), "/config-snapshots/{snapshot_id}"),
    (("POST",), "/config-snapshots"),
    (("POST",), "/inspect"),
    (("POST",), "/logs/checkpoints"),
    (("POST",), "/logs/import"),
    (("POST",), "/logs/media"),
    (("POST",), "/logs/parameter-status"),
    (("POST",), "/logs/runs/delete"),
    (("POST",), "/logs/runs/delete-plan"),
    (("POST",), "/logs/runs/preset-delete"),
    (("POST",), "/logs/runs/preset-delete-plan"),
    (("POST",), "/logs/scalars"),
    (("POST",), "/logs/tags"),
    (("POST",), "/training/jobs"),
    (("POST",), "/training/jobs/{job_id}/cancel"),
    (("POST",), "/training/jobs/{job_id}/reconcile"),
    (("POST",), "/training/run-plan"),
]

ENDPOINT_SCHEMA_MAPPINGS: dict[RouteKey, EndpointSchemaMapping] = {
    (("DELETE",), "/config-snapshots/{snapshot_id}"): EndpointSchemaMapping(
        backend_body_request_schemas=(),
        backend_response_schema=schemas.ConfigSnapshotsResponse,
        frontend_api_function="deleteConfigSnapshot",
        frontend_response_schema="configSnapshotsSchema",
    ),
    (("GET",), "/config-snapshots"): EndpointSchemaMapping(
        backend_body_request_schemas=(),
        backend_response_schema=schemas.ConfigSnapshotsResponse,
        frontend_api_function="fetchConfigSnapshots",
        frontend_response_schema="configSnapshotsSchema",
    ),
    (("GET",), "/config-snapshots/library"): EndpointSchemaMapping(
        backend_body_request_schemas=(),
        backend_response_schema=schemas.ConfigSnapshotLibraryResponse,
        frontend_api_function="fetchConfigSnapshotLibrary",
        frontend_response_schema="configSnapshotLibrarySchema",
    ),
    (("PATCH",), "/config-snapshots/{snapshot_id}"): EndpointSchemaMapping(
        backend_body_request_schemas=(schemas.ConfigSnapshotUpdateRequest,),
        backend_response_schema=schemas.ConfigSnapshotResponse,
        frontend_api_function="updateConfigSnapshot",
        frontend_response_schema="configSnapshotSchema",
    ),
    (("POST",), "/config-snapshots"): EndpointSchemaMapping(
        backend_body_request_schemas=(schemas.ConfigSnapshotCreateRequest,),
        backend_response_schema=schemas.ConfigSnapshotResponse,
        frontend_api_function="createConfigSnapshot",
        frontend_response_schema="configSnapshotSchema",
    ),
    (("DELETE",), "/logs/experiments/{experiment}"): EndpointSchemaMapping(
        backend_body_request_schemas=(),
        backend_response_schema=schemas.LogExperimentDeleteResponse,
        frontend_api_function="deleteLogExperiment",
        frontend_response_schema="logExperimentDeleteSchema",
    ),
    (("GET",), "/capabilities"): EndpointSchemaMapping(
        backend_body_request_schemas=(),
        backend_response_schema=schemas.CapabilitiesResponse,
        frontend_api_function="fetchCapabilities",
        frontend_response_schema="capabilitiesSchema",
    ),
    (("GET",), "/health"): EndpointSchemaMapping(
        backend_body_request_schemas=(),
        backend_response_schema=schemas.HealthResponse,
        frontend_api_function="fetchHealth",
        frontend_response_schema="healthSchema",
    ),
    (("GET",), "/logs/experiments"): EndpointSchemaMapping(
        backend_body_request_schemas=(),
        backend_response_schema=schemas.LogExperimentsResponse,
        frontend_api_function="fetchLogExperiments",
        frontend_response_schema="logExperimentsSchema",
    ),
    (("GET",), "/logs/runs"): EndpointSchemaMapping(
        backend_body_request_schemas=(),
        backend_response_schema=schemas.LogRunsResponse,
        frontend_api_function="fetchLogRuns",
        frontend_response_schema="logRunsSchema",
    ),
    (("GET",), "/logs/runs/{run_id}/artifacts"): EndpointSchemaMapping(
        backend_body_request_schemas=(),
        backend_response_schema=schemas.LogRunArtifactsResponse,
        frontend_api_function="fetchLogRunArtifacts",
        frontend_response_schema="logRunArtifactsSchema",
    ),
    (("GET",), "/logs/runs/{run_id}/monitor-data"): EndpointSchemaMapping(
        backend_body_request_schemas=(),
        backend_response_schema=schemas.MonitorDataResponse,
        frontend_api_function="fetchLogRunMonitorData",
        frontend_response_schema="monitorDataSchema",
    ),
    (("GET",), "/models"): EndpointSchemaMapping(
        backend_body_request_schemas=(),
        backend_response_schema=schemas.ModelsResponse,
        frontend_api_function="fetchModels",
        frontend_response_schema="modelsSchema",
    ),
    (("GET",), "/models/{modelType}/{model}/config-schema"): EndpointSchemaMapping(
        backend_body_request_schemas=(),
        backend_response_schema=schemas.ConfigSchemaResponse,
        frontend_api_function="fetchConfigSchema",
        frontend_response_schema="configSchema",
    ),
    (("GET",), "/models/{modelType}/{model}/datasets"): EndpointSchemaMapping(
        backend_body_request_schemas=(),
        backend_response_schema=schemas.DatasetsResponse,
        frontend_api_function="fetchDatasets",
        frontend_response_schema="datasetsSchema",
    ),
    (("GET",), "/models/{modelType}/{model}/monitors"): EndpointSchemaMapping(
        backend_body_request_schemas=(),
        backend_response_schema=schemas.MonitorsResponse,
        frontend_api_function="fetchMonitors",
        frontend_response_schema="monitorsSchema",
    ),
    (("GET",), "/models/{modelType}/{model}/presets"): EndpointSchemaMapping(
        backend_body_request_schemas=(),
        backend_response_schema=schemas.PresetsResponse,
        frontend_api_function="fetchPresets",
        frontend_response_schema="presetsSchema",
    ),
    (("GET",), "/models/{modelType}/{model}/search-space"): EndpointSchemaMapping(
        backend_body_request_schemas=(),
        backend_response_schema=schemas.SearchSpaceResponse,
        frontend_api_function="fetchSearchSpace",
        frontend_response_schema="searchSpaceSchema",
    ),
    (("GET",), "/training/jobs/{job_id}"): EndpointSchemaMapping(
        backend_body_request_schemas=(),
        backend_response_schema=schemas.TrainingJobResponse,
        frontend_api_function="fetchTrainingJob",
        frontend_response_schema="trainingJobSchema",
    ),
    (("GET",), "/training/jobs/{job_id}/events"): EndpointSchemaMapping(
        backend_body_request_schemas=(),
        backend_response_schema=schemas.TrainingProgressEventsResponse,
        frontend_api_function="fetchTrainingJobEvents",
        frontend_response_schema="trainingJobEventsSchema",
    ),
    (("GET",), "/training/jobs/{job_id}/monitor-data"): EndpointSchemaMapping(
        backend_body_request_schemas=(),
        backend_response_schema=schemas.MonitorDataResponse,
        frontend_api_function="fetchMonitorData",
        frontend_response_schema="monitorDataSchema",
    ),
    (
        ("GET",),
        "/training/jobs/{job_id}/monitor-parameter-status",
    ): EndpointSchemaMapping(
        backend_body_request_schemas=(),
        backend_response_schema=schemas.ParameterStatusResponse,
        frontend_api_function="fetchMonitorParameterStatus",
        frontend_response_schema="parameterStatusSchema",
    ),
    (("POST",), "/inspect"): EndpointSchemaMapping(
        backend_body_request_schemas=(schemas.InspectRequest,),
        backend_response_schema=schemas.InspectResponse,
        frontend_api_function="inspectModel",
        frontend_response_schema="inspectResponseSchema",
    ),
    (("POST",), "/logs/checkpoints"): EndpointSchemaMapping(
        backend_body_request_schemas=(schemas.LogCheckpointsRequest,),
        backend_response_schema=schemas.LogCheckpointsResponse,
        frontend_api_function="fetchLogCheckpoints",
        frontend_response_schema="logCheckpointsSchema",
    ),
    (("POST",), "/logs/import"): EndpointSchemaMapping(
        backend_body_request_schemas=(),
        backend_response_schema=schemas.LogArchiveImportResponse,
        frontend_api_function="importLogArchive",
        frontend_response_schema="logArchiveImportSchema",
    ),
    (("POST",), "/logs/media"): EndpointSchemaMapping(
        backend_body_request_schemas=(schemas.LogMediaRequest,),
        backend_response_schema=schemas.LogMediaResponse,
        frontend_api_function="fetchLogMedia",
        frontend_response_schema="logMediaSchema",
    ),
    (("POST",), "/logs/runs/delete"): EndpointSchemaMapping(
        backend_body_request_schemas=(schemas.LogRunDeleteFiltersRequest,),
        backend_response_schema=schemas.LogRunDeleteResponse,
        frontend_api_function="deleteLogRuns",
        frontend_response_schema="logRunDeleteSchema",
    ),
    (("POST",), "/logs/runs/delete-plan"): EndpointSchemaMapping(
        backend_body_request_schemas=(schemas.LogRunDeleteFiltersRequest,),
        backend_response_schema=schemas.LogRunDeletePlanResponse,
        frontend_api_function="createLogRunDeletePlan",
        frontend_response_schema="logRunDeletePlanSchema",
    ),
    (("POST",), "/logs/parameter-status"): EndpointSchemaMapping(
        backend_body_request_schemas=(schemas.LogParameterStatusRequest,),
        backend_response_schema=schemas.LogParameterStatusResponse,
        frontend_api_function="fetchLogParameterStatus",
        frontend_response_schema="logParameterStatusSchema",
    ),
    (("POST",), "/logs/runs/preset-delete"): EndpointSchemaMapping(
        backend_body_request_schemas=(schemas.LogPresetDeleteRequest,),
        backend_response_schema=schemas.LogRunDeleteResponse,
        frontend_api_function="deleteLogPreset",
        frontend_response_schema="logRunDeleteSchema",
    ),
    (("POST",), "/logs/runs/preset-delete-plan"): EndpointSchemaMapping(
        backend_body_request_schemas=(schemas.LogPresetDeleteRequest,),
        backend_response_schema=schemas.LogRunDeletePlanResponse,
        frontend_api_function="createLogPresetDeletePlan",
        frontend_response_schema="logRunDeletePlanSchema",
    ),
    (("POST",), "/logs/scalars"): EndpointSchemaMapping(
        backend_body_request_schemas=(schemas.LogScalarsRequest,),
        backend_response_schema=schemas.LogScalarsResponse,
        frontend_api_function="fetchLogScalars",
        frontend_response_schema="logScalarsSchema",
    ),
    (("POST",), "/logs/tags"): EndpointSchemaMapping(
        backend_body_request_schemas=(schemas.LogTagsRequest,),
        backend_response_schema=schemas.LogTagsResponse,
        frontend_api_function="fetchLogTags",
        frontend_response_schema="logTagsSchema",
    ),
    (("POST",), "/training/jobs"): EndpointSchemaMapping(
        backend_body_request_schemas=(schemas.TrainingJobCreateRequest,),
        backend_response_schema=schemas.TrainingJobResponse,
        frontend_api_function="createTrainingJob",
        frontend_response_schema="trainingJobSchema",
    ),
    (("POST",), "/training/jobs/{job_id}/cancel"): EndpointSchemaMapping(
        backend_body_request_schemas=(),
        backend_response_schema=schemas.TrainingJobResponse,
        frontend_api_function="cancelTrainingJob",
        frontend_response_schema="trainingJobSchema",
    ),
    (("POST",), "/training/jobs/{job_id}/reconcile"): EndpointSchemaMapping(
        backend_body_request_schemas=(schemas.TrainingJobReconcileRequest,),
        backend_response_schema=schemas.TrainingJobResponse,
        frontend_api_function="reconcileTrainingJob",
        frontend_response_schema="trainingJobSchema",
    ),
    (("POST",), "/training/run-plan"): EndpointSchemaMapping(
        backend_body_request_schemas=(schemas.TrainingRunPlanCreateRequest,),
        backend_response_schema=schemas.TrainingRunPlanResponse,
        frontend_api_function="fetchTrainingRunPlan",
        frontend_response_schema="trainingRunPlanSchema",
    ),
}


@dataclass(frozen=True)
class SchemaParityCase:
    backend_schema: type[schemas.ApiResponseModel]
    frontend_contract: str
    backend_fields: tuple[str, ...]
    frontend_required_fields: tuple[str, ...]
    intentional_frontend_required_looseness: dict[str, str] = field(
        default_factory=dict
    )
    intentional_frontend_default_fields: dict[str, str] = field(default_factory=dict)


CAPABILITIES_FIELDS = (
    "authMode",
    "trainingEnabled",
    "trainingCancellationCapability",
    "trainingResourceLimitsEnforced",
    "logDeletionEnabled",
    "configSnapshotsEnabled",
    "historicalLogsEnabled",
    "liveMonitorDataEnabled",
    "historicalMonitorDataEnabled",
    "uploadsEnabled",
    "maxUploadSize",
    "maxActiveTrainingJobs",
    "trainingJobMemoryLimitBytes",
    "trainingJobCpuLimit",
    "trainingJobProcessLimit",
)
CAPABILITIES_REQUIRED_FIELDS = (
    "authMode",
    "trainingEnabled",
    "trainingCancellationCapability",
    "trainingResourceLimitsEnforced",
    "logDeletionEnabled",
    "configSnapshotsEnabled",
    "historicalLogsEnabled",
    "liveMonitorDataEnabled",
    "historicalMonitorDataEnabled",
    "uploadsEnabled",
    "maxUploadSize",
    "maxActiveTrainingJobs",
    "trainingJobMemoryLimitBytes",
    "trainingJobCpuLimit",
    "trainingJobProcessLimit",
)

CONFIG_FIELD_FIELDS = (
    "key",
    "configKey",
    "flag",
    "label",
    "section",
    "sectionPath",
    "description",
    "type",
    "default",
    "nullable",
    "choices",
    "maximum",
    "locked",
    "lockedValue",
    "lockedReason",
)
CONFIG_FIELD_REQUIRED_FIELDS = (
    "key",
    "configKey",
    "flag",
    "label",
    "section",
    "sectionPath",
    "type",
    "default",
    "nullable",
    "choices",
)
SEARCH_AXIS_FIELDS = (
    "key",
    "configKey",
    "searchKey",
    "label",
    "section",
    "type",
    "values",
    "locked",
    "lockedValue",
    "lockedReason",
    "lockedByPresets",
    "lockReasons",
)
SEARCH_AXIS_REQUIRED_FIELDS = (
    "key",
    "configKey",
    "searchKey",
    "label",
    "section",
    "type",
    "values",
)
GRAPH_CONFIG_FIELD_FIELDS = ("key", "value", "description")
GRAPH_CONFIG_FIELD_REQUIRED_FIELDS = ("key",)
GRAPH_CONFIG_FIELD_LOOSENESS = {
    "value": (
        "graphConfigSchema stores opaque config values as JSON; the frontend "
        "accepts omitted values while backend graph config fields require value "
        "presence."
    ),
}
GRAPH_NODE_FIELDS = (
    "id",
    "label",
    "typeName",
    "description",
    "path",
    "graphRole",
    "parameterCount",
    "parameterSizeBytes",
    "details",
    "config",
)
GRAPH_NODE_REQUIRED_FIELDS = (
    "id",
    "label",
    "typeName",
    "path",
    "graphRole",
    "parameterCount",
    "parameterSizeBytes",
    "details",
    "config",
)
TRAINING_SEARCH_FIELDS = ("mode", "values", "randomSamples")
TRAINING_SEARCH_REQUIRED_FIELDS = ("mode", "values")
TRAINING_RUN_CHANGE_FIELDS = ("key", "label", "value", "source")
TRAINING_RUN_FIELDS = (
    "id",
    "index",
    "status",
    "preset",
    "snapshotId",
    "snapshotName",
    "dataset",
    "experimentTask",
    "changes",
    "overrides",
    "command",
    "commandArgv",
    "commands",
    "totalEpochs",
    "currentEpoch",
    "metrics",
    "logDir",
    "error",
    "errorTraceback",
)
TRAINING_RUN_REQUIRED_FIELDS = (
    "id",
    "index",
    "status",
    "preset",
    "dataset",
    "experimentTask",
    "changes",
    "overrides",
    "command",
    "totalEpochs",
    "currentEpoch",
    "metrics",
    "logDir",
    "error",
)
TRAINING_RUN_PLAN_SUMMARY_FIELDS = (
    "totalRuns",
    "completedRuns",
    "runningRuns",
    "pendingRuns",
    "failedRuns",
    "cancelledRuns",
    "skippedRuns",
    "totalEpochs",
    "completedEpochs",
    "remainingEpochs",
)
TRAINING_RUN_PLAN_FIELDS = (
    "modelType",
    "model",
    "preset",
    "presets",
    "experimentTask",
    "datasets",
    "overrides",
    "search",
    "logFolder",
    "isRandomSearch",
    "runs",
    "summary",
    "snapshotRevisions",
)
TRAINING_RUN_PLAN_REQUIRED_FIELDS = TRAINING_RUN_PLAN_FIELDS[:-1]
TRAINING_JOB_FIELDS = (
    "id",
    "status",
    "modelType",
    "model",
    "preset",
    "presets",
    "experimentTask",
    "datasets",
    "overrides",
    "search",
    "plannedRunCount",
    "runPlan",
    "monitors",
    "logFolder",
    "createdAt",
    "updatedAt",
    "exitCode",
    "pid",
    "cancellationMode",
    "currentPreset",
    "currentDataset",
    "epoch",
    "step",
    "metrics",
    "logDir",
    "events",
    "eventCount",
    "eventCounts",
    "eventsTruncated",
    "clusterGrowth",
    "logTail",
    "logTailTruncated",
    "resultLinks",
)
TRAINING_JOB_REQUIRED_FIELDS = (
    "id",
    "status",
    "modelType",
    "model",
    "preset",
    "experimentTask",
    "datasets",
    "overrides",
    "monitors",
    "logFolder",
    "createdAt",
    "updatedAt",
    "exitCode",
    "pid",
    "currentDataset",
    "epoch",
    "step",
    "metrics",
    "logDir",
    "events",
    "logTail",
    "resultLinks",
)
MONITOR_DATA_FIELDS = (
    "jobId",
    "nodePath",
    "preset",
    "dataset",
    "logDir",
    "eventBytes",
    "skippedEventFiles",
    "truncated",
    "truncationReason",
    "sourceItemCount",
    "returnedItemCount",
    "scalarSeries",
    "histograms",
    "images",
)
MONITOR_DATA_REQUIRED_FIELDS = (
    "jobId",
    "nodePath",
    "scalarSeries",
    "histograms",
    "images",
)
PARAMETER_CHANNEL_STATUS_FIELDS = (
    "status",
    "metric",
    "lastStep",
    "observedPoints",
)
PARAMETER_CHANNEL_STATUS_REQUIRED_FIELDS = (
    "status",
    "observedPoints",
)
PARAMETER_NODE_STATUS_FIELDS = ("nodePath", "weights", "bias")
PARAMETER_STATUS_FIELDS = (
    "sourceId",
    "preset",
    "dataset",
    "logDir",
    "eventBytes",
    "skippedEventFiles",
    "truncated",
    "truncationReason",
    "sourceItemCount",
    "returnedItemCount",
    "nodes",
)
PARAMETER_STATUS_REQUIRED_FIELDS = ("sourceId", "nodes")
LOG_RUN_FIELDS = (
    "id",
    "group",
    "experiment",
    "modelType",
    "model",
    "preset",
    "experimentTask",
    "dataset",
    "runName",
    "timestamp",
    "version",
    "relativePath",
    "hasResult",
    "eventFileCount",
    "checkpointCount",
    "hasHparams",
    "hasLayerMonitorData",
    "metrics",
)
LOG_RUN_REQUIRED_FIELDS = (
    "id",
    "group",
    "modelType",
    "model",
    "preset",
    "dataset",
    "runName",
    "timestamp",
    "version",
    "relativePath",
    "hasResult",
    "eventFileCount",
    "checkpointCount",
    "hasHparams",
    "metrics",
)
LOG_RUN_LOOSENESS = {
    "experiment": (
        "logRunSchema accepts omitted experiments and derives them from "
        "relativePath before piping to the output schema."
    ),
}
LOG_CHECKPOINT_REQUEST_FIELDS = ("runIds",)
LOG_CHECKPOINT_FIELDS = (
    "id",
    "runId",
    "filename",
    "relativePath",
    "epoch",
    "step",
    "sizeBytes",
    "modifiedAt",
)
LOG_ARTIFACT_FIELDS = (
    "id",
    "kind",
    "label",
    "relativePath",
    "sizeBytes",
    "modifiedAt",
)
LOG_RUN_ARTIFACTS_FIELDS = (
    "runId",
    "params",
    "metrics",
    "sourceItemCount",
    "returnedItemCount",
    "truncated",
    "truncationReason",
    "artifacts",
    "checkpoints",
)
LOG_DELETE_FILTER_FIELDS = (
    "experiments",
    "datasets",
    "models",
    "presets",
    "runIds",
)
LOG_DELETE_CANDIDATE_FIELDS = (
    "id",
    "experiment",
    "modelType",
    "model",
    "preset",
    "dataset",
    "runName",
    "version",
    "relativePath",
)
LOG_DELETE_AFFECTED_FIELDS = (
    "experiments",
    "datasets",
    "models",
    "presets",
    "runIds",
)
LOG_DELETE_COUNTS_FIELDS = (
    "runs",
    "experiments",
    "datasets",
    "models",
    "presets",
)
LOG_DELETE_PLAN_FIELDS = (
    "candidateCount",
    "sourceItemCount",
    "returnedItemCount",
    "truncated",
    "truncationReason",
    "counts",
    "affected",
    "candidates",
    "blockedByActiveJobs",
    "canDelete",
)
LOG_DELETE_RESPONSE_FIELDS = (
    *LOG_DELETE_PLAN_FIELDS,
    "deletedRunIds",
    "deletedRunCount",
    "deletedRelativePaths",
)

SCHEMA_PARITY_CASES = (
    SchemaParityCase(schemas.HealthResponse, "healthSchema", ("status",), ("status",)),
    SchemaParityCase(
        schemas.ConfigSnapshotResponse,
        "configSnapshotSchema",
        (
            "id",
            "modelType",
            "model",
            "preset",
            "name",
            "overrides",
            "createdAt",
            "updatedAt",
        ),
        (
            "id",
            "modelType",
            "model",
            "preset",
            "name",
            "overrides",
            "createdAt",
            "updatedAt",
        ),
    ),
    SchemaParityCase(
        schemas.ConfigSnapshotsResponse,
        "configSnapshotsSchema",
        ("modelType", "model", "snapshots"),
        ("modelType", "model", "snapshots"),
    ),
    SchemaParityCase(
        schemas.ConfigSnapshotLibraryResponse,
        "configSnapshotLibrarySchema",
        ("snapshots",),
        ("snapshots",),
    ),
    SchemaParityCase(
        schemas.ConfigSnapshotCreateRequest,
        "createConfigSnapshot input",
        ("modelType", "model", "preset", "name", "overrides"),
        ("modelType", "model", "preset"),
    ),
    SchemaParityCase(
        schemas.ConfigSnapshotUpdateRequest,
        "updateConfigSnapshot input",
        ("name", "overrides"),
        (),
    ),
    SchemaParityCase(
        schemas.CapabilitiesResponse,
        "capabilitiesSchema",
        CAPABILITIES_FIELDS,
        CAPABILITIES_REQUIRED_FIELDS,
    ),
    SchemaParityCase(schemas.ModelsResponse, "modelsSchema", ("models",), ("models",)),
    SchemaParityCase(
        schemas.PresetResponse,
        "presetSchema",
        ("name", "label", "description"),
        ("name", "label", "description"),
    ),
    SchemaParityCase(
        schemas.PresetsResponse,
        "presetsSchema",
        ("modelType", "model", "presets"),
        ("modelType", "model", "presets"),
    ),
    SchemaParityCase(
        schemas.DatasetResponse,
        "datasetSchema",
        ("name", "label", "inputDim", "outputDim"),
        ("name", "label", "inputDim", "outputDim"),
    ),
    SchemaParityCase(
        schemas.DatasetsResponse,
        "datasetsSchema",
        ("modelType", "model", "defaultExperimentTask", "datasetGroups"),
        ("modelType", "model", "defaultExperimentTask", "datasetGroups"),
    ),
    SchemaParityCase(
        schemas.MonitorOptionResponse,
        "monitorOptionSchema",
        ("name", "label", "description", "kinds", "defaultEnabled"),
        ("name", "label", "description", "kinds", "defaultEnabled"),
    ),
    SchemaParityCase(
        schemas.MonitorsResponse,
        "monitorsSchema",
        ("modelType", "model", "monitors"),
        ("modelType", "model", "monitors"),
    ),
    SchemaParityCase(
        schemas.ConfigFieldResponse,
        "configFieldSchema",
        CONFIG_FIELD_FIELDS,
        CONFIG_FIELD_REQUIRED_FIELDS,
    ),
    SchemaParityCase(
        schemas.ConfigSchemaResponse,
        "configSchema",
        ("modelType", "model", "fields"),
        ("modelType", "model", "fields"),
    ),
    SchemaParityCase(
        schemas.SearchAxisResponse,
        "searchAxisSchema",
        SEARCH_AXIS_FIELDS,
        SEARCH_AXIS_REQUIRED_FIELDS,
    ),
    SchemaParityCase(
        schemas.SearchSpaceResponse,
        "searchSpaceSchema",
        ("modelType", "model", "preset", "axes"),
        ("modelType", "model", "axes"),
    ),
    SchemaParityCase(
        schemas.GraphConfigFieldResponse,
        "graphConfigSchema.fields[]",
        GRAPH_CONFIG_FIELD_FIELDS,
        GRAPH_CONFIG_FIELD_REQUIRED_FIELDS,
        intentional_frontend_required_looseness=GRAPH_CONFIG_FIELD_LOOSENESS,
    ),
    SchemaParityCase(
        schemas.GraphConfigResponse,
        "graphConfigSchema",
        ("typeName", "fields"),
        ("typeName", "fields"),
    ),
    SchemaParityCase(
        schemas.GraphNodeResponse,
        "graphNodeSchema",
        GRAPH_NODE_FIELDS,
        GRAPH_NODE_REQUIRED_FIELDS,
    ),
    SchemaParityCase(
        schemas.GraphEdgeResponse,
        "graphEdgeSchema",
        ("id", "source", "target"),
        ("id", "source", "target"),
    ),
    SchemaParityCase(
        schemas.InspectRequest,
        "inspectModel input",
        (
            "modelType",
            "model",
            "preset",
            "overrides",
            "experimentTask",
            "dataset",
            "logRunId",
        ),
        ("modelType", "model", "preset", "overrides"),
    ),
    SchemaParityCase(
        schemas.InspectResponse,
        "inspectResponseSchema",
        (
            "modelType",
            "model",
            "preset",
            "parameterCount",
            "parameterSizeBytes",
            "nodes",
            "edges",
        ),
        (
            "modelType",
            "model",
            "preset",
            "parameterCount",
            "parameterSizeBytes",
            "nodes",
            "edges",
        ),
    ),
    SchemaParityCase(
        schemas.TrainingJobCreateRequest,
        "TrainingJobCreateInput",
        (
            "modelType",
            "model",
            "preset",
            "presets",
            "experimentTask",
            "datasets",
            "overrides",
            "logFolder",
            "monitors",
            "search",
            "runPlan",
            "snapshotIds",
            "snapshotRevisions",
        ),
        (
            "modelType",
            "model",
            "preset",
            "datasets",
            "overrides",
            "logFolder",
            "monitors",
        ),
    ),
    SchemaParityCase(
        schemas.TrainingJobReconcileRequest,
        "TrainingJobReconcileInput",
        ("action", "reason"),
        ("action", "reason"),
    ),
    SchemaParityCase(
        schemas.TrainingRunPlanCreateRequest,
        "TrainingRunPlanCreateInput",
        (
            "modelType",
            "model",
            "preset",
            "presets",
            "experimentTask",
            "datasets",
            "overrides",
            "logFolder",
            "monitors",
            "search",
            "snapshotIds",
        ),
        ("modelType", "model", "preset", "datasets", "overrides"),
    ),
    SchemaParityCase(
        schemas.TrainingSearchResponse,
        "trainingJobSchema.search/trainingRunPlanSchema.search",
        TRAINING_SEARCH_FIELDS,
        TRAINING_SEARCH_REQUIRED_FIELDS,
    ),
    SchemaParityCase(
        schemas.TrainingSearchRequest,
        "TrainingSearchCreateInput",
        TRAINING_SEARCH_FIELDS,
        TRAINING_SEARCH_REQUIRED_FIELDS,
    ),
    SchemaParityCase(
        schemas.TrainingRunChangeResponse,
        "trainingRunChangeSchema",
        TRAINING_RUN_CHANGE_FIELDS,
        TRAINING_RUN_CHANGE_FIELDS,
    ),
    SchemaParityCase(
        schemas.TrainingRunResponse,
        "trainingRunSchema",
        TRAINING_RUN_FIELDS,
        TRAINING_RUN_REQUIRED_FIELDS,
    ),
    SchemaParityCase(
        schemas.SubmittedTrainingRunRequest,
        "TrainingRunSubmitInput",
        ("id", "preset", "snapshotId", "snapshotName", "dataset", "overrides"),
        ("id", "preset", "dataset", "overrides"),
    ),
    SchemaParityCase(
        schemas.TrainingRunPlanSummaryResponse,
        "trainingRunPlanSummarySchema",
        TRAINING_RUN_PLAN_SUMMARY_FIELDS,
        TRAINING_RUN_PLAN_SUMMARY_FIELDS,
    ),
    SchemaParityCase(
        schemas.ConfigSnapshotRevisionResponse,
        "configSnapshotRevisionSchema",
        ("id", "semanticRevision"),
        ("id", "semanticRevision"),
    ),
    SchemaParityCase(
        schemas.TrainingRunPlanResponse,
        "trainingRunPlanSchema",
        TRAINING_RUN_PLAN_FIELDS,
        TRAINING_RUN_PLAN_REQUIRED_FIELDS,
    ),
    SchemaParityCase(
        schemas.SubmittedTrainingRunPlanRequest,
        "TrainingRunPlanSubmitInput",
        ("runs", "snapshotRevisions"),
        ("runs",),
    ),
    SchemaParityCase(
        schemas.TrainingResultLinkResponse,
        "trainingJobSchema.resultLinks[]",
        ("preset", "dataset", "logDir"),
        (),
    ),
    SchemaParityCase(
        schemas.TrainingClusterGrowthAdditionResponse,
        "trainingClusterGrowthAdditionSchema",
        ("coord", "step", "epoch"),
        ("coord", "step", "epoch"),
    ),
    SchemaParityCase(
        schemas.TrainingClusterGrowthResponse,
        "trainingClusterGrowthSchema",
        ("node", "count", "capacityTotal", "additionCount", "additions"),
        ("node", "count", "capacityTotal", "additionCount", "additions"),
    ),
    SchemaParityCase(
        schemas.TrainingJobResponse,
        "trainingJobSchema",
        TRAINING_JOB_FIELDS,
        TRAINING_JOB_REQUIRED_FIELDS,
    ),
    SchemaParityCase(
        schemas.TrainingProgressEventsResponse,
        "trainingJobEventsSchema",
        ("jobId", "offset", "limit", "totalCount", "nextOffset", "events"),
        ("jobId", "offset", "limit", "totalCount", "nextOffset", "events"),
    ),
    SchemaParityCase(
        schemas.ScalarPointResponse,
        "logScalarPointSchema/monitorDataSchema.scalarSeries[].points[]",
        ("step", "wallTime", "value"),
        ("step", "wallTime", "value"),
    ),
    SchemaParityCase(
        schemas.ScalarSeriesResponse,
        "monitorDataSchema.scalarSeries[]",
        (
            "tag",
            "label",
            "points",
            "sourceItemCount",
            "returnedItemCount",
            "truncated",
            "truncationReason",
        ),
        ("tag", "label", "points"),
    ),
    SchemaParityCase(
        schemas.HistogramBucketResponse,
        "monitorDataSchema.histograms[].buckets[]",
        ("left", "right", "count"),
        ("left", "right", "count"),
    ),
    SchemaParityCase(
        schemas.HistogramResponse,
        "monitorDataSchema.histograms[]",
        (
            "tag",
            "step",
            "wallTime",
            "buckets",
            "sourceItemCount",
            "returnedItemCount",
            "truncated",
            "truncationReason",
        ),
        ("tag", "step", "wallTime", "buckets"),
    ),
    SchemaParityCase(
        schemas.ImageResponse,
        "monitorDataSchema.images[]",
        (
            "tag",
            "step",
            "wallTime",
            "mimeType",
            "dataUrl",
            "eventBytes",
            "sourceItemCount",
            "returnedItemCount",
            "truncated",
            "truncationReason",
        ),
        ("tag", "step", "wallTime", "mimeType", "dataUrl"),
    ),
    SchemaParityCase(
        schemas.MonitorDataResponse,
        "monitorDataSchema",
        MONITOR_DATA_FIELDS,
        MONITOR_DATA_REQUIRED_FIELDS,
    ),
    SchemaParityCase(
        schemas.ParameterChannelStatusResponse,
        "parameterChannelStatusSchema",
        PARAMETER_CHANNEL_STATUS_FIELDS,
        PARAMETER_CHANNEL_STATUS_REQUIRED_FIELDS,
    ),
    SchemaParityCase(
        schemas.ParameterNodeStatusResponse,
        "parameterStatusSchema.nodes[]",
        PARAMETER_NODE_STATUS_FIELDS,
        PARAMETER_NODE_STATUS_FIELDS,
    ),
    SchemaParityCase(
        schemas.ParameterStatusResponse,
        "parameterStatusSchema",
        PARAMETER_STATUS_FIELDS,
        PARAMETER_STATUS_REQUIRED_FIELDS,
    ),
    SchemaParityCase(
        schemas.LogRunResponse,
        "logRunSchema",
        LOG_RUN_FIELDS,
        LOG_RUN_REQUIRED_FIELDS,
        intentional_frontend_required_looseness=LOG_RUN_LOOSENESS,
    ),
    SchemaParityCase(
        schemas.LogRunsResponse,
        "logRunsSchema",
        ("total", "limit", "offset", "hasMore", "facets", "runs"),
        ("runs",),
    ),
    SchemaParityCase(
        schemas.LogCheckpointsRequest,
        "fetchLogCheckpoints input",
        LOG_CHECKPOINT_REQUEST_FIELDS,
        LOG_CHECKPOINT_REQUEST_FIELDS,
    ),
    SchemaParityCase(
        schemas.LogCheckpointResponse,
        "logCheckpointSchema",
        LOG_CHECKPOINT_FIELDS,
        LOG_CHECKPOINT_FIELDS,
    ),
    SchemaParityCase(
        schemas.LogCheckpointsResponse,
        "logCheckpointsSchema",
        (
            "sourceItemCount",
            "returnedItemCount",
            "truncated",
            "truncationReason",
            "checkpoints",
        ),
        ("checkpoints",),
    ),
    SchemaParityCase(
        schemas.LogRunArtifactResponse,
        "logRunArtifactSchema",
        LOG_ARTIFACT_FIELDS,
        LOG_ARTIFACT_FIELDS,
    ),
    SchemaParityCase(
        schemas.LogRunArtifactsResponse,
        "logRunArtifactsSchema",
        LOG_RUN_ARTIFACTS_FIELDS,
        LOG_RUN_ARTIFACTS_FIELDS,
    ),
    SchemaParityCase(
        schemas.LogExperimentResponse,
        "logExperimentSchema",
        ("experiment", "runCount", "relativePath"),
        ("experiment", "runCount", "relativePath"),
    ),
    SchemaParityCase(
        schemas.LogExperimentsResponse,
        "logExperimentsSchema",
        ("total", "limit", "offset", "hasMore", "experiments"),
        ("experiments",),
    ),
    SchemaParityCase(
        schemas.LogExperimentDeleteResponse,
        "logExperimentDeleteSchema",
        ("experiment", "deletedRunIds", "deletedRunCount", "deletedRelativePath"),
        ("experiment", "deletedRunIds", "deletedRunCount", "deletedRelativePath"),
    ),
    SchemaParityCase(
        schemas.LogArchiveImportResponse,
        "logArchiveImportSchema",
        ("extractedFileCount", "skippedFileCount", "destinationRoot"),
        ("extractedFileCount", "skippedFileCount", "destinationRoot"),
    ),
    SchemaParityCase(
        schemas.LogRunDeleteFiltersRequest,
        "LogRunDeleteFilters",
        LOG_DELETE_FILTER_FIELDS,
        LOG_DELETE_FILTER_FIELDS,
    ),
    SchemaParityCase(
        schemas.LogPresetDeleteRequest,
        "LogPresetDeleteTarget",
        ("experiment", "preset"),
        ("experiment", "preset"),
    ),
    SchemaParityCase(
        schemas.LogRunDeleteCandidateResponse,
        "logRunDeleteCandidateSchema",
        LOG_DELETE_CANDIDATE_FIELDS,
        LOG_DELETE_CANDIDATE_FIELDS,
    ),
    SchemaParityCase(
        schemas.LogRunDeleteAffectedValuesResponse,
        "logRunDeleteAffectedValuesSchema",
        LOG_DELETE_AFFECTED_FIELDS,
        LOG_DELETE_AFFECTED_FIELDS,
    ),
    SchemaParityCase(
        schemas.LogRunDeleteCountsResponse,
        "logRunDeleteCountsSchema",
        LOG_DELETE_COUNTS_FIELDS,
        LOG_DELETE_COUNTS_FIELDS,
    ),
    SchemaParityCase(
        schemas.LogRunDeleteBlockerResponse,
        "logRunDeleteBlockerSchema",
        ("id", "logFolder", "status"),
        ("id", "logFolder", "status"),
    ),
    SchemaParityCase(
        schemas.LogRunDeletePlanResponse,
        "logRunDeletePlanSchema",
        LOG_DELETE_PLAN_FIELDS,
        (
            "candidateCount",
            "counts",
            "affected",
            "candidates",
            "blockedByActiveJobs",
            "canDelete",
        ),
    ),
    SchemaParityCase(
        schemas.LogRunDeleteResponse,
        "logRunDeleteSchema",
        LOG_DELETE_RESPONSE_FIELDS,
        (
            "candidateCount",
            "counts",
            "affected",
            "candidates",
            "blockedByActiveJobs",
            "canDelete",
            "deletedRunIds",
            "deletedRunCount",
            "deletedRelativePaths",
        ),
    ),
    SchemaParityCase(
        schemas.LogTagsRequest,
        "fetchLogTags input",
        ("runIds",),
        ("runIds",),
    ),
    SchemaParityCase(
        schemas.LogRunTagsResponse,
        "logRunTagsSchema",
        (
            "runId",
            "eventBytes",
            "skippedEventFiles",
            "truncated",
            "truncationReason",
            "sourceItemCount",
            "returnedItemCount",
            "hasLayerMonitorData",
            "scalarTags",
            "histogramTags",
            "imageTags",
            "textTags",
        ),
        ("runId", "scalarTags", "histogramTags", "imageTags"),
    ),
    SchemaParityCase(
        schemas.LogTagsResponse,
        "logTagsSchema",
        ("runs",),
        ("runs",),
    ),
    SchemaParityCase(
        schemas.LogScalarsRequest,
        "fetchLogScalars input",
        ("runIds", "tags", "maxPoints", "sampling"),
        ("runIds", "tags", "maxPoints", "sampling"),
    ),
    SchemaParityCase(
        schemas.LogMediaRequest,
        "fetchLogMedia input",
        ("runIds", "imageTags", "textTags"),
        ("runIds", "imageTags", "textTags"),
    ),
    SchemaParityCase(
        schemas.LogImageSummaryResponse,
        "logImageSummarySchema",
        (
            "runId",
            "tag",
            "step",
            "wallTime",
            "mimeType",
            "dataUrl",
            "eventBytes",
            "sourceItemCount",
            "returnedItemCount",
            "truncated",
            "truncationReason",
        ),
        ("runId", "tag", "step", "wallTime", "mimeType", "dataUrl"),
    ),
    SchemaParityCase(
        schemas.LogTextSummaryResponse,
        "logTextSummarySchema",
        (
            "runId",
            "tag",
            "step",
            "wallTime",
            "text",
            "eventBytes",
            "sourceItemCount",
            "returnedItemCount",
            "truncated",
            "truncationReason",
        ),
        ("runId", "tag", "step", "wallTime", "text"),
    ),
    SchemaParityCase(
        schemas.LogMediaResponse,
        "logMediaSchema",
        (
            "eventBytes",
            "skippedEventFiles",
            "sourceItemCount",
            "returnedItemCount",
            "truncated",
            "truncationReason",
            "images",
            "texts",
        ),
        ("images", "texts"),
    ),
    SchemaParityCase(
        schemas.LogParameterStatusRequest,
        "fetchLogParameterStatus input",
        ("runIds",),
        ("runIds",),
    ),
    SchemaParityCase(
        schemas.LogParameterStatusResponse,
        "logParameterStatusSchema",
        ("runs",),
        ("runs",),
    ),
    SchemaParityCase(
        schemas.LogScalarSeriesResponse,
        "logScalarSeriesSchema",
        ("runId", "tag", "points", "sourcePointCount", "truncated"),
        ("runId", "tag", "points", "sourcePointCount", "truncated"),
    ),
    SchemaParityCase(
        schemas.LogScalarsResponse,
        "logScalarsSchema",
        ("series",),
        ("series",),
    ),
)

SCHEMA_PARITY_BY_BACKEND_SCHEMA = {
    case.backend_schema: case for case in SCHEMA_PARITY_CASES
}
SCHEMA_PARITY_BY_BACKEND_AND_FRONTEND_CONTRACT = {
    (case.backend_schema, case.frontend_contract): case for case in SCHEMA_PARITY_CASES
}
OPENAPI_REQUIRED_FIELD_PARITY_BY_FRONTEND_SCHEMA = {
    (case.backend_schema, case.frontend_contract): case.frontend_required_fields
    for case in SCHEMA_PARITY_CASES
}
INTENTIONAL_FRONTEND_REQUIRED_FIELD_LOOSENESS = {
    (case.backend_schema, case.frontend_contract): (
        case.intentional_frontend_required_looseness
    )
    for case in SCHEMA_PARITY_CASES
    if case.intentional_frontend_required_looseness
}
INTENTIONAL_FRONTEND_DEFAULT_FIELDS = {
    (
        case.backend_schema,
        case.frontend_contract,
    ): case.intentional_frontend_default_fields
    for case in SCHEMA_PARITY_CASES
    if case.intentional_frontend_default_fields
}

HIGH_RISK_SCHEMA_PARITY_GROUPS = {
    "capabilities": (schemas.CapabilitiesResponse,),
    "config snapshots": (
        schemas.ConfigSnapshotResponse,
        schemas.ConfigSnapshotsResponse,
        schemas.ConfigSnapshotLibraryResponse,
        schemas.ConfigSnapshotCreateRequest,
        schemas.ConfigSnapshotUpdateRequest,
    ),
    "model config/search": (
        schemas.ConfigFieldResponse,
        schemas.ConfigSchemaResponse,
        schemas.SearchAxisResponse,
        schemas.SearchSpaceResponse,
    ),
    "inspect graph": (
        schemas.GraphConfigFieldResponse,
        schemas.GraphConfigResponse,
        schemas.GraphNodeResponse,
        schemas.GraphEdgeResponse,
        schemas.InspectRequest,
        schemas.InspectResponse,
    ),
    "training": (
        schemas.TrainingJobCreateRequest,
        schemas.TrainingRunPlanCreateRequest,
        schemas.TrainingSearchResponse,
        schemas.TrainingSearchRequest,
        schemas.TrainingRunChangeResponse,
        schemas.TrainingRunResponse,
        schemas.SubmittedTrainingRunRequest,
        schemas.TrainingRunPlanSummaryResponse,
        schemas.ConfigSnapshotRevisionResponse,
        schemas.TrainingRunPlanResponse,
        schemas.SubmittedTrainingRunPlanRequest,
        schemas.TrainingResultLinkResponse,
        schemas.TrainingClusterGrowthAdditionResponse,
        schemas.TrainingClusterGrowthResponse,
        schemas.TrainingJobResponse,
        schemas.TrainingProgressEventsResponse,
    ),
    "monitor data": (
        schemas.ScalarPointResponse,
        schemas.ScalarSeriesResponse,
        schemas.HistogramBucketResponse,
        schemas.HistogramResponse,
        schemas.ImageResponse,
        schemas.MonitorDataResponse,
        schemas.ParameterChannelStatusResponse,
        schemas.ParameterNodeStatusResponse,
        schemas.ParameterStatusResponse,
    ),
    "logs": (
        schemas.LogRunResponse,
        schemas.LogRunsResponse,
        schemas.LogCheckpointsRequest,
        schemas.LogCheckpointResponse,
        schemas.LogCheckpointsResponse,
        schemas.LogRunArtifactResponse,
        schemas.LogRunArtifactsResponse,
        schemas.LogExperimentResponse,
        schemas.LogExperimentsResponse,
        schemas.LogExperimentDeleteResponse,
        schemas.LogArchiveImportResponse,
        schemas.LogPresetDeleteRequest,
        schemas.LogRunDeleteFiltersRequest,
        schemas.LogRunDeleteCandidateResponse,
        schemas.LogRunDeleteAffectedValuesResponse,
        schemas.LogRunDeleteCountsResponse,
        schemas.LogRunDeleteBlockerResponse,
        schemas.LogRunDeletePlanResponse,
        schemas.LogRunDeleteResponse,
        schemas.LogTagsRequest,
        schemas.LogRunTagsResponse,
        schemas.LogTagsResponse,
        schemas.LogScalarsRequest,
        schemas.LogScalarSeriesResponse,
        schemas.LogScalarsResponse,
        schemas.LogMediaRequest,
        schemas.LogImageSummaryResponse,
        schemas.LogTextSummaryResponse,
        schemas.LogMediaResponse,
        schemas.LogParameterStatusRequest,
        schemas.LogParameterStatusResponse,
    ),
}


def _business_routes_by_key() -> dict[RouteKey, RouteContext]:
    business_prefixes = (
        "/capabilities",
        "/config-snapshots",
        "/health",
        "/models",
        "/inspect",
        "/logs",
        "/training",
    )
    return {
        (tuple(sorted(route.methods or ())), route.path): route
        for route in iter_route_contexts(app.routes)
        if isinstance(route.original_route, APIRoute)
        and route.path is not None
        and route.path.startswith(business_prefixes)
    }


def _body_request_schemas(
    route: RouteContext,
) -> tuple[type[schemas.ApiResponseModel], ...]:
    endpoint = route.endpoint
    if endpoint is None:
        return ()
    annotations = get_type_hints(endpoint)
    return tuple(
        annotations[body_param.name] for body_param in route.dependant.body_params
    )


def _openapi_required_fields(schema_name: str) -> tuple[str, ...]:
    component_schema = app.openapi()["components"]["schemas"][schema_name]
    return tuple(component_schema.get("required", ()))


def _openapi_property_schema(schema_name: str, field_name: str) -> dict[str, object]:
    component_schema = app.openapi()["components"]["schemas"][schema_name]
    return component_schema["properties"][field_name]
