from __future__ import annotations

import asyncio
import os
import unittest
from dataclasses import dataclass, field
from typing import NamedTuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from fastapi.routing import APIRoute

from viewer.backend import schemas
from viewer.backend.api import app

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
    (("POST",), "/inspect/operation-graph"),
    (("POST",), "/logs/checkpoints"),
    (("POST",), "/logs/import"),
    (("POST",), "/logs/media"),
    (("POST",), "/logs/parameter-status"),
    (("POST",), "/logs/runs/delete"),
    (("POST",), "/logs/runs/delete-plan"),
    (("POST",), "/logs/scalars"),
    (("POST",), "/logs/tags"),
    (("POST",), "/training/jobs"),
    (("POST",), "/training/jobs/{job_id}/cancel"),
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
    (("POST",), "/inspect/operation-graph"): EndpointSchemaMapping(
        backend_body_request_schemas=(schemas.InspectRequest,),
        backend_response_schema=schemas.OperationGraphResponse,
        frontend_api_function="inspectOperationGraph",
        frontend_response_schema="operationGraphResponseSchema",
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
    "logDeletionEnabled",
    "configSnapshotsEnabled",
    "historicalLogsEnabled",
    "liveMonitorDataEnabled",
    "historicalMonitorDataEnabled",
    "uploadsEnabled",
    "maxUploadSize",
    "dataSourcesEnabled",
    "dataSources",
)
CAPABILITIES_REQUIRED_FIELDS = (
    "authMode",
    "trainingEnabled",
    "logDeletionEnabled",
    "historicalLogsEnabled",
    "liveMonitorDataEnabled",
    "historicalMonitorDataEnabled",
)
CAPABILITIES_FRONTEND_DEFAULT_FIELDS = {
    "configSnapshotsEnabled": (
        "capabilitiesSchema defaults config-snapshot support on when omitted."
    ),
    "uploadsEnabled": "capabilitiesSchema defaults upload support off when omitted.",
    "maxUploadSize": "capabilitiesSchema defaults upload size to null when omitted.",
    "dataSourcesEnabled": (
        "capabilitiesSchema defaults data-source support off when omitted."
    ),
    "dataSources": "capabilitiesSchema defaults data-source placeholders to [].",
    "trainingCancellationCapability": (
        "capabilitiesSchema defaults strict cancellation support to unsupported "
        "when omitted."
    ),
}

CONFIG_FIELD_FIELDS = (
    "key",
    "configKey",
    "flag",
    "label",
    "section",
    "type",
    "default",
    "nullable",
    "choices",
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
GRAPH_CONFIG_FIELD_FIELDS = ("key", "value")
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
    "path",
    "graphRole",
    "parameterCount",
    "parameterSizeBytes",
    "details",
    "config",
)
OPERATION_GRAPH_NODE_FIELDS = (
    "id",
    "label",
    "opKind",
    "target",
    "modulePath",
    "groupId",
    "details",
)
OPERATION_GRAPH_RESPONSE_FIELDS = (
    "modelType",
    "model",
    "preset",
    "source",
    "status",
    "nodes",
    "edges",
    "warnings",
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
    "changes",
    "overrides",
    "command",
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
    "datasets",
    "overrides",
    "search",
    "logFolder",
    "isRandomSearch",
    "runs",
    "summary",
)
TRAINING_JOB_FIELDS = (
    "id",
    "status",
    "modelType",
    "model",
    "preset",
    "presets",
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
    "resultLinks",
)
TRAINING_JOB_REQUIRED_FIELDS = (
    "id",
    "status",
    "modelType",
    "model",
    "preset",
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
        intentional_frontend_default_fields=CAPABILITIES_FRONTEND_DEFAULT_FIELDS,
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
        ("modelType", "model", "datasets"),
        ("modelType", "model", "datasets"),
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
        GRAPH_NODE_FIELDS,
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
        ("modelType", "model", "preset", "overrides", "dataset"),
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
        schemas.OperationGraphNodeResponse,
        "operationGraphNodeSchema",
        OPERATION_GRAPH_NODE_FIELDS,
        OPERATION_GRAPH_NODE_FIELDS,
    ),
    SchemaParityCase(
        schemas.OperationGraphEdgeResponse,
        "operationGraphEdgeSchema",
        ("id", "source", "target"),
        ("id", "source", "target"),
    ),
    SchemaParityCase(
        schemas.OperationGraphResponse,
        "operationGraphResponseSchema",
        OPERATION_GRAPH_RESPONSE_FIELDS,
        OPERATION_GRAPH_RESPONSE_FIELDS,
    ),
    SchemaParityCase(
        schemas.TrainingJobCreateRequest,
        "TrainingJobCreateInput",
        (
            "modelType",
            "model",
            "preset",
            "presets",
            "datasets",
            "overrides",
            "logFolder",
            "monitors",
            "search",
            "runPlan",
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
        schemas.TrainingRunPlanCreateRequest,
        "TrainingRunPlanCreateInput",
        (
            "modelType",
            "model",
            "preset",
            "presets",
            "datasets",
            "overrides",
            "logFolder",
            "search",
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
        "TrainingSearchSubmitInput",
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
        schemas.SubmittedTrainingRunChangeRequest,
        "TrainingRunSubmitChangeInput",
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
        TRAINING_RUN_FIELDS,
        TRAINING_RUN_REQUIRED_FIELDS,
    ),
    SchemaParityCase(
        schemas.TrainingRunPlanSummaryResponse,
        "trainingRunPlanSummarySchema",
        TRAINING_RUN_PLAN_SUMMARY_FIELDS,
        TRAINING_RUN_PLAN_SUMMARY_FIELDS,
    ),
    SchemaParityCase(
        schemas.SubmittedTrainingRunPlanSummaryRequest,
        "TrainingRunPlanSubmitSummaryInput",
        TRAINING_RUN_PLAN_SUMMARY_FIELDS,
        TRAINING_RUN_PLAN_SUMMARY_FIELDS,
    ),
    SchemaParityCase(
        schemas.TrainingRunPlanResponse,
        "trainingRunPlanSchema",
        TRAINING_RUN_PLAN_FIELDS,
        TRAINING_RUN_PLAN_FIELDS,
    ),
    SchemaParityCase(
        schemas.SubmittedTrainingRunPlanRequest,
        "TrainingRunPlanSubmitInput",
        TRAINING_RUN_PLAN_FIELDS,
        TRAINING_RUN_PLAN_FIELDS,
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
        ("total", "limit", "offset", "hasMore", "runs"),
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
        schemas.SubmittedTrainingRunChangeRequest,
        schemas.TrainingRunResponse,
        schemas.SubmittedTrainingRunRequest,
        schemas.TrainingRunPlanSummaryResponse,
        schemas.SubmittedTrainingRunPlanSummaryRequest,
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

PATH_LIKE_DATASET_FIELDS = {
    "path",
    "root",
    "dir",
    "file",
    "filename",
    "relativePath",
    "absolutePath",
}


def _business_routes_by_key() -> dict[RouteKey, APIRoute]:
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
        for route in app.routes
        if isinstance(route, APIRoute) and route.path.startswith(business_prefixes)
    }


def _body_request_schemas(
    route: APIRoute,
) -> tuple[type[schemas.ApiResponseModel], ...]:
    return tuple(
        getattr(body_param, "type_", None) for body_param in route.dependant.body_params
    )


def _openapi_required_fields(schema_name: str) -> tuple[str, ...]:
    component_schema = app.openapi()["components"]["schemas"][schema_name]
    return tuple(component_schema.get("required", ()))


def _openapi_property_schema(schema_name: str, field_name: str) -> dict[str, object]:
    component_schema = app.openapi()["components"]["schemas"][schema_name]
    return component_schema["properties"][field_name]


class ApiRouteContractTests(unittest.TestCase):
    def test_api_routes_declare_response_models(self) -> None:
        missing = [
            f"{sorted(route.methods)} {route.path}"
            for route in app.routes
            if isinstance(route, APIRoute) and route.response_model is None
        ]

        self.assertEqual(missing, [])

    def test_api_route_inventory_preserves_current_contract(self) -> None:
        business_prefixes = (
            "/capabilities",
            "/config-snapshots",
            "/health",
            "/models",
            "/inspect",
            "/logs",
            "/training",
        )
        routes = sorted(
            (tuple(sorted(route.methods or ())), route.path)
            for route in app.routes
            if isinstance(route, APIRoute) and route.path.startswith(business_prefixes)
        )

        self.assertEqual(routes, EXPECTED_BUSINESS_ROUTES)
        self.assertFalse(
            any(path.startswith("/v1/") or path == "/v1" for _methods, path in routes)
        )

    def test_endpoint_schema_mapping_covers_public_routes(self) -> None:
        self.assertEqual(
            sorted(ENDPOINT_SCHEMA_MAPPINGS),
            EXPECTED_BUSINESS_ROUTES,
        )

        routes_by_key = _business_routes_by_key()
        self.assertEqual(sorted(routes_by_key), EXPECTED_BUSINESS_ROUTES)
        self.assertEqual(set(routes_by_key), set(ENDPOINT_SCHEMA_MAPPINGS))

        for route_key, mapping in ENDPOINT_SCHEMA_MAPPINGS.items():
            with self.subTest(route=route_key):
                route = routes_by_key[route_key]
                self.assertIs(route.response_model, mapping.backend_response_schema)
                self.assertEqual(
                    _body_request_schemas(route),
                    mapping.backend_body_request_schemas,
                )
                self.assertIsInstance(mapping.frontend_api_function, str)
                self.assertTrue(mapping.frontend_api_function)
                self.assertIsInstance(mapping.frontend_response_schema, str)
                self.assertTrue(mapping.frontend_response_schema)

    def test_endpoint_schema_mapping_has_explicit_schema_parity_cases(self) -> None:
        for route_key, mapping in ENDPOINT_SCHEMA_MAPPINGS.items():
            with self.subTest(route=route_key, schema="response"):
                self.assertIn(
                    (
                        mapping.backend_response_schema,
                        mapping.frontend_response_schema,
                    ),
                    SCHEMA_PARITY_BY_BACKEND_AND_FRONTEND_CONTRACT,
                )

            for request_schema in mapping.backend_body_request_schemas:
                with self.subTest(
                    route=route_key,
                    schema=request_schema.__name__,
                ):
                    self.assertIn(request_schema, SCHEMA_PARITY_BY_BACKEND_SCHEMA)


class ApiSchemaContractTests(unittest.TestCase):
    def test_schema_parity_case_inventory_is_unique(self) -> None:
        self.assertEqual(
            len(SCHEMA_PARITY_BY_BACKEND_SCHEMA),
            len(SCHEMA_PARITY_CASES),
        )

    def test_schema_parity_case_field_inventory_is_stable(self) -> None:
        for parity_case in SCHEMA_PARITY_CASES:
            with self.subTest(model=parity_case.backend_schema.__name__):
                backend_fields = set(parity_case.backend_fields)

                self.assertEqual(
                    tuple(parity_case.backend_schema.model_fields),
                    parity_case.backend_fields,
                )
                self.assertEqual(
                    tuple(
                        sorted(
                            set(parity_case.frontend_required_fields) - backend_fields
                        )
                    ),
                    (),
                )
                self.assertEqual(
                    tuple(
                        sorted(
                            set(parity_case.intentional_frontend_required_looseness)
                            - backend_fields
                        )
                    ),
                    (),
                )
                self.assertEqual(
                    tuple(
                        sorted(
                            set(parity_case.intentional_frontend_default_fields)
                            - backend_fields
                        )
                    ),
                    (),
                )

    def test_schema_parity_schemas_reject_extra_fields(self) -> None:
        for parity_case in SCHEMA_PARITY_CASES:
            with self.subTest(model=parity_case.backend_schema.__name__):
                self.assertEqual(
                    parity_case.backend_schema.model_config.get("extra"),
                    "forbid",
                )

    def test_opaque_json_fields_use_named_json_openapi_schemas(self) -> None:
        expected_refs = {
            ("GraphConfigFieldResponse", "value"): "JsonValue-Output",
            ("GraphNodeResponse", "details"): "JsonObject-Output",
            ("OperationGraphNodeResponse", "details"): "JsonObject-Output",
            ("LogRunResponse", "metrics"): "JsonObject-Output",
            ("LogRunArtifactsResponse", "params"): "JsonObject-Output",
            ("LogRunArtifactsResponse", "metrics"): "JsonObject-Output",
            ("TrainingRunResponse", "metrics"): "JsonObject-Output",
            ("TrainingJobResponse", "metrics"): "JsonObject-Output",
        }

        for (schema_name, field_name), ref_name in expected_refs.items():
            with self.subTest(schema=schema_name, field=field_name):
                self.assertEqual(
                    _openapi_property_schema(schema_name, field_name),
                    {"$ref": f"#/components/schemas/{ref_name}"},
                )

    def test_high_risk_nested_schema_parity_groups_are_covered(self) -> None:
        for group, group_schemas in HIGH_RISK_SCHEMA_PARITY_GROUPS.items():
            with self.subTest(group=group):
                missing = [
                    schema.__name__
                    for schema in group_schemas
                    if schema not in SCHEMA_PARITY_BY_BACKEND_SCHEMA
                ]

                self.assertEqual(missing, [])

    def test_openapi_required_fields_have_frontend_required_parity(self) -> None:
        for (
            (
                backend_schema,
                frontend_contract,
            ),
            frontend_required_fields,
        ) in OPENAPI_REQUIRED_FIELD_PARITY_BY_FRONTEND_SCHEMA.items():
            with self.subTest(
                backend_schema=backend_schema.__name__,
                frontend_contract=frontend_contract,
            ):
                frontend_required = set(frontend_required_fields)
                openapi_required = set(
                    _openapi_required_fields(backend_schema.__name__)
                )
                allowed_loose_fields = set(
                    INTENTIONAL_FRONTEND_REQUIRED_FIELD_LOOSENESS.get(
                        (backend_schema, frontend_contract),
                        {},
                    )
                )

                self.assertEqual(
                    tuple(sorted(frontend_required - set(backend_schema.model_fields))),
                    (),
                )
                self.assertEqual(
                    tuple(
                        sorted(
                            openapi_required - frontend_required - allowed_loose_fields
                        )
                    ),
                    (),
                )

    def test_frontend_required_field_looseness_annotations_are_current(self) -> None:
        for (
            (
                backend_schema,
                frontend_contract,
            ),
            looseness_annotations,
        ) in INTENTIONAL_FRONTEND_REQUIRED_FIELD_LOOSENESS.items():
            with self.subTest(
                backend_schema=backend_schema.__name__,
                frontend_contract=frontend_contract,
            ):
                self.assertIn(
                    (backend_schema, frontend_contract),
                    OPENAPI_REQUIRED_FIELD_PARITY_BY_FRONTEND_SCHEMA,
                )
                frontend_required = set(
                    OPENAPI_REQUIRED_FIELD_PARITY_BY_FRONTEND_SCHEMA[
                        (backend_schema, frontend_contract)
                    ]
                )
                openapi_required = set(
                    _openapi_required_fields(backend_schema.__name__)
                )
                annotated_fields = set(looseness_annotations)

                self.assertEqual(
                    tuple(sorted(annotated_fields - openapi_required)),
                    (),
                )
                self.assertEqual(
                    tuple(sorted(annotated_fields & frontend_required)),
                    (),
                )
                self.assertTrue(
                    all(reason for reason in looseness_annotations.values())
                )

    def test_frontend_default_field_annotations_are_current(self) -> None:
        for (
            backend_schema,
            frontend_contract,
        ), default_annotations in INTENTIONAL_FRONTEND_DEFAULT_FIELDS.items():
            with self.subTest(
                backend_schema=backend_schema.__name__,
                frontend_contract=frontend_contract,
            ):
                self.assertIn(
                    (backend_schema, frontend_contract),
                    OPENAPI_REQUIRED_FIELD_PARITY_BY_FRONTEND_SCHEMA,
                )
                frontend_required = set(
                    OPENAPI_REQUIRED_FIELD_PARITY_BY_FRONTEND_SCHEMA[
                        (backend_schema, frontend_contract)
                    ]
                )

                for field_name, reason in default_annotations.items():
                    with self.subTest(field=field_name):
                        self.assertTrue(reason)
                        self.assertIn(field_name, backend_schema.model_fields)
                        self.assertNotIn(field_name, frontend_required)
                        self.assertFalse(
                            backend_schema.model_fields[field_name].is_required()
                        )

    def test_capabilities_schema_defaults_data_source_and_upload_placeholders(
        self,
    ) -> None:
        capabilities = schemas.CapabilitiesResponse(
            authMode="none",
            trainingEnabled=False,
            logDeletionEnabled=False,
        )

        self.assertEqual(capabilities.trainingCancellationCapability, "unsupported")
        self.assertEqual(capabilities.uploadsEnabled, False)
        self.assertIsNone(capabilities.maxUploadSize)
        self.assertEqual(capabilities.dataSourcesEnabled, False)
        self.assertEqual(capabilities.dataSources, [])


class ApiIntegrationContractTests(unittest.TestCase):
    def test_capabilities_endpoint_exposes_local_defaults(self) -> None:
        import httpx

        from viewer.backend.api import app
        from viewer.backend.core.config import get_viewer_api_settings
        from viewer.backend.training_cgroups import requested_cancellation_capability

        async def call_api() -> httpx.Response:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.get("/capabilities")

        response = asyncio.run(call_api())

        self.assertEqual(response.request.url.path, "/capabilities")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "authMode": "none",
                "trainingEnabled": False,
                "trainingCancellationCapability": requested_cancellation_capability(
                    get_viewer_api_settings().training_cancellation_mode
                ),
                "logDeletionEnabled": False,
                "configSnapshotsEnabled": False,
                "historicalLogsEnabled": True,
                "liveMonitorDataEnabled": True,
                "historicalMonitorDataEnabled": True,
                "uploadsEnabled": False,
                "maxUploadSize": get_viewer_api_settings().max_upload_size,
                "dataSourcesEnabled": False,
                "dataSources": [],
            },
        )

    def test_capabilities_endpoint_reports_local_mutation_features(self) -> None:
        import httpx

        from viewer.backend.api import ViewerApiSettings, create_app

        app = create_app(ViewerApiSettings(allow_unsafe_local_mutations=True))

        async def call_api() -> httpx.Response:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.get("/capabilities")

        response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["trainingEnabled"])
        self.assertTrue(response.json()["logDeletionEnabled"])
        self.assertTrue(response.json()["configSnapshotsEnabled"])

    def test_model_dataset_endpoint_exposes_path_free_dataset_metadata(self) -> None:
        import httpx

        from viewer.backend.api import app

        async def call_api() -> httpx.Response:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.get("/models/linears/linear/datasets")

        response = asyncio.run(call_api())

        self.assertEqual(response.request.url.path, "/models/linears/linear/datasets")
        self.assertEqual(response.status_code, 200)

        payload = response.json()
        self.assertEqual(tuple(payload), ("modelType", "model", "datasets"))
        self.assertEqual(payload["modelType"], "linears")
        self.assertEqual(payload["model"], "linear")
        self.assertTrue(payload["datasets"])

        for dataset in payload["datasets"]:
            with self.subTest(dataset=dataset.get("name")):
                self.assertEqual(
                    tuple(dataset),
                    ("name", "label", "inputDim", "outputDim"),
                )
                self.assertTrue(
                    PATH_LIKE_DATASET_FIELDS.isdisjoint(dataset),
                    f"Dataset payload exposed path-like fields: {dataset}",
                )

        dataset_by_name = {dataset["name"]: dataset for dataset in payload["datasets"]}
        self.assertIn("Mnist", dataset_by_name)
        self.assertEqual(dataset_by_name["Mnist"]["inputDim"], 784)
        self.assertEqual(dataset_by_name["Mnist"]["outputDim"], 10)

    def test_api_health_and_inspect(self) -> None:
        import httpx

        from viewer.backend.api import app
        from viewer.backend.api.v1.routers.inspection import inspect
        from viewer.backend.services.inspection import InspectionService

        async def call_api() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                health = await client.get("/health")
                monitors = await client.get("/models/linears/linear/monitors")
                search_space = await client.get(
                    "/models/linears/linear/search-space?preset=baseline"
                )
                return health, monitors, search_space

        health_response, monitors_response, search_space_response = asyncio.run(
            call_api()
        )
        response = asyncio.run(
            inspect(
                schemas.InspectRequest(
                    modelType="linears",
                    model="linear",
                    preset="baseline",
                    dataset="Mnist",
                    overrides={"hidden_dim": "128"},
                ),
                InspectionService(),
            )
        )
        self.assertEqual(health_response.json(), {"status": "ok"})
        self.assertEqual(monitors_response.status_code, 200)
        self.assertEqual(monitors_response.json()["monitors"][0]["name"], "linear")
        self.assertEqual(search_space_response.status_code, 200)
        search_space_payload = search_space_response.json()
        self.assertIn(
            "hidden_dim", {axis["key"] for axis in search_space_payload["axes"]}
        )
        payload = response.model_dump(mode="json")
        self.assertEqual(payload["modelType"], "linears")
        self.assertEqual(payload["model"], "linear")
        self.assertTrue(payload["nodes"])
        self.assertTrue(payload["edges"])
        self.assertIn("parameterCount", payload)
        self.assertIn("parameterSizeBytes", payload)
        self.assertIn("parameterCount", payload["nodes"][0])
        self.assertIn("parameterSizeBytes", payload["nodes"][0])

    def test_inspect_rejects_path_like_dataset_input(self) -> None:
        from viewer.backend.api.v1.routers.inspection import inspect
        from viewer.backend.inspector.errors import InspectorError
        from viewer.backend.services.inspection import InspectionService

        with self.assertRaises(InspectorError) as raised:
            asyncio.run(
                inspect(
                    schemas.InspectRequest(
                        modelType="linears",
                        model="linear",
                        preset="baseline",
                        dataset="./Mnist",
                        overrides={},
                    ),
                    InspectionService(),
                ),
            )

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("./Mnist", raised.exception.detail)
        self.assertIn("filesystem path", raised.exception.detail)
        self.assertIn("server-known dataset name", raised.exception.detail)

    def test_log_scalars_rejects_more_than_max_request_run_ids(self) -> None:
        import httpx

        from viewer.backend.api import app

        async def call_api() -> httpx.Response:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.post(
                    "/logs/scalars",
                    json={
                        "runIds": [f"run-{index}" for index in range(94)],
                        "tags": ["train/loss"],
                        "maxPoints": 500,
                        "sampling": "tail",
                    },
                )

        response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 422)
        detail = response.json()["detail"][0]
        self.assertEqual(detail["type"], "too_long")
        self.assertEqual(detail["loc"], ["body", "runIds"])
        self.assertEqual(detail["ctx"]["max_length"], 50)
        self.assertEqual(detail["ctx"]["actual_length"], 94)

    def test_api_dependency_overrides_can_replace_route_services(self) -> None:
        import httpx

        from viewer.backend.api import create_app
        from viewer.backend.dependencies import get_model_catalog_service

        class FakeModelCatalogService:
            def list_models(self) -> list[dict[str, str]]:
                return [{"modelType": "override", "model": "model"}]

        async def override_model_catalog_service() -> FakeModelCatalogService:
            return FakeModelCatalogService()

        test_app = create_app()
        test_app.dependency_overrides[get_model_catalog_service] = (
            override_model_catalog_service
        )

        async def call_api() -> httpx.Response:
            transport = httpx.ASGITransport(app=test_app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.get("/models")

        response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"models": [{"modelType": "override", "model": "model"}]},
        )

    def test_api_inspector_errors_use_shared_handler(self) -> None:
        import httpx

        from viewer.backend.api import app

        async def call_api() -> httpx.Response:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.get("/models/unknown/model/presets")

        response = asyncio.run(call_api())
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unknown model", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
