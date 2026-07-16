from __future__ import annotations

from typing import Any

from emperor_workbench.model_packages import ModelPackageIdentity
from emperor_workbench.run_history import (
    ActiveLogRunDeleteBlocker,
    LogArchiveImportResult,
    LogCheckpoint,
    LogExperiment,
    LogExperimentDeleteResult,
    LogExperimentPage,
    LogImageSummary,
    LogMedia,
    LogRun,
    LogRunArtifact,
    LogRunArtifacts,
    LogRunDeleteCandidate,
    LogRunDeletePlan,
    LogRunDeleteResult,
    LogRunFacets,
    LogRunPage,
    LogRunTags,
    LogScalarPoint,
    LogScalarSeries,
    LogTextSummary,
)
from emperor_workbench.tensorboard import (
    Histogram,
    ImageSummary,
    MonitorData,
    ParameterChannelStatus,
    ParameterStatus,
    ScalarPoint,
    ScalarSeries,
)

LOG_METADATA_RESPONSE_LIMIT = 500


def _model_identity(model_id: str) -> dict[str, str]:
    identity = ModelPackageIdentity.from_id(model_id)
    return {"modelType": identity.model_type, "model": identity.model}


def log_run_to_payload(run: LogRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "group": run.group,
        "experiment": run.experiment,
        **_model_identity(run.model),
        "preset": run.preset,
        "experimentTask": run.experiment_task,
        "dataset": run.dataset,
        "runName": run.run_name,
        "timestamp": run.timestamp,
        "version": run.version,
        "relativePath": run.relative_path,
        "hasResult": run.has_result,
        "eventFileCount": run.event_file_count,
        "checkpointCount": run.checkpoint_count,
        "hasHparams": run.has_hparams,
        "hasLayerMonitorData": run.has_layer_monitor_data,
        "metrics": run.metrics,
    }


def _facets_to_payload(facets: LogRunFacets) -> dict[str, Any]:
    return {
        "experiments": [
            {
                "experiment": experiment.experiment,
                "runCount": experiment.run_count,
                "datasets": [
                    {"value": facet.value, "count": facet.count}
                    for facet in experiment.datasets
                ],
                "models": [
                    {
                        **_model_identity(facet.model),
                        "count": facet.count,
                    }
                    for facet in experiment.models
                ],
                "presets": [
                    {"value": facet.value, "count": facet.count}
                    for facet in experiment.presets
                ],
            }
            for experiment in facets.experiments
        ]
    }


def log_run_page_to_payload(page: LogRunPage) -> dict[str, Any]:
    return {
        "runs": [log_run_to_payload(run) for run in page.runs],
        "total": page.total,
        "limit": page.limit,
        "offset": page.offset,
        "hasMore": page.has_more,
        "facets": _facets_to_payload(page.facets),
    }


def _experiment_to_payload(experiment: LogExperiment) -> dict[str, Any]:
    return {
        "experiment": experiment.experiment,
        "runCount": experiment.run_count,
        "relativePath": experiment.relative_path,
    }


def log_experiment_page_to_payload(page: LogExperimentPage) -> dict[str, Any]:
    return {
        "experiments": [
            _experiment_to_payload(experiment) for experiment in page.experiments
        ],
        "total": page.total,
        "limit": page.limit,
        "offset": page.offset,
        "hasMore": page.has_more,
    }


def log_checkpoint_to_payload(checkpoint: LogCheckpoint) -> dict[str, Any]:
    return {
        "id": checkpoint.id,
        "runId": checkpoint.run_id,
        "filename": checkpoint.filename,
        "relativePath": checkpoint.relative_path,
        "epoch": checkpoint.epoch,
        "step": checkpoint.step,
        "sizeBytes": checkpoint.size_bytes,
        "modifiedAt": checkpoint.modified_at,
    }


def log_checkpoints_to_payload(checkpoints: list[LogCheckpoint]) -> dict[str, Any]:
    returned = checkpoints[:LOG_METADATA_RESPONSE_LIMIT]
    truncated = len(checkpoints) > len(returned)
    return {
        "sourceItemCount": len(checkpoints),
        "returnedItemCount": len(returned),
        "truncated": truncated,
        "truncationReason": (
            f"checkpoint metadata capped at {LOG_METADATA_RESPONSE_LIMIT} rows"
            if truncated
            else None
        ),
        "checkpoints": [log_checkpoint_to_payload(item) for item in returned],
    }


def _artifact_to_payload(artifact: LogRunArtifact) -> dict[str, Any]:
    return {
        "id": artifact.id,
        "kind": artifact.kind,
        "label": artifact.label,
        "relativePath": artifact.relative_path,
        "sizeBytes": artifact.size_bytes,
        "modifiedAt": artifact.modified_at,
    }


def log_run_artifacts_to_payload(details: LogRunArtifacts) -> dict[str, Any]:
    source_item_count = len(details.artifacts) + len(details.checkpoints)
    returned_artifacts = details.artifacts[:LOG_METADATA_RESPONSE_LIMIT]
    remaining = max(0, LOG_METADATA_RESPONSE_LIMIT - len(returned_artifacts))
    returned_checkpoints = details.checkpoints[:remaining]
    returned_item_count = len(returned_artifacts) + len(returned_checkpoints)
    response_truncated = source_item_count > returned_item_count
    truncated = bool(details.truncation_reasons) or response_truncated
    return {
        "runId": details.run_id,
        "params": details.params,
        "metrics": details.metrics,
        "sourceItemCount": source_item_count,
        "returnedItemCount": returned_item_count,
        "truncated": truncated,
        "truncationReason": (
            details.truncation_reasons[0]
            if details.truncation_reasons
            else (
                f"artifact metadata capped at {LOG_METADATA_RESPONSE_LIMIT} rows"
                if response_truncated
                else None
            )
        ),
        "artifacts": [_artifact_to_payload(item) for item in returned_artifacts],
        "checkpoints": [
            log_checkpoint_to_payload(item) for item in returned_checkpoints
        ],
    }


def log_archive_import_to_payload(result: LogArchiveImportResult) -> dict[str, Any]:
    return {
        "extractedFileCount": result.extracted_file_count,
        "skippedFileCount": result.skipped_file_count,
        "destinationRoot": result.destination_root,
    }


def log_run_tags_to_payload(tags: LogRunTags) -> dict[str, Any]:
    return {
        "runId": tags.run_id,
        "hasLayerMonitorData": tags.has_layer_monitor_data,
        "scalarTags": list(tags.scalar_tags),
        "histogramTags": list(tags.histogram_tags),
        "imageTags": list(tags.image_tags),
        "textTags": list(tags.text_tags),
        "eventBytes": tags.event_bytes,
        "skippedEventFiles": tags.skipped_event_files,
        "truncated": tags.truncated,
        "truncationReason": tags.truncation_reason,
        "sourceItemCount": tags.source_item_count,
        "returnedItemCount": tags.returned_item_count,
    }


def _scalar_point_to_payload(
    point: LogScalarPoint | ScalarPoint,
) -> dict[str, Any]:
    return {"step": point.step, "wallTime": point.wall_time, "value": point.value}


def log_scalar_series_to_payload(series: LogScalarSeries) -> dict[str, Any]:
    return {
        "runId": series.run_id,
        "tag": series.tag,
        "points": [_scalar_point_to_payload(point) for point in series.points],
        "sourcePointCount": series.source_point_count,
        "truncated": series.truncated,
    }


def _image_summary_to_payload(summary: LogImageSummary) -> dict[str, Any]:
    return {
        "runId": summary.run_id,
        "tag": summary.tag,
        "step": summary.step,
        "wallTime": summary.wall_time,
        "mimeType": summary.mime_type,
        "dataUrl": summary.data_url,
        "eventBytes": summary.event_bytes,
        "sourceItemCount": summary.source_item_count,
        "returnedItemCount": summary.returned_item_count,
        "truncated": summary.truncated,
        "truncationReason": summary.truncation_reason,
    }


def _text_summary_to_payload(summary: LogTextSummary) -> dict[str, Any]:
    return {
        "runId": summary.run_id,
        "tag": summary.tag,
        "step": summary.step,
        "wallTime": summary.wall_time,
        "text": summary.text,
        "eventBytes": summary.event_bytes,
        "sourceItemCount": summary.source_item_count,
        "returnedItemCount": summary.returned_item_count,
        "truncated": summary.truncated,
        "truncationReason": summary.truncation_reason,
    }


def log_media_to_payload(media: LogMedia) -> dict[str, Any]:
    return {
        "eventBytes": media.event_bytes,
        "skippedEventFiles": media.skipped_event_files,
        "sourceItemCount": media.source_item_count,
        "returnedItemCount": media.returned_item_count,
        "truncated": media.truncated,
        "truncationReason": media.truncation_reason,
        "images": [_image_summary_to_payload(item) for item in media.images],
        "texts": [_text_summary_to_payload(item) for item in media.texts],
    }


def _monitor_scalar_series_to_payload(
    series: ScalarSeries,
) -> dict[str, Any]:
    return {
        "tag": series.tag,
        "label": series.label,
        "points": [_scalar_point_to_payload(point) for point in series.points],
        "sourceItemCount": series.source_item_count,
        "returnedItemCount": series.returned_item_count,
        "truncated": series.truncated,
        "truncationReason": series.truncation_reason,
    }


def _histogram_to_payload(histogram: Histogram) -> dict[str, Any]:
    return {
        "tag": histogram.tag,
        "step": histogram.step,
        "wallTime": histogram.wall_time,
        "buckets": [
            {"left": bucket.left, "right": bucket.right, "count": bucket.count}
            for bucket in histogram.buckets
        ],
        "sourceItemCount": histogram.source_item_count,
        "returnedItemCount": histogram.returned_item_count,
        "truncated": histogram.truncated,
        "truncationReason": histogram.truncation_reason,
    }


def _monitor_image_to_payload(image: ImageSummary) -> dict[str, Any]:
    return {
        "tag": image.tag,
        "step": image.step,
        "wallTime": image.wall_time,
        "mimeType": image.mime_type,
        "dataUrl": image.data_url,
        "eventBytes": image.event_bytes,
        "sourceItemCount": image.source_item_count,
        "returnedItemCount": image.returned_item_count,
        "truncated": image.truncated,
        "truncationReason": image.truncation_reason,
    }


def monitor_data_to_payload(data: MonitorData) -> dict[str, Any]:
    return {
        "jobId": data.job_id,
        "nodePath": data.node_path,
        "preset": data.preset,
        "dataset": data.dataset,
        "logDir": data.log_dir,
        "eventBytes": data.event_bytes,
        "skippedEventFiles": data.skipped_event_files,
        "truncated": data.truncated,
        "truncationReason": data.truncation_reason,
        "sourceItemCount": data.source_item_count,
        "returnedItemCount": data.returned_item_count,
        "scalarSeries": [
            _monitor_scalar_series_to_payload(series) for series in data.scalar_series
        ],
        "histograms": [
            _histogram_to_payload(histogram) for histogram in data.histograms
        ],
        "images": [_monitor_image_to_payload(image) for image in data.images],
    }


def _parameter_channel_to_payload(
    channel: ParameterChannelStatus,
) -> dict[str, Any]:
    return {
        "status": channel.status,
        "metric": channel.metric,
        "lastStep": channel.last_step,
        "observedPoints": channel.observed_points,
    }


def parameter_status_to_payload(status: ParameterStatus) -> dict[str, Any]:
    return {
        "sourceId": status.source_id,
        "preset": status.preset,
        "dataset": status.dataset,
        "logDir": status.log_dir,
        "eventBytes": status.event_bytes,
        "skippedEventFiles": status.skipped_event_files,
        "truncated": status.truncated,
        "truncationReason": status.truncation_reason,
        "sourceItemCount": status.source_item_count,
        "returnedItemCount": status.returned_item_count,
        "nodes": [
            {
                "nodePath": node.node_path,
                "weights": _parameter_channel_to_payload(node.weights),
                "bias": _parameter_channel_to_payload(node.bias),
            }
            for node in status.nodes
        ],
    }


def log_experiment_delete_to_payload(
    result: LogExperimentDeleteResult,
) -> dict[str, Any]:
    return {
        "experiment": result.experiment,
        "deletedRunIds": list(result.deleted_run_ids),
        "deletedRunCount": result.deleted_run_count,
        "deletedRelativePath": result.deleted_relative_path,
    }


def _delete_candidate_to_payload(
    candidate: LogRunDeleteCandidate,
) -> dict[str, Any]:
    return {
        "id": candidate.id,
        "experiment": candidate.experiment,
        **_model_identity(candidate.model),
        "preset": candidate.preset,
        "dataset": candidate.dataset,
        "runName": candidate.run_name,
        "version": candidate.version,
        "relativePath": candidate.relative_path,
    }


def _delete_blocker_to_payload(
    blocker: ActiveLogRunDeleteBlocker,
) -> dict[str, Any]:
    return {
        "id": blocker.id,
        "logFolder": blocker.log_folder,
        "status": blocker.status,
    }


def _delete_plan_fields(
    candidates: tuple[LogRunDeleteCandidate, ...],
    *,
    blockers: tuple[ActiveLogRunDeleteBlocker, ...],
    can_delete: bool,
) -> dict[str, Any]:
    experiments = sorted({candidate.experiment for candidate in candidates})
    datasets = sorted({candidate.dataset for candidate in candidates})
    models = sorted({candidate.model for candidate in candidates})
    presets = sorted({candidate.preset for candidate in candidates})
    run_ids = sorted({candidate.id for candidate in candidates})
    returned = candidates[:LOG_METADATA_RESPONSE_LIMIT]
    truncated = len(candidates) > len(returned)
    return {
        "candidateCount": len(candidates),
        "sourceItemCount": len(candidates),
        "returnedItemCount": len(returned),
        "truncated": truncated,
        "truncationReason": (
            f"delete candidates capped at {LOG_METADATA_RESPONSE_LIMIT} rows"
            if truncated
            else None
        ),
        "counts": {
            "runs": len(candidates),
            "experiments": len(experiments),
            "datasets": len(datasets),
            "models": len(models),
            "presets": len(presets),
        },
        "affected": {
            "experiments": experiments,
            "datasets": datasets,
            "models": [_model_identity(model) for model in models],
            "presets": presets,
            "runIds": run_ids,
        },
        "candidates": [_delete_candidate_to_payload(item) for item in returned],
        "blockedByActiveJobs": [
            _delete_blocker_to_payload(blocker) for blocker in blockers
        ],
        "canDelete": can_delete,
    }


def log_run_delete_plan_to_payload(plan: LogRunDeletePlan) -> dict[str, Any]:
    return _delete_plan_fields(
        plan.candidates,
        blockers=plan.blocked_by_active_jobs,
        can_delete=plan.can_delete,
    )


def log_run_delete_result_to_payload(result: LogRunDeleteResult) -> dict[str, Any]:
    return {
        "deletedRunIds": list(result.deleted_run_ids),
        "deletedRunCount": len(result.deleted_run_ids),
        "deletedRelativePaths": list(result.deleted_relative_paths),
        **_delete_plan_fields(result.candidates, blockers=(), can_delete=True),
    }


__all__ = [
    "LOG_METADATA_RESPONSE_LIMIT",
    "log_archive_import_to_payload",
    "log_checkpoints_to_payload",
    "log_experiment_delete_to_payload",
    "log_experiment_page_to_payload",
    "log_media_to_payload",
    "monitor_data_to_payload",
    "parameter_status_to_payload",
    "log_run_artifacts_to_payload",
    "log_run_delete_plan_to_payload",
    "log_run_delete_result_to_payload",
    "log_run_page_to_payload",
    "log_run_tags_to_payload",
    "log_scalar_series_to_payload",
]
