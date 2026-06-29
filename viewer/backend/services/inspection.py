"""Model inspection use cases."""

from __future__ import annotations

from typing import Any

from models.dataset_naming import normalize_dataset_name

from viewer.backend.inspector.checkpoint_shapes import (
    CheckpointGraphShapes,
    load_checkpoint_graph_shapes,
)
from viewer.backend.inspector.discovery import load_model_parts
from viewer.backend.inspector.errors import InspectorError
from viewer.backend.inspector.overrides import parse_override_mapping
from viewer.backend.inspector.schema import preset_locks
from viewer.backend.model_identity import require_model_id
from viewer.backend.repositories.log_runs import LogRunRepository
from viewer.backend.training_monitor_locator import normalize_preset_token

_ADAPTIVE_GENERATOR_COUNT_KEYS = {
    "adaptive_generator_stack_num_layers",
}
_ADAPTIVE_FLAG_KEYS = {
    "weight_option_flag",
    "bias_option_flag",
    "diagonal_option_flag",
    "mask_option_flag",
}
_ADAPTIVE_OPTION_KEYS = {
    "weight_option",
    "bias_option",
    "diagonal_option",
    "row_mask_option",
    "input_layer_weight_option",
    "input_layer_bias_option",
    "input_layer_diagonal_option",
    "input_layer_row_mask_option",
    "output_layer_weight_option",
    "output_layer_bias_option",
    "output_layer_diagonal_option",
    "output_layer_row_mask_option",
}


def _normalized_dataset(dataset: str | None) -> str | None:
    return normalize_dataset_name(dataset) if dataset else None


class InspectionService:
    def __init__(self, log_runs: LogRunRepository | None = None) -> None:
        self._log_runs = log_runs

    def inspect(
        self,
        *,
        model_type: str,
        model: str,
        preset: str,
        overrides: dict[str, Any],
        dataset: str | None,
        log_run_id: str | None = None,
    ) -> dict[str, Any]:
        from viewer.backend.inspector.service import inspect_model

        model_id = require_model_id(model_type, model)
        parts = load_model_parts(model_id)
        request_overrides = parse_override_mapping(parts.config_module, overrides)
        checkpoint_shapes: CheckpointGraphShapes | None = None
        checkpoint_structural_fallback = False
        effective_overrides = request_overrides
        if log_run_id:
            saved_run_overrides = parse_override_mapping(
                parts.config_module,
                self._saved_run_overrides(
                    log_run_id=log_run_id,
                    model_id=model_id,
                    preset=preset,
                    dataset=dataset,
                ),
                ignore_unknown=True,
            )
            checkpoint_shapes = self._checkpoint_shapes_for_log_run(log_run_id)
            checkpoint_overrides, checkpoint_parse_fallback = (
                _checkpoint_overrides(
                    model_id=model_id,
                    preset=preset,
                    config_module=parts.config_module,
                    checkpoint_shapes=checkpoint_shapes,
                    saved_run_overrides=saved_run_overrides,
                    request_overrides=request_overrides,
                )
            )
            checkpoint_structural_fallback = checkpoint_parse_fallback
            effective_overrides = {
                **saved_run_overrides,
                **checkpoint_overrides,
                **request_overrides,
            }

        try:
            result = inspect_model(
                model_id,
                preset,
                dataset=dataset,
                parsed_overrides=effective_overrides,
            )
        except InspectorError as exc:
            if not log_run_id or checkpoint_shapes is None or not checkpoint_overrides:
                raise
            fallback_overrides = {
                **saved_run_overrides,
                **request_overrides,
            }
            try:
                result = inspect_model(
                    model_id,
                    preset,
                    dataset=dataset,
                    parsed_overrides=fallback_overrides,
                )
            except InspectorError as fallback_exc:
                raise exc from fallback_exc
            checkpoint_structural_fallback = True
        if checkpoint_shapes is not None:
            _patch_checkpoint_shape_details(
                result,
                checkpoint_shapes,
                structural_fallback=checkpoint_structural_fallback,
            )
        return result

    def _checkpoint_shapes_for_log_run(
        self,
        log_run_id: str,
    ) -> CheckpointGraphShapes | None:
        if self._log_runs is None:
            raise InspectorError("Log run inspection is not configured.")
        checkpoint_paths = list(
            reversed(self._log_runs.checkpoint_paths_for_run(log_run_id))
        )
        return load_checkpoint_graph_shapes(checkpoint_paths)

    def _saved_run_overrides(
        self,
        *,
        log_run_id: str,
        model_id: str,
        preset: str,
        dataset: str | None,
    ) -> dict[str, Any]:
        if self._log_runs is None:
            raise InspectorError("Log run inspection is not configured.")

        run = next(
            (
                candidate
                for candidate in self._log_runs.list_runs()
                if candidate.id == log_run_id
            ),
            None,
        )
        if run is None:
            raise InspectorError(f"Unknown log run id: {log_run_id}")
        if run.model != model_id:
            raise InspectorError(
                f"Log run '{log_run_id}' belongs to model '{run.model}', "
                f"not '{model_id}'."
            )
        if normalize_preset_token(run.preset) != normalize_preset_token(preset):
            raise InspectorError(
                f"Log run '{log_run_id}' belongs to preset '{run.preset}', "
                f"not '{preset}'."
            )
        if dataset and _normalized_dataset(run.dataset) != _normalized_dataset(
            dataset
        ):
            raise InspectorError(
                f"Log run '{log_run_id}' belongs to dataset '{run.dataset}', "
                f"not '{dataset}'."
            )

        artifacts = self._log_runs.artifacts_for_run(log_run_id)
        params = artifacts.get("params")
        return dict(params) if isinstance(params, dict) else {}


def _patch_checkpoint_shape_details(
    result: dict[str, Any],
    checkpoint_shapes: CheckpointGraphShapes,
    *,
    structural_fallback: bool,
) -> None:
    for node in result.get("nodes", []):
        if not isinstance(node, dict):
            continue
        path = node.get("path")
        if not isinstance(path, str):
            continue
        details = node.setdefault("details", {})
        if not isinstance(details, dict):
            continue
        checkpoint_details = checkpoint_shapes.parameter_shapes.get(path)
        if checkpoint_details is not None:
            details.update(checkpoint_details)
        tensor_count = checkpoint_shapes.coverage_counts.get(path, 0)
        checkpoint_detail: dict[str, Any] = {
            "status": "matched" if tensor_count > 0 else "missing",
            "tensorCount": tensor_count,
        }
        if tensor_count == 0:
            checkpoint_detail["reason"] = "noCheckpointTensor"
        if path == "model" and structural_fallback:
            checkpoint_detail["reason"] = "structuralFallback"
            fallback_reasons = (
                checkpoint_shapes.diagnostics.structural_fallback_reasons
            )
            if fallback_reasons:
                checkpoint_detail["fallbackReasons"] = list(fallback_reasons)
        details["checkpoint"] = checkpoint_detail


def _checkpoint_overrides(
    *,
    model_id: str,
    preset: str,
    config_module: Any,
    checkpoint_shapes: CheckpointGraphShapes | None,
    saved_run_overrides: dict[str, Any],
    request_overrides: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    if checkpoint_shapes is None:
        return {}, False

    structural_fallback = bool(
        checkpoint_shapes.diagnostics.structural_fallback_reasons
    )
    override_candidates = _adaptive_safe_checkpoint_candidates(
        model_id=model_id,
        preset=preset,
        checkpoint_overrides=checkpoint_shapes.config_overrides,
        saved_run_overrides=saved_run_overrides,
        request_overrides=request_overrides,
    )
    if len(override_candidates) != len(checkpoint_shapes.config_overrides):
        structural_fallback = True

    try:
        parsed = parse_override_mapping(
            config_module,
            override_candidates,
            ignore_unknown=True,
        )
    except InspectorError:
        return {}, True

    locks = preset_locks(model_id, preset)
    unlocked = {key: value for key, value in parsed.items() if key not in locks}
    if len(unlocked) != len(parsed):
        structural_fallback = True
    return unlocked, structural_fallback


def _adaptive_safe_checkpoint_candidates(
    *,
    model_id: str,
    preset: str,
    checkpoint_overrides: dict[str, Any],
    saved_run_overrides: dict[str, Any],
    request_overrides: dict[str, Any],
) -> dict[str, Any]:
    if not set(checkpoint_overrides) & _ADAPTIVE_GENERATOR_COUNT_KEYS:
        return dict(checkpoint_overrides)
    if _has_selected_adaptive_option(
        model_id=model_id,
        preset=preset,
        saved_run_overrides=saved_run_overrides,
        request_overrides=request_overrides,
    ):
        return dict(checkpoint_overrides)
    return {
        key: value
        for key, value in checkpoint_overrides.items()
        if key not in _ADAPTIVE_GENERATOR_COUNT_KEYS
    }


def _has_selected_adaptive_option(
    *,
    model_id: str,
    preset: str,
    saved_run_overrides: dict[str, Any],
    request_overrides: dict[str, Any],
) -> bool:
    locks = preset_locks(model_id, preset)
    lock_values = {
        key: getattr(lock, "value", None) for key, lock in locks.items()
    }
    for source in (lock_values, saved_run_overrides, request_overrides):
        for key in _ADAPTIVE_FLAG_KEYS:
            if source.get(key) is True:
                return True
        for key in _ADAPTIVE_OPTION_KEYS:
            if source.get(key) is not None:
                return True
    return False
