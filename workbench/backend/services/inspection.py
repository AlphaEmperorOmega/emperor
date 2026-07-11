"""Model inspection use cases."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from emperor.model_packages import normalize_dataset_name

from workbench.backend.inspector.errors import InspectorError
from workbench.backend.model_identity import (
    normalize_preset_token,
    require_model_package,
)
from workbench.backend.run_history import (
    HistoricalInspectionContext,
    HistoricalInspectionSource,
)

if TYPE_CHECKING:
    from emperor.model_packages import ModelPackage

    from workbench.backend.inspector.checkpoint_shapes import CheckpointGraphShapes

_ADAPTIVE_GENERATOR_COUNT_KEYS = {
    "adaptive_generator_stack_num_layers",
}
_ADAPTIVE_FLAG_KEYS = {
    "weight_option_flag",
    "bias_option_flag",
    "diagonal_option_flag",
    "mask_option_flag",
    "router_weight_option_flag",
    "router_bias_option_flag",
    "router_diagonal_option_flag",
    "router_mask_option_flag",
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
    "router_weight_option",
    "router_bias_option",
    "router_diagonal_option",
    "router_row_mask_option",
}


def _normalized_dataset(dataset: str | None) -> str | None:
    return normalize_dataset_name(dataset) if dataset else None


class InspectionService:
    def __init__(
        self,
        historical_runs: HistoricalInspectionSource | None = None,
    ) -> None:
        self._historical_runs = historical_runs

    def inspect(
        self,
        *,
        model_type: str,
        model: str,
        preset: str,
        overrides: dict[str, Any],
        dataset: str | None,
        experiment_task: str | None = None,
        log_run_id: str | None = None,
    ) -> dict[str, Any]:
        from emperor.inspection import (
            InspectionRequest,
            ParsedOverrides,
            parse_overrides,
        )
        from emperor.inspection import (
            inspect_model as inspect_model_semantically,
        )

        from workbench.backend.inspection_errors import call_inspection
        from workbench.backend.inspection_serialization import inspection_result_payload

        package = require_model_package(model_type, model)
        model_id = package.catalog_key
        request_overrides = call_inspection(
            parse_overrides,
            package,
            overrides,
        ).values
        checkpoint_shapes: CheckpointGraphShapes | None = None
        checkpoint_structural_fallback = False
        effective_overrides = request_overrides
        if log_run_id:
            historical_context = self._historical_context(log_run_id)
            saved_run_overrides = call_inspection(
                parse_overrides,
                package,
                self._saved_run_overrides(
                    context=historical_context,
                    model_id=model_id,
                    preset=preset,
                    dataset=dataset,
                ),
                ignore_unknown=True,
            ).values
            checkpoint_shapes = self._checkpoint_shapes_for_context(
                historical_context
            )
            checkpoint_overrides, checkpoint_parse_fallback = (
                _checkpoint_overrides(
                    model_id=model_id,
                    preset=preset,
                    package=package,
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
            semantic_result = call_inspection(
                inspect_model_semantically,
                package,
                InspectionRequest(
                    preset=preset,
                    overrides=ParsedOverrides(effective_overrides),
                    dataset=dataset,
                    experiment_task=experiment_task,
                ),
            )
            result = inspection_result_payload(semantic_result)
        except InspectorError as exc:
            if not log_run_id or checkpoint_shapes is None or not checkpoint_overrides:
                raise
            fallback_overrides = {
                **saved_run_overrides,
                **request_overrides,
            }
            try:
                fallback_result = call_inspection(
                    inspect_model_semantically,
                    package,
                    InspectionRequest(
                        preset=preset,
                        overrides=ParsedOverrides(fallback_overrides),
                        dataset=dataset,
                        experiment_task=experiment_task,
                    ),
                )
                result = inspection_result_payload(fallback_result)
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

    def _historical_context(
        self,
        log_run_id: str,
    ) -> HistoricalInspectionContext:
        if self._historical_runs is None:
            raise InspectorError("Log run inspection is not configured.")
        return self._historical_runs.inspection_context(log_run_id)

    def _checkpoint_shapes_for_context(
        self,
        context: HistoricalInspectionContext,
    ) -> CheckpointGraphShapes | None:
        from workbench.backend.inspector.checkpoint_shapes import (
            load_checkpoint_graph_shapes,
        )

        checkpoint_paths = list(reversed(context.checkpoint_paths))
        return load_checkpoint_graph_shapes(checkpoint_paths)

    def _saved_run_overrides(
        self,
        *,
        context: HistoricalInspectionContext,
        model_id: str,
        preset: str,
        dataset: str | None,
    ) -> dict[str, Any]:
        if context.model != model_id:
            raise InspectorError(
                f"Log run '{context.run_id}' belongs to model '{context.model}', "
                f"not '{model_id}'."
            )
        if normalize_preset_token(context.preset) != normalize_preset_token(preset):
            raise InspectorError(
                f"Log run '{context.run_id}' belongs to preset '{context.preset}', "
                f"not '{preset}'."
            )
        if dataset and _normalized_dataset(
            context.dataset
        ) != _normalized_dataset(dataset):
            raise InspectorError(
                f"Log run '{context.run_id}' belongs to dataset '{context.dataset}', "
                f"not '{dataset}'."
            )

        return dict(context.params)


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
    package: ModelPackage,
    checkpoint_shapes: CheckpointGraphShapes | None,
    saved_run_overrides: dict[str, Any],
    request_overrides: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    from emperor.inspection import InspectionError, parse_overrides, preset_locks

    from workbench.backend.inspection_errors import call_inspection

    if checkpoint_shapes is None:
        return {}, False

    structural_fallback = bool(
        checkpoint_shapes.diagnostics.structural_fallback_reasons
    )
    override_candidates = _adaptive_safe_checkpoint_candidates(
        package=package,
        model_id=model_id,
        preset=preset,
        checkpoint_overrides=checkpoint_shapes.config_overrides,
        saved_run_overrides=saved_run_overrides,
        request_overrides=request_overrides,
    )
    if len(override_candidates) != len(checkpoint_shapes.config_overrides):
        structural_fallback = True

    try:
        parsed = parse_overrides(
            package,
            override_candidates,
            ignore_unknown=True,
        ).values
    except InspectionError:
        return {}, True

    locks = call_inspection(preset_locks, package, preset)
    unlocked = {key: value for key, value in parsed.items() if key not in locks}
    if len(unlocked) != len(parsed):
        structural_fallback = True
    return unlocked, structural_fallback


def _adaptive_safe_checkpoint_candidates(
    *,
    package: ModelPackage,
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
        package=package,
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
    package: ModelPackage,
    model_id: str,
    preset: str,
    saved_run_overrides: dict[str, Any],
    request_overrides: dict[str, Any],
) -> bool:
    from emperor.inspection import preset_locks

    from workbench.backend.inspection_errors import call_inspection

    del model_id
    locks = call_inspection(preset_locks, package, preset)
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
