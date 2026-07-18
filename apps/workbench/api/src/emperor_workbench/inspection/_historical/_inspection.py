from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from model_runtime.inspection import GraphNode, InspectionRequest, InspectionResult
from model_runtime.packages import normalize_dataset_name

from emperor_workbench.inspection._errors import (
    InspectionFailure,
    inspection_failure,
)
from emperor_workbench.inspection._executor import InspectionExecutor
from emperor_workbench.inspection._historical._checkpoint_ranking import (
    rank_historical_checkpoints,
)
from emperor_workbench.inspection._historical._checkpoint_shapes import (
    DEFAULT_CHECKPOINT_LOAD_BUDGETS,
    CheckpointGraphShapes,
    CheckpointLoadBudgets,
    load_checkpoint_graph_shapes,
)
from emperor_workbench.model_packages import (
    ModelPackageFailure,
    SelectedModelPackage,
    normalize_preset_token,
)
from emperor_workbench.run_history import (
    HistoricalInspectionContext,
    HistoricalInspectionSource,
)

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


class HistoricalInspection:
    """Own historical precedence, checkpoint reconstruction, and annotation."""

    def __init__(
        self,
        selected: SelectedModelPackage,
        *,
        executor: InspectionExecutor,
        source: HistoricalInspectionSource,
        checkpoint_budgets: CheckpointLoadBudgets = DEFAULT_CHECKPOINT_LOAD_BUDGETS,
    ) -> None:
        self._selected = selected
        self._executor = executor
        self._source = source
        self._checkpoint_budgets = checkpoint_budgets

    def inspect(
        self,
        *,
        log_run_id: str,
        preset: str,
        request_overrides: Mapping[str, Any],
        dataset: str | None,
        experiment_task: str | None = None,
    ) -> InspectionResult:
        from model_runtime.inspection import ParsedOverrides

        context = self._source.inspection_context(log_run_id)
        request_override_values = dict(request_overrides)
        try:
            saved_run_overrides = self._selected.parse_overrides(
                self._saved_run_overrides(
                    context,
                    preset=preset,
                    dataset=dataset,
                ),
                ignore_unknown=True,
            ).values
        except ModelPackageFailure as exc:
            raise inspection_failure(exc) from exc

        checkpoint_shapes = load_checkpoint_graph_shapes(
            rank_historical_checkpoints(context.checkpoint_candidates),
            budgets=self._checkpoint_budgets,
            package_config_interpreter=(self._selected.checkpoint_config_overrides),
        )
        checkpoint_overrides, checkpoint_structural_fallback = (
            self._checkpoint_overrides(
                preset=preset,
                checkpoint_shapes=checkpoint_shapes,
                saved_run_overrides=saved_run_overrides,
                request_overrides=request_override_values,
            )
        )
        effective_overrides = {
            **saved_run_overrides,
            **checkpoint_overrides,
            **request_override_values,
        }

        try:
            result = self._inspect(
                InspectionRequest(
                    preset=preset,
                    overrides=ParsedOverrides(effective_overrides),
                    dataset=dataset,
                    experiment_task=experiment_task,
                ),
            )
        except InspectionFailure as exc:
            if checkpoint_shapes is None or not checkpoint_overrides:
                raise
            fallback_overrides = {
                **saved_run_overrides,
                **request_override_values,
            }
            try:
                result = self._inspect(
                    InspectionRequest(
                        preset=preset,
                        overrides=ParsedOverrides(fallback_overrides),
                        dataset=dataset,
                        experiment_task=experiment_task,
                    ),
                )
            except InspectionFailure as fallback_exc:
                raise exc from fallback_exc
            checkpoint_structural_fallback = True

        if checkpoint_shapes is not None:
            result = _annotate_checkpoint_shape_details(
                result,
                checkpoint_shapes,
                structural_fallback=checkpoint_structural_fallback,
            )
        return result

    def _inspect(self, request: InspectionRequest) -> InspectionResult:
        return self._executor.inspect(self._selected, request)

    def _saved_run_overrides(
        self,
        context: HistoricalInspectionContext,
        *,
        preset: str,
        dataset: str | None,
    ) -> dict[str, Any]:
        model_id = self._selected.catalog_key
        if context.model != model_id:
            raise InspectionFailure(
                f"Log run '{context.run_id}' belongs to model '{context.model}', "
                f"not '{model_id}'."
            )
        if normalize_preset_token(context.preset) != normalize_preset_token(preset):
            raise InspectionFailure(
                f"Log run '{context.run_id}' belongs to preset "
                f"'{context.preset}', not '{preset}'."
            )
        if dataset and _normalized_dataset(context.dataset) != _normalized_dataset(
            dataset
        ):
            raise InspectionFailure(
                f"Log run '{context.run_id}' belongs to dataset "
                f"'{context.dataset}', not '{dataset}'."
            )
        return dict(context.params)

    def _checkpoint_overrides(
        self,
        *,
        preset: str,
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
            selected=self._selected,
            preset=preset,
            checkpoint_overrides=checkpoint_shapes.config_overrides,
            saved_run_overrides=saved_run_overrides,
            request_overrides=request_overrides,
        )
        if len(override_candidates) != len(checkpoint_shapes.config_overrides):
            structural_fallback = True

        try:
            parsed = self._selected.parse_overrides(
                override_candidates,
                ignore_unknown=True,
            ).values
        except ModelPackageFailure:
            return {}, True

        try:
            locks = self._selected.preset_locks(preset)
        except ModelPackageFailure:
            return {}, True
        unlocked = {key: value for key, value in parsed.items() if key not in locks}
        if len(unlocked) != len(parsed):
            structural_fallback = True
        return unlocked, structural_fallback


def _normalized_dataset(dataset: str | None) -> str | None:
    return normalize_dataset_name(dataset) if dataset else None


def _annotate_checkpoint_shape_details(
    result: InspectionResult,
    checkpoint_shapes: CheckpointGraphShapes,
    *,
    structural_fallback: bool,
) -> InspectionResult:
    nodes: list[GraphNode] = []
    for node in result.nodes:
        details = dict(node.details)
        checkpoint_details = checkpoint_shapes.parameter_shapes.get(node.path)
        if checkpoint_details is not None:
            details.update(
                {
                    _semantic_detail_key(key): value
                    for key, value in checkpoint_details.items()
                }
            )
        tensor_count = checkpoint_shapes.coverage_counts.get(node.path, 0)
        checkpoint_detail: dict[str, Any] = {
            "status": "matched" if tensor_count > 0 else "missing",
            "tensor_count": tensor_count,
        }
        if tensor_count == 0:
            checkpoint_detail["reason"] = "noCheckpointTensor"
        if node.path == "model" and structural_fallback:
            checkpoint_detail["reason"] = "structuralFallback"
            fallback_reasons = checkpoint_shapes.diagnostics.structural_fallback_reasons
            if fallback_reasons:
                checkpoint_detail["fallback_reasons"] = tuple(fallback_reasons)
        details["checkpoint"] = checkpoint_detail
        nodes.append(replace(node, details=details))
    return replace(result, nodes=tuple(nodes))


_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def _semantic_detail_key(key: str) -> str:
    return _CAMEL_BOUNDARY.sub("_", key).lower()


def _adaptive_safe_checkpoint_candidates(
    *,
    selected: SelectedModelPackage,
    preset: str,
    checkpoint_overrides: dict[str, Any],
    saved_run_overrides: dict[str, Any],
    request_overrides: dict[str, Any],
) -> dict[str, Any]:
    if not set(checkpoint_overrides) & _ADAPTIVE_GENERATOR_COUNT_KEYS:
        return dict(checkpoint_overrides)
    if _has_selected_adaptive_option(
        selected=selected,
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
    selected: SelectedModelPackage,
    preset: str,
    saved_run_overrides: dict[str, Any],
    request_overrides: dict[str, Any],
) -> bool:
    try:
        locks = selected.preset_locks(preset)
    except ModelPackageFailure:
        return False
    lock_values = {key: getattr(lock, "value", None) for key, lock in locks.items()}
    for source in (lock_values, saved_run_overrides, request_overrides):
        for key in _ADAPTIVE_FLAG_KEYS:
            if source.get(key) is True:
                return True
        for key in _ADAPTIVE_OPTION_KEYS:
            if source.get(key) is not None:
                return True
    return False


__all__: list[str] = []
