"""Deep Workbench Interface for historical checkpoint Inspection."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from emperor.inspection import GraphNode, InspectionRequest, InspectionResult
from emperor.model_packages import ModelPackage, normalize_dataset_name

from workbench.backend.historical_inspection._checkpoint_ranking import (
    rank_historical_checkpoints,
)
from workbench.backend.historical_inspection._checkpoint_shapes import (
    DEFAULT_CHECKPOINT_LOAD_BUDGETS,
    CheckpointGraphShapes,
    CheckpointLoadBudgets,
    load_checkpoint_graph_shapes,
)
from workbench.backend.inspection_adapter import WorkbenchInspectionAdapter
from workbench.backend.inspection_errors import InspectionFailure
from workbench.backend.model_identity import normalize_preset_token
from workbench.backend.run_history import HistoricalInspectionContext

if TYPE_CHECKING:
    from workbench.backend.inspection_worker import InspectionExecutor

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


@dataclass(frozen=True, slots=True)
class HistoricalInspectionRequest:
    """Already-validated caller choices layered over one frozen Run context."""

    preset: str
    request_overrides: Mapping[str, Any]
    dataset: str | None
    experiment_task: str | None = None


class WorkbenchHistoricalInspection:
    """Own checkpoint policy, reconstruction, precedence, and annotation."""

    def __init__(
        self,
        package: ModelPackage,
        *,
        checkpoint_budgets: CheckpointLoadBudgets = (DEFAULT_CHECKPOINT_LOAD_BUDGETS),
        inspection_executor: InspectionExecutor | None = None,
    ) -> None:
        self._package = package
        self._adapter = WorkbenchInspectionAdapter.from_package(package)
        self._checkpoint_budgets = checkpoint_budgets
        self._inspection_executor = inspection_executor

    def inspect(
        self,
        context: HistoricalInspectionContext,
        request: HistoricalInspectionRequest,
    ) -> InspectionResult:
        from emperor.inspection import InspectionRequest, ParsedOverrides

        request_overrides = dict(request.request_overrides)
        saved_run_overrides = self._adapter.parse_overrides(
            self._saved_run_overrides(context, request),
            ignore_unknown=True,
        ).values
        checkpoint_shapes = load_checkpoint_graph_shapes(
            rank_historical_checkpoints(context.checkpoint_candidates),
            budgets=self._checkpoint_budgets,
            package_config_interpreter=(self._package.checkpoint_config_overrides),
        )
        checkpoint_overrides, checkpoint_structural_fallback = (
            self._checkpoint_overrides(
                preset=request.preset,
                checkpoint_shapes=checkpoint_shapes,
                saved_run_overrides=saved_run_overrides,
                request_overrides=request_overrides,
            )
        )
        effective_overrides = {
            **saved_run_overrides,
            **checkpoint_overrides,
            **request_overrides,
        }

        try:
            result = self._inspect(
                InspectionRequest(
                    preset=request.preset,
                    overrides=ParsedOverrides(effective_overrides),
                    dataset=request.dataset,
                    experiment_task=request.experiment_task,
                ),
            )
        except InspectionFailure as exc:
            if checkpoint_shapes is None or not checkpoint_overrides:
                raise
            fallback_overrides = {
                **saved_run_overrides,
                **request_overrides,
            }
            try:
                result = self._inspect(
                    InspectionRequest(
                        preset=request.preset,
                        overrides=ParsedOverrides(fallback_overrides),
                        dataset=request.dataset,
                        experiment_task=request.experiment_task,
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
        if self._inspection_executor is None:
            return self._adapter.inspect(request)
        return self._inspection_executor.inspect(self._package, request)

    def inspect_payload(
        self,
        context: HistoricalInspectionContext,
        request: HistoricalInspectionRequest,
    ) -> dict[str, Any]:
        """Compatibility Adapter for callers that still need HTTP-shaped data."""
        from workbench.backend.inspection_serialization import (
            inspection_result_payload,
        )

        return inspection_result_payload(self.inspect(context, request))

    def _saved_run_overrides(
        self,
        context: HistoricalInspectionContext,
        request: HistoricalInspectionRequest,
    ) -> dict[str, Any]:
        model_id = self._package.catalog_key
        if context.model != model_id:
            raise InspectionFailure(
                f"Log run '{context.run_id}' belongs to model '{context.model}', "
                f"not '{model_id}'."
            )
        if normalize_preset_token(context.preset) != normalize_preset_token(
            request.preset
        ):
            raise InspectionFailure(
                f"Log run '{context.run_id}' belongs to preset "
                f"'{context.preset}', not '{request.preset}'."
            )
        if request.dataset and _normalized_dataset(
            context.dataset
        ) != _normalized_dataset(request.dataset):
            raise InspectionFailure(
                f"Log run '{context.run_id}' belongs to dataset "
                f"'{context.dataset}', not '{request.dataset}'."
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
            adapter=self._adapter,
            preset=preset,
            checkpoint_overrides=checkpoint_shapes.config_overrides,
            saved_run_overrides=saved_run_overrides,
            request_overrides=request_overrides,
        )
        if len(override_candidates) != len(checkpoint_shapes.config_overrides):
            structural_fallback = True

        try:
            parsed = self._adapter.parse_overrides(
                override_candidates,
                ignore_unknown=True,
            ).values
        except InspectionFailure:
            return {}, True

        locks = self._adapter.preset_locks(preset)
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
    adapter: WorkbenchInspectionAdapter,
    preset: str,
    checkpoint_overrides: dict[str, Any],
    saved_run_overrides: dict[str, Any],
    request_overrides: dict[str, Any],
) -> dict[str, Any]:
    if not set(checkpoint_overrides) & _ADAPTIVE_GENERATOR_COUNT_KEYS:
        return dict(checkpoint_overrides)
    if _has_selected_adaptive_option(
        adapter=adapter,
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
    adapter: WorkbenchInspectionAdapter,
    preset: str,
    saved_run_overrides: dict[str, Any],
    request_overrides: dict[str, Any],
) -> bool:
    locks = adapter.preset_locks(preset)
    lock_values = {key: getattr(lock, "value", None) for key, lock in locks.items()}
    for source in (lock_values, saved_run_overrides, request_overrides):
        for key in _ADAPTIVE_FLAG_KEYS:
            if source.get(key) is True:
                return True
        for key in _ADAPTIVE_OPTION_KEYS:
            if source.get(key) is not None:
                return True
    return False


__all__ = [
    "HistoricalInspectionRequest",
    "WorkbenchHistoricalInspection",
]
