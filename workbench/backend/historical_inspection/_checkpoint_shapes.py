"""Private checkpoint shape interpretation for historical Inspection."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch

TensorShape = tuple[int, ...]

MAX_TENSOR_SHAPES_PER_NODE = 12
MAX_CHECKPOINT_GRAPH_SHAPE_BYTES = 256 * 1024 * 1024
MAX_CHECKPOINT_GRAPH_CANDIDATES = 32
MAX_CHECKPOINT_GRAPH_AGGREGATE_BYTES = 512 * 1024 * 1024
_OVERSIZED_CHECKPOINT_REASON = "checkpointTooLarge"

_DIRECT_STACK_WEIGHT_RE = re.compile(
    r"^main_model\.layers\.(?P<index>\d+)\.model\.weight_params$"
)
_RECURRENT_STACK_WEIGHT_RE = re.compile(
    r"^main_model\.block_model\.layers\.(?P<index>\d+)\.model\.weight_params$"
)
_PRIMARY_STACK_PATTERNS = (
    re.compile(r"^main_model\.layers\.(?P<index>\d+)(?:\.|$)"),
    re.compile(r"^main_model\.block_model\.layers\.(?P<index>\d+)(?:\.|$)"),
    re.compile(r"^main_model\.expert_stack\.layers\.(?P<index>\d+)(?:\.|$)"),
    re.compile(r"^transformer\.layers\.(?P<index>\d+)(?:\.|$)"),
)
_GATE_STACK_RE = re.compile(
    r"^(?P<parent>(?:main_model(?:\.block_model)?\.layers\.\d+|"
    r"main_model\.expert_stack\.layers\.\d+)\.gate_model\.model)"
    r"\.layers\.(?P<index>\d+)(?:\.|$)"
)
_HALTING_STACK_RE = re.compile(
    r"^(?P<parent>(?:main_model(?:\.block_model)?\.layers\.\d+|"
    r"main_model\.expert_stack\.layers\.\d+)\.halting_model"
    r"\.halting_gate_model)\.layers\.(?P<index>\d+)(?:\.|$)"
)
_MEMORY_STACK_RE = re.compile(
    r"^(?P<parent>(?:main_model(?:\.block_model)?\.layers\.\d+|"
    r"main_model\.expert_stack\.layers\.\d+)\.memory_model"
    r"\.memory_model)\.layers\.(?P<index>\d+)(?:\.|$)"
)
_EXPERT_STACK_RE = re.compile(
    r"^(?P<parent>main_model\.expert_stack\.layers\.\d+\.model"
    r"\.expert_modules\.\d+)\.layers\.(?P<index>\d+)(?:\.|$)"
)
_EXPERT_MODULE_RE = re.compile(
    r"^main_model\.expert_stack\.layers\.(?P<outer>\d+)\.model"
    r"\.expert_modules\.(?P<expert>\d+)(?:\.|$)"
)
_ROUTER_STACK_RE = re.compile(
    r"^(?P<parent>main_model\.expert_stack\.layers\.\d+\.model"
    r"\.sampler\.router\.model)\.layers\.(?P<index>\d+)(?:\.|$)"
)
_ROUTER_LAYER_PARAMETER_RE = re.compile(
    r"^main_model\.expert_stack\.layers\.(?P<outer>\d+)\.model\.sampler"
    r"\.router\.model\.layers\.(?P<index>\d+)\.model"
    r"\.(?P<parameter>weight_params|weight|weights|bias_params|bias|biases)$"
)
_ADAPTIVE_GENERATOR_STACK_PATTERNS = (
    re.compile(
        r"^(?P<parent>main_model\.layers\.\d+\.model\.adaptive_behaviour\..*"
        r"\.model)\.layers\.(?P<index>\d+)(?:\.|$)"
    ),
    re.compile(
        r"^(?P<parent>input_model\.model\.adaptive_behaviour\..*"
        r"\.model)\.layers\.(?P<index>\d+)(?:\.|$)"
    ),
    re.compile(
        r"^(?P<parent>output_model\.model\.adaptive_behaviour\..*"
        r"\.model)\.layers\.(?P<index>\d+)(?:\.|$)"
    ),
)


@dataclass(frozen=True)
class CheckpointGraphDiagnostics:
    structural_fallback_reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CheckpointGraphShapes:
    config_overrides: dict[str, Any]
    parameter_shapes: dict[str, dict[str, Any]]
    coverage_counts: dict[str, int]
    tensor_count: int
    diagnostics: CheckpointGraphDiagnostics = field(
        default_factory=CheckpointGraphDiagnostics
    )


@dataclass(frozen=True, slots=True)
class CheckpointLoadBudgets:
    max_candidates: int = MAX_CHECKPOINT_GRAPH_CANDIDATES
    max_file_bytes: int = MAX_CHECKPOINT_GRAPH_SHAPE_BYTES
    max_aggregate_bytes: int = MAX_CHECKPOINT_GRAPH_AGGREGATE_BYTES

    def __post_init__(self) -> None:
        if self.max_candidates < 1:
            raise ValueError("Checkpoint candidate limit must be positive.")
        if self.max_file_bytes < 1:
            raise ValueError("Checkpoint per-file byte limit must be positive.")
        if self.max_aggregate_bytes < 1:
            raise ValueError("Checkpoint aggregate byte limit must be positive.")


DEFAULT_CHECKPOINT_LOAD_BUDGETS = CheckpointLoadBudgets()


class FrozenCheckpointCandidate(Protocol):
    path: Path
    size_bytes: int
    modified_at_ns: int


CheckpointConfigInterpreter = Callable[[Mapping[str, TensorShape]], Mapping[str, Any]]


def load_checkpoint_graph_shapes(
    checkpoint_candidates: Iterable[Path | FrozenCheckpointCandidate],
    *,
    budgets: CheckpointLoadBudgets = DEFAULT_CHECKPOINT_LOAD_BUDGETS,
    package_config_interpreter: CheckpointConfigInterpreter | None = None,
) -> CheckpointGraphShapes | None:
    fallback_reasons: list[str] = []
    attempted_bytes = 0
    for index, candidate in enumerate(checkpoint_candidates):
        if index >= budgets.max_candidates:
            fallback_reasons.append(
                f"checkpointCandidateLimit:{budgets.max_candidates}"
            )
            break
        checkpoint_path, expected_size, expected_modified_at_ns = (
            _checkpoint_candidate_fields(candidate)
        )
        try:
            checkpoint_file = checkpoint_path.open("rb")
        except OSError:
            fallback_reasons.append("checkpointUnavailable")
            continue
        with checkpoint_file:
            try:
                stat = os.fstat(checkpoint_file.fileno())
            except OSError:
                fallback_reasons.append("checkpointUnavailable")
                continue
            checkpoint_size = int(stat.st_size)
            if (
                expected_size is not None
                and expected_modified_at_ns is not None
                and (
                    checkpoint_size != expected_size
                    or int(stat.st_mtime_ns) != expected_modified_at_ns
                )
            ):
                fallback_reasons.append("checkpointChanged")
                continue
            if checkpoint_size > budgets.max_file_bytes:
                fallback_reasons.append(
                    f"{_OVERSIZED_CHECKPOINT_REASON}:"
                    f"{checkpoint_size}>{budgets.max_file_bytes}"
                )
                continue
            if attempted_bytes + checkpoint_size > budgets.max_aggregate_bytes:
                fallback_reasons.append(
                    "checkpointAggregateTooLarge:"
                    f"{attempted_bytes}+{checkpoint_size}>"
                    f"{budgets.max_aggregate_bytes}"
                )
                continue
            attempted_bytes += checkpoint_size
            try:
                checkpoint = torch.load(
                    checkpoint_file,
                    map_location="cpu",
                    weights_only=True,
                )
            except Exception:
                continue
            try:
                loaded_stat = os.fstat(checkpoint_file.fileno())
            except OSError:
                fallback_reasons.append("checkpointUnavailable")
                continue
            if int(loaded_stat.st_size) != checkpoint_size or int(
                loaded_stat.st_mtime_ns
            ) != int(stat.st_mtime_ns):
                fallback_reasons.append("checkpointChanged")
                continue

        state_dict = _state_dict_from_checkpoint(checkpoint)
        if state_dict is None:
            continue
        shapes = checkpoint_graph_shapes_from_state_dict(
            state_dict,
            package_config_interpreter=package_config_interpreter,
        )
        if shapes is not None:
            return shapes
    if fallback_reasons:
        return CheckpointGraphShapes(
            config_overrides={},
            parameter_shapes={},
            coverage_counts={},
            tensor_count=0,
            diagnostics=CheckpointGraphDiagnostics(tuple(fallback_reasons)),
        )
    return None


def _checkpoint_candidate_fields(
    candidate: Path | FrozenCheckpointCandidate,
) -> tuple[Path, int | None, int | None]:
    if isinstance(candidate, Path):
        return candidate, None, None
    return candidate.path, candidate.size_bytes, candidate.modified_at_ns


def checkpoint_graph_shapes_from_state_dict(
    state_dict: Mapping[str, Any],
    *,
    package_config_interpreter: CheckpointConfigInterpreter | None = None,
) -> CheckpointGraphShapes | None:
    tensor_shapes = _tensor_shapes(state_dict)
    if not tensor_shapes:
        return None
    diagnostics: list[str] = []
    config_overrides = _config_overrides(tensor_shapes, diagnostics)
    if package_config_interpreter is not None:
        try:
            package_overrides = dict(package_config_interpreter(tensor_shapes))
        except Exception:
            diagnostics.append("packageCheckpointMetadataUnavailable")
        else:
            config_overrides.update(package_overrides)
    return CheckpointGraphShapes(
        config_overrides=config_overrides,
        parameter_shapes=_parameter_shapes(tensor_shapes),
        coverage_counts=_coverage_counts(tensor_shapes),
        tensor_count=len(tensor_shapes),
        diagnostics=CheckpointGraphDiagnostics(tuple(diagnostics)),
    )


def _state_dict_from_checkpoint(
    checkpoint: object,
) -> Mapping[str, Any] | None:
    if not isinstance(checkpoint, Mapping):
        return None

    nested_state_dict = checkpoint.get("state_dict")
    if isinstance(nested_state_dict, Mapping):
        return nested_state_dict

    if any(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in checkpoint.items()
    ):
        return checkpoint
    return None


def _tensor_shapes(state_dict: Mapping[str, Any]) -> dict[str, TensorShape]:
    shapes: dict[str, TensorShape] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            continue
        shapes[key] = tuple(int(dimension) for dimension in value.shape)
    return shapes


def _config_overrides(
    tensor_shapes: Mapping[str, TensorShape],
    diagnostics: list[str],
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    input_weight = _matrix_shape(tensor_shapes.get("input_model.model.weight_params"))
    output_weight = _matrix_shape(tensor_shapes.get("output_model.model.weight_params"))
    direct_layer_shapes = _stack_weight_shapes(tensor_shapes, _DIRECT_STACK_WEIGHT_RE)
    recurrent_layer_shapes = _stack_weight_shapes(
        tensor_shapes,
        _RECURRENT_STACK_WEIGHT_RE,
    )
    layer_shapes = direct_layer_shapes or recurrent_layer_shapes

    if input_weight is not None:
        overrides["input_dim"] = input_weight[0]
    if output_weight is not None:
        overrides["output_dim"] = output_weight[1]

    hidden_candidates = _hidden_dim_candidates(
        input_weight=input_weight,
        output_weight=output_weight,
        layer_shapes=layer_shapes,
    )
    if hidden_candidates and len(set(hidden_candidates)) == 1:
        overrides["hidden_dim"] = hidden_candidates[0]
    elif len(set(hidden_candidates)) > 1:
        diagnostics.append("hidden_dim:conflictingShapes")

    layer_count = _stack_layer_count_from_patterns(
        tensor_shapes,
        _PRIMARY_STACK_PATTERNS,
        diagnostics=diagnostics,
        label="stack_num_layers",
    )
    if layer_count is not None:
        overrides["stack_num_layers"] = layer_count

    gate_layer_count = _grouped_stack_layer_count(
        tensor_shapes,
        _GATE_STACK_RE,
        diagnostics=diagnostics,
        label="gate_stack_num_layers",
    )
    if gate_layer_count is not None:
        overrides["stack_gate_flag"] = True
        overrides["gate_stack_independent_flag"] = True
        overrides["gate_stack_num_layers"] = gate_layer_count
    gate_hidden_dim = _grouped_stack_hidden_dim(
        tensor_shapes,
        _GATE_STACK_RE,
        diagnostics=diagnostics,
        label="gate_stack_hidden_dim",
    )
    if gate_hidden_dim is not None:
        overrides["stack_gate_flag"] = True
        overrides["gate_stack_independent_flag"] = True
        overrides["gate_stack_hidden_dim"] = gate_hidden_dim

    halting_layer_count = _grouped_stack_layer_count(
        tensor_shapes,
        _HALTING_STACK_RE,
        diagnostics=diagnostics,
        label="halting_stack_num_layers",
    )
    if halting_layer_count is not None:
        overrides["stack_halting_flag"] = True
        overrides["halting_stack_independent_flag"] = True
        overrides["halting_stack_num_layers"] = halting_layer_count
    halting_hidden_dim = _grouped_stack_hidden_dim(
        tensor_shapes,
        _HALTING_STACK_RE,
        diagnostics=diagnostics,
        label="halting_stack_hidden_dim",
    )
    if halting_hidden_dim is not None:
        overrides["stack_halting_flag"] = True
        overrides["halting_stack_independent_flag"] = True
        overrides["halting_stack_hidden_dim"] = halting_hidden_dim

    memory_layer_count = _grouped_stack_layer_count(
        tensor_shapes,
        _MEMORY_STACK_RE,
        diagnostics=diagnostics,
        label="memory_stack_num_layers",
    )
    if memory_layer_count is not None:
        overrides["memory_flag"] = True
        overrides["memory_stack_independent_flag"] = True
        overrides["memory_stack_num_layers"] = memory_layer_count
    memory_hidden_dim = _grouped_stack_hidden_dim(
        tensor_shapes,
        _MEMORY_STACK_RE,
        diagnostics=diagnostics,
        label="memory_stack_hidden_dim",
    )
    if memory_hidden_dim is not None:
        overrides["memory_flag"] = True
        overrides["memory_stack_independent_flag"] = True
        overrides["memory_stack_hidden_dim"] = memory_hidden_dim

    controller_hidden_dims = [
        hidden_dim
        for hidden_dim in (gate_hidden_dim, halting_hidden_dim, memory_hidden_dim)
        if hidden_dim is not None
    ]
    if controller_hidden_dims and len(set(controller_hidden_dims)) == 1:
        overrides["submodule_stack_hidden_dim"] = controller_hidden_dims[0]
    elif len(set(controller_hidden_dims)) > 1:
        diagnostics.append("submodule_stack_hidden_dim:conflictingControllerShapes")

    expert_layer_count = _grouped_stack_layer_count(
        tensor_shapes,
        _EXPERT_STACK_RE,
        diagnostics=diagnostics,
        label="expert_stack_num_layers",
    )
    if expert_layer_count is not None:
        overrides["expert_stack_num_layers"] = expert_layer_count

    router_layer_count = _grouped_stack_layer_count(
        tensor_shapes,
        _ROUTER_STACK_RE,
        diagnostics=diagnostics,
        label="router_stack_num_layers",
    )
    if router_layer_count is not None:
        overrides["router_stack_num_layers"] = router_layer_count

    expert_count = _expert_count(tensor_shapes, diagnostics)
    if expert_count is not None:
        overrides["expert_num_experts"] = expert_count

    adaptive_generator_layer_counts = []
    for pattern in _ADAPTIVE_GENERATOR_STACK_PATTERNS:
        generator_layer_count = _grouped_stack_layer_count(
            tensor_shapes,
            pattern,
            diagnostics=diagnostics,
            label="adaptive_generator_stack_num_layers",
        )
        if generator_layer_count is not None:
            adaptive_generator_layer_counts.append(generator_layer_count)
    if adaptive_generator_layer_counts:
        unique_generator_layer_counts = set(adaptive_generator_layer_counts)
        if len(unique_generator_layer_counts) == 1:
            overrides["adaptive_generator_stack_num_layers"] = (
                adaptive_generator_layer_counts[0]
            )
        else:
            diagnostics.append("adaptive_generator_stack_num_layers:conflictingCounts")

    return overrides


def _matrix_shape(shape: TensorShape | None) -> tuple[int, int] | None:
    if shape is None or len(shape) != 2:
        return None
    return shape[0], shape[1]


def _hidden_dim_candidates(
    *,
    input_weight: tuple[int, int] | None,
    output_weight: tuple[int, int] | None,
    layer_shapes: list[TensorShape],
) -> list[int]:
    candidates: list[int] = []
    if input_weight is not None:
        candidates.append(input_weight[1])
    for layer_shape in layer_shapes:
        matrix = _matrix_shape(layer_shape)
        if matrix is not None:
            candidates.extend(matrix)
    if output_weight is not None:
        candidates.append(output_weight[0])
    return candidates


def _stack_weight_shapes(
    tensor_shapes: Mapping[str, TensorShape],
    pattern: re.Pattern[str],
) -> list[TensorShape]:
    indexed_shapes: dict[int, TensorShape] = {}
    for key, shape in tensor_shapes.items():
        match = pattern.fullmatch(key)
        if match is None:
            continue
        indexed_shapes[int(match.group("index"))] = shape
    contiguous_indices = _contiguous_indices(indexed_shapes)
    if contiguous_indices is None:
        return []
    return [indexed_shapes[index] for index in contiguous_indices]


def _stack_layer_count_from_patterns(
    tensor_shapes: Mapping[str, TensorShape],
    patterns: Iterable[re.Pattern[str]],
    *,
    diagnostics: list[str],
    label: str,
) -> int | None:
    counts: list[int] = []
    for pattern in patterns:
        indices: set[int] = set()
        for key in tensor_shapes:
            match = pattern.match(key)
            if match is not None:
                indices.add(int(match.group("index")))
        contiguous_indices = _contiguous_indices({index: () for index in indices})
        if contiguous_indices is None:
            if indices:
                diagnostics.append(f"{label}:nonContiguous")
            continue
        counts.append(len(contiguous_indices))
    unique_counts = set(counts)
    if len(unique_counts) == 1:
        return counts[0]
    if len(unique_counts) > 1:
        diagnostics.append(f"{label}:conflictingCounts")
    return None


def _grouped_stack_layer_count(
    tensor_shapes: Mapping[str, TensorShape],
    pattern: re.Pattern[str],
    *,
    diagnostics: list[str],
    label: str,
) -> int | None:
    indices_by_parent: dict[str, set[int]] = {}
    for key in tensor_shapes:
        match = pattern.match(key)
        if match is None:
            continue
        indices_by_parent.setdefault(match.group("parent"), set()).add(
            int(match.group("index"))
        )
    if not indices_by_parent:
        return None

    counts: list[int] = []
    for indices in indices_by_parent.values():
        contiguous_indices = _contiguous_indices({index: () for index in indices})
        if contiguous_indices is None:
            diagnostics.append(f"{label}:nonContiguous")
            return None
        counts.append(len(contiguous_indices))

    unique_counts = set(counts)
    if len(unique_counts) != 1:
        diagnostics.append(f"{label}:conflictingCounts")
        return None
    return counts[0]


def _grouped_stack_hidden_dim(
    tensor_shapes: Mapping[str, TensorShape],
    pattern: re.Pattern[str],
    *,
    diagnostics: list[str],
    label: str,
) -> int | None:
    shapes_by_parent: dict[str, dict[int, TensorShape]] = {}
    for key, shape in tensor_shapes.items():
        if not key.endswith(".model.weight_params"):
            continue
        match = pattern.match(key)
        if match is None:
            continue
        shapes_by_parent.setdefault(match.group("parent"), {})[
            int(match.group("index"))
        ] = shape
    if not shapes_by_parent:
        return None

    hidden_dims: list[int] = []
    for indexed_shapes in shapes_by_parent.values():
        contiguous_indices = _contiguous_indices(indexed_shapes)
        if contiguous_indices is None:
            diagnostics.append(f"{label}:nonContiguous")
            return None
        matrices: list[tuple[int, int]] = []
        for index in contiguous_indices:
            matrix = _matrix_shape(indexed_shapes[index])
            if matrix is None:
                diagnostics.append(f"{label}:nonMatrix")
                return None
            matrices.append(matrix)

        hidden_dim, reason = _hidden_dim_from_contiguous_matrices(matrices)
        if reason is not None:
            diagnostics.append(f"{label}:{reason}")
            return None
        if hidden_dim is None:
            return None
        hidden_dims.append(hidden_dim)

    unique_hidden_dims = set(hidden_dims)
    if len(unique_hidden_dims) != 1:
        diagnostics.append(f"{label}:conflictingShapes")
        return None
    return hidden_dims[0]


def _hidden_dim_from_contiguous_matrices(
    matrices: list[tuple[int, int]],
) -> tuple[int | None, str | None]:
    if len(matrices) < 2:
        return None, None
    candidates: list[int] = []
    for matrix in matrices[:-1]:
        candidates.append(matrix[1])
    for matrix in matrices[1:]:
        candidates.append(matrix[0])
    if len(set(candidates)) != 1:
        return None, "conflictingShapes"
    return candidates[0], None


def _expert_count(
    tensor_shapes: Mapping[str, TensorShape],
    diagnostics: list[str],
) -> int | None:
    candidates: list[int] = []
    module_count = _expert_module_count(tensor_shapes, diagnostics)
    if module_count is not None:
        candidates.append(module_count)

    router_count = _router_output_expert_count(tensor_shapes, diagnostics)
    if router_count is not None:
        candidates.append(router_count)

    if not candidates:
        return None
    if len(set(candidates)) != 1:
        diagnostics.append("expert_num_experts:conflictingCounts")
        return None
    return candidates[0]


def _expert_module_count(
    tensor_shapes: Mapping[str, TensorShape],
    diagnostics: list[str],
) -> int | None:
    experts_by_outer_layer: dict[int, set[int]] = {}
    for key in tensor_shapes:
        match = _EXPERT_MODULE_RE.match(key)
        if match is None:
            continue
        experts_by_outer_layer.setdefault(int(match.group("outer")), set()).add(
            int(match.group("expert"))
        )
    if not experts_by_outer_layer:
        return None

    counts: list[int] = []
    for indices in experts_by_outer_layer.values():
        contiguous_indices = _contiguous_indices({index: () for index in indices})
        if contiguous_indices is None:
            diagnostics.append("expert_num_experts:nonContiguous")
            return None
        counts.append(len(contiguous_indices))

    if len(set(counts)) != 1:
        diagnostics.append("expert_num_experts:conflictingModuleCounts")
        return None
    return counts[0]


def _router_output_expert_count(
    tensor_shapes: Mapping[str, TensorShape],
    diagnostics: list[str],
) -> int | None:
    shapes_by_outer_and_layer: dict[int, dict[int, list[tuple[str, TensorShape]]]] = {}
    for key, shape in tensor_shapes.items():
        match = _ROUTER_LAYER_PARAMETER_RE.fullmatch(key)
        if match is None:
            continue
        outer = int(match.group("outer"))
        layer = int(match.group("index"))
        parameter = match.group("parameter")
        shapes_by_outer_and_layer.setdefault(outer, {}).setdefault(layer, []).append(
            (parameter, shape)
        )

    if not shapes_by_outer_and_layer:
        return None

    candidates: list[int] = []
    for layer_shapes in shapes_by_outer_and_layer.values():
        contiguous_layers = _contiguous_indices({index: () for index in layer_shapes})
        if contiguous_layers is None:
            diagnostics.append("router_stack_num_layers:nonContiguous")
            return None
        final_layer_shapes = layer_shapes[contiguous_layers[-1]]
        for parameter, shape in final_layer_shapes:
            if parameter in {"weight_params", "weight", "weights"}:
                matrix = _matrix_shape(shape)
                if matrix is not None:
                    candidates.append(matrix[1])
            elif parameter in {"bias_params", "bias", "biases"} and len(shape) == 1:
                candidates.append(shape[0])

    if not candidates:
        return None
    if len(set(candidates)) != 1:
        diagnostics.append("expert_num_experts:conflictingRouterOutputs")
        return None
    return candidates[0]


def _contiguous_indices(indexed_values: Mapping[int, object]) -> list[int] | None:
    if not indexed_values:
        return None
    indices = sorted(indexed_values)
    if indices != list(range(indices[-1] + 1)):
        return None
    return indices


def _parameter_shapes(
    tensor_shapes: Mapping[str, TensorShape],
) -> dict[str, dict[str, Any]]:
    details_by_path: dict[str, dict[str, Any]] = {}
    direct_tensor_shapes: dict[str, dict[str, str]] = {}
    for key, shape in tensor_shapes.items():
        split_key = _split_tensor_key(key)
        if split_key is None:
            continue
        node_path, parameter_name = split_key
        detail_key = _parameter_detail_key(parameter_name)
        shape_value = _shape_value(shape)
        if detail_key is None:
            direct_tensor_shapes.setdefault(node_path, {})[parameter_name] = shape_value
            continue

        details = details_by_path.setdefault(node_path, {})
        details[detail_key] = shape_value
        details.update(_linear_dimension_details(node_path, parameter_name, shape))
        if _is_embedding_weight(node_path, parameter_name, shape):
            details["numEmbeddings"] = shape[0]
            details["embeddingDim"] = shape[1]
        if _is_conv_like_weight(parameter_name, shape):
            details["outputChannels"] = shape[0]
            details["inputChannels"] = shape[1]
            details["kernelShape"] = _shape_value(shape[2:])

    for node_path, shapes_by_name in direct_tensor_shapes.items():
        sorted_shapes = dict(
            sorted(shapes_by_name.items())[:MAX_TENSOR_SHAPES_PER_NODE]
        )
        if sorted_shapes:
            details_by_path.setdefault(node_path, {})["tensorShapes"] = sorted_shapes
    return details_by_path


def _split_tensor_key(key: str) -> tuple[str, str] | None:
    if "." not in key:
        return None
    node_path, parameter_name = key.rsplit(".", 1)
    if not node_path or not parameter_name:
        return None
    return node_path, parameter_name


def _parameter_detail_key(parameter_name: str) -> str | None:
    if parameter_name in {"weight_params", "weight", "weights"}:
        return "weightShape"
    if parameter_name in {"bias_params", "bias", "biases"}:
        return "biasShape"
    return None


def _linear_dimension_details(
    node_path: str,
    parameter_name: str,
    shape: TensorShape,
) -> dict[str, int | str]:
    if parameter_name != "weight_params":
        return {}
    if _is_embedding_weight(node_path, parameter_name, shape):
        return {}
    matrix = _matrix_shape(shape)
    if matrix is None:
        return {}
    input_dim, output_dim = matrix
    return {
        "inputDim": input_dim,
        "outputDim": output_dim,
        "dims": f"{input_dim} -> {output_dim}",
    }


def _shape_value(shape: TensorShape) -> str:
    if not shape:
        return "scalar"
    return " x ".join(str(dimension) for dimension in shape)


def _is_embedding_weight(
    node_path: str,
    parameter_name: str,
    shape: TensorShape,
) -> bool:
    node_name = node_path.rsplit(".", 1)[-1].lower()
    return (
        parameter_name in {"weight_params", "weight", "weights"}
        and len(shape) == 2
        and "embedding" in node_name
    )


def _is_conv_like_weight(parameter_name: str, shape: TensorShape) -> bool:
    return parameter_name in {"weight_params", "weight", "weights"} and len(shape) >= 3


def _coverage_counts(tensor_shapes: Mapping[str, TensorShape]) -> dict[str, int]:
    counts: dict[str, int] = {"model": len(tensor_shapes)}
    for key in tensor_shapes:
        split_key = _split_tensor_key(key)
        if split_key is None:
            continue
        node_path, _parameter_name = split_key
        for path in _ancestor_paths(node_path):
            if path == "model":
                continue
            counts[path] = counts.get(path, 0) + 1
    return counts


def _ancestor_paths(node_path: str) -> Iterable[str]:
    parts = node_path.split(".")
    for index in range(1, len(parts) + 1):
        yield ".".join(parts[:index])
