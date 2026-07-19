from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

TensorShape = tuple[int, ...]

_DIRECT_LAYER_WEIGHT = re.compile(
    r"^main_model\.layers\.(?P<index>\d+)\.model\.weight_params$"
)
_RECURRENT_LAYER_WEIGHT = re.compile(
    r"^main_model\.block_model\.layers\.(?P<index>\d+)\.model\.weight_params$"
)


def _matrix(shape: TensorShape | None) -> tuple[int, int] | None:
    if shape is None or len(shape) != 2:
        return None
    return shape[0], shape[1]


def _layer_indices(
    tensor_shapes: Mapping[str, TensorShape],
    pattern: re.Pattern[str],
) -> list[int]:
    return sorted(
        {
            int(match.group("index"))
            for key in tensor_shapes
            if (match := pattern.fullmatch(key)) is not None
        }
    )


def _layer_shapes(
    tensor_shapes: Mapping[str, TensorShape],
    pattern: re.Pattern[str],
) -> list[tuple[int, int]]:
    indexed: list[tuple[int, tuple[int, int]]] = []
    for key, shape in tensor_shapes.items():
        match = pattern.fullmatch(key)
        matrix = _matrix(shape)
        if match is not None and matrix is not None:
            indexed.append((int(match.group("index")), matrix))
    return [shape for _index, shape in sorted(indexed)]


def checkpoint_config_overrides(
    tensor_shapes: Mapping[str, TensorShape],
) -> Mapping[str, Any]:
    """Recover the stable Linear package dimensions from its state-dict layout."""

    overrides: dict[str, Any] = {}
    input_weight = _matrix(tensor_shapes.get("input_model.model.weight_params"))
    output_weight = _matrix(tensor_shapes.get("output_model.model.weight_params"))
    if input_weight is not None:
        overrides["input_dim"] = input_weight[0]
    if output_weight is not None:
        overrides["output_dim"] = output_weight[1]

    hidden_candidates: list[int] = []
    if input_weight is not None:
        hidden_candidates.append(input_weight[1])
    if output_weight is not None:
        hidden_candidates.append(output_weight[0])
    layer_shapes = _layer_shapes(tensor_shapes, _DIRECT_LAYER_WEIGHT)
    if not layer_shapes:
        layer_shapes = _layer_shapes(tensor_shapes, _RECURRENT_LAYER_WEIGHT)
    for input_dim, output_dim in layer_shapes:
        hidden_candidates.extend((input_dim, output_dim))
    if hidden_candidates and len(set(hidden_candidates)) == 1:
        overrides["hidden_dim"] = hidden_candidates[0]

    indices = _layer_indices(tensor_shapes, _DIRECT_LAYER_WEIGHT)
    if not indices:
        indices = _layer_indices(tensor_shapes, _RECURRENT_LAYER_WEIGHT)
    if indices and indices == list(range(len(indices))):
        overrides["stack_num_layers"] = len(indices)
    return overrides


__all__ = ["checkpoint_config_overrides"]
