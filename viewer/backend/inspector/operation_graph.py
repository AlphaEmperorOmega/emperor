from __future__ import annotations

import inspect
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from models.catalog import model_identity_payload_from_id
from torch import Tensor
from torch.nn import Module

from viewer.backend.inspector.discovery import option_cli_name
from viewer.backend.inspector.service import build_inspection_target

OPERATION_GRAPH_SOURCE = "torch-export"
INPUT_GROUP_ID = "__inputs__"
OUTPUT_GROUP_ID = "__outputs__"


@dataclass(frozen=True)
class ExportInputs:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class UnsupportedOperationGraph:
    warnings: list[str]


def inspect_operation_graph(
    model_name: str,
    preset_name: str,
    overrides: Mapping[str, Any] | None = None,
    dataset: str | None = None,
    *,
    parsed_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = build_inspection_target(
        model_name,
        preset_name,
        overrides,
        dataset=dataset,
        parsed_overrides=parsed_overrides,
    )
    preset = option_cli_name(target.parts.experiment_options, target.option)

    try:
        target.model.cpu()
        target.model.eval()
    except Exception as exc:
        return _unsupported_response(
            model_name,
            preset,
            [f"Failed to prepare model for torch.export: {_exception_message(exc)}"],
        )

    export_inputs = resolve_synthetic_inputs(
        target.model,
        cfg=target.cfg,
        dataset_type=target.dataset_type,
    )
    if isinstance(export_inputs, UnsupportedOperationGraph):
        return _unsupported_response(model_name, preset, export_inputs.warnings)

    export_fn = getattr(getattr(torch, "export", None), "export", None)
    if not callable(export_fn):
        return _unsupported_response(
            model_name,
            preset,
            ["torch.export.export is not available in this PyTorch runtime."],
        )

    try:
        with torch.no_grad():
            exported_program = export_fn(
                target.model,
                export_inputs.args,
                kwargs=export_inputs.kwargs or None,
            )
    except Exception as exc:
        return _unsupported_response(
            model_name,
            preset,
            [f"torch.export.export failed: {_exception_message(exc)}"],
        )

    try:
        nodes, edges = serialize_exported_program(exported_program, target.model)
    except Exception as exc:
        return _unsupported_response(
            model_name,
            preset,
            [f"Failed to serialize torch.export graph: {_exception_message(exc)}"],
        )
    return {
        **model_identity_payload_from_id(model_name),
        "preset": preset,
        "source": OPERATION_GRAPH_SOURCE,
        "status": "ok",
        "nodes": nodes,
        "edges": edges,
        "warnings": [],
    }


def resolve_synthetic_inputs(
    model: Module,
    *,
    cfg: Any,
    dataset_type: type,
) -> ExportInputs | UnsupportedOperationGraph:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError) as exc:
        return UnsupportedOperationGraph(
            [f"Could not inspect model forward signature: {_exception_message(exc)}"]
        )

    parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.name != "self"
    ]
    if any(
        parameter.kind
        in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        for parameter in parameters
    ):
        return UnsupportedOperationGraph(
            [
                "Synthetic operation tracing does not support variadic forward "
                "signatures."
            ]
        )

    parameter_by_lower_name = {
        parameter.name.lower(): parameter for parameter in parameters
    }
    if "input_ids" in parameter_by_lower_name:
        return _bert_export_inputs(cfg, parameter_by_lower_name)

    image_input = _image_export_input(dataset_type)
    if isinstance(image_input, UnsupportedOperationGraph):
        return image_input

    required_parameters = [
        parameter
        for parameter in parameters
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    ]
    if len(required_parameters) == 1:
        return ExportInputs(args=(image_input,), kwargs={})

    required_keyword_parameters = [
        parameter
        for parameter in parameters
        if parameter.default is inspect.Parameter.empty
        and parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    if len(required_parameters) == 0 and len(required_keyword_parameters) == 1:
        return ExportInputs(
            args=(),
            kwargs={required_keyword_parameters[0].name: image_input},
        )

    image_named_parameter = next(
        (
            parameter
            for parameter in parameters
            if parameter.name.lower() in {"x", "input", "inputs", "image", "images"}
        ),
        None,
    )
    if image_named_parameter is not None:
        if image_named_parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            return ExportInputs(
                args=(),
                kwargs={image_named_parameter.name: image_input},
            )
        return ExportInputs(args=(image_input,), kwargs={})

    parameter_names = ", ".join(parameter.name for parameter in parameters) or "none"
    return UnsupportedOperationGraph(
        [
            "Synthetic operation tracing does not know how to build inputs for "
            f"forward({parameter_names})."
        ]
    )


def serialize_exported_program(
    exported_program: Any,
    model: Module,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    graph_module = exported_program.graph_module
    graph_nodes = list(graph_module.graph.nodes)
    operation_node_id_by_fx_node = {
        fx_node: f"op_{index:04d}" for index, fx_node in enumerate(graph_nodes)
    }
    input_spec_by_name = _input_spec_by_name(exported_program)
    output_spec_by_name = _output_spec_by_name(exported_program)
    owner_by_target = _state_owner_by_target(model)
    owner_by_fx_node: dict[Any, str | None] = {}
    input_kind_by_fx_node: dict[Any, str | None] = {}

    nodes: list[dict[str, Any]] = []
    for fx_node in graph_nodes:
        node_id = operation_node_id_by_fx_node[fx_node]
        details: dict[str, Any] = {}
        module_path: str | None = None
        group_id: str | None = None
        input_kind: str | None = None

        if fx_node.op == "placeholder":
            spec = input_spec_by_name.get(fx_node.name)
            input_kind = _spec_kind(spec)
            if input_kind is not None:
                details["inputKind"] = input_kind
            target_path = getattr(spec, "target", None) if spec is not None else None
            if target_path:
                details["targetPath"] = str(target_path)
                module_path = owner_by_target.get(str(target_path))
                if module_path is None:
                    module_path = _owner_from_state_target(str(target_path))
            if input_kind == "user_input":
                group_id = INPUT_GROUP_ID
        elif fx_node.op == "output":
            output_names = _output_names(fx_node)
            output_kinds = [
                _spec_kind(output_spec_by_name.get(output_name))
                for output_name in output_names
                if output_spec_by_name.get(output_name) is not None
            ]
            if output_kinds:
                details["outputKinds"] = output_kinds
            group_id = OUTPUT_GROUP_ID
        else:
            module_path = _infer_owner_from_inputs(
                fx_node,
                owner_by_fx_node,
                input_kind_by_fx_node,
            )

        if module_path is not None:
            group_id = module_path
        owner_by_fx_node[fx_node] = module_path
        input_kind_by_fx_node[fx_node] = input_kind

        details.update(_tensor_details(fx_node.meta))
        argument_names = [input_node.name for input_node in fx_node.all_input_nodes]
        if argument_names:
            details["inputs"] = argument_names

        nodes.append(
            {
                "id": node_id,
                "label": _node_label(fx_node, details),
                "opKind": fx_node.op,
                "target": _target_string(fx_node.target),
                "modulePath": module_path,
                "groupId": group_id,
                "details": _json_safe_object(details),
            }
        )

    edges: list[dict[str, str]] = []
    edge_counts: Counter[tuple[str, str]] = Counter()
    for fx_node in graph_nodes:
        target_id = operation_node_id_by_fx_node[fx_node]
        for input_node in fx_node.all_input_nodes:
            source_id = operation_node_id_by_fx_node[input_node]
            pair = (source_id, target_id)
            edge_index = edge_counts[pair]
            edge_counts[pair] += 1
            edge_id = (
                f"{source_id}-{target_id}"
                if edge_index == 0
                else f"{source_id}-{target_id}-{edge_index}"
            )
            edges.append({"id": edge_id, "source": source_id, "target": target_id})

    return nodes, edges


def _bert_export_inputs(
    cfg: Any,
    parameter_by_lower_name: dict[str, inspect.Parameter],
) -> ExportInputs | UnsupportedOperationGraph:
    sequence_length = _positive_int(
        getattr(cfg, "sequence_length", None),
        getattr(getattr(cfg, "experiment_config", None), "sequence_length", None),
    )
    if sequence_length is None:
        return UnsupportedOperationGraph(
            ["BERT-style operation tracing requires sequence_length metadata."]
        )
    vocab_size = _positive_int(getattr(cfg, "input_dim", None), default=128)
    input_ids = torch.arange(sequence_length, dtype=torch.long).unsqueeze(0)
    input_ids = input_ids.remainder(max(vocab_size, 1))
    if vocab_size > 2:
        input_ids[:, 0] = 2
    kwargs: dict[str, Tensor] = {
        parameter_by_lower_name["input_ids"].name: input_ids,
    }
    if "attention_mask" in parameter_by_lower_name:
        kwargs[parameter_by_lower_name["attention_mask"].name] = torch.ones_like(
            input_ids
        )
    if "token_type_ids" in parameter_by_lower_name:
        token_type_ids = torch.zeros_like(input_ids)
        if sequence_length > 2:
            token_type_ids[:, sequence_length // 2 :] = 1
        kwargs[parameter_by_lower_name["token_type_ids"].name] = token_type_ids
    return ExportInputs(args=(), kwargs=kwargs)


def _image_export_input(dataset_type: type) -> Tensor | UnsupportedOperationGraph:
    channels = _positive_int(getattr(dataset_type, "num_channels", None))
    height = _positive_int(getattr(dataset_type, "default_height", None))
    width = _positive_int(getattr(dataset_type, "default_width", None))
    if channels is None or height is None or width is None:
        return UnsupportedOperationGraph(
            [
                "Image operation tracing requires dataset num_channels, "
                "default_height, and default_width metadata."
            ]
        )
    return torch.zeros((1, channels, height, width), dtype=torch.float32)


def _positive_int(*values: Any, default: int | None = None) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            integer = int(value)
        except (TypeError, ValueError):
            continue
        if integer > 0:
            return integer
    return default


def _state_owner_by_target(model: Module) -> dict[str, str]:
    owners: dict[str, str] = {}
    for name, _parameter in model.named_parameters(
        recurse=True,
        remove_duplicate=False,
    ):
        owners[name] = _owner_from_state_target(name)
    for name, _buffer in model.named_buffers(recurse=True, remove_duplicate=False):
        owners[name] = _owner_from_state_target(name)
    return owners


def _owner_from_state_target(target: str) -> str | None:
    if "." not in target:
        return "__root__"
    return target.rsplit(".", 1)[0]


def _input_spec_by_name(exported_program: Any) -> dict[str, Any]:
    signature = getattr(exported_program, "graph_signature", None)
    input_specs = getattr(signature, "input_specs", []) if signature is not None else []
    specs: dict[str, Any] = {}
    for spec in input_specs:
        arg = getattr(spec, "arg", None)
        name = getattr(arg, "name", None)
        if name is not None:
            specs[str(name)] = spec
    return specs


def _output_spec_by_name(exported_program: Any) -> dict[str, Any]:
    signature = getattr(exported_program, "graph_signature", None)
    output_specs = (
        getattr(signature, "output_specs", []) if signature is not None else []
    )
    specs: dict[str, Any] = {}
    for spec in output_specs:
        arg = getattr(spec, "arg", None)
        name = getattr(arg, "name", None)
        if name is not None:
            specs[str(name)] = spec
    return specs


def _spec_kind(spec: Any | None) -> str | None:
    if spec is None:
        return None
    kind = getattr(spec, "kind", None)
    if isinstance(kind, Enum):
        return kind.name.lower()
    if kind is None:
        return None
    return str(kind).split(".")[-1].lower()


def _output_names(fx_node: Any) -> list[str]:
    names: list[str] = []

    def visit(value: Any) -> None:
        if hasattr(value, "name"):
            names.append(str(value.name))
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(fx_node.args)
    return names


def _infer_owner_from_inputs(
    fx_node: Any,
    owner_by_fx_node: dict[Any, str | None],
    input_kind_by_fx_node: dict[Any, str | None],
) -> str | None:
    state_owners = [
        owner_by_fx_node.get(input_node)
        for input_node in fx_node.all_input_nodes
        if input_kind_by_fx_node.get(input_node) in {"parameter", "buffer"}
        and owner_by_fx_node.get(input_node) is not None
    ]
    if state_owners:
        return state_owners[0]
    for input_node in fx_node.all_input_nodes:
        owner = owner_by_fx_node.get(input_node)
        if owner is not None:
            return owner
    return None


def _tensor_details(meta: Mapping[str, Any]) -> dict[str, Any]:
    tensor_meta = meta.get("tensor_meta")
    if tensor_meta is not None:
        summary = _tensor_summary(tensor_meta)
        if isinstance(summary, dict):
            return summary
    value = meta.get("val")
    summary = _tensor_summary(value)
    if isinstance(summary, dict):
        return summary
    if isinstance(summary, list):
        return {"values": summary}
    return {}


def _tensor_summary(value: Any) -> dict[str, Any] | list[Any] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [
            summary
            for summary in (_tensor_summary(item) for item in value)
            if summary is not None
        ]
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        return None
    summary: dict[str, Any] = {
        "shape": [int(dimension) for dimension in tuple(shape)],
        "dtype": _dtype_string(dtype),
    }
    requires_grad = getattr(value, "requires_grad", None)
    if requires_grad is not None:
        summary["requiresGrad"] = bool(requires_grad)
    return summary


def _dtype_string(dtype: Any) -> str:
    text = str(dtype)
    return text.removeprefix("torch.")


def _node_label(fx_node: Any, details: Mapping[str, Any]) -> str:
    if fx_node.op == "placeholder":
        input_kind = details.get("inputKind")
        target_path = details.get("targetPath")
        if isinstance(target_path, str) and target_path:
            leaf_name = target_path.rsplit(".", 1)[-1]
            if input_kind == "parameter":
                return f"parameter {leaf_name}"
            if input_kind == "buffer":
                return f"buffer {leaf_name}"
        return f"input {fx_node.name}"
    if fx_node.op == "output":
        return "output"
    target = _target_string(fx_node.target)
    parts = target.split(".")
    if len(parts) >= 2 and parts[-1] in {"default", "Tensor", "Scalar"}:
        return parts[-2]
    return parts[-1] if parts else target


def _target_string(target: Any) -> str:
    if isinstance(target, str):
        return target
    name = getattr(target, "name", None)
    if callable(name):
        try:
            return str(name())
        except Exception:
            pass
    simple_name = getattr(target, "__name__", None)
    if simple_name:
        module = getattr(target, "__module__", None)
        return f"{module}.{simple_name}" if module else str(simple_name)
    return str(target)


def _json_safe_object(value: Mapping[str, Any]) -> dict[str, Any]:
    safe_value = _json_safe(value)
    return safe_value if isinstance(safe_value, dict) else {}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Mapping):
        return {str(key): _json_safe(entry_value) for key, entry_value in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return str(value)


def _unsupported_response(
    model_name: str,
    preset: str,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        **model_identity_payload_from_id(model_name),
        "preset": preset,
        "source": OPERATION_GRAPH_SOURCE,
        "status": "unsupported",
        "nodes": [],
        "edges": [],
        "warnings": warnings,
    }


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    if len(message) > 500:
        message = f"{message[:497]}..."
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__
