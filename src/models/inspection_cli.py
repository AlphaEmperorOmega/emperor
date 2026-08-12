from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from model_runtime.inspection import (
    InspectionError,
    InspectionRequest,
    InspectionResult,
    MethodShapeTrace,
    ModelShapeTrace,
    ParsedOverrides,
    TensorShape,
    inspect_model,
    inspect_model_shapes,
)
from model_runtime.packages import ModelPackage
from models.catalog import model_id_from_parts, model_package
from models.cli_selection import resolve_cli_selection
from models.experiment_cli_parser import get_experiment_parser


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {_camel_case(str(key)): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    if isinstance(value, Enum):
        return value.name
    return value


def _camel_case(key: str) -> str:
    pieces = key.split("_")
    return pieces[0] + "".join(piece[:1].upper() + piece[1:] for piece in pieces[1:])


def _configuration_payload(configuration) -> dict[str, Any] | None:
    if configuration is None:
        return None
    fields = []
    for field in configuration.fields:
        payload = {"key": field.key, "value": _thaw(field.value)}
        if field.description is not None:
            payload["description"] = field.description
        fields.append(payload)
    return {"typeName": configuration.type_name, "fields": fields}


def _node_payload(node) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": node.id,
        "label": node.type_name,
        "typeName": node.type_name,
    }
    if node.description is not None:
        payload["description"] = node.description
    payload.update(
        {
            "path": node.path,
            "graphRole": node.graph_role,
            "parameterCount": node.parameter_count,
            "parameterSizeBytes": node.parameter_size_bytes,
            "details": _thaw(node.details),
            "config": _configuration_payload(node.configuration),
        }
    )
    return payload


def _result_payload(result) -> dict[str, Any]:
    return {
        "modelType": result.identity.model_type,
        "model": result.identity.model,
        "preset": result.preset,
        "parameterCount": result.parameter_count,
        "parameterSizeBytes": result.parameter_size_bytes,
        "nodes": [_node_payload(node) for node in result.nodes],
        "edges": [
            {"id": edge.id, "source": edge.source, "target": edge.target}
            for edge in result.edges
        ],
    }


def _tensor_shape_payload(tensor: TensorShape) -> dict[str, Any]:
    return {
        "name": tensor.name,
        "shape": list(tensor.shape),
        "dtype": tensor.dtype,
        "device": tensor.device,
    }


def _shape_trace_payload(trace: ModelShapeTrace) -> dict[str, Any]:
    return {
        "dataset": trace.dataset,
        "experimentTask": trace.experiment_task,
        "batchSize": trace.batch_size,
        "sampleInputs": [
            _tensor_shape_payload(tensor) for tensor in trace.sample_inputs
        ],
        "modules": [
            {
                "nodeId": module.node_id,
                "calls": [
                    {
                        "inputs": [
                            _tensor_shape_payload(tensor) for tensor in call.inputs
                        ],
                        "outputs": [
                            _tensor_shape_payload(tensor) for tensor in call.outputs
                        ],
                    }
                    for call in module.calls
                ],
            }
            for module in trace.modules
        ],
        "methods": [
            {
                "id": method.id,
                "parentId": method.parent_id,
                "order": method.order,
                "qualifiedName": method.qualified_name,
                "modulePath": method.module_path,
                "sourcePath": method.source_path,
                "firstLine": method.first_line,
                "inputs": [_tensor_shape_payload(tensor) for tensor in method.inputs],
                "variables": [
                    {
                        "order": variable.order,
                        "line": variable.line,
                        "tensors": [
                            _tensor_shape_payload(tensor) for tensor in variable.tensors
                        ],
                    }
                    for variable in method.variables
                ],
                "outputs": [_tensor_shape_payload(tensor) for tensor in method.outputs],
            }
            for method in trace.methods
        ],
    }


def _details_suffix(details: Mapping[str, Any]) -> str:
    pieces: list[str] = []
    if "gate" in details:
        pieces.append(f"gate={'enabled' if details['gate'] else 'off'}")
    if "halting" in details:
        pieces.append(f"halting={'enabled' if details['halting'] else 'off'}")
    if "dims" in details:
        pieces.append(str(details["dims"]).replace(" -> ", "->"))
    if "dropout" in details:
        pieces.append(f"dropout={details['dropout']}")
    if "activation" in details:
        pieces.append(f"activation={details['activation']}")
    if "layerNorm" in details:
        pieces.append(f"layer_norm={details['layerNorm']}")
    recurrent = details.get("recurrent")
    if isinstance(recurrent, Mapping):
        pieces.append(
            "recurrent="
            f"steps:{recurrent.get('maxSteps')},"
            f"gate:{'enabled' if recurrent.get('gate') else 'off'},"
            f"halting:{'enabled' if recurrent.get('halting') else 'off'}"
        )
    return " [" + ", ".join(pieces) + "]" if pieces else ""


def _format_tensor_shape(tensor: TensorShape) -> str:
    dimensions = ",".join(str(dimension) for dimension in tensor.shape)
    device = f"@{tensor.device}" if tensor.device != "cpu" else ""
    return f"{tensor.dtype}[{dimensions}]{device}"


def _format_tensors(tensors: Sequence[TensorShape]) -> str:
    if not tensors:
        return "none"
    return ", ".join(
        f"{tensor.name}={_format_tensor_shape(tensor)}" for tensor in tensors
    )


def _module_shape_suffix(calls) -> str:
    if not calls:
        return " {not called}"
    transitions = [
        f"in: {_format_tensors(call.inputs)} -> out: {_format_tensors(call.outputs)}"
        for call in calls
    ]
    if all(transition == transitions[0] for transition in transitions):
        count = f"{len(calls)} calls; " if len(calls) > 1 else ""
        return " {" + count + transitions[0] + "}"
    numbered = "; ".join(
        f"#{index}: {transition}"
        for index, transition in enumerate(transitions, start=1)
    )
    return f" {{{len(calls)} calls; {numbered}}}"


def _print_tree(payload: Mapping[str, Any], module_traces=()) -> None:
    nodes = list(payload["nodes"])
    node_by_id = {node["id"]: node for node in nodes}
    trace_by_node_id = {trace.node_id: trace for trace in module_traces}
    children: dict[str, list[str]] = {}
    for edge in payload["edges"]:
        children.setdefault(edge["source"], []).append(edge["target"])
    root = nodes[0]
    root_trace = trace_by_node_id.get(root["id"])
    root_shapes = _module_shape_suffix(root_trace.calls) if root_trace else ""
    print(f"model: {root['typeName']}{_details_suffix(root['details'])}{root_shapes}")
    expanded_node_ids = {root["id"]}

    def walk(node_id: str, prefix: str) -> None:
        child_ids = children.get(node_id, [])
        for index, child_id in enumerate(child_ids):
            child = node_by_id[child_id]
            last = index == len(child_ids) - 1
            branch = "`- " if last else "|- "
            next_prefix = prefix + ("   " if last else "|  ")
            trace = trace_by_node_id.get(child_id)
            shape_suffix = _module_shape_suffix(trace.calls) if trace else ""
            if child_id in expanded_node_ids:
                print(
                    f"{prefix}{branch}{child['path'].split('.')[-1]}: "
                    f"{child['typeName']} [reference]"
                )
                continue
            expanded_node_ids.add(child_id)
            print(
                f"{prefix}{branch}{child['path'].split('.')[-1]}: "
                f"{child['typeName']}{_details_suffix(child['details'])}"
                f"{shape_suffix}"
            )
            walk(child_id, next_prefix)

    walk(root["id"], "")


class _MethodTreeRenderer:
    def __init__(self, methods: Sequence[MethodShapeTrace]) -> None:
        self._method_by_id = {method.id: method for method in methods}
        self._children: dict[int | None, list[int]] = {}
        for method in methods:
            self._children.setdefault(method.parent_id, []).append(method.id)
        self._relevant: dict[int, bool] = {}

    def render(self) -> None:
        root_ids = [
            method_id
            for method_id in self._children.get(None, [])
            if self._has_tensor_content(method_id)
        ]
        print("tensor variables (executed Python):")
        for index, method_id in enumerate(root_ids):
            branch = "`- " if index == len(root_ids) - 1 else "|- "
            self._walk(method_id, "", branch)

    def _has_tensor_content(self, method_id: int) -> bool:
        cached = self._relevant.get(method_id)
        if cached is not None:
            return cached
        method = self._method_by_id[method_id]
        result = bool(method.inputs or method.variables or method.outputs) or any(
            self._has_tensor_content(child_id)
            for child_id in self._children.get(method_id, [])
        )
        self._relevant[method_id] = result
        return result

    @staticmethod
    def _method_label(method: MethodShapeTrace) -> str:
        location = f"{method.source_path}:{method.first_line}"
        owner = f"{method.module_path} :: " if method.module_path else ""
        output = f" -> {_format_tensors(method.outputs)}" if method.outputs else ""
        return f"{owner}{method.qualified_name} ({location}){output}"

    def _ordered_entries(self, method: MethodShapeTrace) -> list[tuple[int, str, Any]]:
        entries: list[tuple[int, str, Any]] = []
        if method.inputs:
            entries.append((method.order, "inputs", method.inputs))
        entries.extend(
            (variable.order, "variable", variable) for variable in method.variables
        )
        entries.extend(
            (self._method_by_id[child_id].order, "method", child_id)
            for child_id in self._children.get(method.id, [])
            if self._has_tensor_content(child_id)
        )
        entries.sort(key=lambda entry: entry[0])
        return entries

    def _walk(self, method_id: int, prefix: str, branch: str) -> None:
        method = self._method_by_id[method_id]
        print(f"{prefix}{branch}{self._method_label(method)}")
        entries = self._ordered_entries(method)
        child_prefix = prefix + ("   " if branch == "`- " else "|  ")
        for index, (_order, kind, value) in enumerate(entries):
            last = index == len(entries) - 1
            child_branch = "`- " if last else "|- "
            if kind == "method":
                self._walk(int(value), child_prefix, child_branch)
                continue
            if kind == "inputs":
                label = f"inputs: {_format_tensors(value)}"
            else:
                line = f"line {value.line}" if value.line is not None else "locals"
                label = f"{line}: {_format_tensors(value.tensors)}"
            print(f"{child_prefix}{child_branch}{label}")


def _print_method_tree(methods: Sequence[MethodShapeTrace]) -> None:
    _MethodTreeRenderer(methods).render()


def _parse_args(
    argv: Sequence[str],
) -> tuple[argparse.Namespace, ModelPackage]:
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--model-type", required=True)
    selector.add_argument("--model", required=True)
    selected, _ = selector.parse_known_args(argv)
    model_id = model_id_from_parts(selected.model_type, selected.model)
    package = model_package(model_id) if model_id else None
    if package is None:
        raise SystemExit(
            f"Unknown model: --model-type {selected.model_type} "
            f"--model {selected.model}"
        )
    parser = get_experiment_parser(package)
    parser.prog = "cli.py"
    parser.description = "Inspect model structures generated by experiment presets."
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument(
        "--shape-trace",
        choices=["outputs", "variables"],
        default=None,
        help=(
            "Execute one synthetic batch and print module output shapes, or all "
            "executed Python tensor variables."
        ),
    )
    args = parser.parse_args(list(argv))
    return args, package


@dataclass(frozen=True, slots=True)
class _ResolvedInspection:
    package: ModelPackage
    preset: Any
    request: InspectionRequest
    output_format: str
    shape_trace: str | None


@dataclass(frozen=True, slots=True)
class _InspectionExecution:
    result: InspectionResult
    trace: ModelShapeTrace | None


def _resolve_inspection_request(argv: Sequence[str]) -> _ResolvedInspection:
    args, package = _parse_args(argv)
    if getattr(args, "monitors", None):
        raise InspectionError("Model inspection does not support --monitors.")
    selection = resolve_cli_selection(args, package, package.preset_type)
    if (
        selection.search_mode is not None
        or selection.search_keys
        or selection.search_overrides
    ):
        raise InspectionError("Model inspection does not support search modes.")
    if selection.selected_presets is not None:
        raise InspectionError("Model inspection requires one --preset.")
    preset = package.resolve_preset(args.preset)
    datasets = package.resolve_datasets(
        args.datasets,
        selection.experiment_task,
    )
    request = InspectionRequest(
        preset=package.preset_name(preset),
        dataset=datasets[0].__name__,
        experiment_task=package.task_name(selection.experiment_task),
        overrides=ParsedOverrides(selection.config_overrides),
    )
    return _ResolvedInspection(
        package=package,
        preset=preset,
        request=request,
        output_format=args.format,
        shape_trace=args.shape_trace,
    )


def _execute_inspection(
    resolved: _ResolvedInspection,
) -> _InspectionExecution:
    if resolved.shape_trace is None:
        return _InspectionExecution(
            result=inspect_model(resolved.package, resolved.request),
            trace=None,
        )
    result, trace = inspect_model_shapes(
        resolved.package,
        resolved.request,
        detail=resolved.shape_trace,
    )
    return _InspectionExecution(result=result, trace=trace)


def _execution_payload(execution: _InspectionExecution) -> dict[str, Any]:
    payload = _result_payload(execution.result)
    if execution.trace is not None:
        payload["shapeTrace"] = _shape_trace_payload(execution.trace)
    return payload


def _render_json_inspection(
    payload: Mapping[str, Any],
    request: InspectionRequest,
) -> None:
    encoded = json.dumps(payload, separators=(",", ":"))
    maximum_output_bytes = request.capture_limits.maximum_output_bytes
    encoded_size = len(encoded.encode("utf-8")) + 1
    if encoded_size > maximum_output_bytes:
        raise InspectionError(
            f"Inspection output byte limit of {maximum_output_bytes} exceeded."
        )
    print(encoded)


def _preset_description(package: ModelPackage, preset: Any) -> str:
    description_for_preset = getattr(package.presets, "description_for_preset", None)
    if callable(description_for_preset):
        return description_for_preset(preset)
    return preset.value if isinstance(preset.value, str) else ""


def _render_text_inspection(
    resolved: _ResolvedInspection,
    execution: _InspectionExecution,
    payload: Mapping[str, Any],
) -> None:
    print("=" * 100)
    print(resolved.preset.name)
    description = _preset_description(resolved.package, resolved.preset)
    if description:
        print(f"description: {description}")
    trace = execution.trace
    if trace is not None:
        print(
            "shape sample: "
            f"dataset={trace.dataset}, task={trace.experiment_task}, "
            f"batch={trace.batch_size}, mode=eval/no_grad"
        )
    _print_tree(payload, trace.modules if trace is not None else ())
    if trace is not None and resolved.shape_trace == "variables":
        print()
        _print_method_tree(trace.methods)


def run_inspection(argv: Sequence[str]) -> int:
    try:
        resolved = _resolve_inspection_request(argv)
        execution = _execute_inspection(resolved)
        payload = _execution_payload(execution)
        if resolved.output_format == "json":
            _render_json_inspection(payload, resolved.request)
        else:
            _render_text_inspection(resolved, execution, payload)
        return 0
    except InspectionError as exc:
        print(str(exc), file=sys.stderr)
        return 1


__all__ = ["run_inspection"]
