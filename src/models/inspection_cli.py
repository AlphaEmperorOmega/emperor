from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

from model_runtime.inspection import (
    InspectionError,
    InspectionRequest,
    ParsedOverrides,
    inspect_model,
)
from models.catalog import model_id_from_parts, model_package
from models.parser import (
    get_experiment_parser,
    resolve_dataset_names,
    resolve_experiment_mode,
)


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


def _print_tree(payload: Mapping[str, Any]) -> None:
    nodes = list(payload["nodes"])
    node_by_id = {node["id"]: node for node in nodes}
    children: dict[str, list[str]] = {}
    for edge in payload["edges"]:
        children.setdefault(edge["source"], []).append(edge["target"])
    root = nodes[0]
    print(f"model: {root['typeName']}{_details_suffix(root['details'])}")

    def walk(node_id: str, prefix: str) -> None:
        child_ids = children.get(node_id, [])
        for index, child_id in enumerate(child_ids):
            child = node_by_id[child_id]
            last = index == len(child_ids) - 1
            branch = "`- " if last else "|- "
            next_prefix = prefix + ("   " if last else "|  ")
            print(
                f"{prefix}{branch}{child['path'].split('.')[-1]}: "
                f"{child['typeName']}{_details_suffix(child['details'])}"
            )
            walk(child_id, next_prefix)

    walk(root["id"], "")


def _parse_args(argv: Sequence[str]):
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
    parser = get_experiment_parser(
        package.preset_type.names(),
        package.runtime_defaults.__name__.rsplit(".", 1)[0],
    )
    parser.prog = "cli.py"
    parser.description = "Inspect model structures generated by experiment presets."
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args(list(argv))
    return args, package


def run_inspection(argv: Sequence[str]) -> int:
    try:
        args, package = _parse_args(argv)
        if getattr(args, "monitors", None):
            raise InspectionError(
                "--print-model inspection does not support --monitors."
            )
        mode = resolve_experiment_mode(args, package.preset_type)
        if mode.search_mode is not None or mode.search_keys or mode.search_overrides:
            raise InspectionError(
                "--print-model inspection does not support search modes."
            )
        if mode.selected_presets is not None:
            raise InspectionError("--print-model inspection requires one --preset.")
        preset = package.resolve_preset(args.preset)
        datasets = resolve_dataset_names(
            package.dataset_options_for_task(mode.experiment_task),
            args.datasets,
        )
        result = inspect_model(
            package,
            InspectionRequest(
                preset=package.preset_name(preset),
                dataset=datasets[0].__name__,
                experiment_task=package.task_name(mode.experiment_task),
                overrides=ParsedOverrides(mode.config_overrides),
            ),
        )
        payload = _result_payload(result)
        if args.format == "json":
            print(json.dumps(payload, indent=2))
            return 0
        print("=" * 100)
        print(preset.name)
        description_for_preset = getattr(
            package.presets, "description_for_preset", None
        )
        description = (
            description_for_preset(preset)
            if callable(description_for_preset)
            else (preset.value if isinstance(preset.value, str) else "")
        )
        if description:
            print(f"description: {description}")
        _print_tree(payload)
        return 0
    except InspectionError as exc:
        print(str(exc), file=sys.stderr)
        return 1


__all__ = ["run_inspection"]
