from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from viewer.backend.inspector.discovery import (
    load_model_parts,
    option_cli_name,
    resolve_dataset,
)
from viewer.backend.inspector.errors import InspectorError
from viewer.backend.inspector.graph import serialize_graph
from viewer.backend.inspector.overrides import parse_override_mapping
from viewer.backend.inspector.schema import preset_locks


def reject_locked_overrides(
    model_name: str,
    preset_name: str,
    config_overrides: Mapping[str, Any] | None,
) -> None:
    locks = preset_locks(model_name, preset_name)
    locked_keys = sorted(set(config_overrides or {}) & set(locks))
    if not locked_keys:
        return
    details = ", ".join(f"{key} ({locks[key].reason})" for key in locked_keys)
    raise InspectorError(
        f"Preset '{preset_name}' does not allow overriding locked fields: {details}"
    )


def build_config(
    model_name: str,
    preset_name: str,
    dataset_name: str | None = None,
    config_overrides: dict[str, Any] | None = None,
):
    parts = load_model_parts(model_name)
    try:
        option = parts.experiment_options.get_option(preset_name)
    except Exception as exc:
        raise InspectorError(
            f"Unknown preset '{preset_name}' for model '{model_name}'."
        ) from exc

    try:
        dataset = resolve_dataset(parts, dataset_name)
        configs = parts.presets.get_config(
            option,
            dataset,
            config_overrides=config_overrides or {},
        )
    except Exception as exc:
        raise InspectorError(
            f"Failed to build preset '{preset_name}' for model '{model_name}': {exc}"
        ) from exc

    if not configs:
        raise InspectorError(
            f"Preset '{preset_name}' for model '{model_name}' did not produce configs."
        )
    return parts, option, configs[0]


def inspect_model(
    model_name: str,
    preset_name: str,
    overrides: Mapping[str, Any] | None = None,
    dataset: str | None = None,
    *,
    parsed_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parts = load_model_parts(model_name)
    config_overrides = (
        parsed_overrides
        if parsed_overrides is not None
        else parse_override_mapping(parts.config_module, overrides)
    )
    reject_locked_overrides(model_name, preset_name, config_overrides)
    _parts, option, cfg = build_config(
        model_name,
        preset_name,
        dataset_name=dataset,
        config_overrides=config_overrides,
    )

    try:
        model = parts.model_type(cfg)
    except Exception as exc:
        raise InspectorError(
            f"Failed to instantiate model '{model_name}' preset '{preset_name}': {exc}"
        ) from exc

    nodes, edges = serialize_graph(model)
    return {
        "model": model_name,
        "preset": option_cli_name(parts.experiment_options, option),
        "parameterCount": nodes[0]["parameterCount"] if nodes else 0,
        "nodes": nodes,
        "edges": edges,
    }
