import argparse
import importlib
from enum import Enum
from typing import Any

from torch.nn import Module

from models.catalog import model_id_from_parts, module_path_for_model_id
from models.experiment_cli_parser import get_experiment_parser
from models.experiment_mode import resolve_experiment_mode
from models.model_metadata import load_model_metadata


def _load_model_parts(model_name: str) -> tuple[type[Enum], Any, type[Module], type]:
    module_path = module_path_for_model_id(model_name)
    if module_path is None:
        raise ValueError(f"Unknown model: {model_name}")
    metadata = load_model_metadata(model_name)
    presets_module = importlib.import_module(f"{module_path}.presets")
    model_module = importlib.import_module(f"{module_path}.model")
    return (
        presets_module.ExperimentPreset,
        presets_module.ExperimentPresets(),
        model_module.Model,
        metadata.dataset_options_for_task(metadata.default_experiment_task)[0],
    )


def _module_details(module: Module) -> str:
    details = []
    gate_attribute = None
    if hasattr(module, "gate_model"):
        gate_attribute = "gate_model"
    elif hasattr(module, "recurrent_gate"):
        gate_attribute = "recurrent_gate"
    if gate_attribute is not None:
        details.append(
            "gate="
            + (
                "enabled"
                if getattr(module, gate_attribute, None) is not None
                else "off"
            )
        )
    if hasattr(module, "halting_model"):
        details.append(
            "halting="
            + (
                "enabled"
                if getattr(module, "halting_model", None) is not None
                else "off"
            )
        )
    if hasattr(module, "input_dim") and hasattr(module, "output_dim"):
        details.append(f"{module.input_dim}->{module.output_dim}")
    if hasattr(module, "dropout_probability"):
        details.append(f"dropout={module.dropout_probability}")
    if not details:
        return ""
    return " [" + ", ".join(details) + "]"


def _print_module_tree(module: Module) -> None:
    print(f"model: {type(module).__name__}{_module_details(module)}")
    _print_tree_children(module, "")


def _print_tree_children(module: Module, prefix: str) -> None:
    children = list(module.named_children())
    for index, (child_name, child) in enumerate(children):
        branch = "`- " if index == len(children) - 1 else "|- "
        next_prefix = prefix + ("   " if index == len(children) - 1 else "|  ")
        print(
            f"{prefix}{branch}{child_name}: "
            f"{type(child).__name__}{_module_details(child)}"
        )
        _print_tree_children(child, next_prefix)


def _preset_description(preset: Enum, presets: Any | None = None) -> str:
    description_for_preset = getattr(presets, "description_for_preset", None)
    description = (
        description_for_preset(preset) if callable(description_for_preset) else None
    )
    if isinstance(description, str):
        return description
    return preset.value if isinstance(preset.value, str) else ""


def _print_model_structure(
    preset: Enum,
    presets: Any,
    config_index: int,
    config_count: int,
    model: Module,
) -> None:
    suffix = f" config {config_index + 1}/{config_count}" if config_count > 1 else ""

    print("=" * 100)
    print(f"{preset.name}{suffix}")
    description = _preset_description(preset, presets)
    if description:
        print(f"description: {description}")
    _print_module_tree(model)


def print_experiment_models(
    experiment_preset_enum: type[Enum],
    presets: Any,
    model_type: type[Module],
    dataset: type,
    args: argparse.Namespace,
    selected_presets: list[Enum] | None = None,
    search_mode: Any = None,
    search_keys: list[str] | None = None,
    config_overrides: dict | None = None,
    search_overrides: dict | None = None,
) -> None:
    preset_members = (
        selected_presets
        if selected_presets is not None
        else list(experiment_preset_enum)
        if args.all_presets
        else [experiment_preset_enum.get_member(args.preset)]
    )

    for preset in preset_members:
        configs = presets.get_config(
            preset,
            dataset,
            search_mode=search_mode,
            search_keys=search_keys,
            config_overrides=config_overrides,
            search_overrides=search_overrides,
        )
        for index, cfg in enumerate(configs):
            model = model_type(cfg)
            _print_model_structure(
                preset,
                presets,
                index,
                len(configs),
                model,
            )


def _parse_args() -> argparse.Namespace:
    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument(
        "--model-type",
        required=True,
        help="Model type to inspect.",
    )
    model_parser.add_argument(
        "--model",
        required=True,
        help="Model package name to inspect.",
    )
    model_args, _ = model_parser.parse_known_args()
    model_id = model_id_from_parts(model_args.model_type, model_args.model)
    if model_id is None:
        raise SystemExit(
            f"Unknown model: --model-type {model_args.model_type} "
            f"--model {model_args.model}"
        )
    experiment_preset_enum, _, _, _ = _load_model_parts(model_id)
    parser = get_experiment_parser(
        experiment_preset_enum.names(),
        module_path_for_model_id(model_id),
    )
    parser.description = "Print model structures generated by experiment presets."
    parser.add_argument("--model-type", required=True, help="Model type to inspect.")
    parser.add_argument("--model", required=True, help="Model package name to inspect.")
    args = parser.parse_args()
    args.model_id = model_id_from_parts(args.model_type, args.model)
    if args.model_id is None:
        raise SystemExit(
            f"Unknown model: --model-type {args.model_type} --model {args.model}"
        )
    return args


def main() -> None:
    args = _parse_args()
    experiment_preset_enum, presets, model_type, dataset = _load_model_parts(
        args.model_id
    )
    mode = resolve_experiment_mode(args, experiment_preset_enum)
    print_experiment_models(
        experiment_preset_enum,
        presets,
        model_type,
        dataset,
        args,
        selected_presets=mode.selected_presets,
        search_mode=mode.search_mode,
        search_keys=mode.search_keys,
        config_overrides=mode.config_overrides,
        search_overrides=mode.search_overrides,
    )


if __name__ == "__main__":
    main()
