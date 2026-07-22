from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

from models.catalog import (
    discover_model_identities,
    discover_model_identities_for_type,
    discover_model_types,
    model_id_from_parts,
    model_type_exists,
)
from models.config_overrides import (
    print_config_options,
    print_dataset_options,
    print_monitor_options,
    print_preset_options,
)

COMMAND = "mise run experiment --"
MODEL_TYPE_ARG = "<type>"
MODEL_NAME_ARG = "<name>"
MODEL_SELECTOR_ARG = f"--model-type {MODEL_TYPE_ARG} --model {MODEL_NAME_ARG}"
PRESET_ARG = "<preset>"
OPTS_ARG = "[options]"
INSPECTION_FLAGS = {
    "--print-model": None,
    "--print-model-shapes": "outputs",
    "--print-model-tensor-shapes": "variables",
}


def _captured_lines(action: Callable[[], None]) -> list[str]:
    """Capture a catalog printer while preserving its exact line-oriented output."""

    import contextlib
    import io

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        action()
    return output.getvalue().splitlines()


def list_model_types() -> None:
    print("Available model types:")
    for model_type in discover_model_types():
        print(f"  --model-type {model_type}")


def list_models(model_type: str = "") -> None:
    if model_type:
        identities = discover_model_identities_for_type(model_type)
        print(f"Available models for --model-type {model_type}:")
        for identity in identities:
            print(f"  --model {identity.model}")
        return
    print("Available models:")
    for identity in discover_model_identities():
        print(f"  --model-type {identity.model_type} --model {identity.model}")


def list_datasets(model_type: str, model: str) -> None:
    model_id = f"{model_type}/{model}"
    print(f"Available datasets for --model-type {model_type} --model {model}:")
    for dataset in _captured_lines(lambda: print_dataset_options(model_id)):
        if dataset:
            print(f"  --datasets {dataset}")


def list_monitors(model_type: str, model: str) -> None:
    model_id = f"{model_type}/{model}"
    print(f"Available monitors for --model-type {model_type} --model {model}:")
    for monitor in _captured_lines(lambda: print_monitor_options(model_id)):
        if monitor:
            print(f"  --monitors {monitor}")


def list_flags() -> None:
    rows = (
        (
            "--list-model-types",
            "Show available model categories",
            f"{COMMAND} --list-model-types",
        ),
        (
            "--list-models",
            "Show available models",
            f"{COMMAND} --list-models\n"
            f"                          {COMMAND} --model-type linears "
            "--list-models",
        ),
        (
            "--list-datasets",
            "Show available datasets for a model",
            f"{COMMAND} {MODEL_SELECTOR_ARG} --list-datasets",
        ),
        (
            "--list-monitors",
            "Show available monitor callbacks for a model",
            f"{COMMAND} {MODEL_SELECTOR_ARG} --list-monitors",
        ),
        (
            f"--model-type {MODEL_TYPE_ARG}",
            "Select a model category",
            f"{COMMAND} {MODEL_SELECTOR_ARG} --list-presets",
        ),
        (
            f"--model {MODEL_NAME_ARG}",
            "Select a model package name",
            f"{COMMAND} {MODEL_SELECTOR_ARG} --preset {PRESET_ARG}",
        ),
        (
            "--list-presets",
            "Show available presets for a model",
            f"{COMMAND} {MODEL_SELECTOR_ARG} --list-presets",
        ),
        (
            "--list-config",
            "Show overridable config flags and defaults",
            f"{COMMAND} {MODEL_SELECTOR_ARG} --list-config",
        ),
        (
            f"--preset {PRESET_ARG}",
            "Run one model preset",
            f"{COMMAND} {MODEL_SELECTOR_ARG} --preset {PRESET_ARG}",
        ),
        (
            "--resume-checkpoint <path>",
            "Continue one Run from a trusted Lightning checkpoint",
            f"{COMMAND} {MODEL_SELECTOR_ARG} --preset {PRESET_ARG} "
            "--datasets mnist --resume-checkpoint logs/.../last.ckpt "
            "--config --num-epochs 30",
        ),
        (
            "--presets p1 p2",
            "Run selected model presets sequentially",
            f"{COMMAND} {MODEL_SELECTOR_ARG} --presets baseline gating --grid-search",
        ),
        (
            "--print-model",
            "Print model structure instead of running training",
            f"{COMMAND} {MODEL_SELECTOR_ARG} --preset {PRESET_ARG} --print-model",
        ),
        (
            "--print-model-shapes",
            "Print the model tree with executed module input/output shapes",
            f"{COMMAND} {MODEL_SELECTOR_ARG} --preset {PRESET_ARG} "
            "--print-model-shapes",
        ),
        (
            "--print-model-tensor-shapes",
            "Also print every executed Python tensor variable and method output",
            f"{COMMAND} {MODEL_SELECTOR_ARG} --preset {PRESET_ARG} "
            "--print-model-tensor-shapes",
        ),
        (
            "--monitors <names...>",
            "Enable monitor callbacks for a training run",
            f"{COMMAND} {MODEL_SELECTOR_ARG} --preset {PRESET_ARG} "
            "--monitors linear halting",
        ),
        (
            "--all-presets",
            "Run all model presets sequentially",
            f"{COMMAND} {MODEL_SELECTOR_ARG} --all-presets",
        ),
        (
            "--grid-search",
            "Run grid search for the selected option",
            f"{COMMAND} {MODEL_SELECTOR_ARG} --preset {PRESET_ARG} --grid-search",
        ),
        (
            "--random-search <n>",
            "Run random search with <n> sampled combinations",
            f"{COMMAND} {MODEL_SELECTOR_ARG} --preset {PRESET_ARG} --random-search <n>",
        ),
        (
            "--search-keys <keys>",
            "Restrict sweep to named config-file search axes",
            f"{COMMAND} {MODEL_SELECTOR_ARG} --preset {PRESET_ARG} "
            "--grid-search --search-keys HIDDEN_DIM STACK_NUM_LAYERS",
        ),
        (
            "--search-set <k=v,..>",
            "Sweep command-line values for one search axis",
            f"{COMMAND} {MODEL_SELECTOR_ARG} --preset {PRESET_ARG} "
            "--grid-search --search-set hidden_dim=64,128",
        ),
        (
            "--config <flags...>",
            "Override model config values without editing config.py",
            f"{COMMAND} {MODEL_SELECTOR_ARG} --preset {PRESET_ARG} "
            "--config --num-epochs 30 "
            "--callback-early-stopping-patience 0",
        ),
        (
            "--logdir <dir>",
            "Store model logs in <dir>",
            f"{COMMAND} {MODEL_SELECTOR_ARG} --preset {PRESET_ARG} --logdir <dir>",
        ),
    )
    print("Available flags:")
    print()
    flag_width = max(24, *(len(flag) for flag, _description, _example in rows))
    example_indent = " " * (flag_width + 2)
    for flag, description, example in rows:
        print(f"  {flag:{flag_width}} ## {description}")
        print(f"{example_indent}{example}")
        if flag != rows[-1][0]:
            print("-" * 95)


def show_no_argument_usage_help() -> None:
    print(f"Usage: {COMMAND} {MODEL_SELECTOR_ARG} {OPTS_ARG}")
    print()
    list_flags()


def show_model_list_usage(model_type: str) -> None:
    if model_type:
        print(
            f"Usage: {COMMAND} --model-type {model_type} "
            f"--model {MODEL_NAME_ARG} {OPTS_ARG}"
        )
    else:
        print(f"Usage: {COMMAND} {MODEL_SELECTOR_ARG} {OPTS_ARG}")
    print()
    list_models(model_type)


def show_model_type_list_usage() -> None:
    print(f"Usage: {COMMAND} {MODEL_SELECTOR_ARG} {OPTS_ARG}")
    print()
    list_model_types()


def show_dataset_list_usage(model_type: str, model: str) -> None:
    print(f"Usage: {COMMAND} --model-type {model_type} --model {model} {OPTS_ARG}")
    print()
    list_datasets(model_type, model)


def show_monitor_list_usage(model_type: str, model: str) -> None:
    print(f"Usage: {COMMAND} --model-type {model_type} --model {model} {OPTS_ARG}")
    print()
    list_monitors(model_type, model)


def show_unknown_model_error(model_type: str, model: str) -> None:
    print(f"Error: unknown model '--model-type {model_type} --model {model}'")
    print()
    print(f"Run '{COMMAND} --list-models' to see available models.")
    print(f"Run '{COMMAND}' to see available flags.")


def show_unknown_model_type_error(model_type: str) -> None:
    print(f"Error: unknown model type '--model-type {model_type}'.")
    print()
    print(f"Run '{COMMAND} --list-model-types' to see available model types.")


def show_missing_model_selector_error() -> None:
    print(
        "Error: model commands require "
        f"--model-type {MODEL_TYPE_ARG} --model {MODEL_NAME_ARG}."
    )
    print()
    print(f"Run '{COMMAND} --list-models' to see available models.")
    print(f"Run '{COMMAND}' to see available flags.")


def show_missing_flag_value_error(flag: str) -> None:
    print(f"Error: {flag} requires a value.")
    print()
    print(f"Run '{COMMAND}' to see available flags.")


def show_positional_model_error(value: str) -> None:
    model_type = MODEL_TYPE_ARG
    model = MODEL_NAME_ARG
    if "/" in value:
        model_type, model = value.split("/", 1)
    print(
        "Positional model ids are no longer supported. Use "
        f"--model-type {model_type} --model {model} ..."
    )


def _print_model_errors(
    model_type: str,
    model: str,
    arguments: Sequence[str],
) -> bool:
    selected_flags = [flag for flag in INSPECTION_FLAGS if flag in arguments]
    if not selected_flags:
        return False
    if len(selected_flags) > 1:
        print(
            "Error: choose only one model inspection flag: "
            + ", ".join(INSPECTION_FLAGS)
            + "."
        )
        return True
    selected_flag = selected_flags[0]
    if "--all-presets" in arguments or "--presets" in arguments:
        print(
            f"Error: {selected_flag} requires --preset {PRESET_ARG} and cannot be "
            "used with --all-presets or --presets."
        )
        print()
        print(
            f"Run '{COMMAND} --model-type {model_type} --model {model} "
            f"--preset {PRESET_ARG} {selected_flag}' to inspect one preset."
        )
        return True
    if "--monitors" in arguments:
        print(
            "Error: --monitors applies to training runs and cannot be used with "
            f"{selected_flag}."
        )
        print()
        print(
            f"Run '{COMMAND} --model-type {model_type} --model {model} "
            f"--preset {PRESET_ARG} {selected_flag}' without --monitors."
        )
        return True
    return False


def run_model_command(
    model_type: str,
    model: str,
    arguments: Sequence[str],
) -> int:
    if _print_model_errors(model_type, model, arguments):
        return 1
    selected_flag = next(
        (flag for flag in INSPECTION_FLAGS if flag in arguments),
        None,
    )
    if selected_flag is not None:
        from models.inspection_cli import run_inspection

        inspection_arguments = [
            argument for argument in arguments if argument != selected_flag
        ]
        shape_trace = INSPECTION_FLAGS[selected_flag]
        if shape_trace is not None:
            inspection_arguments.extend(["--shape-trace", shape_trace])
        return run_inspection(
            ["--model-type", model_type, "--model", model, *inspection_arguments]
        )
    catalog_key = model_id_from_parts(model_type, model)
    if catalog_key is None:
        return 1
    from models.package_cli import run_model_package_cli

    return run_model_package_cli(catalog_key, arguments)


def run_experiment(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        show_no_argument_usage_help()
        return 0
    if not arguments[0].startswith("--"):
        show_positional_model_error(arguments[0])
        return 1

    model_type = ""
    model = ""
    list_model_types_requested = False
    list_models_requested = False
    list_datasets_requested = False
    list_monitors_requested = False
    forwarded: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--list-model-types":
            list_model_types_requested = True
        elif argument == "--list-models":
            list_models_requested = True
        elif argument == "--list-datasets":
            list_datasets_requested = True
        elif argument == "--list-monitors":
            list_monitors_requested = True
        elif argument in {"--model-type", "--model"}:
            if index + 1 >= len(arguments) or arguments[index + 1] == "":
                show_missing_flag_value_error(argument)
                return 1
            if argument == "--model-type":
                model_type = arguments[index + 1]
            else:
                model = arguments[index + 1]
            index += 1
        else:
            forwarded.append(argument)
        index += 1

    if list_model_types_requested:
        show_model_type_list_usage()
        return 0
    if list_models_requested:
        if model_type and not model_type_exists(model_type):
            show_unknown_model_type_error(model_type)
            return 1
        show_model_list_usage(model_type)
        return 0
    if list_datasets_requested or list_monitors_requested:
        if not model_type or not model:
            show_missing_model_selector_error()
            return 1
        if model_id_from_parts(model_type, model) is None:
            show_unknown_model_error(model_type, model)
            return 1
        if list_datasets_requested:
            show_dataset_list_usage(model_type, model)
        else:
            show_monitor_list_usage(model_type, model)
        return 0
    if not model_type or not model:
        show_missing_model_selector_error()
        return 1
    model_id = model_id_from_parts(model_type, model)
    if model_id is None:
        show_unknown_model_error(model_type, model)
        return 1
    if forwarded and forwarded[0] == "--list-presets":
        print_preset_options(model_id)
        return 0
    if forwarded and forwarded[0] == "--list-config":
        print_config_options(model_id)
        return 0
    if forwarded == ["--preset"]:
        print_preset_options(model_id)
        return 0
    return run_model_command(model_type, model, forwarded)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "experiment":
        arguments.pop(0)
    if arguments and arguments[0] == "inspect":
        from models.inspection_cli import run_inspection

        return run_inspection(arguments[1:])
    return run_experiment(arguments)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_experiment", "run_model_command"]
