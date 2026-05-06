import argparse
import importlib
from collections import Counter
from enum import Enum
from types import MethodType
from typing import Any

import torch
from torch.nn import Module

from emperor.datasets.image.classification.cifar_10 import Cifar10
from emperor.datasets.image.classification.cifar_100 import Cifar100
from emperor.datasets.image.classification.fashion_mnist import FashionMNIST
from emperor.datasets.image.classification.mnist import Mnist


DATASETS = {
    "mnist": Mnist,
    "fashion_mnist": FashionMNIST,
    "cifar10": Cifar10,
    "cifar100": Cifar100,
}

MODEL_MODULES = {
    "linear": "models.linear",
    "linear_adaptive": "models.linear_adaptive",
}


def _load_model_parts(model_name: str) -> tuple[type[Enum], Any, type[Module]]:
    module_path = MODEL_MODULES[model_name]
    presets_module = importlib.import_module(f"{module_path}.presets")
    model_module = importlib.import_module(f"{module_path}.model")
    return (
        presets_module.ExperimentOptions,
        presets_module.ExperimentPresets(),
        model_module.Model,
    )


def _count_parameters(model: Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def _count_layer_features(model: Module) -> tuple[int, int, int]:
    layers = 0
    gated = 0
    halting = 0
    for module in model.modules():
        if hasattr(module, "gate_model") or hasattr(module, "halting_model"):
            layers += 1
            if getattr(module, "gate_model", None) is not None:
                gated += 1
            if getattr(module, "halting_model", None) is not None:
                halting += 1
    return layers, gated, halting


def _module_type_counts(model: Module) -> str:
    counts = Counter(type(module).__name__ for module in model.modules())
    return ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))


def _input_shape(dataset: type, batch_size: int) -> tuple[int, ...]:
    if dataset in (Mnist, FashionMNIST):
        return batch_size, 1, 28, 28
    if dataset in (Cifar10, Cifar100):
        return batch_size, 3, 32, 32
    raise ValueError(f"No dummy input shape configured for {dataset.__name__}.")


def _register_call_counter(
    module: Module,
    calls: Counter[str],
    key: str,
    handles: list[Any],
) -> None:
    def _hook(_module: Module, _inputs: tuple[Any, ...], _output: Any) -> None:
        calls[key] += 1

    handles.append(module.register_forward_hook(_hook))


def _register_method_counter(
    module: Module,
    method_name: str,
    calls: Counter[str],
    key: str,
    restore_callbacks: list[Any],
) -> None:
    original_method = getattr(module, method_name)

    def _wrapped(self: Module, *args: Any, **kwargs: Any) -> Any:
        calls[key] += 1
        return original_method(*args, **kwargs)

    setattr(module, method_name, MethodType(_wrapped, module))
    restore_callbacks.append(lambda: setattr(module, method_name, original_method))


def _check_feature_execution(model: Module, dataset: type, batch_size: int) -> None:
    calls: Counter[str] = Counter()
    handles = []
    restore_callbacks = []

    for module in model.modules():
        gate_model = getattr(module, "gate_model", None)
        if gate_model is not None:
            _register_call_counter(gate_model, calls, "gate_model", handles)

        halting_model = getattr(module, "halting_model", None)
        if halting_model is not None:
            _register_method_counter(
                halting_model,
                "update_halting_state",
                calls,
                "halting_model",
                restore_callbacks,
            )

    try:
        model.eval()
        with torch.no_grad():
            model(torch.randn(_input_shape(dataset, batch_size), device=model.device))
    finally:
        for handle in handles:
            handle.remove()
        for restore_callback in restore_callbacks:
            restore_callback()

    print(
        "forward calls: "
        f"gate_model={calls['gate_model']} "
        f"halting_model={calls['halting_model']}"
    )


def _module_details(module: Module) -> str:
    details = []
    if hasattr(module, "gate_model"):
        details.append(
            "gate="
            + ("enabled" if getattr(module, "gate_model", None) is not None else "off")
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


def _option_description(option: Enum) -> str:
    value = option.value
    if isinstance(value, str):
        return value
    return ""


def _print_model(
    option: Enum,
    config_index: int,
    config_count: int,
    model: Module,
    dataset: type,
    batch_size: int,
    compact: bool,
    structure: bool,
    check_forward: bool,
) -> None:
    total_params, trainable_params = _count_parameters(model)
    layers, gated_layers, halting_layers = _count_layer_features(model)
    suffix = f" config {config_index + 1}/{config_count}" if config_count > 1 else ""

    print("=" * 100)
    print(f"{option.name}{suffix}")
    description = _option_description(option)
    if description:
        print(f"description: {description}")
    print(f"parameters: total={total_params:,} trainable={trainable_params:,}")
    print(f"layers: total={layers} gated={gated_layers} halting={halting_layers}")
    if check_forward:
        _check_feature_execution(model, dataset, batch_size)

    if compact:
        print(f"module types: {_module_type_counts(model)}")
        return

    if structure:
        _print_module_tree(model)
        return

    print(model)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the concrete models generated by experiment presets."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_MODULES),
        default="linear",
        help="Model package to inspect.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="mnist",
        help="Dataset class passed to the preset builder.",
    )
    parser.add_argument(
        "--option",
        help="Single experiment option to print. Defaults to every option.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print counts and feature presence without the full module tree.",
    )
    parser.add_argument(
        "--structure",
        action="store_true",
        help="Print a class tree with dimensions and gate/halting state.",
    )
    parser.add_argument(
        "--check-forward",
        action="store_true",
        help="Run one dummy batch and count gate/halting forward calls.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Dummy batch size used by --check-forward.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    experiment_options, presets, model_type = _load_model_parts(args.model)
    dataset = DATASETS[args.dataset]

    if args.option is None:
        options = list(experiment_options)
    else:
        options = [experiment_options.get_option(args.option)]

    for option in options:
        configs = presets.get_config(option, dataset)
        for index, cfg in enumerate(configs):
            model = model_type(cfg)
            _print_model(
                option,
                index,
                len(configs),
                model,
                dataset,
                args.batch_size,
                args.compact,
                args.structure,
                args.check_forward,
            )


if __name__ == "__main__":
    main()
