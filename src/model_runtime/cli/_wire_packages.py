from __future__ import annotations

from enum import Enum
from typing import Any

from emperor.experiments import ExperimentTask, experiment_task_name
from model_runtime.cli._wire_shared import (
    WireCodecError,
    json_value_to_wire,
    wire_bool,
    wire_fields,
    wire_int,
    wire_list,
    wire_mapping,
    wire_scalar,
    wire_string,
    wire_string_list,
)
from model_runtime.packages import (
    ModelIdentity,
    ModelPackage,
    dataset_label,
    dataset_name,
)

_EXPERIMENT_TASK_NAMES = {experiment_task_name(task) for task in ExperimentTask}


def _experiment_task(value: object, path: str) -> str:
    selected = wire_string(value, path)
    if selected not in _EXPERIMENT_TASK_NAMES:
        raise WireCodecError(f"{path} is not a supported Experiment Task.")
    return selected


def identity_to_wire(identity: ModelIdentity) -> dict[str, str]:
    return {
        "model_type": wire_string(identity.model_type, "$.identity.model_type"),
        "model": wire_string(identity.model, "$.identity.model"),
    }


def identity_from_wire(payload: object) -> ModelIdentity:
    raw = wire_fields(
        payload,
        path="$.identity",
        required=("model_type", "model"),
    )
    return ModelIdentity(
        wire_string(raw["model_type"], "$.identity.model_type"),
        wire_string(raw["model"], "$.identity.model"),
    )


def _runtime_default_to_wire(value: object, path: str) -> object:
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, type):
        return value.__name__
    return json_value_to_wire(value, path=path)


def package_metadata_to_wire(package: ModelPackage) -> dict[str, Any]:
    runtime_defaults = {
        key: _runtime_default_to_wire(value, f"$.runtime_defaults.{key}")
        for key, value in vars(package.runtime_defaults).items()
        if key.isupper()
        and (value is None or isinstance(value, (str, int, float, bool, Enum, type)))
    }
    payload = {
        "identity": identity_to_wire(package.identity),
        "catalog_key": package.catalog_key,
        "presets": [
            {
                "name": package.preset_name(preset),
                "key": preset.name,
                "label": preset.name,
                "description": package.preset_description(preset),
            }
            for preset in package.preset_type
        ],
        "default_experiment_task": package.task_name(package.default_experiment_task),
        "dataset_groups": [
            {
                "experiment_task": package.task_name(task),
                "label": package.task_label(task),
                "datasets": [
                    {
                        "name": dataset_name(dataset),
                        "label": dataset_label(dataset),
                        "input_dim": int(
                            getattr(dataset, "flattened_input_dim", 0) or 0
                        ),
                        "output_dim": int(getattr(dataset, "num_classes", 0) or 0),
                    }
                    for dataset in datasets
                ],
            }
            for task, datasets in package.dataset_metadata.items()
        ],
        "monitors": [option.to_api() for option in package.monitor_options()],
        "runtime_defaults": runtime_defaults,
    }
    return package_metadata_from_wire(payload)


def package_metadata_from_wire(payload: object) -> dict[str, Any]:
    raw = wire_fields(
        payload,
        path="$",
        required=(
            "identity",
            "catalog_key",
            "presets",
            "default_experiment_task",
            "dataset_groups",
            "monitors",
            "runtime_defaults",
        ),
    )
    identity = identity_from_wire(raw["identity"])
    catalog_key = wire_string(raw["catalog_key"], "$.catalog_key")
    if catalog_key != identity.catalog_key:
        raise WireCodecError("$.catalog_key must match $.identity.")

    presets: list[dict[str, str]] = []
    for index, item in enumerate(wire_list(raw["presets"], "$.presets")):
        path = f"$.presets[{index}]"
        preset = wire_fields(
            item,
            path=path,
            required=("name", "key", "label", "description"),
        )
        presets.append(
            {
                field: wire_string(preset[field], f"{path}.{field}")
                for field in ("name", "key", "label", "description")
            }
        )

    dataset_groups: list[dict[str, Any]] = []
    for group_index, item in enumerate(
        wire_list(raw["dataset_groups"], "$.dataset_groups")
    ):
        path = f"$.dataset_groups[{group_index}]"
        group = wire_fields(
            item,
            path=path,
            required=("experiment_task", "label", "datasets"),
        )
        datasets: list[dict[str, Any]] = []
        for dataset_index, dataset_item in enumerate(
            wire_list(group["datasets"], f"{path}.datasets")
        ):
            dataset_path = f"{path}.datasets[{dataset_index}]"
            dataset = wire_fields(
                dataset_item,
                path=dataset_path,
                required=("name", "label", "input_dim", "output_dim"),
            )
            datasets.append(
                {
                    "name": wire_string(dataset["name"], f"{dataset_path}.name"),
                    "label": wire_string(dataset["label"], f"{dataset_path}.label"),
                    "input_dim": wire_int(
                        dataset["input_dim"],
                        f"{dataset_path}.input_dim",
                        minimum=0,
                    ),
                    "output_dim": wire_int(
                        dataset["output_dim"],
                        f"{dataset_path}.output_dim",
                        minimum=0,
                    ),
                }
            )
        dataset_groups.append(
            {
                "experiment_task": _experiment_task(
                    group["experiment_task"],
                    f"{path}.experiment_task",
                ),
                "label": wire_string(group["label"], f"{path}.label"),
                "datasets": datasets,
            }
        )

    monitors: list[dict[str, Any]] = []
    for index, item in enumerate(wire_list(raw["monitors"], "$.monitors")):
        path = f"$.monitors[{index}]"
        monitor = wire_fields(
            item,
            path=path,
            required=(
                "name",
                "label",
                "description",
                "kinds",
                "defaultEnabled",
            ),
        )
        monitors.append(
            {
                "name": wire_string(monitor["name"], f"{path}.name"),
                "label": wire_string(monitor["label"], f"{path}.label"),
                "description": wire_string(
                    monitor["description"],
                    f"{path}.description",
                ),
                "kinds": list(wire_string_list(monitor["kinds"], f"{path}.kinds")),
                "defaultEnabled": wire_bool(
                    monitor["defaultEnabled"],
                    f"{path}.defaultEnabled",
                ),
            }
        )

    runtime_defaults = wire_mapping(
        raw["runtime_defaults"],
        "$.runtime_defaults",
    )
    return {
        "identity": identity_to_wire(identity),
        "catalog_key": catalog_key,
        "presets": presets,
        "default_experiment_task": _experiment_task(
            raw["default_experiment_task"],
            "$.default_experiment_task",
        ),
        "dataset_groups": dataset_groups,
        "monitors": monitors,
        "runtime_defaults": {
            key: wire_scalar(value, f"$.runtime_defaults.{key}")
            for key, value in runtime_defaults.items()
        },
    }


__all__ = [
    "identity_from_wire",
    "identity_to_wire",
    "package_metadata_from_wire",
    "package_metadata_to_wire",
]
