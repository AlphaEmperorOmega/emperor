from models.experiment_cli_parser import (
    _ExperimentParser,
    get_experiment_parser,
    preset_name_to_cli,
)
from models.experiment_mode import (
    ExperimentMode,
    model_monitor_options,
    resolve_experiment_mode,
    resolve_monitor_callbacks,
    resolve_monitor_options,
)
from models.dataset_naming import dataset_cli_name, dataset_name


def resolve_dataset_names(
    dataset_options: list[type],
    dataset_names: list[str] | None,
) -> list[type]:
    if not dataset_names:
        return list(dataset_options)
    by_name = {}
    for dataset in dataset_options:
        by_name[dataset_name(dataset)] = dataset
        by_name[dataset_name(dataset).lower()] = dataset
        by_name[dataset_cli_name(dataset)] = dataset
    resolved = []
    unknown = []
    seen = set()
    for name in dataset_names:
        dataset = by_name.get(name) or by_name.get(name.lower())
        if dataset is None:
            unknown.append(name)
            continue
        if dataset_name(dataset) in seen:
            continue
        seen.add(dataset_name(dataset))
        resolved.append(dataset)
    if unknown:
        valid = ", ".join(dataset_cli_name(dataset) for dataset in dataset_options)
        raise ValueError(f"Unknown --datasets: {unknown}. Valid datasets: {valid}")
    return resolved
