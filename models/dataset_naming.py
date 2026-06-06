import re


def dataset_name(dataset: type) -> str:
    return dataset.__name__


def dataset_label(dataset: type) -> str:
    name = dataset_name(dataset)
    label = re.sub(r"(?<!^)(?=[A-Z])", " ", name).replace("_", " ")
    return re.sub(r"\s+", " ", label).strip()


def dataset_cli_name(dataset: type) -> str:
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", dataset_name(dataset))
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def normalize_dataset_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
