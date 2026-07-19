import re


def dataset_name(dataset: type) -> str:
    return dataset.__name__


def dataset_label(dataset: type) -> str:
    name = dataset_name(dataset)
    label = re.sub(r"(?<!^)(?=[A-Z])", " ", name).replace("_", " ")
    return re.sub(r"\s+", " ", label).strip()


def dataset_class_name_to_cli_name(name: str) -> str:
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", name)
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def dataset_cli_name(dataset: type) -> str:
    return dataset_class_name_to_cli_name(dataset_name(dataset))


def normalize_dataset_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
