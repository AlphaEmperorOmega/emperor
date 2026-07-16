from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._experiment import SequenceClassifierExperiment

__all__ = ("SequenceClassifierExperiment",)

_LAZY_EXPORTS = {
    "SequenceClassifierExperiment": (
        "emperor.experiments.sequence_classifier._experiment",
        "SequenceClassifierExperiment",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from error

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
