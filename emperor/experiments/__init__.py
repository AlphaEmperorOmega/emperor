from importlib import import_module
from typing import TYPE_CHECKING, Any

from .tasks import ExperimentTask

if TYPE_CHECKING:
    from .language_model import LanguageModelExperiment, LanguageModelStepOutput
    from .translation import TranslationExperiment, TranslationStepOutput

__all__ = [
    "ExperimentTask",
    "LanguageModelExperiment",
    "LanguageModelStepOutput",
    "TranslationExperiment",
    "TranslationStepOutput",
]

_LAZY_EXPORTS = {
    "LanguageModelExperiment": (
        "emperor.experiments.language_model",
        "LanguageModelExperiment",
    ),
    "LanguageModelStepOutput": (
        "emperor.experiments.language_model",
        "LanguageModelStepOutput",
    ),
    "TranslationExperiment": (
        "emperor.experiments.translation",
        "TranslationExperiment",
    ),
    "TranslationStepOutput": (
        "emperor.experiments.translation",
        "TranslationStepOutput",
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
