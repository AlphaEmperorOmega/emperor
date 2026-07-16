"""Public Interface for supported causal language-modeling datasets."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emperor.datasets.text.language_modeling._penn_treebank import PennTreebank
    from emperor.datasets.text.language_modeling._wiki_text_2 import WikiText2

__all__ = ("PennTreebank", "WikiText2")

_LAZY_EXPORTS = {
    "PennTreebank": (
        "emperor.datasets.text.language_modeling._penn_treebank",
        "PennTreebank",
    ),
    "WikiText2": (
        "emperor.datasets.text.language_modeling._wiki_text_2",
        "WikiText2",
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
