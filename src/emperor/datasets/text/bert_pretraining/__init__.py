"""Public Interface for supported BERT pretraining datasets."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emperor.datasets.text.bert_pretraining._datasets import (
        PennTreebankBertPretraining,
        WikiText2BertPretraining,
    )
    from emperor.datasets.text.bert_pretraining._tokenizer import (
        BERT_PRETRAINING_TARGET_VOCAB_SIZE,
    )

__all__ = (
    "BERT_PRETRAINING_TARGET_VOCAB_SIZE",
    "PennTreebankBertPretraining",
    "WikiText2BertPretraining",
)

_LAZY_EXPORTS = {
    "BERT_PRETRAINING_TARGET_VOCAB_SIZE": (
        "emperor.datasets.text.bert_pretraining._tokenizer",
        "BERT_PRETRAINING_TARGET_VOCAB_SIZE",
    ),
    "PennTreebankBertPretraining": (
        "emperor.datasets.text.bert_pretraining._datasets",
        "PennTreebankBertPretraining",
    ),
    "WikiText2BertPretraining": (
        "emperor.datasets.text.bert_pretraining._datasets",
        "WikiText2BertPretraining",
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
