"""Compatibility imports for the public Runtime Options Interface."""

from models.bert.linear.runtime_options import (
    BertEmbeddingOptions,
    BertMlmHeadOptions,
    BertNspHeadOptions,
)

LEGACY_RUNTIME_OPTIONS_MODULE = True

__all__ = [
    "BertEmbeddingOptions",
    "BertMlmHeadOptions",
    "BertNspHeadOptions",
]
