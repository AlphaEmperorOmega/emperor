"""Compatibility imports for the public Runtime Options Interface."""

from models.bert.linear.runtime_options import (
    TransformerEncoderOptions,
    TransformerPositionalEmbeddingOptions,
    TransformerAttentionOptions,
    TransformerFeedForwardOptions,
    VitPatchOptions,
    VitOutputOptions,
)

LEGACY_RUNTIME_OPTIONS_MODULE = True

__all__ = [
    "TransformerEncoderOptions",
    "TransformerPositionalEmbeddingOptions",
    "TransformerAttentionOptions",
    "TransformerFeedForwardOptions",
    "VitPatchOptions",
    "VitOutputOptions",
]
