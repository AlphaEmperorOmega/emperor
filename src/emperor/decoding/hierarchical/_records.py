from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

# Emperor reserves separate controls rather than reusing unused UTF-8 byte IDs
# as paper v2 does. These output IDs are not the encoder's [W] input embedding.
BYTE_VALUES = 256
END_OF_TOKEN = 256
END_OF_DOCUMENT = 257
OUTPUT_SYMBOLS = 258


@dataclass(frozen=True)
class HierarchicalByteDecoderOutput:
    """Packed predictions in batch/token/byte order; loss is auxiliary only."""

    logits: Tensor
    token_offsets: Tensor
    loss: Tensor


@dataclass(frozen=True)
class ByteGenerationOptions:
    """Sampling, when enabled, always uses the fixed byte/control alphabet."""

    do_sample: bool = False
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 1.0
    seed: int | None = None


@dataclass(frozen=True)
class ByteTokenGenerationOutput:
    text: str
    stop_reason: str
    new_bytes: int


@dataclass(frozen=True)
class HierarchicalTextGenerationOutput:
    text: str
    stop_reason: str
    new_bytes: int
    new_tokens: int
