from dataclasses import dataclass
from emperor.base.config import ConfigBase, optional_field

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.transformer.core.config import (
        TransformerEncoderStackConfig,
        TransformerDecoderStackConfig,
    )


@dataclass
class TransformerConfig(ConfigBase):
    encoder_stack_config: "TransformerEncoderStackConfig | None" = optional_field(
        "Encoder stack configuration."
    )
    decoder_stack_config: "TransformerDecoderStackConfig | None" = optional_field(
        "Decoder stack configuration."
    )

    def _registry_owner(self) -> type:
        from emperor.transformer.model import Transformer

        return Transformer
