from Emperor.base.enums import BaseOptions
from Emperor.transformer.utils.models import Transformer
from Emperor.transformer.utils.stack import (
    TransformerDecoderStack,
    TransformerEncoderStack,
)


class TransformerOptions(BaseOptions):
    DEFAULT = Transformer
    ENCODER = TransformerEncoderStack
    DECODER = TransformerDecoderStack
