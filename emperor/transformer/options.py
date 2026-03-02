from emperor.base.enums import BaseOptions
from emperor.transformer.utils.models import Transformer
from emperor.transformer.utils.stack import (
    TransformerDecoderStack,
    TransformerEncoderStack,
)


class TransformerOptions(BaseOptions):
    DEFAULT = Transformer
    ENCODER = TransformerEncoderStack
    DECODER = TransformerDecoderStack
