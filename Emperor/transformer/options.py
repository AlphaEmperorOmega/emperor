from Emperor.base.enums import BaseOptions
from Emperor.transformer.utils.stack import (
    Transformer,
    TransformerDecoder,
    TransformerEncoder,
)


class TransformerOptions(BaseOptions):
    DEFAULT = Transformer
    ENCODER = TransformerEncoder
    DECODER = TransformerDecoder
