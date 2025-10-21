from Emperor.base.enums import BaseOptions
from Emperor.transformer.layer import (
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)


class TransformerOptions(BaseOptions):
    DEFAULT = Transformer
    ENCODER = TransformerEncoder
    ENCODER_LAYER = TransformerEncoderLayer
    DECODER = TransformerDecoder
    DECODER_LAYER = TransformerDecoderLayer
