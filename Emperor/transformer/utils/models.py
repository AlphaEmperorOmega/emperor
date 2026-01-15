from torch import Tensor
from Emperor.base.utils import Module
from Emperor.transformer.utils.embedding.selector import PositionalEmbeddingSelector
from Emperor.transformer.utils.patch.selector import PatchSelector
from Emperor.transformer.utils.stack import (
    TransformerDecoderStack,
    TransformerEncoderStack,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class BERTVITModel(Module):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__()
        self.patch_model = PatchSelector(cfg).build()
        self.positional_embedding_model = PositionalEmbeddingSelector(cfg).build()
        self.encoder_model = TransformerEncoderStack(cfg)

    def forward(
        self,
        tokens_tensor: Tensor,
    ) -> Tensor:
        X = self.patch_model(tokens_tensor)
        X = self.positional_embedding_model(X)
        X = self.encoder_model(X)
        return X


class Transformer(Module):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__()
        self.encoder_model = TransformerEncoderStack(cfg)
        self.decoder_model = TransformerDecoderStack(cfg)

    def forward(
        self,
        source_token_embeddings: Tensor,
        target_token_embeddings: Tensor,
        source_attention_mask: Tensor | None = None,
        target_attention_mask: Tensor | None = None,
        memory_attention_mask: Tensor | None = None,
        source_key_padding_mask: Tensor | None = None,
        target_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        # source_is_causal: bool | None = None,
        # target_is_causal: bool | None = None,
        # memory_is_causal: bool | None = None,
    ) -> Tensor:
        memory, memory_loss = self.encoder_model(
            source_token_embeddings=source_token_embeddings,
            source_key_padding_mask=source_key_padding_mask,
            attention_mask=source_attention_mask,
            # TODO: Find out if this is necessary in the future
            # soruce_is_causal,
        )
        output, output_loss = self.decoder_model(
            target_token_embeddings=target_token_embeddings,
            encoder_output=memory,
            target_key_padding_mask=target_key_padding_mask,
            encoder_key_padding_mask=memory_key_padding_mask,
            attention_mask=target_attention_mask,
            encoder_attention_mask=memory_attention_mask,
            # TODO: Find out if this is necessary in the future
            # target_is_causal,
            # memory_is_causal,
        )

        return output, memory_loss + output_loss
