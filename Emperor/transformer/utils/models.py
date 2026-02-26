from torch import Tensor
from Emperor.base.layer import LayerStackConfig
from Emperor.base.utils import Module
from dataclasses import dataclass, field
from Emperor.base.utils import ConfigBase
from Emperor.linears.utils.layers import LinearLayerConfig
from Emperor.linears.utils.stack import LinearLayerStack
from Emperor.embedding.absolute.config import AbsolutePositionalEmbeddingConfig
from Emperor.embedding.absolute.factory import AbsolutePositionalEmbeddingFactory
from Emperor.transformer.utils.layers import TransformerConfig
from Emperor.transformer.utils.patch.options.base import PatchConfig
from Emperor.transformer.utils.patch.selector import PatchSelector
from Emperor.transformer.utils.stack import (
    TransformerEncoderStack,
    TransformerDecoderStack,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


@dataclass
class VITExperimentConfig(ConfigBase):
    patch_config: "PatchConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )
    positional_embedding_config: "AbsolutePositionalEmbeddingConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )
    encoder_config: "TransformerConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )
    output_config: "LinearLayerConfig | LayerStackConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )


class TransformerBase(Module):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.input_dim = self.cfg.input_dim
        self.hidden_dim = self.cfg.hidden_dim
        self.output_dim = self.cfg.output_dim

        self.main_cfg = self._resolve_main_config(self.cfg, cfg)
        self.patch_config = self.main_cfg.patch_config
        self.embedding_config = self.main_cfg.positional_embedding_config
        self.encoder_config = self.main_cfg.encoder_config
        self.output_config = self.main_cfg.output_config

        self.patch = PatchSelector(self.patch_config).build()
        self.embedding = AbsolutePositionalEmbeddingFactory(
            self.embedding_config
        ).build()
        self.output = LinearLayerStack(self.output_config).build_model()


class TransformerEncoderModel(TransformerBase):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__(cfg)

        self.main_cfg = self._resolve_main_config(self.cfg, cfg)
        self.patch_config = self.main_cfg.patch_config
        self.embedding_config = self.main_cfg.positional_embedding_config
        self.encoder_config = self.main_cfg.encoder_config
        self.output_config = self.main_cfg.output_config

        self.patch = PatchSelector(self.patch_config).build()
        self.positional_embedding = AbsolutePositionalEmbeddingFactory(
            self.embedding_config
        ).build()
        self.transformer = TransformerEncoderStack(self.encoder_config)
        self.output = LinearLayerStack(self.output_config).build_model()

    def forward(self, tokens_tensor: Tensor) -> Tensor:
        X = self.patch(tokens_tensor)
        X = self.embedding(X)
        X, loss = self.transformer(X)
        X = self.__select_class_tokens(X)
        X = self.output(X)
        return X

    def __select_class_tokens(self, X: Tensor) -> Tensor:
        return X[:, 0, :]


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
