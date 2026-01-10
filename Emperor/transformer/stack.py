import copy
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import ModuleList
from dataclasses import dataclass, field
from Emperor.base.utils import ConfigBase, Module
from Emperor.transformer.layers import TransformerDecoderLayer, TransformerEncoderLayer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig
    from Emperor.transformer.layers import TransformerLayerBase


@dataclass
class TransformerConfig(ConfigBase):
    num_layers: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    source_sequence_length: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    target_sequence_length: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    layer_norm_dim: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    causal_attention_mask_flag: bool | None = field(
        default=None,
        metadata={"help": ""},
    )


class TransformerBase(Module):
    def __init__(self, cfg: "TransformerConfig | ModelConfig"):
        super().__init__()
        self.cfg: "TransformerConfig" = cfg
        self.num_layers = self.cfg.num_layers
        self.source_sequence_length = self.cfg.source_sequence_length
        self.target_sequence_length = self.cfg.target_sequence_length
        self.causal_attention_mask_flag = self.cfg.causal_attention_mask_flag
        self.layer_norm_dim = self.cfg.layer_norm_dim

        self.layer_norm_module = self.__init_layer_norm_module()
        self.layers = self.__create_layers()

    def __create_layers(self) -> ModuleList:
        model = self._create_transformer_layer()
        return ModuleList([copy.deepcopy(model) for _ in range(self.num_layers)])

    def _create_transformer_layer(self) -> "TransformerLayerBase":
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the _create_model method."
        )

    def __init_layer_norm_module(self) -> nn.Module | None:
        if self.layer_norm_dim is None:
            return None
        assert self.layer_norm_dim > 0, (
            f"Expected layer_norm_dim must be greater than 0, received {self.layer_norm_dim}"
        )
        return nn.LayerNorm(self.layer_norm_dim)

    def _is_attention_mask_causal(
        self,
        attention_mask: Tensor | None = None,
    ) -> bool:
        if self.causal_attention_mask_flag:
            return True

        if self.causal_attention_mask_flag is None and attention_mask is not None:
            causal_mask = self.__generate_causal_mask(
                attention_mask.device, attention_mask.dtype
            )
            if attention_mask.size() == causal_mask.size():
                return bool((attention_mask == causal_mask).all())
        return False

    def __generate_causal_mask(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        mask_shape = (self.source_sequence_length, self.source_sequence_length)
        negative_infinity_tensor = torch.full(
            mask_shape, float("-inf"), dtype=dtype, device=device
        )

        return torch.triu(negative_infinity_tensor, diagonal=1)


class TransformerEncoder(TransformerBase):
    def __init__(
        self,
        cfg: "TransformerConfig | ModelConfig",
        overrides: "TransformerConfig | None" = None,
    ):
        config = getattr(cfg, "transformer_config", cfg)
        self.cfg: "TransformerConfig" = self._overwrite_config(config, overrides)

        self.main_config = cfg
        super().__init__(self.cfg)
        self.__perform_encoder_checks()

    def __perform_encoder_checks(self):
        assert self.source_sequence_length == self.target_sequence_length, (
            "Source and target sequence length must be equal in TransformerEncoder"
        )

    def _create_transformer_layer(self) -> "TransformerLayerBase":
        return TransformerEncoderLayer(self.main_config)

    def forward(
        self,
        source_token_embeddings: Tensor,
        source_key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        # FIXME: At the moment this is not used because i tought that this
        # can be used as a hyper parameter, but it a boolean that checks
        # if the input `attention_mask` is causal or not
        # this is used in MultiHeadAttention in `scaled_dot_product_attention` method
        # to create an attention mask dinamically if one is not given
        #
        # is_causal = self.__is_attention_mask_causal(attention_mask, is_causal, seq_len)

        total_loss = 0.0
        output = source_token_embeddings
        for encoder_layer in self.layers:
            output, layer_loss = encoder_layer(
                source_token_embeddings=output,
                source_key_padding_mask=source_key_padding_mask,
                attention_mask=attention_mask,
                # is_causal=is_causal,
            )
            total_loss += layer_loss

        if self.layer_norm_module is not None:
            output = self.layer_norm_module(output)

        return output, total_loss


class TransformerDecoder(TransformerBase):
    def __init__(
        self,
        cfg: "TransformerConfig | ModelConfig",
        overrides: "TransformerConfig | None" = None,
    ):
        config = getattr(cfg, "transformer_config", cfg)
        self.cfg: "TransformerConfig" = self._overwrite_config(config, overrides)
        self.main_config = cfg
        super().__init__(self.cfg)

    def _create_transformer_layer(self) -> "TransformerLayerBase":
        return TransformerDecoderLayer(self.main_config)

    def forward(
        self,
        target_token_embeddings: Tensor,
        encoder_output: Tensor,
        target_key_padding_mask: Tensor | None = None,
        encoder_key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        # encoder_is_causal: bool | None = None,
    ) -> tuple[Tensor, Tensor]:
        # FIXME:
        ## - At the moment this is not used because i tought that this
        # can be used as a hyper parameter, but it a boolean that checks
        # if the input `attention_mask` is causal or not
        # this is used in MultiHeadAttention in `scaled_dot_product_attention` method
        # to create an attention mask dinamically if one is not given
        #
        # is_causal = self._is_attention_mask_causal(attention_mask, is_causal, seq_len)

        total_loss = 0.0
        output = target_token_embeddings
        for decoder_layer in self.layers:
            output, layer_loss = decoder_layer(
                target_token_embeddings=output,
                encoder_output=encoder_output,
                key_padding_mask=target_key_padding_mask,
                encoder_padding_mask=encoder_key_padding_mask,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                # is_causal=is_causal,
                # memory_is_causal=encoder_is_causal,
            )
            total_loss += layer_loss

        if self.layer_norm_module is not None:
            output = self.layer_norm_module(output)

        return output, total_loss


class Transformer(Module):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__()
        self.encoder_model = TransformerEncoder(cfg)
        self.decoder_model = TransformerDecoder(cfg)

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
