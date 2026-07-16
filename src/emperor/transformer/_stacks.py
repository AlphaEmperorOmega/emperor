import copy
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import ModuleList

from emperor.nn import Module
from emperor.transformer._validation import TransformerValidator

if TYPE_CHECKING:
    from emperor.transformer._config import (
        TransformerDecoderStackConfig,
        TransformerEncoderStackConfig,
    )


class TransformerEncoderStack(Module):
    VALIDATOR = TransformerValidator

    def __init__(
        self,
        cfg: "TransformerEncoderStackConfig",
        overrides: "TransformerEncoderStackConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "transformer_encoder_stack_config", cfg)
        self.cfg: TransformerEncoderStackConfig = self._override_config(
            config, overrides
        )

        self.num_layers: int = self.cfg.num_layers
        self.embedding_dim: int = self.cfg.embedding_dim
        self.source_sequence_length: int = self.cfg.source_sequence_length
        self.target_sequence_length: int = self.cfg.target_sequence_length
        self.causal_attention_mask_flag: bool = self.cfg.causal_attention_mask_flag

        self.VALIDATOR.validate_encoder_stack(self)

        self.layers = self.__create_layers()
        self.layer_norm_module = nn.LayerNorm(self.embedding_dim)

    def __create_layers(self) -> ModuleList:
        prototype_layer = self.cfg.layer_config.build()
        return ModuleList(
            [copy.deepcopy(prototype_layer) for _ in range(self.num_layers)]
        )

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

    def __resolve_causal_attention_mask(
        self,
        embeddings: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor | None:
        if not self._is_attention_mask_causal(attention_mask):
            return attention_mask
        if attention_mask is not None:
            return attention_mask
        return self.__generate_causal_mask(embeddings.device, embeddings.dtype)

    def forward(
        self,
        source_token_embeddings: Tensor,
        source_key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        self.VALIDATOR.validate_encoder_stack_forward_inputs(
            self, source_token_embeddings
        )
        attention_mask = self.__resolve_causal_attention_mask(
            source_token_embeddings, attention_mask
        )
        total_loss = source_token_embeddings.new_zeros(())
        output = source_token_embeddings
        for encoder_layer in self.layers:
            output, layer_loss = encoder_layer(
                source_token_embeddings=output,
                source_key_padding_mask=source_key_padding_mask,
                attention_mask=attention_mask,
            )
            total_loss = total_loss + layer_loss

        if self.layer_norm_module is not None:
            output = self.layer_norm_module(output)

        return output, total_loss


class TransformerDecoderStack(Module):
    VALIDATOR = TransformerValidator

    def __init__(
        self,
        cfg: "TransformerDecoderStackConfig",
        overrides: "TransformerDecoderStackConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "transformer_decoder_stack_config", cfg)
        self.cfg: TransformerDecoderStackConfig = self._override_config(
            config, overrides
        )

        self.num_layers: int = self.cfg.num_layers
        self.embedding_dim: int = self.cfg.embedding_dim
        self.source_sequence_length: int = self.cfg.source_sequence_length
        self.target_sequence_length: int = self.cfg.target_sequence_length
        self.causal_attention_mask_flag: bool = self.cfg.causal_attention_mask_flag

        self.VALIDATOR.validate_decoder_stack(self)

        self.layers = self.__create_layers()
        self.layer_norm_module = nn.LayerNorm(self.embedding_dim)

    def __create_layers(self) -> ModuleList:
        prototype_layer = self.cfg.layer_config.build()
        return ModuleList(
            [copy.deepcopy(prototype_layer) for _ in range(self.num_layers)]
        )

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
        mask_shape = (self.target_sequence_length, self.target_sequence_length)
        negative_infinity_tensor = torch.full(
            mask_shape, float("-inf"), dtype=dtype, device=device
        )
        return torch.triu(negative_infinity_tensor, diagonal=1)

    def __resolve_causal_attention_mask(
        self,
        embeddings: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor | None:
        if not self._is_attention_mask_causal(attention_mask):
            return attention_mask
        if attention_mask is not None:
            return attention_mask
        return self.__generate_causal_mask(embeddings.device, embeddings.dtype)

    def forward(
        self,
        target_token_embeddings: Tensor,
        encoder_output: Tensor | None = None,
        target_key_padding_mask: Tensor | None = None,
        encoder_key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        self.VALIDATOR.validate_decoder_stack_forward_inputs(
            self, target_token_embeddings, encoder_output
        )
        attention_mask = self.__resolve_causal_attention_mask(
            target_token_embeddings, attention_mask
        )
        total_loss = target_token_embeddings.new_zeros(())
        output = target_token_embeddings
        for decoder_layer in self.layers:
            output, layer_loss = decoder_layer(
                target_token_embeddings=output,
                encoder_output=encoder_output,
                key_padding_mask=target_key_padding_mask,
                encoder_padding_mask=encoder_key_padding_mask,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
            total_loss = total_loss + layer_loss

        if self.layer_norm_module is not None:
            output = self.layer_norm_module(output)

        return output, total_loss
