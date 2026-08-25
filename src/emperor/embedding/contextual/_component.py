from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from emperor.embedding.contextual._encoding import Utf8BitEncoder
from emperor.embedding.contextual._state import ByteContextualEmbeddingState
from emperor.embedding.contextual._validation import (
    ByteContextualEmbeddingValidator,
)
from emperor.nn import Module

if TYPE_CHECKING:
    from collections.abc import Sequence

    from emperor.embedding.contextual._config import ByteContextualEmbeddingConfig


class ByteContextualEmbedding(Module):
    VALIDATOR = ByteContextualEmbeddingValidator

    def __init__(
        self,
        cfg: ByteContextualEmbeddingConfig,
        overrides: ByteContextualEmbeddingConfig | None = None,
    ) -> None:
        self.VALIDATOR.validate_config_type(cfg)
        self.VALIDATOR.validate_overrides_type(overrides)
        super().__init__()
        self.cfg = self.__resolve_config(cfg, overrides)
        self.VALIDATOR.validate(self)

        self.max_token_bytes = self.cfg.max_token_bytes
        self.hidden_dim = self.cfg.hidden_dim
        self.encoder = Utf8BitEncoder(self.max_token_bytes)
        self.byte_moe = self.cfg.byte_moe_config.build()
        self.positional_embedding = self.cfg.positional_embedding_config.build()
        self.prefix_kernel = self.cfg.prefix_kernel_config.build()
        self.context_moe = self.cfg.context_moe_config.build()
        self.gamma = nn.Parameter(
            torch.tensor(float(self.cfg.residual_scale_initial_value))
        )

    @staticmethod
    def __resolve_config(
        cfg: ByteContextualEmbeddingConfig,
        overrides: ByteContextualEmbeddingConfig | None,
    ) -> ByteContextualEmbeddingConfig:
        if overrides is None:
            return cfg
        resolved = copy.deepcopy(cfg)
        if overrides.max_token_bytes != 4:
            resolved.max_token_bytes = overrides.max_token_bytes
        for field_name in (
            "hidden_dim",
            "byte_moe_config",
            "positional_embedding_config",
            "prefix_kernel_config",
            "context_moe_config",
        ):
            override_value = getattr(overrides, field_name)
            if override_value is not None:
                setattr(resolved, field_name, override_value)
        if overrides.residual_scale_initial_value != 1e-3:
            resolved.residual_scale_initial_value = (
                overrides.residual_scale_initial_value
            )
        return resolved

    def forward(
        self,
        token_texts: Sequence[Sequence[str]],
        attention_mask: Tensor | None = None,
    ) -> ByteContextualEmbeddingState:
        batch_size, sequence_length = self.VALIDATOR.validate_forward_inputs(
            token_texts,
            attention_mask,
            maximum_sequence_length=self.cfg.positional_embedding_config.num_embeddings,
        )
        resolved_mask = self.__resolve_attention_mask(
            attention_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        bits, byte_mask = self.encoder(token_texts, device=self.gamma.device)
        byte_features = torch.cat((bits, byte_mask), dim=-1).to(dtype=self.gamma.dtype)
        draft, byte_moe_auxiliary_loss = self.__route_valid_rows(
            self.byte_moe,
            byte_features,
            resolved_mask,
            output_dim=self.hidden_dim,
        )

        position_ids = resolved_mask.long().cumsum(dim=1)
        position_ids = position_ids.masked_fill(~resolved_mask, 0)
        learned_position = self.positional_embedding(
            position_ids,
            positions=position_ids,
        )
        sequence_mask = resolved_mask.unsqueeze(-1)
        positioned_draft = (draft + learned_position) * sequence_mask
        prefix_context = self.prefix_kernel(positioned_draft, resolved_mask)

        joint_context = torch.cat((positioned_draft, prefix_context), dim=-1)
        contextual_delta, context_moe_auxiliary_loss = self.__route_valid_rows(
            self.context_moe,
            joint_context,
            resolved_mask,
            output_dim=self.hidden_dim,
        )
        hidden = positioned_draft + self.gamma * contextual_delta
        hidden = hidden * sequence_mask
        loss = byte_moe_auxiliary_loss + context_moe_auxiliary_loss
        return ByteContextualEmbeddingState(
            hidden=hidden,
            byte_moe_auxiliary_loss=byte_moe_auxiliary_loss,
            context_moe_auxiliary_loss=context_moe_auxiliary_loss,
            loss=loss,
        )

    def __resolve_attention_mask(
        self,
        attention_mask: Tensor | None,
        *,
        batch_size: int,
        sequence_length: int,
    ) -> Tensor:
        if attention_mask is None:
            return torch.ones(
                (batch_size, sequence_length),
                dtype=torch.bool,
                device=self.gamma.device,
            )
        return attention_mask.to(device=self.gamma.device)

    @staticmethod
    def __route_valid_rows(
        mixture,
        values: Tensor,
        attention_mask: Tensor,
        *,
        output_dim: int,
    ) -> tuple[Tensor, Tensor]:
        batch_size, sequence_length, input_dim = values.shape
        flat_values = values.reshape(batch_size * sequence_length, input_dim)
        flat_mask = attention_mask.reshape(batch_size * sequence_length)
        scattered_output = values.new_zeros((batch_size * sequence_length, output_dim))
        if not flat_mask.any().item():
            return (
                scattered_output.reshape(batch_size, sequence_length, output_dim),
                values.new_zeros(()),
            )
        valid_values = flat_values[flat_mask]
        valid_output, _, auxiliary_loss = mixture(valid_values)
        scattered_output[flat_mask] = valid_output
        return (
            scattered_output.reshape(batch_size, sequence_length, output_dim),
            auxiliary_loss,
        )
