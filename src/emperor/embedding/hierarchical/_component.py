from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from emperor.embedding.hierarchical._config import HierarchicalByteEmbeddingConfig
from emperor.embedding.hierarchical._validation import (
    HierarchicalByteEmbeddingValidator,
)
from emperor.layers import LayerState
from emperor.nn import Module


class HierarchicalByteEmbedding(Module):
    """Independent byte attention and leading-symbol pooling for supplied tokens."""

    VALIDATOR = HierarchicalByteEmbeddingValidator

    def __init__(
        self,
        cfg: HierarchicalByteEmbeddingConfig,
        overrides: HierarchicalByteEmbeddingConfig | None = None,
    ) -> None:
        self.VALIDATOR.validate_config_types(cfg, overrides)
        super().__init__()
        self.cfg = copy.deepcopy(self._override_config(cfg, overrides))
        self._max_group_tokens = self.VALIDATOR.resolve_config(self.cfg)
        self.byte_embedding = nn.Embedding(257, self.cfg.byte_embedding_dim)
        self.byte_position = self.cfg.byte_position_config.build()
        self.encoder = self.cfg.encoder_config.build()
        self.projection = self._build_from_config(
            self.cfg.projection_config,
            input_dim=self.cfg.byte_embedding_dim,
            output_dim=self.cfg.output_dim,
        )

    def forward(
        self,
        token_texts: Sequence[Sequence[str]],
        attention_mask: Tensor | None = None,
    ) -> LayerState:
        batch_size, token_count = self.VALIDATOR.validate_forward_inputs(
            token_texts, attention_mask
        )
        flat_tokens = [token for row in token_texts for token in row]
        encoded_tokens = [
            self.__encode_token(token, index, token_count)
            for index, token in enumerate(flat_tokens)
        ]
        valid = (
            [True] * len(flat_tokens)
            if attention_mask is None
            else attention_mask.reshape(-1).tolist()
        )
        valid_indices = [index for index, enabled in enumerate(valid) if enabled]
        groups = defaultdict(list)
        for compact_index, original_index in enumerate(valid_indices):
            token_bytes = encoded_tokens[original_index]
            groups[len(token_bytes)].append((compact_index, token_bytes))
        for length, group in groups.items():
            if len(group) > self._max_group_tokens:
                raise ValueError(
                    f"{len(group)} tokens with {length} UTF-8 bytes exceed encoder "
                    f"batch_size bound {self._max_group_tokens}"
                )

        weights = self.byte_embedding.weight
        loss = weights.new_zeros(())
        if not valid_indices:
            return LayerState(
                hidden=weights.new_zeros(
                    (batch_size, token_count, self.cfg.output_dim)
                ),
                loss=loss,
            )
        pooled = None
        for group in groups.values():
            indices = [index for index, _ in group]
            byte_ids = torch.tensor(
                [[256, *token_bytes] for _, token_bytes in group],
                dtype=torch.long,
                device=weights.device,
            )
            features = self.byte_embedding(byte_ids) + self.byte_position(byte_ids)
            encoded, group_loss = self.encoder(source_token_embeddings=features)
            if pooled is None:
                pooled = encoded.new_zeros(
                    (len(valid_indices), self.cfg.byte_embedding_dim)
                )
            pooled[indices] = encoded[:, 0]
            loss = loss + group_loss * (len(group) / len(valid_indices))
        projected = self.projection(LayerState(hidden=pooled))
        if projected.loss is not None:
            loss = loss + projected.loss
        hidden = projected.hidden.new_zeros((len(flat_tokens), self.cfg.output_dim))
        hidden[valid_indices] = projected.hidden
        return LayerState(
            hidden=hidden.reshape(batch_size, token_count, self.cfg.output_dim),
            loss=loss,
        )

    def __encode_token(self, token: str, index: int, token_count: int) -> bytes:
        location = f"token_texts[{index // token_count}][{index % token_count}]"
        try:
            token_bytes = token.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError(f"{location} is not a valid UTF-8 string") from error
        if len(token_bytes) > self.cfg.max_token_bytes:
            raise ValueError(
                f"{location} contains {len(token_bytes)} UTF-8 bytes; "
                f"max_token_bytes is {self.cfg.max_token_bytes}"
            )
        return token_bytes
