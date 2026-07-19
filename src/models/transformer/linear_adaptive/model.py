from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from emperor.attention import AttentionLayerState
from emperor.experiments.translation import TranslationExperiment
from emperor.transformer import TransformerDecoderLayerState

if TYPE_CHECKING:
    from emperor.config import ModelConfig


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


class Model(TranslationExperiment):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.experiment_config = config.experiment_config
        cfg = self.experiment_config
        self.source_sequence_length = cfg.source_sequence_length
        self.target_sequence_length = cfg.target_sequence_length
        self.bos_token_id = cfg.bos_token_id
        self.eos_token_id = cfg.eos_token_id
        self.embedding_scale = math.sqrt(cfg.model_dim)
        shared = nn.Embedding(
            cfg.vocab_size, cfg.model_dim, padding_idx=cfg.pad_token_id
        )
        self.shared_embedding = shared
        self.source_embedding = shared
        self.target_embedding = shared
        self.source_positional_embedding = (
            cfg.source_positional_embedding_config.build()
        )
        self.target_positional_embedding = (
            cfg.target_positional_embedding_config.build()
        )
        self.embedding_dropout = nn.Dropout(cfg.dropout_probability)
        self.encoder = cfg.encoder_config.build()
        self.decoder = cfg.decoder_config.build()
        self.encoder_layer_norm = nn.LayerNorm(cfg.model_dim)
        self.decoder_layer_norm = nn.LayerNorm(cfg.model_dim)
        self.output_projection = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False)
        self.output_projection.weight = self.shared_embedding.weight

    def forward(
        self, source_ids: Tensor, target_input_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        self._validate_ids(source_ids, self.source_sequence_length, "source")
        self._validate_ids(
            target_input_ids,
            self.target_sequence_length,
            "target_input",
        )
        if source_ids.size(0) != target_input_ids.size(0):
            raise ValueError(
                "source_ids and target_input_ids must have matching batch sizes."
            )
        encoded, source_mask, encoder_loss = self._encode(source_ids)
        decoded, decoder_loss = self._decode(target_input_ids, encoded, source_mask)
        auxiliary_loss = (encoder_loss + decoder_loss).reshape(())
        return self.output_projection(decoded), auxiliary_loss

    def _embed(self, ids: Tensor, embedding, position) -> Tensor:
        return self.embedding_dropout(
            embedding(ids) * self.embedding_scale + position(ids)
        )

    def _encode(self, source_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        self._validate_ids(source_ids, self.source_sequence_length, "source")
        source_ids = source_ids.to(self.device, dtype=torch.long)
        padding_mask = source_ids.eq(self.pad_token_id)
        state = self.encoder(
            AttentionLayerState(
                hidden=self._embed(
                    source_ids, self.source_embedding, self.source_positional_embedding
                ),
                key_padding_mask=padding_mask,
            )
        )
        output = self.encoder_layer_norm(state.hidden)
        return (
            output,
            padding_mask,
            state.loss if state.loss is not None else output.new_zeros(()),
        )

    def _decode(
        self, target_ids: Tensor, encoder_output: Tensor, source_padding_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        self._validate_ids(target_ids, self.target_sequence_length, "target")
        target_ids = target_ids.to(self.device, dtype=torch.long)
        state = self.decoder(
            TransformerDecoderLayerState(
                hidden=self._embed(
                    target_ids, self.target_embedding, self.target_positional_embedding
                ),
                target_key_padding_mask=target_ids.eq(self.pad_token_id),
                encoder_output=encoder_output,
                encoder_padding_mask=source_padding_mask,
            )
        )
        output = self.decoder_layer_norm(state.hidden)
        return output, state.loss if state.loss is not None else output.new_zeros(())

    def _validate_ids(self, ids: Tensor, maximum: int, name: str) -> None:
        if not isinstance(ids, Tensor):
            raise TypeError(f"{name}_ids must be a torch.Tensor.")
        if ids.ndim != 2:
            raise ValueError(f"{name}_ids must be a rank-2 [batch, sequence] tensor.")
        if ids.size(0) == 0 or ids.size(1) == 0:
            raise ValueError(
                f"{name}_ids must have non-empty batch and sequence dimensions."
            )
        if ids.size(1) > maximum:
            raise ValueError(
                f"{name} sequence length {ids.size(1)} exceeds maximum {maximum}."
            )
        if ids.dtype not in _INTEGER_DTYPES:
            raise TypeError(f"{name}_ids must contain integer token IDs.")
        if bool(torch.any(ids < 0).item()) or bool(
            torch.any(ids >= self.vocab_size).item()
        ):
            raise ValueError(f"{name}_ids must be in the range [0, {self.vocab_size}).")

    @torch.no_grad()
    def generate(self, source_ids: Tensor, max_length: int | None = None) -> Tensor:
        if max_length is not None and (
            isinstance(max_length, bool) or not isinstance(max_length, int)
        ):
            raise TypeError("max_length must be an integer or None.")
        maximum = self.target_sequence_length if max_length is None else max_length
        if maximum < 1 or maximum > self.target_sequence_length:
            raise ValueError(
                f"max_length must be in [1, {self.target_sequence_length}]"
            )
        self._validate_ids(source_ids, self.source_sequence_length, "source")

        was_training = self.training
        try:
            self.eval()
            source_ids = source_ids.to(self.device, dtype=torch.long)
            encoded, source_mask, _ = self._encode(source_ids)
            generated = torch.full(
                (source_ids.size(0), maximum),
                self.pad_token_id,
                dtype=torch.long,
                device=source_ids.device,
            )
            generated[:, 0] = self.bos_token_id
            finished = torch.zeros(
                source_ids.size(0), dtype=torch.bool, device=source_ids.device
            )
            for position in range(1, maximum):
                decoded, _ = self._decode(generated[:, :position], encoded, source_mask)
                next_ids = self.output_projection(decoded[:, -1]).argmax(dim=-1)
                next_ids = torch.where(
                    finished,
                    torch.full_like(next_ids, self.pad_token_id),
                    next_ids,
                )
                generated[:, position] = next_ids
                finished |= next_ids.eq(self.eos_token_id)
                if bool(finished.all().item()):
                    break
            return generated
        finally:
            self.train(was_training)


__all__ = ["Model"]
