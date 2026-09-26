import copy

import torch
from torch import Tensor, nn

from emperor.layers import LayerState
from emperor.nn import Module

from ._config import HierarchicalByteDecoderConfig
from ._generation import _generate_token
from ._records import (
    BYTE_VALUES,
    OUTPUT_SYMBOLS,
    ByteGenerationOptions,
    ByteTokenGenerationOutput,
    HierarchicalByteDecoderOutput,
)
from ._validation import HierarchicalByteDecoderValidator


class HierarchicalByteDecoder(Module):
    """HAT v2 conditioning-prefix decoder using native Emperor components.

    Equations 3–5: project preceding backbone context, prepend it to raw target
    byte embeddings, and predict the next byte at each causal position. For
    target 'mat', inputs [context, m, a, t] align with labels [m, a, t, EOW].
    This uses neither encoder activations nor the later release's cross-attention.
    Independent weights and 258 output symbols are deliberate adaptations; see
    docs/hierarchical-implementation-review.md. Returned loss is auxiliary only.
    """

    VALIDATOR = HierarchicalByteDecoderValidator

    def __init__(
        self,
        cfg: HierarchicalByteDecoderConfig,
        overrides: HierarchicalByteDecoderConfig | None = None,
    ):
        self.VALIDATOR.validate_config_types(cfg, overrides)
        super().__init__()
        self.cfg = copy.deepcopy(self._override_config(cfg, overrides))
        self._max_group_tokens = self.VALIDATOR.resolve_config(self.cfg)
        # Only literal bytes enter this table; EOW/EOS are prediction targets.
        self.byte_embedding = nn.Embedding(BYTE_VALUES, self.cfg.byte_embedding_dim)
        self.byte_position = self.cfg.byte_position_config.build()
        self.conditioning_projection = self.cfg.conditioning_projection_config.build()
        self.decoder = self.cfg.decoder_config.build()
        self.output_projection = self.cfg.output_projection_config.build()

    def generate_token(
        self,
        conditioning: Tensor,
        *,
        prefix: str = "",
        max_new_bytes: int | None = None,
        options: ByteGenerationOptions | None = None,
        generator: torch.Generator | None = None,
    ) -> ByteTokenGenerationOutput:
        return _generate_token(
            self,
            conditioning,
            prefix,
            max_new_bytes,
            options or ByteGenerationOptions(),
            generator,
        )

    def forward(
        self,
        conditioning: Tensor,
        byte_prefix_ids: Tensor,
        byte_lengths: Tensor,
        attention_mask: Tensor | None = None,
    ) -> HierarchicalByteDecoderOutput:
        prefixes, lengths, valid = self.VALIDATOR.validate_forward_inputs(
            self.cfg,
            conditioning,
            byte_prefix_ids,
            byte_lengths,
            attention_mask,
            self.byte_embedding.weight.device,
        )
        flat_lengths = lengths.reshape(-1)
        flat_valid = valid.reshape(-1)
        counts = torch.where(flat_valid, flat_lengths + 1, 0)
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        total_positions = int(offsets[-1])
        if total_positions == 0:
            return HierarchicalByteDecoderOutput(
                conditioning.new_empty((0, OUTPUT_SYMBOLS)),
                offsets,
                conditioning.new_zeros(()),
            )
        indices = flat_valid.nonzero().flatten()
        selected_lengths = flat_lengths[indices]
        for length in selected_lengths.unique().tolist():
            group_count = int((selected_lengths == length).sum())
            if group_count > self._max_group_tokens:
                raise ValueError(
                    f"{group_count} prefixes with {length} bytes exceed decoder batch_size bound {self._max_group_tokens}"
                )
        context = self.conditioning_projection(
            LayerState(
                hidden=conditioning.reshape(-1, self.cfg.conditioning_dim)[indices]
            )
        )
        loss = context.hidden.new_zeros(()) if context.loss is None else context.loss
        packed_hidden = None
        flat_prefixes = prefixes.reshape(lengths.numel(), prefixes.shape[-1])
        for length in selected_lengths.unique().tolist():
            compact_indices = (selected_lengths == length).nonzero().flatten()
            original_indices = indices[compact_indices]
            byte_ids = flat_prefixes[original_indices, :length]
            # Bidirectional encoder outputs would expose future target bytes.
            features = torch.cat(
                (context.hidden[compact_indices, None], self.byte_embedding(byte_ids)),
                dim=1,
            )
            position_ids = byte_ids.new_zeros((len(compact_indices), length + 1))
            features = features + self.byte_position(position_ids)
            encoded, group_loss = self.decoder(target_token_embeddings=features)
            if packed_hidden is None:
                packed_hidden = encoded.new_zeros(
                    (total_positions, self.cfg.byte_embedding_dim)
                )
            positions = offsets[original_indices, None] + torch.arange(
                length + 1, device=indices.device
            )
            packed_hidden[positions.flatten()] = encoded.reshape(
                -1, self.cfg.byte_embedding_dim
            )
            loss = loss + group_loss * (positions.numel() / total_positions)
        projected = self.output_projection(LayerState(hidden=packed_hidden))
        if projected.loss is not None:
            loss = loss + projected.loss
        return HierarchicalByteDecoderOutput(
            projected.hidden, offsets, loss.reshape(())
        )
