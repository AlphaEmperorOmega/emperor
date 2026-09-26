import copy

import torch
from torch import nn

from emperor.layers import LayerState
from emperor.nn import Module
from emperor.transformer import TransformerDecoderLayerState

from ._batch import HierarchicalLanguageModelBatch
from ._generation import _validate_options
from ._records import (
    ByteGenerationOptions,
    HierarchicalByteDecoderOutput,
    HierarchicalTextGenerationOutput,
)
from ._text import HierarchicalTextCodec
from ._validation import HierarchicalLanguageModelValidator


class HierarchicalLanguageModel(Module):
    """Compose independent token encoding, causal context, and byte prediction.

    Context/target shifting belongs to the batch codec; the backbone output at
    slot i predicts that slot's next-token target. BOS is a learned vector added
    by this integration. See docs/hierarchical-implementation-review.md for the
    v2 mapping and the differences from the released TFree-HAT models.
    """

    VALIDATOR = HierarchicalLanguageModelValidator

    def __init__(self, cfg, overrides=None):
        self.VALIDATOR.validate_config_types(cfg, overrides)
        super().__init__()
        self.cfg = copy.deepcopy(self._override_config(cfg, overrides))
        self._max_batch = self.VALIDATOR.resolve_config(self.cfg)
        self.encoder = self.cfg.embedding_config.build()
        self.backbone = self.cfg.backbone_config.build()
        self.decoder = self.cfg.decoding_config.build()
        self.positions = self.cfg.position_config.build()
        self.input_normalization = self._build_normalization(
            self.cfg.embedding_normalization_config
        )
        self.output_normalization = self._build_normalization(
            self.cfg.output_normalization_config
        )
        self.dropout = nn.Dropout(self.cfg.dropout_probability)
        self.beginning_of_document = nn.Parameter(
            torch.empty(self.cfg.embedding_config.output_dim)
        )
        nn.init.normal_(
            self.beginning_of_document, std=self.beginning_of_document.numel() ** -0.5
        )

    @staticmethod
    def _build_normalization(config):
        return nn.Identity() if config is None else config.build_normalization()

    def _condition(self, batch):
        self.VALIDATOR.validate_batch(batch, self.cfg.sequence_length)
        device = self.beginning_of_document.device
        mask = batch.attention_mask.to(device)
        bos = batch.bos_mask.to(device)
        state = self.encoder(batch.context_texts, attention_mask=mask & ~bos)
        hidden = torch.where(bos[..., None], self.beginning_of_document, state.hidden)
        lengths = mask.sum(-1)
        total = int(lengths.sum())
        output = torch.zeros_like(hidden)
        loss = state.loss
        for length in lengths.unique().tolist():
            if not length:
                continue
            rows = (lengths == length).nonzero().flatten()
            if len(rows) > self._max_batch:
                raise ValueError("backbone batch_size bound exceeded")
            values = hidden[rows, :length]
            position_marker = (self.cfg.position_config.padding_idx or 0) + 1
            positions = torch.full(
                values.shape[:2], position_marker, dtype=torch.long, device=device
            )
            values = self.dropout(
                self.input_normalization(values + self.positions(positions))
            )
            decoded = self.backbone(TransformerDecoderLayerState(hidden=values))
            output[rows, :length] = self.output_normalization(decoded.hidden)
            if decoded.loss is not None:
                loss = loss + decoded.loss * (len(rows) * length / total)
        return LayerState(hidden=output, loss=loss)

    def forward(
        self, batch: HierarchicalLanguageModelBatch
    ) -> HierarchicalByteDecoderOutput:
        state = self._condition(batch)
        decoded = self.decoder(
            state.hidden,
            batch.byte_prefix_ids,
            batch.byte_lengths,
            batch.attention_mask,
        )
        return HierarchicalByteDecoderOutput(
            decoded.logits, decoded.token_offsets, decoded.loss + state.loss
        )

    def generate_text(
        self,
        prompt: str = "",
        *,
        max_new_tokens: int = 32,
        max_new_bytes: int = 256,
        options: ByteGenerationOptions | None = None,
        prompt_is_complete: bool = False,
    ) -> HierarchicalTextGenerationOutput:
        options = options or ByteGenerationOptions()
        _validate_options(options)
        for name, value in (
            ("max_new_tokens", max_new_tokens),
            ("max_new_bytes", max_new_bytes),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(prompt_is_complete) is not bool:
            raise TypeError("prompt_is_complete must be bool")
        codec = HierarchicalTextCodec(self.cfg.embedding_config.max_token_bytes)
        completed = list(codec.split_text(prompt))
        partial = ""
        if (
            completed
            and not prompt_is_complete
            and not completed[-1][-1].isspace()
            and len(completed[-1].encode()) < codec.max_token_bytes
        ):
            partial = completed.pop()
        text, new_bytes, new_tokens = prompt, 0, 0
        reason = "token_budget"
        generator = (
            torch.Generator().manual_seed(options.seed)
            if options.seed is not None
            else None
        )
        training_states = [(child, child.training) for child in self.modules()]
        try:
            self.eval()
            with torch.no_grad():
                while new_tokens < max_new_tokens:
                    if new_bytes >= max_new_bytes:
                        reason = "byte_budget"
                        break
                    contexts = ([""] + completed)[-self.cfg.sequence_length :]
                    bos = [len(completed) < self.cfg.sequence_length] + [False] * (
                        len(contexts) - 1
                    )
                    batch = HierarchicalLanguageModelBatch.collate(
                        [
                            {
                                "context_texts": contexts,
                                "target_texts": [None] * len(contexts),
                                "bos_mask": bos,
                            }
                        ]
                    )
                    conditioning = self._condition(batch).hidden[0, -1]
                    generated = self.decoder.generate_token(
                        conditioning,
                        prefix=partial,
                        max_new_bytes=max_new_bytes - new_bytes,
                        options=options,
                        generator=generator,
                    )
                    text += generated.text[len(partial) :]
                    new_bytes += generated.new_bytes
                    if generated.stop_reason == "end_of_document":
                        reason = "end_of_document"
                        break
                    completed.append(generated.text)
                    partial = ""
                    new_tokens += 1
                    if generated.stop_reason == "byte_budget":
                        reason = "byte_budget"
                        break
        finally:
            for child, training in training_states:
                child.training = training
        return HierarchicalTextGenerationOutput(text, reason, new_bytes, new_tokens)
