"""Generic token scoring and per-sequence Top-K sampling, independent of layers."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

import torch
from torch import Tensor

from emperor.nn import Module
from emperor.sampler._config import RouterConfig
from emperor.sampler._token_config import TokenSamplerConfig
from emperor.sampler._token_validation import TokenSamplerValidator


@dataclass(frozen=True)
class TokenSamplingResult:
    indices: Tensor
    weights: Tensor
    valid: Tensor
    sequence_length: int


class TokenSamplerModel(Module):
    VALIDATOR = TokenSamplerValidator

    def __init__(
        self, cfg: TokenSamplerConfig, overrides: TokenSamplerConfig | None = None
    ) -> None:
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.VALIDATOR.validate_config(self.cfg)
        self.router = self.cfg.router_config.build(
            RouterConfig(input_dim=self.cfg.input_dim)
        )

    def forward(
        self, hidden: Tensor, padding_mask: Tensor | None = None
    ) -> TokenSamplingResult:
        self.VALIDATOR.validate_hidden(hidden, self.cfg.input_dim)
        padding = self.VALIDATOR.padding(hidden, padding_mask)
        flattened = hidden.reshape(-1, hidden.shape[-1])
        logits = self.router.compute_logit_scores(flattened)
        self.VALIDATOR.validate_logits(logits, flattened)
        scores = logits.reshape(hidden.shape[:-1])
        count = max(1, floor(self.cfg.selection_ratio * hidden.shape[-2]))
        # Rank logits to avoid sigmoid saturation changing the Top-K ordering.
        indices = scores.masked_fill(padding, -torch.inf).topk(count, dim=-1).indices
        valid = ~padding.gather(-1, indices)
        # Fill unused capacity with a valid query; consumers mask its key/update.
        first_valid = (~padding).to(torch.int64).argmax(dim=-1, keepdim=True)
        indices = torch.where(valid, indices, first_valid)
        order = (indices * 2 + (~valid).to(torch.int64)).argsort(dim=-1, stable=True)
        indices = indices.gather(-1, order)
        valid = valid.gather(-1, order)
        weights = scores.gather(-1, indices).sigmoid().masked_fill(~valid, 0)
        return TokenSamplingResult(indices, weights, valid, hidden.shape[-2])
