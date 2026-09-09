from __future__ import annotations

import math
from dataclasses import replace
from numbers import Real

import torch
from torch import Tensor

from emperor._validation import ValidatorBase
from emperor.sampler._config import RouterConfig
from emperor.sampler._token_config import TokenSamplerConfig


class TokenSamplerValidator(ValidatorBase):
    @classmethod
    def validate_config(cls, cfg: TokenSamplerConfig) -> None:
        if not isinstance(cfg, TokenSamplerConfig):
            raise TypeError("TokenSamplerModel requires TokenSamplerConfig.")
        cls.validate_required_fields(cfg)
        if isinstance(cfg.input_dim, bool) or not isinstance(cfg.input_dim, int):
            raise TypeError("input_dim must be an integer.")
        if cfg.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        ratio = cfg.selection_ratio
        if isinstance(ratio, bool) or not isinstance(ratio, Real):
            raise TypeError("selection_ratio must be a real number.")
        if not math.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError("selection_ratio must be finite and in (0, 1].")
        if not isinstance(cfg.router_config, RouterConfig):
            raise TypeError("router_config must be a RouterConfig.")
        if (
            type(cfg.router_config.num_experts) is not int
            or cfg.router_config.num_experts != 1
            or cfg.router_config.noisy_topk_flag is not False
        ):
            raise ValueError(
                "Token sampler router requires num_experts=1 and noisy_topk_flag=False."
            )
        router_cfg = replace(cfg.router_config, input_dim=cfg.input_dim)
        router_cfg.registry_owner().VALIDATOR.validate_config(router_cfg)

    @staticmethod
    def validate_hidden(hidden: Tensor, input_dim: int) -> None:
        if not isinstance(hidden, Tensor) or not hidden.is_floating_point():
            raise TypeError("Token sampler hidden must be a floating-point Tensor.")
        if hidden.ndim < 2 or hidden.shape[-1] != input_dim:
            raise ValueError("Token sampler requires [..., tokens, input_dim] hidden.")
        if any(size == 0 for size in hidden.shape):
            raise ValueError("Token sampler requires non-empty dimensions.")

    @staticmethod
    def validate_logits(logits: Tensor, flattened_hidden: Tensor) -> None:
        if not isinstance(logits, Tensor) or not logits.is_floating_point():
            raise TypeError("Token router must return floating-point logits.")
        if logits.shape != (flattened_hidden.shape[0], 1):
            raise ValueError("Token router must return one logit per input token.")
        if (
            logits.device != flattened_hidden.device
            or logits.dtype != flattened_hidden.dtype
        ):
            raise ValueError("Token router must preserve hidden device and dtype.")

    @staticmethod
    def padding(hidden: Tensor, padding_mask: Tensor | None) -> Tensor:
        padding = torch.zeros(hidden.shape[:-1], dtype=torch.bool, device=hidden.device)
        if padding_mask is not None:
            if not isinstance(padding_mask, Tensor):
                raise TypeError("Token padding mask must be a Tensor.")
            if padding_mask.shape != padding.shape:
                raise ValueError("Token padding mask must match hidden.shape[:-1].")
            if padding_mask.device != hidden.device:
                raise ValueError("Token padding mask must be on the hidden device.")
            if (
                padding_mask.dtype != torch.bool
                and not padding_mask.is_floating_point()
            ):
                raise TypeError("Token padding mask must be boolean or floating-point.")
            padding = (
                padding_mask
                if padding_mask.dtype == torch.bool
                else padding_mask.isneginf()
            )
        if padding.all(dim=-1).any():
            raise ValueError(
                "Token sampling requires at least one unpadded token per sequence."
            )
        return padding
