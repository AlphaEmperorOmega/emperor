import math
from numbers import Integral, Real
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.layers import LastLayerBiasOptions, LayerConfig, LayerStackConfig

if TYPE_CHECKING:
    from emperor.halting._base import HaltingBase
    from emperor.halting._config import HaltingConfig


class StickBreakingValidator:
    OPTIONAL_FIELDS = {
        "dropout_probability",
        "override_config",
    }

    @classmethod
    def validate(cls, model: "HaltingBase") -> None:
        cfg = model.cfg
        cls._validate_required_fields(cfg)
        cls._validate_input_dim(cfg.input_dim)
        cls._validate_threshold(cfg.threshold)
        cls._validate_dropout_probability(cfg.dropout_probability)
        cls._validate_hidden_state_mode(cfg.hidden_state_mode)
        cls._validate_halting_gate_config(cfg.halting_gate_config)
        cls._validate_halting_gate_layer_config(cfg.halting_gate_config.layer_config)

    @classmethod
    def validate_config(cls, cfg: "HaltingConfig") -> None:
        cls.validate(SimpleNamespace(cfg=cfg, threshold=cfg.threshold))

    @classmethod
    def _validate_required_fields(cls, cfg: "HaltingConfig") -> None:
        for field_name in cfg.__dataclass_fields__:
            if field_name in cls.OPTIONAL_FIELDS:
                continue
            if getattr(cfg, field_name) is None:
                raise ValueError(
                    f"{field_name} is required for {type(cfg).__name__}, received None"
                )

    @staticmethod
    def _validate_input_dim(input_dim: int) -> None:
        if isinstance(input_dim, bool) or not isinstance(input_dim, Integral):
            raise TypeError(
                "input_dim must be a positive integer, "
                f"received {type(input_dim).__name__}"
            )
        if input_dim <= 0:
            raise ValueError(f"input_dim must be greater than 0, received {input_dim}")

    @staticmethod
    def _validate_threshold(threshold: float) -> None:
        if isinstance(threshold, bool) or not isinstance(threshold, Real):
            raise TypeError(
                f"threshold must be a number, received {type(threshold).__name__}"
            )
        if not math.isfinite(float(threshold)) or not 0.0 < threshold <= 1.0:
            raise ValueError(
                "threshold must be finite and between 0.0 (exclusive) and "
                f"1.0 (inclusive), received {threshold}"
            )

    @staticmethod
    def _validate_dropout_probability(dropout_probability: float | None) -> None:
        if dropout_probability is None:
            return
        if isinstance(dropout_probability, bool) or not isinstance(
            dropout_probability, Real
        ):
            raise TypeError(
                "dropout_probability must be a number or None, "
                f"received {type(dropout_probability).__name__}"
            )
        if (
            not math.isfinite(float(dropout_probability))
            or not 0.0 <= dropout_probability <= 1.0
        ):
            raise ValueError(
                "dropout_probability must be finite and between 0.0 and 1.0, "
                f"received {dropout_probability}"
            )

    @staticmethod
    def _validate_hidden_state_mode(hidden_state_mode) -> None:
        from emperor.halting._config import HaltingHiddenStateModeOptions

        if not isinstance(hidden_state_mode, HaltingHiddenStateModeOptions):
            raise TypeError(
                "hidden_state_mode must be a HaltingHiddenStateModeOptions "
                f"value, received {type(hidden_state_mode).__name__}"
            )

    @classmethod
    def _validate_halting_gate_config(
        cls,
        halting_gate_config: "LayerStackConfig | None",
    ) -> None:
        if not isinstance(halting_gate_config, LayerStackConfig):
            raise TypeError(
                f"halting_gate_config must be an instance of LayerStackConfig, "
                f"got {type(halting_gate_config).__name__}"
            )
        if halting_gate_config.output_dim != 2:
            raise ValueError(
                "halting_gate_config.output_dim must be 2 "
                "(continuation and halting logits), "
                f"received {halting_gate_config.output_dim}"
            )
        if halting_gate_config.last_layer_bias_option != LastLayerBiasOptions.DISABLED:
            raise ValueError(
                f"halting_gate_config.last_layer_bias_option must be DISABLED, "
                f"received {halting_gate_config.last_layer_bias_option}"
            )
        if halting_gate_config.shared_halting_config is not None:
            raise ValueError(
                "halting_gate_config.shared_halting_config must be None, "
                "nested halting is not allowed"
            )
        if cls._is_gate_config_active(halting_gate_config.shared_gate_config):
            raise ValueError(
                "halting_gate_config.shared_gate_config must be inactive, "
                "nested gates are not allowed in halting"
            )

    @classmethod
    def _validate_halting_gate_layer_config(
        cls,
        layer_config: "LayerConfig | None",
    ) -> None:
        if layer_config is None:
            return
        if not isinstance(layer_config, LayerConfig):
            raise TypeError(
                "halting_gate_config.layer_config must be a LayerConfig or None, "
                f"got {type(layer_config).__name__}"
            )
        if cls._is_gate_config_active(layer_config.gate_config):
            raise ValueError(
                "halting_gate_config.layer_config.gate_config must be None, "
                "nested gates are not allowed in halting"
            )
        if layer_config.halting_config is not None:
            raise ValueError(
                "halting_gate_config.layer_config.halting_config must be None, "
                "nested halting is not allowed"
            )

    @staticmethod
    def _is_gate_config_active(gate_config) -> bool:
        return gate_config is not None

    @staticmethod
    def validate_hidden_tensor(
        hidden: Tensor,
        input_dim: int,
        field_name: str = "model_hidden_state",
    ) -> None:
        if not isinstance(hidden, Tensor):
            raise TypeError(
                f"{field_name} must be a Tensor, received {type(hidden).__name__}"
            )
        if hidden.dim() < 2:
            raise ValueError(
                f"{field_name} must have rank >= 2 with feature-last layout, "
                f"received {hidden.dim()}D tensor with shape {tuple(hidden.shape)}"
            )
        if hidden.shape[-1] != input_dim:
            raise ValueError(
                f"{field_name} final dimension must be {input_dim}, "
                f"received {hidden.shape[-1]} with shape {tuple(hidden.shape)}"
            )

    @staticmethod
    def validate_tensor_shape(
        tensor: Tensor,
        expected_shape: torch.Size,
        field_name: str,
    ) -> None:
        if tensor.shape != expected_shape:
            raise ValueError(
                f"{field_name} must have shape {tuple(expected_shape)}, "
                f"received {tuple(tensor.shape)}"
            )

    @staticmethod
    def validate_pad_mask(
        pad_mask: Tensor | None,
        hidden: Tensor,
        *,
        required_by: str | None = None,
    ) -> None:
        if pad_mask is None:
            if required_by is not None:
                raise TypeError(
                    f"pad_mask must be a Tensor for {required_by}, received None"
                )
            return
        if not isinstance(pad_mask, Tensor):
            raise TypeError(
                f"pad_mask must be a Tensor or None, received {type(pad_mask).__name__}"
            )
        expected_shape = hidden.shape[:-1]
        if pad_mask.shape != expected_shape:
            raise ValueError(
                f"pad_mask must have shape {tuple(expected_shape)}, "
                f"received {tuple(pad_mask.shape)}"
            )
        if pad_mask.dtype != torch.bool and not torch.is_floating_point(pad_mask):
            raise TypeError(
                f"pad_mask must use bool or floating dtype, received {pad_mask.dtype}"
            )
        if pad_mask.dtype != torch.bool and (
            not torch.isfinite(pad_mask).all().item()
            or (pad_mask < 0.0).any().item()
            or (pad_mask > 1.0).any().item()
        ):
            raise ValueError("pad_mask values must be finite and between 0.0 and 1.0")
