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


class _HaltingContractValidator:
    @classmethod
    def validate_minimum_step_capability(
        cls,
        cfg: "HaltingConfig",
        *,
        owner_name: str,
    ) -> None:
        configured_min_steps = getattr(cfg, "min_steps", None)
        cls._validate_min_steps(configured_min_steps)
        min_steps = 1 if configured_min_steps is None else configured_min_steps
        if min_steps == 1:
            return
        owner = cfg._registry_owner()
        if getattr(owner, "supports_minimum_step_delay", False) is not True:
            owner_type_name = getattr(owner, "__name__", type(owner).__name__)
            raise ValueError(
                f"halting_config.min_steps={min_steps} requires minimum-step "
                f"delay support for {owner_name}; {owner_type_name} implements "
                "only the legacy lifecycle."
            )

    @classmethod
    def validate_owner_step_contract(
        cls,
        cfg: "HaltingConfig",
        *,
        owner_step_limit: int | None,
        owner_name: str,
    ) -> None:
        cls.validate_minimum_step_capability(cfg, owner_name=owner_name)
        configured_min_steps = getattr(cfg, "min_steps", None)
        min_steps = 1 if configured_min_steps is None else configured_min_steps
        if min_steps == 1:
            return
        if owner_step_limit is None:
            raise ValueError(
                f"halting_config.min_steps={min_steps} is not supported for "
                f"{owner_name}; this owner has no defined owner step limit."
            )
        if isinstance(owner_step_limit, bool) or not isinstance(
            owner_step_limit,
            Integral,
        ):
            raise TypeError(
                "owner_step_limit must be an integer or None, "
                f"received {type(owner_step_limit).__name__}"
            )
        if owner_step_limit < 1:
            raise ValueError(
                "owner_step_limit must be greater than or equal to 1, "
                f"received {owner_step_limit}"
            )
        if min_steps > owner_step_limit:
            raise ValueError(
                "min_steps must be less than or equal to the owner step limit, "
                f"received min_steps={min_steps} and "
                f"owner_step_limit={owner_step_limit}."
            )

    @staticmethod
    def _validate_ponder_cost_weight(ponder_cost_weight: float | None) -> None:
        if ponder_cost_weight is None:
            return
        if isinstance(ponder_cost_weight, bool) or not isinstance(
            ponder_cost_weight, Real
        ):
            raise TypeError(
                "ponder_cost_weight must be a number or None, "
                f"received {type(ponder_cost_weight).__name__}"
            )
        if not math.isfinite(float(ponder_cost_weight)) or ponder_cost_weight < 0:
            raise ValueError(
                "ponder_cost_weight must be finite and greater than or equal to 0, "
                f"received {ponder_cost_weight}"
            )

    @staticmethod
    def _validate_min_steps(min_steps: int | None) -> None:
        if min_steps is None:
            return
        if isinstance(min_steps, bool) or not isinstance(min_steps, Integral):
            raise TypeError(
                "min_steps must be an integer or None, "
                f"received {type(min_steps).__name__}"
            )
        if min_steps < 1:
            raise ValueError(
                f"min_steps must be greater than or equal to 1, received {min_steps}"
            )


class StickBreakingValidator(_HaltingContractValidator):
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
        cls._validate_min_steps(cfg.min_steps)
        cls._validate_ponder_cost_weight(cfg.ponder_cost_weight)
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


class SoftHaltingValidator(StickBreakingValidator):
    OPTIONAL_FIELDS = {
        *StickBreakingValidator.OPTIONAL_FIELDS,
        "halting_gate_config",
    }

    @classmethod
    def validate(cls, model: "HaltingBase") -> None:
        cfg = model.cfg
        cls._validate_required_fields(cfg)
        cls._validate_input_dim(cfg.input_dim)
        cls._validate_threshold(cfg.threshold)
        cls._validate_min_steps(cfg.min_steps)
        cls._validate_ponder_cost_weight(cfg.ponder_cost_weight)
        cls._validate_dropout_probability(cfg.dropout_probability)
        cls._validate_hidden_state_mode(cfg.hidden_state_mode)
        if cfg.halting_gate_config is None:
            return
        cls._validate_halting_gate_config(cfg.halting_gate_config)
        cls._validate_halting_gate_layer_config(cfg.halting_gate_config.layer_config)
