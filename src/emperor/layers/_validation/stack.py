from __future__ import annotations

from typing import TYPE_CHECKING

from emperor._validation import ValidatorBase
from emperor.layers._validation.common import (
    _HALTING_CONFIG_FIELDS,
    _MEMORY_CONFIG_FIELDS,
    _matches_config_contract,
    _validate_halting_lifecycle_owner,
)
from emperor.layers._validation.gate import LayerGateValidator

if TYPE_CHECKING:
    from emperor.layers._config import LayerStackConfig
    from emperor.layers._stack import LayerStack


class LayerStackValidator(ValidatorBase):
    GATE_VALIDATOR = LayerGateValidator

    OPTIONAL_FIELDS = {
        "layer_type",
        "shared_gate_config",
        "shared_halting_config",
        "shared_memory_config",
        "override_config",
    }

    @classmethod
    def validate(cls, model: LayerStack) -> None:
        cfg = model.cfg
        cls.validate_required_fields(cfg)
        cls.validate_field_types(cfg)
        cls.validate_dimensions(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim,
            num_layers=cfg.num_layers,
        )
        cls._validate_gate_config(cfg)
        cls._validate_halting_config(cfg)
        cls._validate_memory_config(cfg)

    @classmethod
    def _validate_gate_config(cls, cfg: LayerStackConfig) -> None:
        cls.GATE_VALIDATOR.validate_shared_gate_config_type(cfg.shared_gate_config)
        if cls.GATE_VALIDATOR.is_gate_config_active(
            cfg.shared_gate_config
        ) and cls.GATE_VALIDATOR.is_gate_config_active(cfg.layer_config.gate_config):
            raise ValueError(
                "shared_gate_config and layer_config.gate_config are mutually "
                "exclusive. Put shared gate controllers on LayerStackConfig and "
                "per-layer gate controllers on LayerConfig."
            )
        cls.GATE_VALIDATOR.validate_layer_gate_config(
            cfg.layer_config.gate_config,
            owner_name="LayerStackConfig.layer_config",
        )
        cls.GATE_VALIDATOR.validate_layer_gate_config(
            cfg.shared_gate_config,
            owner_name="LayerStackConfig.shared_gate_config",
        )
        if cls.GATE_VALIDATOR.is_gate_config_active(cfg.shared_gate_config):
            cls._validate_shared_gate_dimensions(cfg)

    @classmethod
    def _validate_halting_config(cls, cfg: LayerStackConfig) -> None:
        cls._validate_shared_halting_config_type(cfg.shared_halting_config)
        if (
            cfg.shared_halting_config is not None
            and cfg.layer_config.halting_config is not None
        ):
            raise ValueError(
                "shared_halting_config and layer_config.halting_config are mutually "
                "exclusive. Put shared halting controllers on LayerStackConfig and "
                "per-layer halting controllers on LayerConfig."
            )
        halting_config = (
            cfg.shared_halting_config
            if cfg.shared_halting_config is not None
            else cfg.layer_config.halting_config
        )
        if halting_config is not None and cfg.num_layers < 2:
            raise ValueError(
                f"num_layers must be at least 2 when halting_config is provided, "
                f"got {cfg.num_layers}. The halting mechanism requires multiple "
                f"steps to accumulate halting probabilities across layers."
            )
        if halting_config is not None and (
            cfg.input_dim != cfg.hidden_dim or cfg.hidden_dim != cfg.output_dim
        ):
            raise ValueError(
                "input_dim, hidden_dim, and output_dim must all be equal when "
                "halting_config is provided, "
                f"got input_dim={cfg.input_dim}, hidden_dim={cfg.hidden_dim}, "
                f"output_dim={cfg.output_dim}. Halting accumulates hidden states "
                "across steps, which requires consistent dimensions."
            )

    @classmethod
    def _validate_memory_config(cls, cfg: LayerStackConfig) -> None:
        cls._validate_shared_memory_config_type(cfg.shared_memory_config)
        if (
            cfg.shared_memory_config is not None
            and cfg.layer_config.memory_config is not None
        ):
            raise ValueError(
                "shared_memory_config and layer_config.memory_config are mutually "
                "exclusive. Put shared memory controllers on LayerStackConfig and "
                "per-layer memory controllers on LayerConfig."
            )
        if cfg.shared_memory_config is not None and (
            cfg.input_dim != cfg.hidden_dim or cfg.hidden_dim != cfg.output_dim
        ):
            raise ValueError(
                f"input_dim, hidden_dim, and output_dim must all be equal when "
                f"shared_memory_config is provided, got input_dim={cfg.input_dim}, "
                f"hidden_dim={cfg.hidden_dim}, output_dim={cfg.output_dim}. "
                f"Shared memory uses one module across all layers and requires "
                f"consistent dimensions."
            )

    @staticmethod
    def _validate_shared_gate_dimensions(cfg: LayerStackConfig) -> None:
        if cfg.hidden_dim != cfg.output_dim:
            raise ValueError(
                f"hidden_dim and output_dim must be equal when "
                f"shared_gate_config is provided, got "
                f"hidden_dim={cfg.hidden_dim} and output_dim={cfg.output_dim}. "
                f"Shared gates use one module across all layer outputs and "
                f"require one gate dimension."
            )

    @staticmethod
    def _validate_shared_halting_config_type(shared_halting_config) -> None:
        if shared_halting_config is None:
            return
        if not _matches_config_contract(
            shared_halting_config,
            _HALTING_CONFIG_FIELDS,
        ):
            raise TypeError(
                "shared_halting_config must be an instance of HaltingConfig for "
                f"LayerStackConfig, got {type(shared_halting_config).__name__}"
            )
        _validate_halting_lifecycle_owner(
            shared_halting_config,
            field_name="shared_halting_config",
            owner_name="LayerStackConfig",
        )

    @staticmethod
    def _validate_shared_memory_config_type(shared_memory_config) -> None:
        if shared_memory_config is None:
            return
        if not _matches_config_contract(shared_memory_config, _MEMORY_CONFIG_FIELDS):
            raise TypeError(
                "shared_memory_config must be an instance of DynamicMemoryConfig "
                f"for LayerStackConfig, got {type(shared_memory_config).__name__}"
            )
