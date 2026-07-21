from __future__ import annotations

from typing import TYPE_CHECKING

from emperor._validation import ValidatorBase
from emperor.config import ConfigBase
from emperor.layers._options import LayerNormPositionOptions
from emperor.layers._validation.common import (
    _HALTING_CONFIG_FIELDS,
    _MEMORY_CONFIG_FIELDS,
    _matches_config_contract,
    _validate_halting_lifecycle_owner,
)
from emperor.layers._validation.gate import LayerGateValidator
from emperor.layers._validation.residual import ResidualConnectionValidator

if TYPE_CHECKING:
    from emperor.halting import HaltingConfig
    from emperor.layers._config import GateConfig, LayerConfig, ResidualConfig
    from emperor.layers._layer import Layer


class LayerValidator(ValidatorBase):
    GATE_VALIDATOR = LayerGateValidator
    RESIDUAL_VALIDATOR = ResidualConnectionValidator

    OPTIONAL_FIELDS = {
        "gate_config",
        "halting_config",
        "memory_config",
        "layer_model_config",
        "residual_config",
        "override_config",
    }

    @classmethod
    def validate(cls, model: Layer) -> None:
        cfg = model.cfg
        cls.validate_required_fields(cfg)
        cls.validate_field_types(cfg)
        cls.validate_dimensions(input_dim=cfg.input_dim, output_dim=cfg.output_dim)
        cls._validate_dropout_probability(cfg.dropout_probability)
        cls._validate_residual_config(cfg.residual_config)
        cls._validate_residual_dimensions(
            cfg.input_dim,
            cfg.output_dim,
            cfg.residual_config,
        )
        cls._validate_gate_config(cfg.gate_config)
        cls._validate_model_config(cfg.layer_model_config)
        cls._validate_layer_norm_with_spatial_model(cfg)
        cls._validate_residual_with_strided_model(cfg)
        cls._validate_halting_config(cfg.halting_config)
        cls._validate_memory_config(cfg.memory_config)
        cls._validate_halting_dimensions(
            cfg.input_dim, cfg.output_dim, cfg.halting_config
        )

    @staticmethod
    def _validate_dropout_probability(dropout_probability: float) -> None:
        if dropout_probability < 0.0 or dropout_probability > 1.0:
            raise ValueError(
                "dropout_probability must be between 0.0 and 1.0, "
                f"received {dropout_probability}"
            )

    @staticmethod
    def _validate_residual_dimensions(
        input_dim: int,
        output_dim: int,
        residual_config: ResidualConfig | None,
    ) -> None:
        if residual_config is None:
            return
        residual_connection_option = residual_config.option
        if input_dim != output_dim:
            raise ValueError(
                "input_dim and output_dim must be equal when "
                f"residual_config.option is {residual_connection_option}, "
                f"got input_dim={input_dim} and output_dim={output_dim}."
            )

    @staticmethod
    def _validate_model_config(model_config: ConfigBase | None) -> None:
        if model_config is None:
            raise ValueError(
                "layer_model_config is required, Layer needs it to build the model"
            )
        if not isinstance(model_config, ConfigBase):
            raise TypeError(
                f"model_config must be an instance of ConfigBase, "
                f"got {type(model_config).__name__}"
            )

    @staticmethod
    def _validate_layer_norm_with_spatial_model(cfg: LayerConfig) -> None:
        layer_model_config = cfg.layer_model_config
        if not hasattr(layer_model_config, "kernel_size"):
            return
        if cfg.layer_norm_position == LayerNormPositionOptions.DISABLED:
            return
        raise ValueError(
            f"layer_norm_position must be DISABLED when layer_model_config "
            f"is a spatial (Conv2d-like) module, received "
            f"{cfg.layer_norm_position}. nn.LayerNorm normalizes over the last "
            f"tensor dim; for (B, C, H, W) inputs that is W, which is not "
            f"channel normalization. Use BatchNorm2d or GroupNorm externally, "
            f"or disable layer norm."
        )

    @staticmethod
    def _validate_residual_with_strided_model(cfg: LayerConfig) -> None:
        layer_model_config = cfg.layer_model_config
        stride = getattr(layer_model_config, "stride", None)
        if stride is None or stride <= 1:
            return
        if cfg.residual_config is None:
            return
        raise ValueError(
            f"residual_config.option cannot be "
            f"{cfg.residual_config.option} when layer_model_config has "
            f"stride > 1 (received stride={stride}). Spatial reduction "
            f"breaks the residual connection shape contract."
        )

    @classmethod
    def _validate_gate_config(cls, gate_config: GateConfig | None) -> None:
        cls.GATE_VALIDATOR.validate_layer_gate_config(
            gate_config, owner_name="LayerConfig.gate_config"
        )

    @classmethod
    def _validate_residual_config(
        cls,
        residual_config: ResidualConfig | None,
    ) -> None:
        cls.RESIDUAL_VALIDATOR.validate_residual_config(
            residual_config,
            owner_name="LayerConfig",
        )

    @staticmethod
    def _validate_halting_config(
        halting_config: HaltingConfig | None,
    ) -> None:
        if halting_config is None:
            return
        if not _matches_config_contract(halting_config, _HALTING_CONFIG_FIELDS):
            raise TypeError(
                "halting_config must be an instance of HaltingConfig, "
                f"got {type(halting_config).__name__}"
            )
        _validate_halting_lifecycle_owner(
            halting_config,
            field_name="halting_config",
            owner_name="LayerConfig",
        )

    @staticmethod
    def _validate_memory_config(
        memory_config,
    ) -> None:
        if memory_config is None:
            return
        if not _matches_config_contract(memory_config, _MEMORY_CONFIG_FIELDS):
            raise TypeError(
                f"memory_config must be an instance of DynamicMemoryConfig, "
                f"got {type(memory_config).__name__}."
            )

    @staticmethod
    def _validate_halting_dimensions(
        input_dim: int,
        output_dim: int,
        halting_config: HaltingConfig | None,
    ) -> None:
        if halting_config is not None and input_dim != output_dim:
            raise ValueError(
                "input_dim and output_dim must be equal when halting_config "
                "is provided, "
                f"got input_dim={input_dim} and output_dim={output_dim}. "
                "Halting accumulates hidden states across steps, which requires "
                "consistent dimensions."
            )
