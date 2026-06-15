from __future__ import annotations

from dataclasses import fields

from emperor.base.utils import ConfigBase
from emperor.base.options import LayerNormPositionOptions
from emperor.base.validator import ValidatorBase
from .residual import ResidualConnectionOptions
from .gate._validator import LayerGateValidator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from emperor.halting.config import HaltingConfig

    from .config import (
        GateConfig,
        LayerConfig,
        LayerStackConfig,
        RecurrentLayerConfig,
    )
    from .state import LayerState


class LayerValidator(ValidatorBase):
    OPTIONAL_FIELDS = {
        "gate_config",
        "halting_config",
        "memory_config",
        "layer_model_config",
        "override_config",
    }

    @staticmethod
    def validate(cfg: LayerConfig) -> None:
        LayerValidator.validate_required_fields(cfg)
        LayerValidator.validate_field_types(cfg)
        LayerValidator.validate_dimensions(
            input_dim=cfg.input_dim, output_dim=cfg.output_dim
        )
        LayerValidator.__validate_dropout_probability(cfg.dropout_probability)
        LayerValidator.__validate_residual_dimensions(
            cfg.input_dim, cfg.output_dim, cfg.residual_connection_option
        )
        LayerValidator.__validate_gate_config(cfg.gate_config)
        LayerValidator.__validate_model_config(cfg.layer_model_config)
        LayerValidator.__validate_layer_norm_with_spatial_model(cfg)
        LayerValidator.__validate_residual_with_strided_model(cfg)
        LayerValidator.__validate_halting_config(cfg.halting_config)
        LayerValidator.__validate_memory_config(cfg.memory_config)
        LayerValidator.__validate_halting_dimensions(
            cfg.input_dim, cfg.output_dim, cfg.halting_config
        )

    @staticmethod
    def __validate_dropout_probability(dropout_probability: float) -> None:
        if dropout_probability < 0.0 or dropout_probability > 1.0:
            raise ValueError(
                f"dropout_probability must be between 0.0 and 1.0, received {dropout_probability}"
            )

    @staticmethod
    def __validate_residual_dimensions(
        input_dim: int,
        output_dim: int,
        residual_connection_option: ResidualConnectionOptions,
    ) -> None:
        if (
            residual_connection_option != ResidualConnectionOptions.DISABLED
            and input_dim != output_dim
        ):
            raise ValueError(
                "input_dim and output_dim must be equal when "
                f"residual_connection_option is {residual_connection_option}, "
                f"got input_dim={input_dim} and output_dim={output_dim}."
            )

    @staticmethod
    def __validate_model_config(model_config: ConfigBase | None) -> None:
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
    def __validate_layer_norm_with_spatial_model(cfg: LayerConfig) -> None:
        layer_model_config = cfg.layer_model_config
        if layer_model_config is None:
            return
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
    def __validate_residual_with_strided_model(cfg: LayerConfig) -> None:
        layer_model_config = cfg.layer_model_config
        if layer_model_config is None:
            return
        stride = getattr(layer_model_config, "stride", None)
        if stride is None or stride <= 1:
            return
        if cfg.residual_connection_option == ResidualConnectionOptions.DISABLED:
            return
        raise ValueError(
            f"residual_connection_option cannot be "
            f"{cfg.residual_connection_option} when layer_model_config has "
            f"stride > 1 (received stride={stride}). Spatial reduction "
            f"breaks the residual connection shape contract."
        )

    @staticmethod
    def __validate_gate_config(gate_config: "GateConfig | None") -> None:
        LayerGateValidator.validate_layer_gate_config(
            gate_config, owner_name="LayerConfig.gate_config"
        )

    @staticmethod
    def __validate_halting_config(
        halting_config: "HaltingConfig | None",
    ) -> None:
        if halting_config is not None:
            from emperor.halting.config import HaltingConfig

            if not isinstance(halting_config, HaltingConfig):
                raise TypeError(
                    f"halting_config must be an instance of HaltingConfig, "
                    f"got {type(halting_config).__name__}"
                )

    @staticmethod
    def __validate_memory_config(
        memory_config,
    ) -> None:
        if memory_config is None:
            return
        from emperor.memory.config import DynamicMemoryConfig

        if not isinstance(memory_config, DynamicMemoryConfig):
            raise TypeError(
                f"memory_config must be an instance of DynamicMemoryConfig, "
                f"got {type(memory_config).__name__}."
            )

    @staticmethod
    def __validate_halting_dimensions(
        input_dim: int,
        output_dim: int,
        halting_config: "HaltingConfig | None",
    ) -> None:
        if halting_config is not None and input_dim != output_dim:
            raise ValueError(
                f"input_dim and output_dim must be equal when halting_config is provided, "
                f"got input_dim={input_dim} and output_dim={output_dim}. "
                f"Halting accumulates hidden states across steps, which requires consistent dimensions."
            )


class LayerStackValidator(ValidatorBase):
    OPTIONAL_FIELDS = {
        "layer_type",
        "shared_gate_config",
        "shared_halting_config",
        "shared_memory_config",
        "override_config",
    }

    @staticmethod
    def validate(cfg: "LayerStackConfig") -> None:
        LayerStackValidator.validate_required_fields(cfg)
        LayerStackValidator.validate_field_types(cfg)
        LayerStackValidator.validate_dimensions(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim,
            num_layers=cfg.num_layers,
        )
        LayerStackValidator.__validate_num_layers(cfg.num_layers)
        LayerStackValidator.__validate_layer_config(cfg.layer_config)
        LayerStackValidator.__validate_gate_config(cfg)
        LayerStackValidator.__validate_halting_config(cfg)
        LayerStackValidator.__validate_memory_config(cfg)

    @staticmethod
    def __validate_num_layers(num_layers: int) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, received {num_layers}.")

    @staticmethod
    def __validate_layer_config(layer_config: "LayerConfig | None") -> None:
        from .config import LayerConfig

        if layer_config is None:
            raise ValueError(f"layer_config is required, received None")
        if not isinstance(layer_config, LayerConfig):
            raise TypeError(
                f"layer_config must be an instance of LayerConfig, "
                f"got {type(layer_config).__name__}"
            )

    @staticmethod
    def __validate_gate_config(cfg: "LayerStackConfig") -> None:
        if cfg.layer_config is None:
            return
        LayerGateValidator.validate_shared_gate_config_type(cfg.shared_gate_config)
        if (
            LayerGateValidator.is_gate_config_active(cfg.shared_gate_config)
            and LayerGateValidator.is_gate_config_active(cfg.layer_config.gate_config)
        ):
            raise ValueError(
                "shared_gate_config and layer_config.gate_config are mutually "
                "exclusive. Put shared gate controllers on LayerStackConfig and "
                "per-layer gate controllers on LayerConfig."
            )
        LayerGateValidator.validate_layer_gate_config(
            cfg.layer_config.gate_config,
            owner_name="LayerStackConfig.layer_config",
        )
        LayerGateValidator.validate_layer_gate_config(
            cfg.shared_gate_config,
            owner_name="LayerStackConfig.shared_gate_config",
        )
        if LayerGateValidator.is_gate_config_active(cfg.shared_gate_config):
            LayerStackValidator.__validate_shared_gate_dimensions(cfg)

    @staticmethod
    def __validate_halting_config(cfg: "LayerStackConfig") -> None:
        if cfg.layer_config is None:
            return
        LayerStackValidator.__validate_shared_halting_config_type(
            cfg.shared_halting_config
        )
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
                f"got {cfg.num_layers}. The halting mechanism requires multiple steps to accumulate "
                f"halting probabilities across layers."
            )
        if halting_config is not None and (
            cfg.input_dim != cfg.hidden_dim or cfg.hidden_dim != cfg.output_dim
        ):
            raise ValueError(
                f"input_dim, hidden_dim, and output_dim must all be equal when halting_config is provided, "
                f"got input_dim={cfg.input_dim}, hidden_dim={cfg.hidden_dim}, output_dim={cfg.output_dim}. "
                f"Halting accumulates hidden states across steps, which requires consistent dimensions."
            )

    @staticmethod
    def __validate_memory_config(cfg: "LayerStackConfig") -> None:
        if cfg.layer_config is None:
            return
        LayerStackValidator.__validate_shared_memory_config_type(
            cfg.shared_memory_config
        )
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
    def __validate_shared_gate_dimensions(cfg: "LayerStackConfig") -> None:
        if cfg.hidden_dim != cfg.output_dim:
            raise ValueError(
                f"hidden_dim and output_dim must be equal when "
                f"shared_gate_config is provided, got "
                f"hidden_dim={cfg.hidden_dim} and output_dim={cfg.output_dim}. "
                f"Shared gates use one module across all layer outputs and "
                f"require one gate dimension."
            )

    @staticmethod
    def __validate_shared_halting_config_type(shared_halting_config) -> None:
        if shared_halting_config is None:
            return

        from emperor.halting.config import HaltingConfig

        if not isinstance(shared_halting_config, HaltingConfig):
            raise TypeError(
                "shared_halting_config must be an instance of HaltingConfig for "
                f"LayerStackConfig, got {type(shared_halting_config).__name__}"
            )

    @staticmethod
    def __validate_shared_memory_config_type(shared_memory_config) -> None:
        if shared_memory_config is None:
            return

        from emperor.memory.config import DynamicMemoryConfig

        if not isinstance(shared_memory_config, DynamicMemoryConfig):
            raise TypeError(
                "shared_memory_config must be an instance of DynamicMemoryConfig "
                f"for LayerStackConfig, got {type(shared_memory_config).__name__}"
            )


class RecurrentLayerValidator(ValidatorBase):
    OPTIONAL_FIELDS = {
        "gate_config",
        "halting_config",
        "memory_config",
        "override_config",
    }

    @staticmethod
    def validate(cfg: RecurrentLayerConfig) -> None:
        RecurrentLayerValidator.validate_required_fields(cfg)
        RecurrentLayerValidator.__validate_integer_field(
            "input_dim",
            cfg.input_dim,
        )
        RecurrentLayerValidator.__validate_integer_field(
            "output_dim",
            cfg.output_dim,
        )
        RecurrentLayerValidator.__validate_integer_field(
            "max_steps",
            cfg.max_steps,
        )
        RecurrentLayerValidator.validate_dimensions(
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            max_steps=cfg.max_steps,
        )
        RecurrentLayerValidator.__validate_stable_dimensions(
            cfg.input_dim,
            cfg.output_dim,
        )
        RecurrentLayerValidator.__validate_recurrent_layer_norm_position(
            cfg.recurrent_layer_norm_position
        )
        RecurrentLayerValidator.__validate_block_config(cfg.block_config)
        RecurrentLayerValidator.__validate_residual_connection_option(
            cfg.residual_connection_option
        )
        RecurrentLayerValidator.__validate_gate_config(cfg.gate_config)
        RecurrentLayerValidator.__validate_halting_config(cfg.halting_config)
        RecurrentLayerValidator.__validate_memory_config(cfg.memory_config)

    @staticmethod
    def validate_state(state: "LayerState", expected_feature_dim: int) -> None:
        from .state import LayerState

        if not isinstance(state, LayerState):
            raise TypeError(
                f"state must be an instance of LayerState for RecurrentLayer, "
                f"got {type(state).__name__}"
            )
        RecurrentLayerValidator.validate_hidden(
            state.hidden,
            expected_feature_dim,
            "state.hidden",
        )

    @staticmethod
    def validate_hidden(
        hidden: "Tensor",
        expected_feature_dim: int,
        field_name: str = "hidden",
    ) -> None:
        if hidden.dim() < 2:
            raise ValueError(
                f"{field_name} must have rank >= 2 with feature-last layout, "
                f"got {hidden.dim()}D tensor with shape {tuple(hidden.shape)}"
            )
        actual_feature_dim = hidden.shape[-1]
        if actual_feature_dim != expected_feature_dim:
            raise ValueError(
                f"{field_name} last dimension must be {expected_feature_dim} "
                f"for RecurrentLayer, got {actual_feature_dim} with shape "
                f"{tuple(hidden.shape)}"
            )

    @staticmethod
    def validate_candidate(
        candidate: "Tensor",
        previous_hidden: "Tensor",
        expected_feature_dim: int,
    ) -> None:
        RecurrentLayerValidator.validate_hidden(candidate, expected_feature_dim)
        if candidate.shape != previous_hidden.shape:
            raise ValueError(
                f"recurrent block must preserve hidden shape, got candidate "
                f"shape {tuple(candidate.shape)} and previous shape "
                f"{tuple(previous_hidden.shape)}"
            )

    @staticmethod
    def __validate_integer_field(field_name: str, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError(
                f"{field_name} must be int for RecurrentLayerConfig, "
                f"got {type(value).__name__}"
            )

    @staticmethod
    def __validate_stable_dimensions(input_dim: int, output_dim: int) -> None:
        if input_dim != output_dim:
            raise ValueError(
                f"input_dim and output_dim must be equal for RecurrentLayerConfig, "
                f"got input_dim={input_dim} and output_dim={output_dim}."
            )

    @staticmethod
    def __validate_block_config(block_config: ConfigBase) -> None:
        if not isinstance(block_config, ConfigBase):
            raise TypeError(
                f"block_config must be an instance of ConfigBase for "
                f"RecurrentLayerConfig, "
                f"got {type(block_config).__name__}"
            )

        field_names = {field.name for field in fields(block_config)}
        missing_fields = {"input_dim", "output_dim"} - field_names
        if missing_fields:
            missing_field_list = ", ".join(sorted(missing_fields))
            raise TypeError(
                f"block_config must declare dataclass fields input_dim and "
                f"output_dim for RecurrentLayerConfig; "
                f"{type(block_config).__name__} is missing {missing_field_list}"
            )

    @staticmethod
    def __validate_recurrent_layer_norm_position(
        recurrent_layer_norm_position: LayerNormPositionOptions | None,
    ) -> None:
        if recurrent_layer_norm_position is None:
            raise ValueError(
                "recurrent_layer_norm_position is required for "
                "RecurrentLayerConfig, received None"
            )
        if not isinstance(recurrent_layer_norm_position, LayerNormPositionOptions):
            raise TypeError(
                "recurrent_layer_norm_position must be a "
                "LayerNormPositionOptions value for RecurrentLayerConfig, "
                f"got {type(recurrent_layer_norm_position).__name__}"
            )

    @staticmethod
    def __validate_gate_config(gate_config: "GateConfig | None") -> None:
        LayerGateValidator.validate_recurrent_gate_config(
            gate_config,
            owner_name="RecurrentLayerConfig.gate_config",
        )

    @staticmethod
    def __validate_residual_connection_option(
        residual_connection_option: ResidualConnectionOptions | None,
    ) -> None:
        if residual_connection_option is None:
            raise ValueError(
                "residual_connection_option is required for RecurrentLayerConfig, "
                "received None"
            )
        if not isinstance(residual_connection_option, ResidualConnectionOptions):
            raise TypeError(
                "residual_connection_option must be a ResidualConnectionOptions "
                f"value for RecurrentLayerConfig, got "
                f"{type(residual_connection_option).__name__}"
            )

    @staticmethod
    def __validate_halting_config(
        halting_config: "HaltingConfig | None",
    ) -> None:
        if halting_config is None:
            return

        from emperor.halting.config import HaltingConfig

        if not isinstance(halting_config, HaltingConfig):
            raise TypeError(
                f"halting_config must be an instance of HaltingConfig for "
                f"RecurrentLayerConfig, got {type(halting_config).__name__}"
            )

        try:
            owner = halting_config._registry_owner()
        except NotImplementedError as exc:
            raise ValueError(
                "halting_config must be a concrete halting config for "
                "RecurrentLayerConfig"
            ) from exc

        if not hasattr(owner, "finalize_weighted_accumulation"):
            raise ValueError(
                f"halting_config {type(halting_config).__name__} builds "
                f"{owner.__name__}, which does not expose "
                "finalize_weighted_accumulation required by RecurrentLayer"
            )

    @staticmethod
    def __validate_memory_config(memory_config) -> None:
        if memory_config is None:
            return

        from emperor.memory.config import DynamicMemoryConfig

        if not isinstance(memory_config, DynamicMemoryConfig):
            raise TypeError(
                f"memory_config must be an instance of DynamicMemoryConfig for "
                f"RecurrentLayerConfig, got {type(memory_config).__name__}"
            )
