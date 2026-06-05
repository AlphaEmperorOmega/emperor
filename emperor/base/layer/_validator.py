from dataclasses import fields

from emperor.base.utils import ConfigBase
from emperor.base.options import LayerNormPositionOptions
from emperor.base.validator import ValidatorBase
from emperor.base.layer.config import (
    LayerConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from emperor.base.layer.state import LayerState
    from emperor.halting.config import HaltingConfig


def _validate_gate_config(
    gate_config: "LayerStackConfig | None",
    owner_name: str | None = None,
) -> None:
    if gate_config is None:
        return
    if not isinstance(gate_config, LayerStackConfig):
        owner_context = f" for {owner_name}" if owner_name is not None else ""
        raise TypeError(
            f"gate_config must be an instance of LayerStackConfig{owner_context}, "
            f"got {type(gate_config).__name__}"
        )
    layer_config = gate_config.layer_config
    if layer_config is None:
        return

    if layer_config.gate_config is not None:
        raise ValueError(
            "gate_config.layer_config.gate_config must be None, nested gates are not allowed"
        )
    if layer_config.halting_config is not None:
        raise ValueError(
            "gate_config.layer_config.halting_config must be None, halting is not allowed in gates"
        )
    if layer_config.shared_halting_flag:
        raise ValueError(
            "gate_config.layer_config.shared_halting_flag must be False, halting is not allowed in gates"
        )


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
            cfg.input_dim, cfg.output_dim, cfg.residual_flag
        )
        LayerValidator.__validate_model_config(cfg.layer_model_config)
        LayerValidator.__validate_layer_norm_with_spatial_model(cfg)
        LayerValidator.__validate_residual_with_strided_model(cfg)
        LayerValidator.__validate_gate_config(cfg.gate_config)
        LayerValidator.__validate_halting_config(
            cfg.halting_config, cfg.shared_halting_flag
        )
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
        input_dim: int, output_dim: int, residual_flag: bool | None
    ) -> None:
        if residual_flag and input_dim != output_dim:
            raise ValueError(
                f"input_dim and output_dim must be equal when residual_flag is True, "
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
        if not cfg.residual_flag:
            return
        raise ValueError(
            f"residual_flag cannot be True when layer_model_config has "
            f"stride > 1 (received stride={stride}). Spatial reduction "
            f"breaks the residual addition shape contract."
        )

    @staticmethod
    def __validate_gate_config(gate_config: "LayerStackConfig | None") -> None:
        _validate_gate_config(gate_config)

    @staticmethod
    def __validate_halting_config(
        halting_config: "HaltingConfig | None",
        shared_halting_flag: bool | None,
    ) -> None:
        if halting_config is None and shared_halting_flag:
            raise ValueError(
                "shared_halting_flag must be False when no halting_config is provided"
            )
        if halting_config is not None:
            from emperor.halting.config import HaltingConfig

            if not isinstance(halting_config, HaltingConfig):
                raise TypeError(
                    f"halting_config must be an instance of HaltingConfig, "
                    f"got {type(halting_config).__name__}"
                )

    @staticmethod
    def __validate_memory_config(memory_config) -> None:
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
        LayerStackValidator.__validate_halting_config(cfg)

    @staticmethod
    def __validate_num_layers(num_layers: int) -> None:
        if num_layers < 1:
            raise ValueError(
                f"num_layers must be at least 1, received {num_layers}."
            )

    @staticmethod
    def __validate_layer_config(layer_config: "LayerConfig | None") -> None:
        if layer_config is None:
            raise ValueError(f"layer_config is required, received None")
        if not isinstance(layer_config, LayerConfig):
            raise TypeError(
                f"layer_config must be an instance of LayerConfig, "
                f"got {type(layer_config).__name__}"
            )

    @staticmethod
    def __validate_halting_config(cfg: "LayerStackConfig") -> None:
        if cfg.layer_config is None:
            return
        if cfg.layer_config.halting_config is not None and cfg.num_layers < 2:
            raise ValueError(
                f"num_layers must be at least 2 when halting_config is provided, "
                f"got {cfg.num_layers}. The halting mechanism requires multiple steps to accumulate "
                f"halting probabilities across layers."
            )
        if cfg.layer_config.halting_config is not None and (
            cfg.input_dim != cfg.hidden_dim or cfg.hidden_dim != cfg.output_dim
        ):
            raise ValueError(
                f"input_dim, hidden_dim, and output_dim must all be equal when halting_config is provided, "
                f"got input_dim={cfg.input_dim}, hidden_dim={cfg.hidden_dim}, output_dim={cfg.output_dim}. "
                f"Halting accumulates hidden states across steps, which requires consistent dimensions."
            )


class RecurrentLayerValidator(ValidatorBase):
    OPTIONAL_FIELDS = {
        "gate_config",
        "halting_config",
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
        RecurrentLayerValidator.__validate_block_config(cfg.block_config)
        RecurrentLayerValidator.__validate_gate_config(cfg.gate_config)
        RecurrentLayerValidator.__validate_halting_config(cfg.halting_config)

    @staticmethod
    def validate_state(state: "LayerState", expected_feature_dim: int) -> None:
        from emperor.base.layer.state import LayerState

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
    def __validate_gate_config(gate_config: "LayerStackConfig | None") -> None:
        _validate_gate_config(gate_config, owner_name="RecurrentLayerConfig")

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
