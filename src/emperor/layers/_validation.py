from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

from emperor._validation import ValidatorBase
from emperor.config import ConfigBase
from emperor.layers._options import (
    ActivationOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
    ResidualConnectionOptions,
)

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.halting import HaltingConfig
    from emperor.layers._composition.gate import LayerGate
    from emperor.layers._composition.residual import ResidualConnection
    from emperor.layers._config import (
        GateConfig,
        LayerConfig,
        LayerStackConfig,
        ResidualConfig,
    )
    from emperor.layers._layer import Layer
    from emperor.layers._recurrent import RecurrentLayer
    from emperor.layers._stack import LayerStack
    from emperor.layers._state import LayerState
    from emperor.linears import LinearLayerConfig


def _config_classes():
    from emperor.layers._config import LayerConfig, LayerStackConfig

    return LayerConfig, LayerStackConfig


def _gate_config_class():
    from emperor.layers._config import GateConfig

    return GateConfig


def _residual_config_class():
    from emperor.layers._config import ResidualConfig

    return ResidualConfig


def _linear_layer_config_class():
    from emperor.linears import LinearLayerConfig

    return LinearLayerConfig


def _gate_option_field_path(owner_name: str | None = None) -> str:
    return f"{owner_name}.option" if owner_name is not None else "gate_config.option"


_HALTING_CONFIG_FIELDS = (
    "input_dim",
    "threshold",
    "halting_dropout",
    "hidden_state_mode",
    "halting_gate_config",
)
_MEMORY_CONFIG_FIELDS = (
    "input_dim",
    "output_dim",
    "memory_position_option",
    "test_time_training_learning_rate",
    "test_time_training_num_inner_steps",
    "model_config",
)


def _matches_config_contract(config: object, field_names: tuple[str, ...]) -> bool:
    return isinstance(config, ConfigBase) and all(
        hasattr(config, field_name) for field_name in field_names
    )


class ResidualConnectionValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"residual_dim", "model_config"}
    DATA_DEPENDENT_OPTIONS = (
        ResidualConnectionOptions.WEIGHTED_RESIDUAL,
        ResidualConnectionOptions.WEIGHTED_BLEND,
    )

    @staticmethod
    def option_names() -> str:
        return ", ".join(option.name for option in ResidualConnectionOptions)

    @classmethod
    def validate(cls, model: ResidualConnection) -> None:
        cfg = model.cfg
        if not isinstance(cfg, _residual_config_class()):
            raise TypeError(
                "ResidualConnection cfg must be a ResidualConfig, "
                f"got {type(cfg).__name__}."
            )
        cls.validate_option(cfg.option, owner_name="ResidualConfig.option")
        cls.validate_required_fields(cfg)
        cls.validate_field_types(cfg)
        cls._validate_data_dependent_model_config(cfg.option, cfg.model_config)
        if cfg.model_config is None:
            return
        cls._validate_data_dependent_residual_dim(cfg.residual_dim)

    @classmethod
    def validate_residual_config(
        cls,
        residual_config: ResidualConfig | None,
        owner_name: str,
    ) -> None:
        if residual_config is None:
            return
        if not isinstance(residual_config, _residual_config_class()):
            raise TypeError(
                f"residual_config must be an instance of ResidualConfig for "
                f"{owner_name}, got {type(residual_config).__name__}"
            )
        cls.validate_option(
            residual_config.option,
            owner_name=f"{owner_name}.residual_config.option",
        )
        cls._validate_data_dependent_model_config(
            residual_config.option,
            residual_config.model_config,
        )

    @classmethod
    def validate_option(
        cls,
        option: ResidualConnectionOptions | None,
        owner_name: str = "residual_config.option",
    ) -> None:
        if option is None:
            raise ValueError(
                f"{owner_name} is required when residual_config is provided; pass "
                f"one of ResidualConnectionOptions: {cls.option_names()}. Use "
                "residual_config=None to disable the residual connection."
            )
        if not isinstance(option, ResidualConnectionOptions):
            raise TypeError(
                f"{owner_name} must be a ResidualConnectionOptions value, got "
                f"{type(option).__name__}. Valid values are: {cls.option_names()}."
            )

    @staticmethod
    def validate_raw_mix_coefficient(
        raw_mix_coefficient: Tensor | None,
        option: ResidualConnectionOptions,
    ) -> Tensor:
        if raw_mix_coefficient is None:
            raise RuntimeError(
                f"{option} requires either raw_weight or a coefficient model."
            )
        return raw_mix_coefficient

    @staticmethod
    def _validate_data_dependent_residual_dim(residual_dim: int | None) -> None:
        if isinstance(residual_dim, bool) or not isinstance(residual_dim, int):
            raise TypeError(
                "ResidualConfig.residual_dim must be an int when model_config is "
                "provided, "
                f"got {type(residual_dim).__name__}."
            )
        if residual_dim <= 0:
            raise ValueError(
                "ResidualConfig.residual_dim must be greater than 0 when model_config "
                "is provided, "
                f"got {residual_dim}."
            )

    @classmethod
    def _validate_data_dependent_model_config(
        cls,
        option: ResidualConnectionOptions,
        model_config: LinearLayerConfig | None,
    ) -> None:
        if model_config is None:
            return
        if option not in cls.DATA_DEPENDENT_OPTIONS:
            supported_options = ", ".join(
                supported_option.name for supported_option in cls.DATA_DEPENDENT_OPTIONS
            )
            raise ValueError(
                "ResidualConfig.model_config can only generate coefficients for "
                f"weighted residual modes: {supported_options}; got {option.name}."
            )
        if not isinstance(model_config, _linear_layer_config_class()):
            raise TypeError(
                "ResidualConfig.model_config must be a LinearLayerConfig when "
                "provided, "
                f"got {type(model_config).__name__}."
            )
        if model_config.bias_flag is not True:
            raise ValueError(
                "ResidualConfig.model_config.bias_flag must be True so the initial "
                "mixing coefficient can be represented."
            )


class LayerGateValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"gate_dim", "activation"}

    @staticmethod
    def option_names() -> str:
        return ", ".join(option.name for option in LayerGateOptions)

    @classmethod
    def validate(cls, model: LayerGate) -> None:
        cfg = model.cfg
        if not isinstance(cfg, _gate_config_class()):
            raise TypeError(
                f"LayerGate cfg must be a GateConfig, got {type(cfg).__name__}."
            )
        cls.validate_option(cfg.option, owner_name="GateConfig.option")
        cls.validate_required_fields(cfg)
        cls.validate_field_types(cfg)
        cls.validate_activation(cfg.activation, owner_name="GateConfig.activation")
        cls._validate_dimensions(cfg)
        cls.validate_gate_model_config(cfg.model_config)

    @classmethod
    def _validate_dimensions(cls, cfg: GateConfig) -> None:
        if cfg.gate_dim is not None:
            cls.validate_dimensions(gate_dim=cfg.gate_dim)

    @classmethod
    def validate_option(
        cls,
        option: LayerGateOptions | None,
        owner_name: str = "gate_config.option",
    ) -> None:
        if option is None:
            raise ValueError(
                f"{owner_name} is required when gate_config is provided; pass one "
                f"of LayerGateOptions: {cls.option_names()}. Use gate_config=None "
                "to disable the gate."
            )
        if not isinstance(option, LayerGateOptions):
            raise TypeError(
                f"{owner_name} must be a LayerGateOptions value, got "
                f"{type(option).__name__}. Valid values are: {cls.option_names()}."
            )

    @staticmethod
    def validate_activation(
        activation: ActivationOptions | None,
        owner_name: str = "gate_config.activation",
    ) -> None:
        if activation is None:
            return
        if not isinstance(activation, ActivationOptions):
            raise TypeError(
                f"{owner_name} must be an ActivationOptions value or None, got "
                f"{type(activation).__name__}."
            )

    @staticmethod
    def validate_gate_model(model: object | None) -> None:
        if model is None:
            raise ValueError("LayerGate requires a gate model when enabled.")

    @staticmethod
    def validate_gate_output(
        gate_output: object,
        current: Tensor,
        option: LayerGateOptions,
    ) -> None:
        import torch

        if not torch.is_tensor(gate_output):
            raise TypeError(
                "LayerGate model must return a Tensor or LayerState.hidden Tensor, "
                f"got {type(gate_output).__name__}."
            )
        if gate_output.shape != current.shape:
            option_name = (
                option.name if isinstance(option, LayerGateOptions) else option
            )
            raise ValueError(
                f"{option_name} requires gate output and current shapes to match, "
                f"got gate output shape {tuple(gate_output.shape)} and current shape "
                f"{tuple(current.shape)}."
            )

    @staticmethod
    def is_gate_config_active(gate_config: GateConfig | None) -> bool:
        return isinstance(gate_config, _gate_config_class())

    @classmethod
    def validate_gate_model_config(
        cls,
        gate_model_config: LayerStackConfig | None,
        owner_name: str | None = None,
    ) -> None:
        LayerConfig, LayerStackConfig = _config_classes()
        owner_context = f" for {owner_name}" if owner_name is not None else ""
        field_path = (
            f"{owner_name}.model_config"
            if owner_name is not None
            else "gate_config.model_config"
        )
        if gate_model_config is None:
            raise ValueError(f"{field_path} is required when gate_config is provided")
        if not isinstance(gate_model_config, LayerStackConfig):
            raise TypeError(
                f"{field_path} must be an instance of LayerStackConfig{owner_context}, "
                f"got {type(gate_model_config).__name__}"
            )
        layer_config = gate_model_config.layer_config
        if layer_config is None:
            raise ValueError(
                f"{field_path}.layer_config is required when gate_config is provided"
            )

        if type(layer_config) is not LayerConfig:
            raise TypeError(
                f"{field_path}.layer_config must be exactly LayerConfig"
                f"{owner_context}, got {type(layer_config).__name__}. "
                "Configured gate stacks must be tensor-only controllers and must "
                "not depend on caller LayerState fields."
            )
        if layer_config.gate_config is not None:
            cls.validate_layer_gate_config(
                layer_config.gate_config,
                owner_name=f"{field_path}.layer_config.gate_config",
            )
        if cls.is_gate_config_active(layer_config.gate_config):
            raise ValueError(
                f"{field_path}.layer_config.gate_config must be inactive, "
                "nested gates are not allowed"
            )
        if gate_model_config.shared_gate_config is not None:
            cls.validate_layer_gate_config(
                gate_model_config.shared_gate_config,
                owner_name=f"{field_path}.shared_gate_config",
            )
        if cls.is_gate_config_active(gate_model_config.shared_gate_config):
            raise ValueError(
                f"{field_path}.shared_gate_config must be inactive, "
                "nested gates are not allowed"
            )
        if layer_config.halting_config is not None:
            raise ValueError(
                f"{field_path}.layer_config.halting_config must be None, "
                "halting is not allowed in gates"
            )
        if gate_model_config.shared_halting_config is not None:
            raise ValueError(
                f"{field_path}.shared_halting_config must be None, "
                "halting is not allowed in gates"
            )

    @classmethod
    def validate_layer_gate_config(
        cls,
        gate_config: GateConfig | None,
        owner_name: str | None = None,
    ) -> None:
        if gate_config is None:
            return
        owner_context = f" for {owner_name}" if owner_name is not None else ""
        if not isinstance(gate_config, _gate_config_class()):
            raise TypeError(
                f"gate_config must be an instance of GateConfig{owner_context}, "
                f"got {type(gate_config).__name__}"
            )
        cls.validate_option(
            gate_config.option,
            owner_name=_gate_option_field_path(owner_name),
        )
        cls.validate_activation(
            gate_config.activation,
            owner_name=(
                f"{owner_name}.activation"
                if owner_name is not None
                else "gate_config.activation"
            ),
        )
        cls.validate_gate_model_config(
            gate_config.model_config,
            owner_name=owner_name,
        )

    @classmethod
    def validate_recurrent_gate_config(
        cls,
        gate_config: GateConfig | None,
        owner_name: str | None = None,
    ) -> None:
        cls.validate_layer_gate_config(gate_config, owner_name)

    @staticmethod
    def validate_shared_gate_config_type(shared_gate_config) -> None:
        if shared_gate_config is None:
            return

        if not isinstance(shared_gate_config, _gate_config_class()):
            raise TypeError(
                "shared_gate_config must be an instance of GateConfig "
                f"for LayerStackConfig, got {type(shared_gate_config).__name__}"
            )


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
        if halting_config is not None and not _matches_config_contract(
            halting_config,
            _HALTING_CONFIG_FIELDS,
        ):
            raise TypeError(
                "halting_config must be an instance of HaltingConfig, "
                f"got {type(halting_config).__name__}"
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
                "steps to accumulate halting probabilities across layers."
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

    @staticmethod
    def _validate_shared_memory_config_type(shared_memory_config) -> None:
        if shared_memory_config is None:
            return
        if not _matches_config_contract(shared_memory_config, _MEMORY_CONFIG_FIELDS):
            raise TypeError(
                "shared_memory_config must be an instance of DynamicMemoryConfig "
                f"for LayerStackConfig, got {type(shared_memory_config).__name__}"
            )


class RecurrentLayerValidator(ValidatorBase):
    GATE_VALIDATOR = LayerGateValidator
    RESIDUAL_VALIDATOR = ResidualConnectionValidator

    OPTIONAL_FIELDS = {
        "gate_config",
        "halting_config",
        "memory_config",
        "residual_config",
        "override_config",
    }

    @classmethod
    def validate(cls, model: RecurrentLayer) -> None:
        cfg = model.cfg
        cls.validate_required_fields(cfg)
        cls._validate_integer_field(
            "input_dim",
            cfg.input_dim,
        )
        cls._validate_integer_field(
            "output_dim",
            cfg.output_dim,
        )
        cls._validate_integer_field(
            "max_steps",
            cfg.max_steps,
        )
        cls.validate_dimensions(
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            max_steps=cfg.max_steps,
        )
        cls._validate_stable_dimensions(
            cfg.input_dim,
            cfg.output_dim,
        )
        cls._validate_recurrent_layer_norm_position(cfg.recurrent_layer_norm_position)
        cls._validate_block_config(cfg.block_config)
        cls._validate_residual_config(cfg.residual_config)
        cls._validate_gate_config(cfg.gate_config)
        cls._validate_halting_config(cfg.halting_config)
        cls._validate_memory_config(cfg.memory_config)

    @classmethod
    def validate_state(cls, state: LayerState, expected_feature_dim: int) -> None:
        from emperor.layers._state import LayerState

        if not isinstance(state, LayerState):
            raise TypeError(
                f"state must be an instance of LayerState for RecurrentLayer, "
                f"got {type(state).__name__}"
            )
        cls.validate_hidden(
            state.hidden,
            expected_feature_dim,
            "state.hidden",
        )

    @staticmethod
    def validate_hidden(
        hidden: Tensor,
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

    @classmethod
    def validate_candidate(
        cls,
        candidate: Tensor,
        previous_hidden: Tensor,
        expected_feature_dim: int,
    ) -> None:
        cls.validate_hidden(candidate, expected_feature_dim)
        if candidate.shape != previous_hidden.shape:
            raise ValueError(
                f"recurrent block must preserve hidden shape, got candidate "
                f"shape {tuple(candidate.shape)} and previous shape "
                f"{tuple(previous_hidden.shape)}"
            )

    @staticmethod
    def _validate_integer_field(field_name: str, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError(
                f"{field_name} must be int for RecurrentLayerConfig, "
                f"got {type(value).__name__}"
            )

    @staticmethod
    def _validate_stable_dimensions(input_dim: int, output_dim: int) -> None:
        if input_dim != output_dim:
            raise ValueError(
                f"input_dim and output_dim must be equal for RecurrentLayerConfig, "
                f"got input_dim={input_dim} and output_dim={output_dim}."
            )

    @staticmethod
    def _validate_block_config(block_config: ConfigBase) -> None:
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
    def _validate_recurrent_layer_norm_position(
        recurrent_layer_norm_position: LayerNormPositionOptions | None,
    ) -> None:
        if not isinstance(recurrent_layer_norm_position, LayerNormPositionOptions):
            raise TypeError(
                "recurrent_layer_norm_position must be a "
                "LayerNormPositionOptions value for RecurrentLayerConfig, "
                f"got {type(recurrent_layer_norm_position).__name__}"
            )

    @classmethod
    def _validate_gate_config(cls, gate_config: GateConfig | None) -> None:
        cls.GATE_VALIDATOR.validate_recurrent_gate_config(
            gate_config,
            owner_name="RecurrentLayerConfig.gate_config",
        )

    @classmethod
    def _validate_residual_config(
        cls,
        residual_config: ResidualConfig | None,
    ) -> None:
        cls.RESIDUAL_VALIDATOR.validate_residual_config(
            residual_config,
            owner_name="RecurrentLayerConfig",
        )

    @staticmethod
    def _validate_halting_config(
        halting_config: HaltingConfig | None,
    ) -> None:
        if halting_config is None:
            return
        if not _matches_config_contract(
            halting_config,
            _HALTING_CONFIG_FIELDS,
        ):
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
    def _validate_memory_config(memory_config) -> None:
        if memory_config is None:
            return
        if not _matches_config_contract(memory_config, _MEMORY_CONFIG_FIELDS):
            raise TypeError(
                f"memory_config must be an instance of DynamicMemoryConfig for "
                f"RecurrentLayerConfig, got {type(memory_config).__name__}"
            )
