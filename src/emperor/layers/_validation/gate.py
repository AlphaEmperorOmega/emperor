from __future__ import annotations

from typing import TYPE_CHECKING

from emperor._validation import ValidatorBase
from emperor.layers._options import ActivationOptions, LayerGateOptions
from emperor.layers._validation.common import (
    _config_classes,
    _gate_config_class,
    _gate_option_field_path,
)

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.layers._composition.gate import LayerGate
    from emperor.layers._config import GateConfig, LayerStackConfig


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
