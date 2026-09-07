from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING, TypeGuard

from emperor._validation import ValidatorBase, _validate_grouped_row_preservation
from emperor.config import ConfigBase
from emperor.layers._composition.gate.validation import LayerGateValidator
from emperor.layers._composition.residual.validation import (
    ResidualConnectionValidator,
)
from emperor.layers._options import (
    ActivationOptions,
    LayerNormPositionOptions,
    NormalizationOptions,
)
from emperor.layers._validation.common import (
    _HALTING_CONFIG_FIELDS,
    _MEMORY_CONFIG_FIELDS,
    _matches_config_contract,
    _validate_halting_lifecycle_owner,
    _validate_halting_owner_step_contract,
    _validate_no_grouping_with_context_controllers,
)

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.halting import HaltingConfig, HaltingInterface, HaltingStateBase
    from emperor.layers._composition.gate import LayerGate
    from emperor.layers._composition.residual.base import ResidualConnectionAbstract
    from emperor.layers._composition.residual.config import ResidualConfig
    from emperor.layers._config import GateConfig, LayerConfig
    from emperor.layers._layer.core import Layer
    from emperor.layers._layer.pipeline.halting import LayerHaltingDelegate
    from emperor.layers._layer.pipeline.memory import LayerMemoryDelegate
    from emperor.layers._layer.pipeline.normalization import (
        LayerNormalizationDelegate,
    )
    from emperor.layers._layer.pipeline.postprocessing import (
        LayerPostprocessingDelegate,
    )
    from emperor.layers._layer.pipeline.residual import LayerResidualDelegate
    from emperor.memory import MemoryInterface
    from emperor.nn import Module


def _implements_halting_interface(
    model: object,
) -> TypeGuard[HaltingInterface[HaltingStateBase]]:
    return callable(getattr(model, "update_halting_state", None)) and callable(
        getattr(model, "finalize_weighted_accumulation", None)
    )


def _implements_memory_interface(model: object) -> TypeGuard[MemoryInterface]:
    return callable(model) and hasattr(model, "memory_position_option")


class LayerHaltingDelegateValidator(ValidatorBase):
    @classmethod
    def validate(cls, model: LayerHaltingDelegate) -> None:
        cls.validate_resolved_output_dim(model.cfg.output_dim)

    @staticmethod
    def validate_resolved_output_dim(output_dim: int | None) -> None:
        if output_dim is None:
            raise ValueError("Layer halting requires a resolved output_dim.")

    @staticmethod
    def validate_built_halting_model_interface(
        model: Module | None,
    ) -> HaltingInterface[HaltingStateBase] | None:
        if model is None:
            return None
        if not _implements_halting_interface(model):
            raise TypeError(
                "halting_config must build a model implementing HaltingInterface."
            )
        return model


class LayerMemoryDelegateValidator(ValidatorBase):
    @classmethod
    def validate(cls, model: LayerMemoryDelegate) -> None:
        cls.validate_resolved_dimensions(
            model.cfg.input_dim,
            model.cfg.output_dim,
        )

    @staticmethod
    def validate_resolved_dimensions(
        input_dim: int | None,
        output_dim: int | None,
    ) -> None:
        if input_dim is None or output_dim is None:
            raise ValueError(
                "Layer memory requires resolved input and output dimensions."
            )

    @staticmethod
    def validate_built_memory_model_interface(
        model: Module | None,
    ) -> MemoryInterface | None:
        if model is None:
            return None
        if not _implements_memory_interface(model):
            raise TypeError(
                "memory_config must build a model implementing MemoryInterface."
            )
        return model


class LayerNormalizationDelegateValidator(ValidatorBase):
    @classmethod
    def validate(cls, model: LayerNormalizationDelegate) -> None:
        cls.validate_normalization_option(model.cfg.normalization)
        cls.validate_resolved_position(model.cfg.layer_norm_position)
        cls.validate_resolved_dimensions(
            model.cfg.input_dim,
            model.cfg.output_dim,
        )

    @staticmethod
    def validate_normalization_option(
        normalization: NormalizationOptions | None,
    ) -> None:
        if normalization is not None and not isinstance(
            normalization, NormalizationOptions
        ):
            raise TypeError(
                "normalization must be a NormalizationOptions value or None, "
                f"got {type(normalization).__name__}."
            )

    @staticmethod
    def validate_resolved_position(
        position: LayerNormPositionOptions | None,
    ) -> None:
        if position is None:
            raise ValueError("Layer normalization requires a resolved position.")

    @staticmethod
    def validate_resolved_dimensions(
        input_dim: int | None,
        output_dim: int | None,
    ) -> None:
        if input_dim is None or output_dim is None:
            raise ValueError(
                "Layer normalization requires resolved input and output dimensions."
            )


class ElementwiseNormalizationValidator(ValidatorBase):
    @staticmethod
    def validate_forward_input(hidden: Tensor, dimension: int) -> None:
        if hidden.ndim == 0 or hidden.shape[-1] != dimension:
            raise ValueError(
                f"Normalization requires last dimension {dimension}, "
                f"got shape {tuple(hidden.shape)}."
            )
        if not hidden.is_floating_point():
            raise TypeError("Normalization requires floating-point inputs.")


class LayerPostprocessingDelegateValidator(ValidatorBase):
    @classmethod
    def validate(cls, model: LayerPostprocessingDelegate) -> None:
        cls.validate_resolved_activation(model.cfg.activation)
        cls.validate_resolved_output_dim(model.cfg.output_dim)
        cls.validate_resolved_dropout_probability(model.cfg.dropout_probability)

    @staticmethod
    def validate_resolved_activation(
        activation: ActivationOptions | None,
    ) -> None:
        if activation is None:
            raise ValueError("Layer postprocessing requires a resolved activation.")

    @staticmethod
    def validate_resolved_output_dim(output_dim: int | None) -> None:
        if output_dim is None:
            raise ValueError("Layer postprocessing requires a resolved output_dim.")

    @staticmethod
    def validate_resolved_dropout_probability(
        dropout_probability: float | None,
    ) -> None:
        if dropout_probability is None:
            raise ValueError(
                "Layer postprocessing requires a resolved dropout_probability."
            )

    @staticmethod
    def validate_built_gate_type(gate: Module | None) -> LayerGate | None:
        if gate is None:
            return None
        from emperor.layers._composition.gate import LayerGate

        if not isinstance(gate, LayerGate):
            raise TypeError("gate_config must build a LayerGate.")
        return gate


class LayerResidualDelegateValidator(ValidatorBase):
    @classmethod
    def validate(cls, model: LayerResidualDelegate) -> None:
        cls.validate_resolved_output_dim(model.cfg.output_dim)

    @staticmethod
    def validate_resolved_output_dim(output_dim: int | None) -> None:
        if output_dim is None:
            raise ValueError(
                "Layer residual processing requires a resolved output_dim."
            )

    @staticmethod
    def validate_built_residual_connection_type(
        connection: Module | None,
    ) -> ResidualConnectionAbstract | None:
        if connection is None:
            return None
        from emperor.layers._composition.residual.base import (
            ResidualConnectionAbstract,
        )

        if not isinstance(connection, ResidualConnectionAbstract):
            raise TypeError("residual_config must build a ResidualConnectionAbstract.")
        return connection

    @staticmethod
    def validate_forward_local_state_lifecycle_requirement(
        connection: ResidualConnectionAbstract,
    ) -> None:
        from emperor.layers._composition.residual.base import (
            ResidualRuntimeRequirement,
        )

        lifecycle = connection.residual_state_lifecycle
        requires_forward_local_state = (
            ResidualRuntimeRequirement.FORWARD_LOCAL_STATE
            in connection.RUNTIME_REQUIREMENTS
        )
        if requires_forward_local_state and lifecycle is None:
            raise RuntimeError(
                f"{type(connection).__name__} declares forward-local residual "
                "state but does not provide a ResidualStateLifecycle."
            )


class LayerValidator(ValidatorBase):
    GATE_VALIDATOR = LayerGateValidator
    RESIDUAL_VALIDATOR = ResidualConnectionValidator
    NORMALIZATION_VALIDATOR = LayerNormalizationDelegateValidator

    OPTIONAL_FIELDS = {
        "gate_config",
        "halting_config",
        "memory_config",
        "layer_model_config",
        "residual_config",
        "override_config",
        "normalization",
    }

    @classmethod
    def validate(cls, model: Layer) -> None:
        cfg = model.cfg
        cls.validate_required_fields(cfg)
        cls.validate_field_types(cfg)
        cls.NORMALIZATION_VALIDATOR.validate_normalization_option(cfg.normalization)
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
        _validate_grouped_row_preservation(cfg, root=type(cfg).__name__)
        cls._validate_halting_dimensions(
            cfg.input_dim, cfg.output_dim, cfg.halting_config
        )
        _validate_no_grouping_with_context_controllers(
            cfg,
            owner_name="LayerConfig",
            controllers=(
                ("halting_config", cfg.halting_config),
                ("memory_config", cfg.memory_config),
            ),
        )

    @staticmethod
    def validate_layer_model(model: Module | None) -> Module:
        if model is None:
            raise RuntimeError("layer_model_config must build a model.")
        return model

    @staticmethod
    def _validate_dropout_probability(dropout_probability: float) -> None:
        if (
            not isfinite(dropout_probability)
            or dropout_probability < 0.0
            or dropout_probability > 1.0
        ):
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
        if input_dim != output_dim:
            raise ValueError(
                "input_dim and output_dim must be equal when "
                f"residual_config is {type(residual_config).__name__}, "
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
            f"{cfg.layer_norm_position}. Layer normalization operates over the last "
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
            f"residual_config cannot be {type(cfg.residual_config).__name__} "
            "when layer_model_config has "
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
        _validate_halting_owner_step_contract(
            halting_config,
            owner_step_limit=None,
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
