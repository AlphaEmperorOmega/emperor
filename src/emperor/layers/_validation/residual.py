from __future__ import annotations

from typing import TYPE_CHECKING, Never

from emperor._validation import ValidatorBase
from emperor.layers._options import ResidualConnectionOptions
from emperor.layers._validation.common import (
    _linear_layer_config_class,
    _residual_config_class,
)

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.layers._composition.attention_residual import (
        AttentionResidual,
        AttentionResidualState,
    )
    from emperor.layers._composition.residual import ResidualConnection
    from emperor.layers._config import AttentionResidualConfig, ResidualConfig
    from emperor.linears import LinearLayerConfig


class ResidualConnectionValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"residual_dim", "model_config", "attention_config"}
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
        cls._validate_attention_residual_config(cfg.option, cfg.attention_config)
        if cfg.option == ResidualConnectionOptions.ATTENTION_RESIDUAL:
            cls._validate_attention_residual_dim(cfg.residual_dim)
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
        cls._validate_attention_residual_config(
            residual_config.option,
            residual_config.attention_config,
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
    ) -> None:
        if raw_mix_coefficient is None:
            raise RuntimeError(
                f"{option} requires either raw_weight or a coefficient model."
            )

    @staticmethod
    def reject_unsupported_mixing_coefficient_option(
        option: object,
    ) -> Never:
        raise ValueError(f"Residual option does not use mixing coefficients: {option}.")

    @staticmethod
    def validate_attention_residual_available(
        attention_residual: AttentionResidual | None,
    ) -> None:
        if attention_residual is None:
            raise RuntimeError(
                "new_state is only available for ATTENTION_RESIDUAL connections."
            )

    @staticmethod
    def validate_attention_residual_state(
        residual_state: AttentionResidualState | None,
    ) -> None:
        if residual_state is None:
            raise ValueError(
                "residual_state is required for ATTENTION_RESIDUAL; create one "
                "with new_state(initial_source)."
            )

    @staticmethod
    def reject_unsupported_runtime_option(option: object) -> Never:
        raise ValueError(
            f"Unsupported residual connection option {option} for ResidualConnection."
        )

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

    @staticmethod
    def _validate_attention_residual_dim(residual_dim: int | None) -> None:
        if isinstance(residual_dim, bool) or not isinstance(residual_dim, int):
            raise TypeError(
                "ResidualConfig.residual_dim must be an int for "
                "ATTENTION_RESIDUAL, "
                f"got {type(residual_dim).__name__}."
            )
        if residual_dim <= 0:
            raise ValueError(
                "ResidualConfig.residual_dim must be greater than 0 for "
                f"ATTENTION_RESIDUAL, got {residual_dim}."
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

    @staticmethod
    def _validate_attention_residual_config(
        option: ResidualConnectionOptions,
        attention_config: AttentionResidualConfig | None,
    ) -> None:
        if attention_config is None:
            return
        if option != ResidualConnectionOptions.ATTENTION_RESIDUAL:
            raise ValueError(
                "ResidualConfig.attention_config is only supported for "
                "ATTENTION_RESIDUAL."
            )
        from emperor.layers._config import AttentionResidualConfig

        if not isinstance(attention_config, AttentionResidualConfig):
            raise TypeError(
                "ResidualConfig.attention_config must be an "
                "AttentionResidualConfig when provided, "
                f"got {type(attention_config).__name__}."
            )
