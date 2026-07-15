from typing import TYPE_CHECKING

from emperor.base.layer.config import LayerConfig, LayerStackConfig
from emperor.base.options import LastLayerBiasOptions

if TYPE_CHECKING:
    from emperor.halting.config import HaltingConfig
    from emperor.halting.core.base import HaltingBase


class StickBreakingValidator:
    OPTIONAL_FIELDS = {
        "halting_dropout",
        "override_config",
    }

    @classmethod
    def validate(cls, model: "HaltingBase") -> None:
        cfg = model.cfg
        cls._validate_required_fields(cfg)
        cls._validate_input_dim(cfg.input_dim)
        cls._validate_threshold(cfg.threshold)
        cls._validate_halting_gate_config(cfg.halting_gate_config)
        cls._validate_halting_gate_layer_config(
            cfg.halting_gate_config.layer_config
        )

    @classmethod
    def _validate_required_fields(cls, cfg: "HaltingConfig") -> None:
        for field_name in cfg.__dataclass_fields__:
            if field_name in cls.OPTIONAL_FIELDS:
                continue
            if getattr(cfg, field_name) is None:
                raise ValueError(f"{field_name} is required, received None")

    @staticmethod
    def _validate_input_dim(input_dim: int) -> None:
        if input_dim <= 0:
            raise ValueError(
                f"input_dim must be greater than 0, received {input_dim}"
            )

    @staticmethod
    def _validate_threshold(threshold: float) -> None:
        if threshold <= 0.0 or threshold > 1.0:
            raise ValueError(
                "threshold must be between 0.0 (exclusive) and 1.0 "
                f"(inclusive), received {threshold}"
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
