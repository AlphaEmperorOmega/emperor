from emperor.base.layer.config import LayerStackConfig
from emperor.base.enums import LastLayerBiasOptions

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.halting.config import HaltingConfig


class StickBreakingValidator:
    OPTIONAL_FIELDS = {
        "halting_dropout",
        "override_config",
    }

    @staticmethod
    def validate(cfg: "HaltingConfig") -> None:
        StickBreakingValidator.__validate_required_fields(cfg)
        StickBreakingValidator.__validate_input_dim(cfg.input_dim)
        StickBreakingValidator.__validate_threshold(cfg.threshold)
        StickBreakingValidator.__validate_halting_gate_config(cfg.halting_gate_config)
        StickBreakingValidator.__validate_halting_gate_layer_config(
            cfg.halting_gate_config.layer_config
        )

    @staticmethod
    def __validate_required_fields(cfg: "HaltingConfig") -> None:
        for field_name in cfg.__dataclass_fields__:
            if field_name in StickBreakingValidator.OPTIONAL_FIELDS:
                continue
            if getattr(cfg, field_name) is None:
                raise ValueError(f"{field_name} is required, received None")

    @staticmethod
    def __validate_input_dim(input_dim: int) -> None:
        if input_dim <= 0:
            raise ValueError(
                f"input_dim must be greater than 0, received {input_dim}"
            )

    @staticmethod
    def __validate_threshold(threshold: float) -> None:
        if threshold <= 0.0 or threshold > 1.0:
            raise ValueError(
                f"threshold must be between 0.0 (exclusive) and 1.0 (inclusive), received {threshold}"
            )

    @staticmethod
    def __validate_halting_gate_config(
        halting_gate_config: "LayerStackConfig | None",
    ) -> None:
        if not isinstance(halting_gate_config, LayerStackConfig):
            raise TypeError(
                f"halting_gate_config must be an instance of LayerStackConfig, "
                f"got {type(halting_gate_config).__name__}"
            )
        if halting_gate_config.output_dim != 2:
            raise ValueError(
                f"halting_gate_config.output_dim must be 2 (continuation and halting logits), "
                f"received {halting_gate_config.output_dim}"
            )
        if halting_gate_config.last_layer_bias_option != LastLayerBiasOptions.DISABLED:
            raise ValueError(
                f"halting_gate_config.last_layer_bias_option must be DISABLED, "
                f"received {halting_gate_config.last_layer_bias_option}"
            )

    @staticmethod
    def __validate_halting_gate_layer_config(
        layer_config: "LayerConfig | None",
    ) -> None:
        if layer_config is None:
            return
        if layer_config.gate_config is not None:
            raise ValueError(
                "halting_gate_config.layer_config.gate_config must be None, nested gates are not allowed in halting"
            )
        if layer_config.halting_config is not None:
            raise ValueError(
                "halting_gate_config.layer_config.halting_config must be None, nested halting is not allowed"
            )
        if layer_config.shared_halting_flag:
            raise ValueError(
                "halting_gate_config.layer_config.shared_halting_flag must be False, nested halting is not allowed"
            )

