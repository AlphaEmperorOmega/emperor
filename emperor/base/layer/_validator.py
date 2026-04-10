from emperor.base.utils import ConfigBase
from emperor.base.layer.config import LayerConfig, LayerStackConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.halting.config import HaltingConfig


class LayerValidator:
    OPTIONAL_FIELDS = {
        "gate_config",
        "halting_config",
        "model_config",
        "override_config",
    }

    @staticmethod
    def validate(cfg: LayerConfig) -> None:
        LayerValidator.__validate_required_fields(cfg)
        LayerValidator.__validate_dimensions(cfg.input_dim, cfg.output_dim)
        LayerValidator.__validate_dropout_probability(cfg.dropout_probability)
        LayerValidator.__validate_model_config(cfg.model_config)
        LayerValidator.__validate_gate_config(cfg.gate_config)
        LayerValidator.__validate_halting_config(
            cfg.halting_config, cfg.shared_halting_flag
        )
        LayerValidator.__validate_halting_dimensions(
            cfg.input_dim, cfg.output_dim, cfg.halting_config
        )

    @staticmethod
    def __validate_required_fields(cfg: LayerConfig) -> None:
        for field_name in cfg.__dataclass_fields__:
            if field_name in LayerValidator.OPTIONAL_FIELDS:
                continue
            if getattr(cfg, field_name) is None:
                raise ValueError(f"{field_name} is required, received None")

    @staticmethod
    def __validate_dimensions(input_dim: int, output_dim: int) -> None:
        if input_dim <= 0:
            raise ValueError(f"input_dim must be greater than 0, received {input_dim}")
        if output_dim <= 0:
            raise ValueError(
                f"output_dim must be greater than 0, received {output_dim}"
            )

    @staticmethod
    def __validate_dropout_probability(dropout_probability: float) -> None:
        if dropout_probability < 0.0 or dropout_probability > 1.0:
            raise ValueError(
                f"dropout_probability must be between 0.0 and 1.0, received {dropout_probability}"
            )

    @staticmethod
    def __validate_model_config(model_config: ConfigBase | None) -> None:
        if model_config is None:
            raise ValueError(
                "model_config is required, Layer needs it to build the model"
            )
        if not isinstance(model_config, ConfigBase):
            raise TypeError(
                f"model_config must be an instance of ConfigBase, "
                f"got {type(model_config).__name__}"
            )

    @staticmethod
    def __validate_gate_config(gate_config: "LayerStackConfig | None") -> None:
        if gate_config is None:
            return
        if not isinstance(gate_config, LayerStackConfig):
            raise TypeError(
                f"gate_config must be an instance of LayerStackConfig, "
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


class LayerStackValidator:
    @staticmethod
    def validate(cfg: "LayerStackConfig") -> None:
        LayerStackValidator.__validate_required_fields(cfg)
        LayerStackValidator.__validate_dimensions(cfg)
        LayerStackValidator.__validate_halting_config(cfg)

    @staticmethod
    def __validate_required_fields(cfg: "LayerStackConfig") -> None:
        if cfg.input_dim is None:
            raise ValueError(f"input_dim is required, received {cfg.input_dim}")
        if cfg.hidden_dim is None:
            raise ValueError(f"hidden_dim is required, received {cfg.hidden_dim}")
        if cfg.output_dim is None:
            raise ValueError(f"output_dim is required, received {cfg.output_dim}")
        if cfg.num_layers is None:
            raise ValueError(f"num_layers is required, received {cfg.num_layers}")
        if cfg.last_layer_bias_option is None:
            raise ValueError(
                f"last_layer_bias_option is required, received {cfg.last_layer_bias_option}"
            )
        if cfg.apply_output_pipeline_flag is None:
            raise ValueError(
                f"apply_output_pipeline_flag is required, received {cfg.apply_output_pipeline_flag}"
            )
        if cfg.layer_config is None:
            raise ValueError(f"layer_config is required, received {cfg.layer_config}")
        if not isinstance(cfg.layer_config, LayerConfig):
            raise TypeError(
                f"layer_config must be an instance of LayerConfig, "
                f"got {type(cfg.layer_config).__name__}"
            )

    @staticmethod
    def __validate_dimensions(cfg: "LayerStackConfig") -> None:
        if cfg.input_dim is not None and cfg.input_dim <= 0:
            raise ValueError(
                f"input_dim must be greater than 0, received {cfg.input_dim}"
            )
        if cfg.hidden_dim is not None and cfg.hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be greater than 0, received {cfg.hidden_dim}"
            )
        if cfg.output_dim is not None and cfg.output_dim <= 0:
            raise ValueError(
                f"output_dim must be greater than 0, received {cfg.output_dim}"
            )
        if cfg.num_layers is not None and cfg.num_layers <= 0:
            raise ValueError(
                f"num_layers must be greater than 0, received {cfg.num_layers}"
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
