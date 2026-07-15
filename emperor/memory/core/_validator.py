import math
from typing import TYPE_CHECKING

import torch

from emperor.base.layer import LayerStackConfig
from emperor.base.validator import ValidatorBase
from emperor.memory.options import MemoryPositionOptions

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.memory.config import AttentionDynamicMemoryConfig
    from emperor.memory.core.base import DynamicMemoryAbstract


class AdaptiveGeneratorValidatorBase:
    @classmethod
    def validate_generator_model(cls, generator_model) -> None:
        from torch.nn import Sequential

        from emperor.base.layer import Layer, LayerStack

        if isinstance(generator_model, Layer):
            cls._validate_generator_layer(generator_model)
            return
        if isinstance(generator_model, (Sequential, LayerStack)):
            cls._validate_generator_sequence(generator_model)
            return
        raise TypeError(
            "Expected model_config.build(...) to return a Layer, Sequential, or "
            f"LayerStack, received {type(generator_model).__name__}."
        )

    @classmethod
    def validate_test_time_training_generator_model(cls, generator_model) -> None:
        cls.validate_generator_model(generator_model)
        trainable_parameters = [
            parameter
            for parameter in generator_model.parameters()
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError(
                "Test-time-training memory requires model_config.build(...) to "
                "return a generator with at least one trainable parameter."
            )

    @classmethod
    def _validate_generator_sequence(cls, generator_sequence) -> None:
        from emperor.base.layer import Layer

        for generator_layer in generator_sequence:
            if not isinstance(generator_layer, Layer):
                raise TypeError(
                    "Expected each generator sequence item to be a Layer, "
                    f"received {type(generator_layer).__name__}."
                )
            cls._validate_generator_layer(generator_layer)

    @staticmethod
    def _validate_generator_layer(generator_layer) -> None:
        from emperor.linears.core.layers import LinearLayer

        if not isinstance(generator_layer.model, LinearLayer):
            raise TypeError(
                "Expected each generator Layer to wrap a LinearLayer, "
                f"received {type(generator_layer.model).__name__}."
            )


class DynamicMemoryValidator(AdaptiveGeneratorValidatorBase, ValidatorBase):
    OPTIONAL_FIELDS = {
        "num_memory_slots",
        "test_time_training_learning_rate",
        "test_time_training_num_inner_steps",
    }

    @classmethod
    def validate(cls, model: "DynamicMemoryAbstract") -> None:
        from emperor.memory.config import DynamicMemoryConfig

        if not hasattr(model, "cfg") or not isinstance(model.cfg, DynamicMemoryConfig):
            raise TypeError(
                "DynamicMemoryValidator.validate expected a DynamicMemoryAbstract "
                f"with a DynamicMemoryConfig cfg, received {type(model).__name__}."
            )
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls._validate_positive_int_field(model.cfg, "input_dim", model.cfg.input_dim)
        cls._validate_positive_int_field(model.cfg, "output_dim", model.cfg.output_dim)
        cls._validate_memory_position_option(model.cfg)
        cls._validate_model_config(model.cfg)
        cls._validate_test_time_training_config(model.cfg)

    @staticmethod
    def _validate_positive_int_field(cfg, field_name: str, value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"{field_name} must be a positive integer for "
                f"{cfg.__class__.__name__}, received {type(value).__name__}."
            )
        if value <= 0:
            raise ValueError(
                f"{field_name} must be greater than 0 for "
                f"{cfg.__class__.__name__}, received {value}."
            )

    @staticmethod
    def _validate_memory_position_option(cfg) -> None:
        if not isinstance(cfg.memory_position_option, MemoryPositionOptions):
            raise TypeError(
                "memory_position_option must be a MemoryPositionOptions value for "
                f"{cfg.__class__.__name__}, received "
                f"{type(cfg.memory_position_option).__name__}."
            )

    @classmethod
    def _validate_model_config(cls, cfg) -> None:
        if cfg.model_config is None:
            raise ValueError(
                f"model_config is required for {cfg.__class__.__name__}, received None."
            )
        if not isinstance(cfg.model_config, LayerStackConfig):
            raise TypeError(
                "model_config must be a LayerStackConfig for "
                f"{cfg.__class__.__name__}, received "
                f"{type(cfg.model_config).__name__}."
            )
        cls._validate_generator_config(cfg.model_config)

    @staticmethod
    def _validate_generator_config(model_config: LayerStackConfig) -> None:
        layer_config = model_config.layer_config
        restricted_fields = (
            (
                "model_config.shared_memory_config",
                model_config.shared_memory_config,
                "shared memory is not allowed in memory generators",
            ),
            (
                "model_config.shared_halting_config",
                model_config.shared_halting_config,
                "shared halting is not allowed in memory generators",
            ),
        )
        if layer_config is not None:
            restricted_fields += (
                (
                    "model_config.layer_config.gate_config",
                    layer_config.gate_config,
                    "nested gates are not allowed in memory generators",
                ),
                (
                    "model_config.layer_config.halting_config",
                    layer_config.halting_config,
                    "halting is not allowed in memory generators",
                ),
                (
                    "model_config.layer_config.memory_config",
                    layer_config.memory_config,
                    "nested memory is not allowed in memory generators",
                ),
            )
        for field_name, value, reason in restricted_fields:
            if value is not None:
                raise ValueError(f"{field_name} must be None, {reason}.")

    @staticmethod
    def _validate_test_time_training_config(cfg) -> None:
        learning_rate = cfg.test_time_training_learning_rate
        num_inner_steps = cfg.test_time_training_num_inner_steps
        learning_rate_set = learning_rate is not None
        num_inner_steps_set = num_inner_steps is not None

        if learning_rate_set != num_inner_steps_set:
            raise ValueError(
                "test_time_training_learning_rate and "
                "test_time_training_num_inner_steps must be provided together "
                f"for {cfg.__class__.__name__}."
            )
        if not learning_rate_set:
            return

        if not isinstance(learning_rate, (int, float)) or isinstance(
            learning_rate, bool
        ):
            raise TypeError(
                "test_time_training_learning_rate must be a positive float for "
                f"{cfg.__class__.__name__}, received "
                f"{type(learning_rate).__name__}."
            )
        if not math.isfinite(float(learning_rate)):
            raise ValueError(
                "test_time_training_learning_rate must be finite for "
                f"{cfg.__class__.__name__}, received {learning_rate}."
            )
        if learning_rate <= 0:
            raise ValueError(
                "test_time_training_learning_rate must be greater than 0 for "
                f"{cfg.__class__.__name__}, received {learning_rate}."
            )
        if not isinstance(num_inner_steps, int) or isinstance(num_inner_steps, bool):
            raise TypeError(
                "test_time_training_num_inner_steps must be a positive integer for "
                f"{cfg.__class__.__name__}, received "
                f"{type(num_inner_steps).__name__}."
            )
        if num_inner_steps <= 0:
            raise ValueError(
                "test_time_training_num_inner_steps must be greater than 0 for "
                f"{cfg.__class__.__name__}, received {num_inner_steps}."
            )

    @staticmethod
    def validate_attention_num_memory_slots(
        cfg: "AttentionDynamicMemoryConfig",
    ) -> None:
        num_memory_slots = cfg.num_memory_slots
        if num_memory_slots is None:
            raise ValueError(
                "num_memory_slots is required for AttentionDynamicMemoryConfig, "
                "received None."
            )
        if not isinstance(num_memory_slots, int):
            raise TypeError(
                "num_memory_slots must be int for AttentionDynamicMemoryConfig, "
                f"received {type(num_memory_slots).__name__}."
            )
        if isinstance(num_memory_slots, bool):
            raise TypeError(
                "num_memory_slots must be int for AttentionDynamicMemoryConfig, "
                "received bool."
            )
        if num_memory_slots <= 0:
            raise ValueError(
                "num_memory_slots must be greater than 0 for "
                f"AttentionDynamicMemoryConfig, received {num_memory_slots}."
            )

    @staticmethod
    def validate_forward_inputs(logits: "Tensor", expected_dim: int) -> None:
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                "logits must be a torch.Tensor for DynamicMemory forward, "
                f"received {type(logits).__name__}."
            )
        if logits.ndim < 2:
            raise ValueError(
                "logits must have rank >= 2 for DynamicMemory forward, "
                f"received rank {logits.ndim} with shape {tuple(logits.shape)}."
            )
        if logits.shape[-1] != expected_dim:
            raise ValueError(
                "logits final dimension must match the configured memory_dim "
                f"for DynamicMemory forward, expected {expected_dim}, "
                f"received {logits.shape[-1]} with shape {tuple(logits.shape)}."
            )
