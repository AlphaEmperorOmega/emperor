from typing import TYPE_CHECKING

import torch

from emperor.base.validator import ValidatorBase
from emperor.base.layer import LayerStackConfig

if TYPE_CHECKING:
    from torch import Tensor
    from emperor.memory.config import AttentionDynamicMemoryConfig
    from emperor.memory.core.base import DynamicMemoryAbstract


class AdaptiveGeneratorValidatorBase:
    @staticmethod
    def validate_generator_model(generator_model) -> None:
        from torch.nn import Sequential
        from emperor.base.layer import Layer

        if isinstance(generator_model, Layer):
            AdaptiveGeneratorValidatorBase.validate_generator_layer(generator_model)
            return
        if isinstance(generator_model, Sequential):
            AdaptiveGeneratorValidatorBase.validate_generator_sequence(
                generator_model
            )
            return
        raise TypeError(
            "Expected model_config.build(...) to return a Layer or Sequential, "
            f"received {type(generator_model).__name__}."
        )

    @staticmethod
    def validate_generator_sequence(generator_sequence) -> None:
        from emperor.base.layer import Layer

        for generator_layer in generator_sequence:
            if not isinstance(generator_layer, Layer):
                raise TypeError(
                    "Expected each generator sequence item to be a Layer, "
                    f"received {type(generator_layer).__name__}."
                )
            AdaptiveGeneratorValidatorBase.validate_generator_layer(generator_layer)

    @staticmethod
    def validate_generator_layer(generator_layer) -> None:
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

    @staticmethod
    def validate(model: "DynamicMemoryAbstract") -> None:
        from emperor.memory.config import DynamicMemoryConfig

        if not hasattr(model, "cfg") or not isinstance(
            model.cfg, DynamicMemoryConfig
        ):
            raise TypeError(
                "DynamicMemoryValidator.validate expected a DynamicMemoryAbstract "
                f"with a DynamicMemoryConfig cfg, received {type(model).__name__}."
            )
        DynamicMemoryValidator.validate_required_fields(model.cfg)
        DynamicMemoryValidator.validate_field_types(model.cfg)
        DynamicMemoryValidator.validate_dimensions(
            input_dim=model.cfg.input_dim,
            output_dim=model.cfg.output_dim,
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
