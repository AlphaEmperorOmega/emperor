from typing import TYPE_CHECKING

from emperor.base.validator import ValidatorBase
from emperor.base.layer import LayerStackConfig

if TYPE_CHECKING:
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
        DynamicMemoryValidator.validate_required_fields(model.cfg)
        DynamicMemoryValidator.validate_field_types(model.cfg)
        DynamicMemoryValidator.validate_dimensions(
            input_dim=model.cfg.input_dim,
            output_dim=model.cfg.output_dim,
        )
