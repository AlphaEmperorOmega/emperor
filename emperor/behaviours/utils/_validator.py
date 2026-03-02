from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.behaviours.model import AdaptiveParameterBehaviour


class AdaptiveParameterBehaviourValidator:
    _FIELDS = {
        "input_dim": {
            "type": int,
            "validate": lambda v: v > 0 or "must be > 0",
        },
        "output_dim": {
            "type": int,
            "validate": lambda v: v > 0 or "must be > 0",
        },
        "generator_depth": {
            "type": "DynamicDepthOptions",
        },
        "diagonal_option": {
            "type": "DynamicDiagonalOptions",
        },
        "bias_option": {
            "type": "DynamicBiasOptions",
        },
        "memory_option": {
            "type": "LinearMemoryOptions",
        },
        "memory_size_option": {
            "type": "LinearMemorySizeOptions",
        },
        "memory_position_option": {
            "type": "LinearMemoryPositionOptions",
        },
    }

    def __init__(self, model: "AdaptiveParameterBehaviour"):
        self.model = model
        self._resolve_enum_types()
        self.validate()

    def _resolve_enum_types(self) -> None:
        from emperor.behaviours.utils.enums import (
            DynamicBiasOptions,
            DynamicDepthOptions,
            DynamicDiagonalOptions,
            LinearMemoryOptions,
            LinearMemoryPositionOptions,
            LinearMemorySizeOptions,
        )

        self._TYPES = {
            "DynamicBiasOptions": DynamicBiasOptions,
            "DynamicDepthOptions": DynamicDepthOptions,
            "DynamicDiagonalOptions": DynamicDiagonalOptions,
            "LinearMemoryOptions": LinearMemoryOptions,
            "LinearMemorySizeOptions": LinearMemorySizeOptions,
            "LinearMemoryPositionOptions": LinearMemoryPositionOptions,
        }

    def validate(self) -> None:
        for name, rules in self._FIELDS.items():
            val = getattr(self.model, name)

            if val is None:
                if rules.get("optional"):
                    continue
                raise ValueError(
                    f"Configuration Error: '{name}' is required for "
                    f"{self.model.__class__.__name__}."
                )

            expected = rules.get("type")
            expected_type = self._TYPES.get(expected, expected)
            if not isinstance(val, expected_type):
                raise TypeError(
                    f"Type Error: '{name}' on {self.model.__class__.__name__} "
                    f"expected {expected_type.__name__}, got "
                    f"{type(val).__name__} (value={val!r})."
                )

            validator = rules.get("validate")
            if validator is not None:
                result = validator(val)
                if result is not True and result is not None:
                    raise ValueError(
                        f"Configuration Error: '{name}' {result} (value={val!r})."
                    )
