from typing import Any, Dict, Type


class BaseModelValidator:
    _FIELDS: Dict[str, Dict[str, Any]] = {}
    _TYPES: Dict[str, Type] = {}

    def __init__(self, model: Any):
        self.model = model
        self._register_custom_types()
        self.validate()

    def _register_custom_types(self) -> None:
        pass

    def validate(self) -> None:
        for name, rules in self._FIELDS.items():
            if not hasattr(self.model, name):
                raise ValueError(
                    f"Configuration Error: '{name}' is missing on "
                    f"{self.model.__class__.__name__}."
                )

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

            if expected_type is int and type(val) is not int:
                raise TypeError(
                    f"Type Error: '{name}' on {self.model.__class__.__name__} "
                    f"expected int, got {type(val).__name__} (value={val!r})."
                )

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


class LinearBaseValidator(BaseModelValidator):
    _FIELDS = {
        "input_dim": {"type": int, "validate": lambda v: v > 0 or "must be > 0"},
        "output_dim": {"type": int, "validate": lambda v: v > 0 or "must be > 0"},
        "bias_flag": {"type": bool},
        "data_monitor": {"type": "TensorMonitor", "optional": True},
        "parameter_monitor": {"type": "StatisticsMonitor", "optional": True},
    }

    def _resolve_types(self) -> None:
        from emperor.linears.core.monitors import (
            TensorMonitor,
            StatisticsMonitor,
        )

        self._TYPES = {
            "TensorMonitor": TensorMonitor,
            "StatisticsMonitor": StatisticsMonitor,
        }


class AdaptiveParameterAugmentationValidator(BaseModelValidator):
    _FIELDS = {
        "input_dim": {"type": int, "validate": lambda v: v > 0 or "must be > 0"},
        "output_dim": {"type": int, "validate": lambda v: v > 0 or "must be > 0"},
        "generator_depth": {"type": "DynamicDepthOptions"},
        "diagonal_option": {"type": "DynamicDiagonalOptions"},
        "bias_option": {"type": "DynamicBiasOptions"},
        "memory_option": {"type": "LinearMemoryOptions"},
        "memory_size_option": {"type": "LinearMemorySizeOptions"},
        "memory_position_option": {"type": "LinearMemoryPositionOptions"},
    }

    def _resolve_types(self) -> None:
        from emperor.augmentations.adaptive_parameters.options import (
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
