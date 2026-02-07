from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.linears.utils.layers import LinearBase


class LinearBaseValidator:
    _FIELDS = {
        "input_dim": {"type": int, "validate": lambda v: v > 0 or "must be > 0"},
        "output_dim": {"type": int, "validate": lambda v: v > 0 or "must be > 0"},
        "bias_flag": {"type": bool},
        "data_monitor": {"type": "TensorMonitor", "optional": True},
        "parameter_monitor": {"type": "StatisticsMonitor", "optional": True},
    }

    def __init__(self, model: "LinearBase"):
        self.model = model
        self._resolve_monitor_types()
        self.validate()

    def _resolve_monitor_types(self) -> None:
        from Emperor.linears.utils._monitors import StatisticsMonitor, TensorMonitor

        self._TYPES = {
            "TensorMonitor": TensorMonitor,
            "StatisticsMonitor": StatisticsMonitor,
        }

    def validate(self) -> None:
        for name, rules in self._FIELDS.items():
            val = getattr(self.model, name, None)

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
