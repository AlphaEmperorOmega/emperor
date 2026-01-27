from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.linears.utils.layers import LinearBase


class LinearBaseValidator:
    def __init__(self, model: "LinearBase"):
        self.model = model
        self.__ensure_required_values_present()
        self.__ensure_types()
        self.__ensure_positive_dimensions()

    def __ensure_required_values_present(self) -> None:
        required_attrs = [
            "input_dim",
            "output_dim",
            "bias_flag",
        ]
        for name in required_attrs:
            if getattr(self.model, name, None) is None:
                raise ValueError(
                    f"Configuration Error: '{name}' is None for {self.model.__class__.__name__}."
                )

    def __ensure_types(self) -> None:
        from Emperor.linears.utils.monitors import DataMonitor, ParameterMonitor

        if not isinstance(self.model.input_dim, int):
            raise TypeError(
                f"Type Error: 'input_dim' should be int, got {type(self.model.input_dim).__name__}."
            )
        if not isinstance(self.model.output_dim, int):
            raise TypeError(
                f"Type Error: 'output_dim' should be int, got {type(self.model.output_dim).__name__}."
            )
        if not isinstance(self.model.bias_flag, bool):
            raise TypeError(
                f"Type Error: 'bias_flag' should be bool, got {type(self.model.bias_flag).__name__}."
            )
        if self.model.data_monitor is not None and not (
            isinstance(self.model.data_monitor, type)
            and issubclass(self.model.data_monitor, DataMonitor)
        ):
            raise TypeError(
                "Type Error: 'data_monitor' should be a DataMonitor subclass or None."
            )
        if self.model.parameter_monitor is not None and not (
            isinstance(self.model.parameter_monitor, type)
            and issubclass(self.model.parameter_monitor, ParameterMonitor)
        ):
            raise TypeError(
                "Type Error: 'parameter_monitor' should be a ParameterMonitor subclass or None."
            )

    def __ensure_positive_dimensions(self) -> None:
        if self.model.input_dim <= 0:
            raise ValueError("Configuration Error: 'input_dim' must be > 0")
        if self.model.output_dim <= 0:
            raise ValueError("Configuration Error: 'output_dim' must be > 0")
