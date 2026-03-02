from emperor.parametric.utils.mixtures.types.utils.enums import ClipParameterOptions

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.parametric.utils.mixtures.base import AdaptiveMixtureBase


class _AdaptiveMixtureBaseValidator:
    def __init__(self, model: "AdaptiveMixtureBase"):
        self.model = model
        self.__ensure_values_are_not_none()
        self.__ensure_correct_input_types()

    def __ensure_values_are_not_none(self) -> None:
        if self.model.input_dim is None:
            raise ValueError("Configuration Error: 'input_dim' is None.")
        if self.model.output_dim is None:
            raise ValueError("Configuration Error: 'output_dim' is None.")
        if self.model.top_k is None:
            raise ValueError("Configuration Error: 'top_k' is None.")
        if self.model.num_experts is None:
            raise ValueError("Configuration Error: 'num_experts' is None.")
        if self.model.weighted_parameters_flag is None:
            raise ValueError("Configuration Error: 'weighted_parameters_flag' is None.")
        if self.model.clip_parameter_option is None:
            raise ValueError("Configuration Error: 'clip_parameter_option' is None.")
        if self.model.clip_range is None:
            raise ValueError("Configuration Error: 'clip_range' is None.")

    def __ensure_correct_input_types(self) -> None:
        if not isinstance(self.model.input_dim, int):
            raise TypeError(
                f"Type Error: 'input_dim' should be int, but got {type(self.model.input_dim).__name__}."
            )
        if not isinstance(self.model.output_dim, int):
            raise TypeError(
                f"Type Error: 'output_dim' should be int, but got {type(self.model.output_dim).__name__}."
            )
        if not isinstance(self.model.top_k, int):
            raise TypeError(
                f"Type Error: 'top_k' should be int, but got {type(self.model.top_k).__name__}."
            )
        if not isinstance(self.model.num_experts, int):
            raise TypeError(
                f"Type Error: 'num_experts' should be int, but got {type(self.model.num_experts).__name__}."
            )
        if not isinstance(self.model.weighted_parameters_flag, bool):
            raise TypeError(
                f"Type Error: 'weighted_parameters_flag' should be bool, but got {type(self.model.weighted_parameters_flag).__name__}."
            )
        if not isinstance(self.model.clip_parameter_option, ClipParameterOptions):
            raise TypeError(
                f"Type Error: 'clip_parameter_option' should be ClipParameterOptions, but got {type(self.model.clip_parameter_option).__name__}."
            )
        if not isinstance(self.model.clip_range, float):
            raise TypeError(
                f"Type Error: 'clip_range' should be float, but got {type(self.model.clip_range).__name__}."
            )
