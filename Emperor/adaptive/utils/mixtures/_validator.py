from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.adaptive.utils.mixture import AdaptiveMixtureBase


class _AdaptiveMixtureBaseValidator:
    def __init__(self, model: "AdaptiveMixtureBase"):
        self.model = model
        self.__ensure_values_are_not_none()
        self.__ensure_correct_input_types()
        self.__ensure_weighted_parameters_for_full_mixture()

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

    def __ensure_weighted_parameters_for_full_mixture(self) -> None:
        is_full_mixture = self.model.num_experts == self.model.top_k
        is_weighted_parameters = self.model.weighted_parameters_flag is True
        if is_full_mixture and is_weighted_parameters:
            raise ValueError(
                "If `full_mixture` is performed the `weighted_parameters_flag` must be True. Because the `weight_bank` or `bias_bank` needs to be broadcasted across the batch."
            )
