from typing import TYPE_CHECKING

from torch import Tensor

from Emperor.adaptive.utils.enums import ClipParameterOptions


if TYPE_CHECKING:
    from Emperor.adaptive.utils.mixtures.generator import GeneratorMixtureBase
    from Emperor.adaptive.utils.mixture import AdaptiveMixtureBase


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


class _GeneratorMixtureValidator:
    def __init__(self, model: "GeneratorMixtureBase"):
        self.model = model
        self.__ensure_weighted_parameters_for_full_mixture()

    def ensure_mixture_weighted_flag_is_false(self) -> None:
        is_expert_weight_flag = (
            self.model.input_vector_generator.weighted_parameters_flag
        )

        if self.model.weighted_parameters_flag and is_expert_weight_flag:
            raise ValueError(
                "Both 'weighted_parameters_flag' and 'input_vector_generator.weighted_parameters_flag' are set to True. Only one can be `True` when computing weighted parameters"
            )

    def ensure_probabilities_exist_for_weighted_flag(
        self, probabilities: Tensor | None
    ) -> None:
        if self.model.weighted_parameters_flag and probabilities is None:
            raise ValueError(
                "'weighted_parameters_flag' is set to True, but no probabilities are provided. Probabilities must be explicitly provided to proceed."
            )

    def ensure_input_batch_is_2D_tensor(self, input_batch: Tensor) -> None:
        if input_batch.dim() != 2:
            raise ValueError(
                f"Input batch must be a 2D tensor, but got {input_batch.ndim}D."
            )

    def ensure_clip_range(self, input_batch: Tensor) -> None:
        is_valid_option = self.model.clip_parameter_option != ClipParameterOptions.NONE
        is_valid_range = self.model.clip_range >= 0.0
        if is_valid_option and is_valid_range:
            raise ValueError(
                f"Invalid clip range: {self.model.clip_range}. It must be non-negative."
            )

    def __ensure_weighted_parameters_for_full_mixture(self) -> None:
        is_full_mixture = self.model.num_experts == self.model.top_k
        is_weighted = self.model.weighted_parameters_flag is True
        is_expert_weighted = (
            self.model.input_vector_generator.weighted_parameters_flag is True
        )
        if is_full_mixture and not (is_weighted or is_expert_weighted):
            raise ValueError(
                "Configuration Error: When performing `full_mixture`, at least one of `weighted_parameters_flag` or `input_vector_generator.weighted_parameters_flag` must be True, as you cannot have auxiliary losses for full mixture."
            )
