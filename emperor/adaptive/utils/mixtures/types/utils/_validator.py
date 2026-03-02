from torch import Tensor

from emperor.adaptive.utils.mixtures.types.utils.enums import ClipParameterOptions

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.adaptive.utils.mixtures.types.vector import VectorMixtureBase
    from emperor.adaptive.utils.mixtures.types.generator import GeneratorMixtureBase


class _VectorMixtureValidator:
    def __init__(self, model: "VectorMixtureBase"):
        self.model = model
        self.__ensure_input_output_dim_is_equal()

    def __ensure_input_output_dim_is_equal(self) -> None:
        if self.model.input_dim != self.model.output_dim:
            raise ValueError(
                "`input_dim` and `output_dim` dimensions must be equal for the vector mixture model. Fix this later if you want to use different dimensions"
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
