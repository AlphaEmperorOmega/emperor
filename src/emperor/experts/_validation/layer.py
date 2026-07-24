"""Private mixture-of-experts layer validation."""

from typing import TYPE_CHECKING

from emperor._validation import ValidatorBase

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.experts._layers.layer import MixtureOfExpertsLayer
    from emperor.experts._state import MixtureOfExpertsLayerState


class MixtureOfExpertsLayerValidator(ValidatorBase):
    """Validate RowLayout contracts at a routed expert-layer boundary."""

    @classmethod
    def validate(cls, model: "MixtureOfExpertsLayer") -> None:
        from emperor.layers import Layer

        Layer.VALIDATOR.validate(model)

    @staticmethod
    def validate_layout_can_cross_routing(
        layer: "MixtureOfExpertsLayer",
        state: "MixtureOfExpertsLayerState",
        main_model_input: "Tensor",
    ) -> None:
        row_layout = state.row_layout
        if row_layout is None:
            return
        if row_layout.row_count != main_model_input.size(0):
            raise ValueError(
                f"MixtureOfExpertsLayer row_layout row_count={row_layout.row_count} "
                "does not match input row count "
                f"{main_model_input.size(0)}."
            )
        if not layer.model.compute_expert_mixture_flag and layer.model.top_k != 1:
            raise ValueError(
                "MixtureOfExpertsLayer cannot preserve RowLayout when routing "
                "returns multiple unreduced expert rows per input; enable expert "
                "mixture reduction or use top_k=1."
            )

    @staticmethod
    def validate_layout_restored(
        state: "MixtureOfExpertsLayerState",
        output: "Tensor",
    ) -> None:
        row_layout = state.row_layout
        if row_layout is None:
            return
        if output.dim() == 0 or output.size(0) != row_layout.row_count:
            raise ValueError(
                "MixtureOfExpertsLayer did not restore one output row per "
                f"RowLayout entry: expected {row_layout.row_count}, received "
                f"shape {tuple(output.shape)}."
            )
