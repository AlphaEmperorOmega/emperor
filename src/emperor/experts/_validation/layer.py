"""Private mixture-of-experts layer validation."""

from typing import TYPE_CHECKING

from emperor.layers._layer.validation import LayerValidator

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.experts._layers.layer import MixtureOfExpertsLayer


class MixtureOfExpertsLayerValidator(LayerValidator):
    """Validate the row count promised by the configured routing mode."""

    @staticmethod
    def validate_output_rows(
        layer: "MixtureOfExpertsLayer",
        main_model_input: "Tensor",
        output: "Tensor",
    ) -> None:
        if not layer.model.compute_expert_mixture_flag and layer.model.top_k != 1:
            return
        expected_rows = main_model_input.size(0)
        if output.dim() == 0 or output.size(0) != expected_rows:
            raise ValueError(
                "MixtureOfExpertsLayer did not restore one output row per input: "
                f"expected {expected_rows}, received shape {tuple(output.shape)}."
            )
