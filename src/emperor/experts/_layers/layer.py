from torch import Tensor

from emperor.experts._state import MixtureOfExpertsLayerState
from emperor.layers import Layer


class MixtureOfExpertsLayer(Layer):
    def _handle_model_processing(
        self,
        main_model_input: Tensor,
        state: MixtureOfExpertsLayerState,
    ) -> Tensor:
        output, skip_mask, loss = self.model(
            main_model_input,
            state.probabilities,
            state.indices,
            state.skip_mask,
        )
        state.skip_mask = skip_mask
        state.loss = loss if state.loss is None else state.loss + loss
        return output
