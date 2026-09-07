from emperor.experts._state import MixtureOfExpertsLayerState
from emperor.experts._validation.layer import MixtureOfExpertsLayerValidator
from emperor.layers import Layer


class MixtureOfExpertsLayer(Layer):
    VALIDATOR = MixtureOfExpertsLayerValidator

    def _handle_model_processing(
        self,
        state: MixtureOfExpertsLayerState,
    ) -> MixtureOfExpertsLayerState:
        main_model_input = state.hidden
        output, skip_mask, loss = self.model(
            main_model_input,
            state.probabilities,
            state.indices,
            state.skip_mask,
        )
        state.skip_mask = skip_mask
        state.loss = loss if state.loss is None else state.loss + loss
        self.VALIDATOR.validate_output_rows(self, main_model_input, output)
        state.hidden = output
        return state
