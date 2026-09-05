from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from emperor.halting._base import HaltingBase, HaltingStateBase
from emperor.halting._config import HaltingHiddenStateModeOptions
from emperor.halting._validation import StickBreakingValidator
from emperor.halting._variants._initialization import zero_gate_parameters
from emperor.layers import Layer, LayerStack, LayerStackConfig

if TYPE_CHECKING:
    from emperor.config import ModelConfig
    from emperor.halting._config import HaltingConfig


@dataclass
class StickBreakingState(HaltingStateBase):
    halt_mask: Tensor = field(
        metadata={
            "help": (
                "Boolean mask indicating which tokens have accumulated enough "
                "halt probability to stop computing"
            )
        },
    )
    log_continuation: Tensor = field(
        metadata={
            "help": (
                "Log of the remaining stick length after all breaks so far; "
                "the cumulative log probability of not yet having halted"
            )
        },
    )
    accumulated_hidden: Tensor = field(
        metadata={
            "help": (
                "Weighted sum of hidden states accumulated so far, where each "
                "step contributes proportionally to its halt probability"
            )
        },
    )
    output_hidden: Tensor = field(
        metadata={
            "help": (
                "Hidden state returned by the halting mechanism after applying "
                "its hidden-state mode and halted-position masking"
            )
        },
    )
    accumulated_halt_probabilities: Tensor = field(
        metadata={
            "help": (
                "Total halt probability spent across all steps so far; halting "
                "is triggered when this exceeds the threshold"
            )
        },
    )
    step_count: int | Tensor = field(
        metadata={
            "help": (
                "Current step index, used to compute the expected number of "
                "steps for regularisation"
            )
        },
    )
    accumulated_ponder_cost: Tensor = field(
        metadata={
            "help": (
                "Running sum of halt_prob * step across all steps; used to "
                "compute the expected computation depth"
            )
        },
    )


class StickBreaking(HaltingBase[StickBreakingState]):
    VALIDATOR = StickBreakingValidator
    supports_minimum_step_delay = True

    def __init__(
        self,
        cfg: "HaltingConfig | ModelConfig",
        overrides: "HaltingConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "halting_config", cfg)
        self.cfg: HaltingConfig = self._override_config(config, overrides)
        self.VALIDATOR.validate(self)

        self.input_dim: int = self.cfg.input_dim
        self.threshold: float = self.cfg.threshold
        self.min_steps: int = self.cfg.min_steps
        self.ponder_cost_weight: float = self.cfg.ponder_cost_weight
        self.halting_gate_config: LayerStackConfig = self.cfg.halting_gate_config
        self.hidden_state_mode: HaltingHiddenStateModeOptions = (
            self.cfg.hidden_state_mode
        )

        self.halting_gate_model = self.__build_halting_gate_model()
        self.__init_gate_weights()

    def __build_halting_gate_model(self) -> "Layer | LayerStack":
        override = type(self.halting_gate_config)(input_dim=self.input_dim)
        return self.halting_gate_config.build(overrides=override)

    def __init_gate_weights(self) -> None:
        output_layer = (
            self.halting_gate_model[-1]
            if isinstance(self.halting_gate_model, LayerStack)
            else self.halting_gate_model
        )
        zero_gate_parameters(output_layer.model)

    def update_halting_state(
        self,
        previous_state: StickBreakingState | None,
        model_hidden_state: Tensor,
    ) -> tuple[StickBreakingState, Tensor]:
        self.VALIDATOR.validate_hidden_tensor(
            model_hidden_state,
            self.input_dim,
        )
        if previous_state is None:
            state = self.__initialize_first_step(model_hidden_state)
        else:
            state = self.__advance_step(previous_state, model_hidden_state)
        return state, state.output_hidden

    def __initialize_first_step(
        self,
        model_hidden_state: Tensor,
    ) -> StickBreakingState:
        if self.min_steps > 1:
            return self.__dormant_state(model_hidden_state, step_count=0)
        return self.__init_state(
            self.__compute_gate_logits(model_hidden_state),
            model_hidden_state,
            step_count=0,
        )

    def __advance_step(
        self,
        previous_state: StickBreakingState,
        model_hidden_state: Tensor,
    ) -> StickBreakingState:
        if self.min_steps == 1:
            return self.__update_state(
                previous_state,
                self.__compute_gate_logits(model_hidden_state),
                model_hidden_state,
            )

        updated_step_count = previous_state.step_count + 1
        accumulation_start_index = self.min_steps - 1
        if isinstance(updated_step_count, Tensor):
            return self.__advance_row_steps(
                previous_state, model_hidden_state, updated_step_count
            )
        if updated_step_count < accumulation_start_index:
            return self.__dormant_state(
                model_hidden_state,
                step_count=updated_step_count,
            )

        current_log_gates = self.__compute_gate_logits(model_hidden_state)
        if previous_state.step_count < accumulation_start_index:
            return self.__init_state(
                current_log_gates,
                model_hidden_state,
                step_count=updated_step_count,
            )
        return self.__update_state(
            previous_state,
            current_log_gates,
            model_hidden_state,
        )

    def __advance_row_steps(
        self,
        previous_state: StickBreakingState,
        model_hidden_state: Tensor,
        updated_step_count: Tensor,
    ) -> StickBreakingState:
        eligible_mask = updated_step_count >= self.min_steps - 1
        dormant_state = self.__dormant_state(
            model_hidden_state, step_count=updated_step_count
        )
        if not bool(eligible_mask.any().item()):
            return dormant_state
        # A dormant row has zero spent mass and accumulated state, so its first
        # eligible update is the same equation as initial stick construction.
        updated_state = self.__update_state(
            previous_state,
            self.__compute_gate_logits(model_hidden_state),
            model_hidden_state,
        )
        for name, updated_value in vars(updated_state).items():
            row_mask = eligible_mask
            while row_mask.dim() < updated_value.dim():
                row_mask = row_mask.unsqueeze(-1)
            setattr(
                updated_state,
                name,
                torch.where(row_mask, updated_value, getattr(dormant_state, name)),
            )
        return updated_state

    @staticmethod
    def __dormant_state(
        model_hidden_state: Tensor,
        *,
        step_count: int | Tensor,
    ) -> StickBreakingState:
        leading_zeros = model_hidden_state.new_zeros(model_hidden_state.shape[:-1])
        return StickBreakingState(
            halt_mask=leading_zeros.bool(),
            log_continuation=leading_zeros,
            accumulated_hidden=torch.zeros_like(model_hidden_state),
            output_hidden=model_hidden_state,
            accumulated_halt_probabilities=leading_zeros,
            step_count=step_count,
            accumulated_ponder_cost=model_hidden_state.new_zeros(()),
        )

    def __compute_gate_logits(self, hidden_state: Tensor) -> Tensor:
        original_shape = hidden_state.shape
        flat = hidden_state.reshape(-1, original_shape[-1])
        halting_gate_state = Layer.run_model_from_hidden(self.halting_gate_model, flat)
        logits = halting_gate_state.hidden
        logits = logits.reshape(*original_shape[:-1], 2)
        if self.training:
            logits = logits + torch.randn_like(logits)
        return F.log_softmax(logits, dim=-1)

    def __init_state(
        self,
        log_softmax_gates: Tensor,
        model_hidden_state: Tensor,
        *,
        step_count: int,
    ) -> StickBreakingState:
        log_continuation, log_halting = torch.unbind(log_softmax_gates, dim=-1)
        halting_probability = torch.exp(log_halting)
        halt_mask = halting_probability >= self.threshold
        weighted_hidden = halting_probability.unsqueeze(-1) * model_hidden_state
        output_hidden = self.__output_hidden_for_mode(
            weighted_hidden, model_hidden_state
        )
        return StickBreakingState(
            halt_mask=halt_mask,
            log_continuation=log_continuation,
            accumulated_hidden=weighted_hidden,
            output_hidden=output_hidden,
            accumulated_halt_probabilities=halting_probability,
            step_count=step_count,
            accumulated_ponder_cost=model_hidden_state.new_zeros(()),
        )

    def __update_state(
        self,
        previous_state: StickBreakingState,
        log_softmax_gates: Tensor,
        model_hidden_state: Tensor,
    ) -> StickBreakingState:
        updated_step_count = previous_state.step_count + 1
        log_continuation, halting_probability = self.__compute_step_halting_probability(
            previous_state, log_softmax_gates
        )
        accumulated_halting_probability = (
            previous_state.accumulated_halt_probabilities + halting_probability
        )
        halt_mask = accumulated_halting_probability >= self.threshold
        weighted_hidden = halting_probability.unsqueeze(-1) * model_hidden_state
        updated_accumulated_hidden = previous_state.accumulated_hidden + weighted_hidden
        output_hidden = self.__output_hidden_for_mode(
            updated_accumulated_hidden, model_hidden_state
        )
        output_hidden = self.__preserve_previous_output_hidden(
            previous_state.output_hidden,
            output_hidden,
            previous_state.halt_mask,
        )
        optional_step_index = updated_step_count - (self.min_steps - 1)
        step_contribution = halting_probability * optional_step_index
        updated_accumulated_ponder_cost = (
            previous_state.accumulated_ponder_cost + step_contribution
        )
        return StickBreakingState(
            halt_mask=halt_mask,
            log_continuation=log_continuation,
            accumulated_hidden=updated_accumulated_hidden,
            output_hidden=output_hidden,
            accumulated_halt_probabilities=accumulated_halting_probability,
            step_count=updated_step_count,
            accumulated_ponder_cost=updated_accumulated_ponder_cost,
        )

    def __output_hidden_for_mode(
        self,
        accumulated_hidden: Tensor,
        model_hidden_state: Tensor,
    ) -> Tensor:
        if self.hidden_state_mode == HaltingHiddenStateModeOptions.ACCUMULATED:
            return accumulated_hidden
        return model_hidden_state

    def __preserve_previous_output_hidden(
        self,
        previous_output_hidden: Tensor,
        candidate_output_hidden: Tensor,
        halt_mask: Tensor,
    ) -> Tensor:
        while halt_mask.dim() < candidate_output_hidden.dim():
            halt_mask = halt_mask.unsqueeze(-1)
        return torch.where(halt_mask, previous_output_hidden, candidate_output_hidden)

    def __compute_step_halting_probability(
        self,
        previous_state: StickBreakingState,
        current_log_gates: Tensor,
    ) -> tuple[Tensor, Tensor]:
        previous_step_log_continuation = previous_state.log_continuation.unsqueeze(-1)
        current_log_halting = previous_step_log_continuation + current_log_gates
        log_continuation, log_halting = torch.unbind(current_log_halting, dim=-1)
        halting_probability = torch.exp(log_halting)
        previous_halt_mask = previous_state.halt_mask
        halting_probability = halting_probability.masked_fill(previous_halt_mask, 0.0)
        return log_continuation, halting_probability

    def finalize_weighted_accumulation(
        self,
        state: StickBreakingState,
        current_hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self.VALIDATOR.validate_hidden_tensor(
            current_hidden,
            self.input_dim,
            "current_hidden",
        )
        self.VALIDATOR.validate_tensor_shape(
            current_hidden,
            state.accumulated_hidden.shape,
            "current_hidden",
        )
        remaining_probabilities = 1 - state.accumulated_halt_probabilities
        remaining_probabilities = remaining_probabilities.masked_fill(
            state.halt_mask, 0.0
        )
        weighted_remaining_hidden = (
            remaining_probabilities.unsqueeze(-1) * current_hidden
        )
        soft_halted_hidden = state.accumulated_hidden + weighted_remaining_hidden
        next_optional_step_index = state.step_count - (self.min_steps - 1) + 1
        if isinstance(next_optional_step_index, Tensor):
            next_optional_step_index = next_optional_step_index.clamp_min(0)
        else:
            next_optional_step_index = max(0, next_optional_step_index)
        remaining_step_contribution = remaining_probabilities * next_optional_step_index
        raw_ponder_loss = state.accumulated_ponder_cost + remaining_step_contribution
        return soft_halted_hidden, self._apply_ponder_cost_weight(
            state,
            raw_ponder_loss,
        )
