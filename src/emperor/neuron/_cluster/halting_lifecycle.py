"""Mask-aware adapter for Neuron's finished halting lifecycle."""

import copy
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

if TYPE_CHECKING:
    from emperor.halting import HaltingBase, HaltingStateBase


class _NeuronHaltingLifecycle:
    """Advance only computed route rows through a shared halting strategy."""

    @classmethod
    def update(
        cls,
        halting_model: "HaltingBase | None",
        previous_state: "HaltingStateBase | None",
        current_hidden: Tensor,
        weighted_candidate: Tensor,
        update_mask: Tensor,
    ) -> "HaltingStateBase | None":
        if halting_model is None or not bool(update_mask.any().item()):
            return previous_state

        halting_input = torch.where(
            update_mask.unsqueeze(-1),
            weighted_candidate,
            current_hidden,
        )
        prepared_previous_state = cls.__copy_previous_state(previous_state)
        halting_state, _ = halting_model.update_halting_state(
            prepared_previous_state,
            halting_input,
        )
        if prepared_previous_state is not None:
            halting_state = cls.__merge_state_row_updates(
                prepared_previous_state,
                halting_state,
                update_mask,
            )
        cls.__record_route_metadata(
            halting_state,
            prepared_previous_state,
            halting_input,
            update_mask,
        )
        halting_usage_tracker = getattr(halting_model, "_usage_tracker", None)
        if halting_usage_tracker is not None:
            halting_usage_tracker.replace_last_step(halting_state)
        return halting_state

    @staticmethod
    def __copy_previous_state(
        previous_state: "HaltingStateBase | None",
    ) -> "HaltingStateBase | None":
        if previous_state is None:
            return None
        return copy.copy(previous_state)

    @classmethod
    def __merge_state_row_updates(
        cls,
        previous_state: "HaltingStateBase",
        updated_state: "HaltingStateBase",
        update_mask: Tensor,
    ) -> "HaltingStateBase":
        for attribute_name, previous_value in vars(previous_state).items():
            if not hasattr(updated_state, attribute_name):
                setattr(updated_state, attribute_name, previous_value)

        for attribute_name, updated_value in vars(updated_state).items():
            previous_value = getattr(previous_state, attribute_name, None)
            if not cls.__is_row_aligned_tensor(updated_value, update_mask):
                continue
            if not isinstance(previous_value, Tensor):
                raise ValueError(
                    "Halting row-aligned state must retain its tensor schema and "
                    "shape across sparse Neuron updates."
                )
            if previous_value.dim() == 0:
                previous_value = previous_value.expand_as(updated_value)
            if previous_value.shape != updated_value.shape:
                raise ValueError(
                    "Halting row-aligned state must retain its tensor schema and "
                    "shape across sparse Neuron updates."
                )
            expanded_update_mask = update_mask
            while expanded_update_mask.dim() < updated_value.dim():
                expanded_update_mask = expanded_update_mask.unsqueeze(-1)
            setattr(
                updated_state,
                attribute_name,
                torch.where(
                    expanded_update_mask,
                    updated_value,
                    previous_value,
                ),
            )
        return updated_state

    @staticmethod
    def __is_row_aligned_tensor(value: Any, update_mask: Tensor) -> bool:
        return (
            isinstance(value, Tensor)
            and value.dim() >= 1
            and value.shape[0] == update_mask.shape[0]
        )

    @classmethod
    def __record_route_metadata(
        cls,
        halting_state: "HaltingStateBase",
        previous_state: "HaltingStateBase | None",
        halting_input: Tensor,
        update_mask: Tensor,
    ) -> None:
        previous_advanced_mask = (
            previous_state.advanced_mask
            if previous_state is not None and hasattr(previous_state, "advanced_mask")
            else torch.zeros_like(update_mask)
        )
        advanced_mask = previous_advanced_mask | update_mask
        halting_state.valid_mask = advanced_mask
        halting_state.advanced_mask = advanced_mask

        previous_raw_hidden = (
            previous_state.raw_hidden
            if previous_state is not None and hasattr(previous_state, "raw_hidden")
            else halting_input
        )
        halting_state.raw_hidden = torch.where(
            update_mask.unsqueeze(-1),
            halting_input,
            previous_raw_hidden,
        )

        if hasattr(halting_state, "log_continuation"):
            halting_state.continuation_probability = (
                halting_state.log_continuation.exp()
            )
        elif not hasattr(halting_state, "continuation_probability"):
            halting_state.continuation_probability = update_mask.to(halting_input.dtype)

        if hasattr(halting_state, "step_count"):
            step_count = halting_state.step_count
            if not isinstance(step_count, Tensor) or step_count.dim() == 0:
                step_count = torch.full(
                    update_mask.shape,
                    int(step_count),
                    dtype=torch.long,
                    device=update_mask.device,
                )
            halting_state.step_count = step_count
            halting_state.step_indices = step_count
        else:
            previous_step_indices = (
                previous_state.step_indices
                if previous_state is not None
                and hasattr(previous_state, "step_indices")
                else torch.zeros_like(update_mask, dtype=torch.long)
            )
            halting_state.step_indices = (
                previous_step_indices
                if previous_state is None
                else previous_step_indices + update_mask.to(torch.long)
            )

        accumulated_ponder_cost = getattr(
            halting_state,
            "accumulated_ponder_cost",
            None,
        )
        if (
            isinstance(accumulated_ponder_cost, Tensor)
            and accumulated_ponder_cost.dim() == 0
        ):
            halting_state.accumulated_ponder_cost = accumulated_ponder_cost.expand(
                update_mask.shape
            ).clone()
        halt_mask = cls.halt_mask(halting_state)
        halting_state.stop_requested = bool(
            halt_mask is not None and (halt_mask | ~advanced_mask).all().item()
        )
        halting_state.finalized = False

    @classmethod
    def finalize(
        cls,
        halting_model: "HaltingBase",
        halting_state: "HaltingStateBase",
        current_hidden: Tensor,
        beam_scores: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        finalized_hidden, ponder_loss = halting_model.finalize_weighted_accumulation(
            halting_state,
            current_hidden,
        )
        reduced_ponder_loss = cls.__reduce_ponder_loss(
            ponder_loss,
            beam_scores,
            getattr(halting_state, "advanced_mask", None),
        )
        halting_usage_tracker = getattr(halting_model, "_usage_tracker", None)
        if halting_usage_tracker is not None:
            halting_usage_tracker.record_final(reduced_ponder_loss, halting_state)
        return finalized_hidden, reduced_ponder_loss

    @staticmethod
    def __reduce_ponder_loss(
        ponder_loss: Tensor,
        beam_scores: Tensor | None,
        advanced_mask: Tensor | None,
    ) -> Tensor:
        if ponder_loss.dim() == 0:
            return ponder_loss
        if advanced_mask is None or ponder_loss.shape[0] != advanced_mask.shape[0]:
            raise ValueError(
                "Vector Neuron ponder loss requires an aligned advanced mask."
            )
        valid_slot_mask = advanced_mask.bool()
        if beam_scores is not None:
            if ponder_loss.shape[0] != beam_scores.shape[0]:
                raise ValueError(
                    "Vector Neuron ponder loss requires aligned beam scores."
                )
            finite_beam_mask = torch.isfinite(beam_scores)
            valid_slot_mask = valid_slot_mask & finite_beam_mask

        while valid_slot_mask.dim() < ponder_loss.dim():
            valid_slot_mask = valid_slot_mask.unsqueeze(-1)
        valid_slot_mask = valid_slot_mask.expand_as(ponder_loss)
        valid_slot_count = valid_slot_mask.sum().clamp_min(1).to(ponder_loss.dtype)
        valid_ponder_loss = ponder_loss.masked_fill(~valid_slot_mask, 0.0)
        return valid_ponder_loss.sum() / valid_slot_count

    @staticmethod
    def halt_mask(halting_state: "HaltingStateBase | None") -> Tensor | None:
        if halting_state is None:
            return None
        halt_mask = halting_state.halt_mask.bool()
        if halt_mask.dim() == 1:
            return halt_mask
        return halt_mask.reshape(halt_mask.shape[0], -1).all(dim=1)

    @classmethod
    def halt_mask_tensor(
        cls,
        halting_state: "HaltingStateBase | None",
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        halt_mask = cls.halt_mask(halting_state)
        if halt_mask is None:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        return halt_mask

    @staticmethod
    def gather_state_rows(
        halting_state: "HaltingStateBase | None",
        row_indices: Tensor,
    ) -> "HaltingStateBase | None":
        if halting_state is None:
            return None
        gathered_state = copy.copy(halting_state)
        selected_row_count = row_indices.shape[0]
        for attribute_name, attribute_value in vars(halting_state).items():
            if (
                isinstance(attribute_value, Tensor)
                and attribute_value.dim() >= 1
                and attribute_value.shape[0] == selected_row_count
            ):
                setattr(
                    gathered_state,
                    attribute_name,
                    attribute_value.index_select(0, row_indices),
                )
        return gathered_state
