import torch

from dataclasses import dataclass, fields
from torch import Tensor
from torch.nn import Sequential
from emperor.base.utils import ConfigBase, Module
from emperor.base.layer.config import (
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.base.layer.layer import Layer
from emperor.base.layer.state import LayerState
from emperor.base.layer._validator import RecurrentLayerValidator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.halting.config import HaltingConfig
    from emperor.halting.utils.options.base import HaltingBase, HaltingStateBase


@dataclass
class _RecurrentRunState:
    hidden: Tensor
    loss: Tensor | None
    halting_state: "HaltingStateBase | None" = None


class RecurrentLayer(Module):
    def __init__(
        self,
        cfg: RecurrentLayerConfig,
        overrides: RecurrentLayerConfig | None = None,
    ):
        super().__init__()
        self.cfg: RecurrentLayerConfig = self._override_config(cfg, overrides)
        RecurrentLayerValidator.validate(self.cfg)

        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.max_steps: int = self.cfg.max_steps
        self.block_config: ConfigBase | None = self.cfg.block_config
        self.gate_config: LayerStackConfig | None = self.cfg.gate_config
        self.halting_config: "HaltingConfig | None" = self.cfg.halting_config

        self.block_model = self.__build_block_model()
        self.gate_model = self.__build_gate_model()
        self.halting_model = self.__build_halting_model()

    def __build_block_model(self) -> "Layer | Sequential | Module":
        if self.block_config is None:
            raise ValueError("block_config is required for RecurrentLayer")
        return self.block_config.build(overrides=self.__resolve_block_overrides())

    def __resolve_block_overrides(
        self,
    ) -> ConfigBase:
        if self.block_config is None:
            raise ValueError("block_config is required for RecurrentLayer")

        block_fields = {field.name for field in fields(self.block_config)}
        override_kwargs = {
            field_name: self.output_dim
            for field_name in ("input_dim", "output_dim")
            if field_name in block_fields
        }
        return type(self.block_config)(**override_kwargs)

    def __build_gate_model(self) -> "Layer | Sequential | None":
        return self.__build_from_config(
            self.gate_config,
            input_dim=self.output_dim,
            output_dim=self.output_dim,
        )

    def __build_halting_model(self) -> "HaltingBase | None":
        return self.__build_from_config(
            self.halting_config,
            input_dim=self.output_dim,
        )

    def __build_from_config(
        self,
        config: "ConfigBase | None",
        **kwargs,
    ) -> "Module | None":
        if config is None:
            return None
        return config.build(overrides=type(config)(**kwargs))

    def forward(self, state: LayerState) -> LayerState:
        RecurrentLayerValidator.validate_state(state, self.input_dim)

        original_halting_state = state.halting_state
        run_state = self.__run_recurrent_steps(
            _RecurrentRunState(
                hidden=state.hidden,
                loss=state.loss,
            )
        )
        run_state = self.__maybe_finalize_recurrent_halting(run_state)
        return self.__restore_output_state(
            state,
            run_state,
            original_halting_state,
        )

    def __run_recurrent_steps(
        self,
        run_state: _RecurrentRunState,
    ) -> _RecurrentRunState:
        for _ in range(self.max_steps):
            previous_hidden = run_state.hidden
            already_halted_mask = self.__get_halt_mask(run_state.halting_state)
            candidate_state = self.__run_recurrent_block_step(run_state)
            run_state = self.__run_recurrent_controllers(
                candidate_state,
                previous_hidden,
                already_halted_mask,
            )

            if self.__all_items_halted(run_state.halting_state):
                break

        return run_state

    def __run_recurrent_block_step(
        self,
        run_state: _RecurrentRunState,
    ) -> _RecurrentRunState:
        block_state = LayerState(
            hidden=run_state.hidden,
            loss=run_state.loss,
            halting_state=None,
        )
        block_output = self.block_model(block_state)
        candidate = block_output.hidden
        RecurrentLayerValidator.validate_candidate(
            candidate,
            run_state.hidden,
            self.output_dim,
        )
        return _RecurrentRunState(
            hidden=candidate,
            loss=block_output.loss,
            halting_state=run_state.halting_state,
        )

    def __run_recurrent_controllers(
        self,
        candidate_state: _RecurrentRunState,
        previous_hidden: Tensor,
        already_halted_mask: Tensor | None,
    ) -> _RecurrentRunState:
        hidden = self.__maybe_apply_recurrent_gate(
            candidate_state.hidden,
            previous_hidden,
        )
        hidden = self.__preserve_already_halted_hidden(
            previous_hidden,
            hidden,
            already_halted_mask,
        )
        halting_state, hidden = self.__maybe_update_recurrent_halting_state(
            candidate_state.halting_state,
            hidden,
            previous_hidden,
        )
        hidden = self.__preserve_already_halted_hidden(
            previous_hidden,
            hidden,
            already_halted_mask,
        )
        return _RecurrentRunState(
            hidden=hidden,
            loss=candidate_state.loss,
            halting_state=halting_state,
        )

    def __maybe_update_recurrent_halting_state(
        self,
        halting_state: "HaltingStateBase | None",
        hidden: Tensor,
        previous_hidden: Tensor,
    ) -> tuple["HaltingStateBase | None", Tensor]:
        if self.halting_model is None:
            return halting_state, hidden

        halting_state, halting_hidden = (
            self.halting_model.update_halting_state(
                halting_state,
                hidden,
            )
        )
        RecurrentLayerValidator.validate_candidate(
            halting_hidden,
            previous_hidden,
            self.output_dim,
        )
        return halting_state, halting_hidden

    def __maybe_apply_recurrent_gate(
        self,
        candidate: Tensor,
        previous_hidden: Tensor,
    ) -> Tensor:
        if self.gate_model is None:
            return candidate
        gate_logits = Layer.forward_with_state(self.gate_model, candidate)
        RecurrentLayerValidator.validate_candidate(
            gate_logits,
            previous_hidden,
            self.output_dim,
        )
        gate = torch.sigmoid(gate_logits)
        return gate * candidate + (1.0 - gate) * previous_hidden

    def __preserve_already_halted_hidden(
        self,
        previous_hidden: Tensor,
        candidate_hidden: Tensor,
        already_halted_mask: Tensor | None,
    ) -> Tensor:
        return self.__preserve_halted_hidden(
            previous_hidden,
            candidate_hidden,
            already_halted_mask,
        )

    def __maybe_finalize_recurrent_halting(
        self,
        run_state: _RecurrentRunState,
    ) -> _RecurrentRunState:
        halting_state = run_state.halting_state
        if self.halting_model is None or halting_state is None:
            return run_state

        pre_finalization_hidden = run_state.hidden
        finalized_hidden, recurrent_loss = (
            self.halting_model.finalize_weighted_accumulation(
                halting_state,
                run_state.hidden,
            )
        )
        finalized_hidden = self.__preserve_halted_hidden(
            pre_finalization_hidden,
            finalized_hidden,
            self.__get_halt_mask(halting_state),
        )
        return _RecurrentRunState(
            hidden=finalized_hidden,
            loss=self.__accumulate_auxiliary_loss(
                run_state.loss,
                recurrent_loss,
            ),
            halting_state=halting_state,
        )

    def __accumulate_auxiliary_loss(
        self,
        loss: Tensor | None,
        auxiliary_loss: Tensor,
    ) -> Tensor:
        return auxiliary_loss if loss is None else loss + auxiliary_loss

    def __restore_output_state(
        self,
        state: LayerState,
        run_state: _RecurrentRunState,
        original_halting_state: "HaltingStateBase | None",
    ) -> LayerState:
        state.hidden = run_state.hidden
        state.loss = run_state.loss
        state.halting_state = original_halting_state
        return state

    def __get_halt_mask(
        self,
        halting_state: "HaltingStateBase | None",
    ) -> Tensor | None:
        if halting_state is None:
            return None
        return getattr(halting_state, "halt_mask", None)

    def __all_items_halted(self, halting_state: "HaltingStateBase | None") -> bool:
        halt_mask = self.__get_halt_mask(halting_state)
        if halt_mask is None:
            return False
        return bool(halt_mask.all().item())

    def __preserve_halted_hidden(
        self,
        previous_hidden: Tensor,
        candidate_hidden: Tensor,
        halt_mask: Tensor | None,
    ) -> Tensor:
        if halt_mask is None:
            return candidate_hidden
        while halt_mask.dim() < candidate_hidden.dim():
            halt_mask = halt_mask.unsqueeze(-1)
        return torch.where(halt_mask, previous_hidden, candidate_hidden)
