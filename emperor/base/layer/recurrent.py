import torch

from torch import Tensor
from torch.nn import Sequential
from emperor.base.utils import ConfigBase, Module
from emperor.base.layer.config import (
    LayerConfig,
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
        self.block_config: LayerConfig | LayerStackConfig = self.cfg.block_config
        self.gate_config: LayerStackConfig | None = self.cfg.gate_config
        self.halting_config: "HaltingConfig | None" = self.cfg.halting_config

        self.block_model = self.__build_block_model()
        self.gate_model = self.__build_gate_model()
        self.halting_model = self.__build_halting_model()

    def __build_block_model(self) -> "Layer | Sequential":
        recurrent_dim = self.output_dim
        if isinstance(self.block_config, LayerStackConfig):
            overrides = LayerStackConfig(
                input_dim=recurrent_dim,
                hidden_dim=recurrent_dim,
                output_dim=recurrent_dim,
            )
        else:
            overrides = LayerConfig(
                input_dim=recurrent_dim,
                output_dim=recurrent_dim,
            )
        return self.block_config.build(overrides=overrides)

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
        recurrent_halting_state = None
        hidden = state.hidden
        loss = state.loss

        for _ in range(self.max_steps):
            previous_hidden = hidden
            previous_halt_mask = self.__get_halt_mask(recurrent_halting_state)

            block_state = LayerState(
                hidden=hidden,
                loss=loss,
                halting_state=None,
            )
            block_output = self.block_model(block_state)
            candidate = block_output.hidden
            loss = block_output.loss
            RecurrentLayerValidator.validate_candidate(
                candidate,
                previous_hidden,
                self.output_dim,
            )

            hidden = self.__maybe_apply_recurrent_gate(candidate, previous_hidden)
            hidden = self.__preserve_halted_hidden(
                previous_hidden,
                hidden,
                previous_halt_mask,
            )

            if self.halting_model is None:
                continue

            recurrent_halting_state, halting_hidden = (
                self.halting_model.update_halting_state(
                    recurrent_halting_state,
                    hidden,
                )
            )
            RecurrentLayerValidator.validate_candidate(
                halting_hidden,
                previous_hidden,
                self.output_dim,
            )
            hidden = self.__preserve_halted_hidden(
                previous_hidden,
                halting_hidden,
                previous_halt_mask,
            )
            if self.__all_items_halted(recurrent_halting_state):
                break

        if self.halting_model is not None and recurrent_halting_state is not None:
            hidden, loss = self.__finalize_recurrent_halting(
                recurrent_halting_state,
                hidden,
                loss,
            )

        state.hidden = hidden
        state.loss = loss
        state.halting_state = original_halting_state
        return state

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

    def __finalize_recurrent_halting(
        self,
        recurrent_halting_state: "HaltingStateBase",
        hidden: Tensor,
        loss: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        frozen_hidden = hidden
        finalized_hidden, recurrent_loss = (
            self.halting_model.finalize_weighted_accumulation(
                recurrent_halting_state,
                hidden,
            )
        )
        finalized_hidden = self.__preserve_halted_hidden(
            frozen_hidden,
            finalized_hidden,
            self.__get_halt_mask(recurrent_halting_state),
        )
        loss = recurrent_loss if loss is None else loss + recurrent_loss
        return finalized_hidden, loss

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
