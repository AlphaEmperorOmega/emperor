from typing import TYPE_CHECKING

from torch import Tensor

from emperor.experts._config import MixtureOfExpertsModelConfig
from emperor.experts._options import RoutingInitializationMode
from emperor.experts._state import MixtureOfExpertsLayerState
from emperor.experts._validation.model import MixtureOfExpertsModelValidator
from emperor.layers import LayerState
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.sampler import SamplerModel


class MixtureOfExpertsModel(Module):
    VALIDATOR = MixtureOfExpertsModelValidator

    def __init__(
        self,
        cfg: MixtureOfExpertsModelConfig,
        overrides: MixtureOfExpertsModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg: MixtureOfExpertsModelConfig = self._override_config(cfg, overrides)

        self.top_k = self.cfg.top_k
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.routing_initialization_mode = self.cfg.routing_initialization_mode
        self.sampler_config = self.cfg.sampler_config
        self.stack_config = self.cfg.stack_config

        self.VALIDATOR.validate(self)
        self.shared_sampler = self.__maybe_create_shared_sampler()
        self.expert_stack = self.__build_expert_stack()

    def __maybe_create_shared_sampler(self) -> "SamplerModel | None":
        if self.routing_initialization_mode != RoutingInitializationMode.SHARED:
            return None
        return self.sampler_config.build_with_router_input_dim(self.input_dim)

    def __build_expert_stack(self) -> Module:
        return self.stack_config.build()

    def forward(self, state: LayerState) -> MixtureOfExpertsLayerState:
        existing_routing_probabilities = getattr(state, "probabilities", None)
        existing_expert_indices = getattr(state, "indices", None)
        existing_skip_mask = getattr(state, "skip_mask", None)
        probabilities, indices, skip_mask, shared_sampler_loss = (
            self.__maybe_apply_shared_routing(
                state.hidden,
                existing_routing_probabilities,
                existing_expert_indices,
                existing_skip_mask,
            )
        )
        mixture_of_experts_state = MixtureOfExpertsLayerState(
            hidden=state.hidden,
            probabilities=probabilities,
            indices=indices,
            skip_mask=skip_mask,
            loss=state.loss,
        )
        mixture_of_experts_state = self.expert_stack(mixture_of_experts_state)
        mixture_of_experts_state.loss = self.__combine_losses(
            shared_sampler_loss, mixture_of_experts_state.loss
        )
        return mixture_of_experts_state

    def __maybe_apply_shared_routing(
        self,
        hidden: Tensor,
        probabilities: Tensor | None,
        indices: Tensor | None,
        skip_mask: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        if self.shared_sampler is None:
            return probabilities, indices, skip_mask, None
        probabilities, indices, skip_mask, sampler_loss = (
            self.shared_sampler.sample_probabilities_and_indices(hidden, skip_mask)
        )
        return probabilities, indices, skip_mask, sampler_loss

    def __combine_losses(
        self,
        shared_sampler_loss: Tensor | None,
        expert_stack_loss: Tensor | None,
    ) -> Tensor | None:
        if shared_sampler_loss is None:
            return expert_stack_loss
        if expert_stack_loss is None:
            return shared_sampler_loss
        return shared_sampler_loss + expert_stack_loss
