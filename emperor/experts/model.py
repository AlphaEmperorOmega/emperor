from torch import Tensor
from emperor.base.utils import Module
from emperor.base.layer import LayerStackConfig
from emperor.experts.core.options import RoutingInitializationMode
from emperor.experts.core.state import MixtureOfExpertsLayerState
from emperor.experts.core._validator import MixtureOfExpertsModelValidator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.sampler.model import SamplerModel


class MixtureOfExpertsModel(Module):
    def __init__(
        self,
        cfg: LayerStackConfig,
        overrides: LayerStackConfig | None = None,
    ) -> None:
        super().__init__()
        self.stack_config = self._override_config(cfg, overrides)
        self.cfg = self.stack_config.layer_config.layer_model_config

        self.top_k = self.cfg.top_k
        self.input_dim = self.cfg.input_dim
        self.routing_initialization_mode = self.cfg.routing_initialization_mode
        self.sampler_config = self.cfg.sampler_config

        MixtureOfExpertsModelValidator.validate(self)
        self.shared_sampler = self._maybe_create_shared_sampler()
        self.expert_stack = self._build_expert_stack()

    def _maybe_create_shared_sampler(self) -> "SamplerModel | None":
        if self.routing_initialization_mode != RoutingInitializationMode.SHARED:
            return None
        from emperor.sampler.core.config import RouterConfig, SamplerConfig

        router_overrides = RouterConfig(input_dim=self.input_dim)
        router_config = self._override_config(
            self.sampler_config.router_config, router_overrides
        )
        sampler_overrides = SamplerConfig(router_config=router_config)
        return self.sampler_config.build(sampler_overrides)

    def _build_expert_stack(self) -> Module:
        return self.stack_config.build()

    def forward(
        self,
        hidden: Tensor,
        probabilities: Tensor | None = None,
        indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        probabilities, indices, shared_sampler_loss = (
            self.__maybe_apply_shared_routing(hidden, probabilities, indices)
        )
        state = MixtureOfExpertsLayerState(
            hidden=hidden,
            probabilities=probabilities,
            indices=indices,
            loss=None,
        )
        state = self.expert_stack(state)
        return state.hidden, self.__combine_losses(
            shared_sampler_loss, state.loss
        )

    def __maybe_apply_shared_routing(
        self,
        hidden: Tensor,
        probabilities: Tensor | None,
        indices: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        if self.shared_sampler is None:
            return probabilities, indices, None
        probabilities, indices, _, sampler_loss = (
            self.shared_sampler.sample_probabilities_and_indices(hidden, None)
        )
        return probabilities, indices, sampler_loss

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
