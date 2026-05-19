import torch

from torch import Tensor
from torch.nn import Sequential
from emperor.base.utils import Module
from emperor.sampler.model import SamplerModel
from emperor.base.layer import Layer, LayerStackConfig
from emperor.experts.core.layers import MixtureOfExperts
from emperor.experts.core.options import RoutingInitializationMode
from emperor.experts.core.stack import MixtureOfExpertsStack
from emperor.experts.core.config import MixtureOfExpertsConfig
from emperor.sampler.core.config import RouterConfig, SamplerConfig
from emperor.sampler.core.routers import RouterModel
from emperor.experts.core._validator import MixtureOfExpertsModelValidator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class MixtureOfExpertsModel(Module):
    def __init__(
        self,
        cfg: "MixtureOfExpertsConfig | LayerStackConfig | ModelConfig",
        overrides: "MixtureOfExpertsConfig | LayerStackConfig | None" = None,
    ) -> None:
        super().__init__()
        self.stack_config = self.__resolve_stack_config(cfg, overrides)
        self.cfg = self.__resolve_mixture_config(cfg, overrides)
        self.main_cfg: "MixtureOfExpertsConfig" = self.cfg

        self.top_k = self.cfg.top_k
        self.input_dim = self.cfg.input_dim
        self.routing_initialization_mode = self.cfg.routing_initialization_mode
        self.router_config = self.cfg.router_config
        self.sampler_config = self.cfg.sampler_config

        self.router, self.sampler = self.__maybe_create_router_and_sampler()
        self.expert_stack = self._create_expert_stack()
        MixtureOfExpertsModelValidator.validate(self)

    def __resolve_stack_config(
        self,
        cfg: "MixtureOfExpertsConfig | LayerStackConfig | ModelConfig",
        overrides: "MixtureOfExpertsConfig | LayerStackConfig | None",
    ) -> "LayerStackConfig | None":
        config = getattr(cfg, "layer_stack_config", cfg)
        if not isinstance(config, LayerStackConfig):
            return None
        if isinstance(overrides, LayerStackConfig):
            return self._override_config(config, overrides)
        return config

    def __resolve_mixture_config(
        self,
        cfg: "MixtureOfExpertsConfig | LayerStackConfig | ModelConfig",
        overrides: "MixtureOfExpertsConfig | LayerStackConfig | None",
    ) -> "MixtureOfExpertsConfig":
        config = getattr(cfg, "mixture_of_experts_config", cfg)
        config = getattr(config, "layer_stack_config", config)
        if isinstance(config, LayerStackConfig):
            config = config.layer_config.layer_model_config
        if isinstance(overrides, MixtureOfExpertsConfig):
            return self._override_config(config, overrides)
        return config

    def get_top_k(self) -> int:
        return self.top_k

    def _create_expert_stack(self) -> Layer | Sequential | MixtureOfExperts:
        if self.stack_config is not None:
            return MixtureOfExpertsStack(self.stack_config).build()
        return MixtureOfExperts(self.cfg)

    def __maybe_create_router_and_sampler(
        self,
    ) -> tuple[RouterModel | None, SamplerModel | None]:
        if self.routing_initialization_mode == RoutingInitializationMode.SHARED:
            router_overrides = RouterConfig(input_dim=self.input_dim)
            router_config = self._override_config(self.router_config, router_overrides)
            sampler_overrides = SamplerConfig(router_config=router_config)
            sampler_config = self._override_config(self.sampler_config, sampler_overrides)
            sampler = SamplerModel(sampler_config)
            return sampler.router, sampler
        return None, None

    def forward(
        self,
        input: Tensor,
        probabilities: Tensor | None = None,
        indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        probabilities, indices, sampler_loss = self._maybe_compute_expert_indices(
            input, probabilities, indices
        )

        if self.stack_config is None:
            output, expert_loss = self.expert_stack(input, probabilities, indices)
        else:
            inputs = {
                "input_batch": input,
                "probabilities": probabilities,
                "indices": indices,
                "loss": None,
            }
            output, expert_loss = self.expert_stack(inputs)

        total_loss = sampler_loss + expert_loss
        return output, total_loss

    def _maybe_compute_expert_indices(
        self,
        inputs: Tensor,
        probabilities: Tensor | None = None,
        indices: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor]:
        if self.routing_initialization_mode != RoutingInitializationMode.SHARED or (
            indices is not None or probabilities is not None
        ):
            return probabilities, indices, torch.tensor(0.0)

        # TODO: In the future see if `skip_mask` needs to be implemented
        skip_mask = None
        probabilities, indices, skip_mask, sampler_loss = (
            self.sampler.sample_probabilities_and_indices(inputs, skip_mask)
        )
        return probabilities, indices, sampler_loss
