from torch import Tensor
from torch.nn import Sequential
from emperor.base.utils import Module
from emperor.base.layer import Layer, LayerStackConfig
from emperor.experts.core.layers import MixtureOfExperts
from emperor.experts.core.options import RoutingInitializationMode
from emperor.experts.core.stack import MixtureOfExpertsStack
from emperor.experts.core.config import MixtureOfExpertsConfig
from emperor.experts.core._validator import MixtureOfExpertsModelValidator

class MixtureOfExpertsModel(Module):
    def __init__(
        self,
        cfg: "MixtureOfExpertsConfig | LayerStackConfig",
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

        self.expert_stack = self._create_expert_stack()
        MixtureOfExpertsModelValidator.validate(self)

    def __resolve_stack_config(
        self,
        cfg: "MixtureOfExpertsConfig | LayerStackConfig",
        overrides: "MixtureOfExpertsConfig | LayerStackConfig | None",
    ) -> "LayerStackConfig | None":
        if not isinstance(cfg, LayerStackConfig):
            return None
        if isinstance(overrides, LayerStackConfig):
            return self._override_config(cfg, overrides)
        return cfg

    def __resolve_mixture_config(
        self,
        cfg: "MixtureOfExpertsConfig | LayerStackConfig",
        overrides: "MixtureOfExpertsConfig | LayerStackConfig | None",
    ) -> "MixtureOfExpertsConfig":
        config = cfg
        if isinstance(config, LayerStackConfig):
            config = config.layer_config.layer_model_config
        if isinstance(overrides, MixtureOfExpertsConfig):
            return self._override_config(config, overrides)
        return config

    def get_top_k(self) -> int:
        return self.top_k

    def _create_expert_stack(self) -> Layer | Sequential | MixtureOfExperts:
        if self.routing_initialization_mode == RoutingInitializationMode.SHARED:
            return MixtureOfExperts(self.__create_shared_mixture_config())
        if self.stack_config is not None:
            return MixtureOfExpertsStack(self.stack_config).build()
        return MixtureOfExperts(self.cfg)

    def __create_shared_mixture_config(self) -> MixtureOfExpertsConfig:
        overrides = MixtureOfExpertsConfig(
            routing_initialization_mode=RoutingInitializationMode.LAYER,
        )
        if self.stack_config is None:
            return self._override_config(self.cfg, overrides)

        expert_model_config = self._override_config(
            self.cfg.expert_model_config,
            LayerStackConfig(num_layers=self.stack_config.num_layers),
        )
        overrides.expert_model_config = expert_model_config
        return self._override_config(self.cfg, overrides)

    def forward(
        self,
        input: Tensor,
        probabilities: Tensor | None = None,
        indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if isinstance(self.expert_stack, MixtureOfExperts):
            return self.expert_stack(input, probabilities, indices)
        inputs = {
            "input_batch": input,
            "probabilities": probabilities,
            "indices": indices,
            "loss": None,
        }
        return self.expert_stack(inputs)
