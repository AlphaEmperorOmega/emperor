from torch import Tensor

from emperor.base.layer import Layer, LayerState
from emperor.base.module import Module
from emperor.parametric.core._validator import ParametricHandlerValidator
from emperor.parametric.core.mixtures.config import (
    GeneratorBiasMixtureConfig,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.parametric.core.config import ParametricLayerConfig


class ParameterHandlerBase(Module):
    def __init__(
        self,
        cfg: "ParametricLayerConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.weight_mixture_config = self.cfg.weight_mixture_config
        self.bias_mixture_config = self.cfg.bias_mixture_config
        self.routing_initialization_mode = self.cfg.routing_initialization_mode
        self.router_config = self.cfg.router_config
        self.sampler_config = self.cfg.sampler_config
        ParametricHandlerValidator.validate(self)

    def build_sampler_models(self) -> tuple:
        from emperor.parametric.core.config import AdaptiveRouterOptions

        if self.routing_initialization_mode == AdaptiveRouterOptions.SHARED_ROUTER:
            return self._init_shared_sampler()
        return self._init_independent_sampler()

    def _init_shared_sampler(self) -> tuple:
        raise NotImplementedError(
            "The method `_init_shared_sampler` must be implemented in the child class."
        )

    def _init_independent_sampler(self) -> tuple:
        raise NotImplementedError(
            "The method `_init_independent_sampler` must be implemented in the child class."
        )

    def _build_weight_router_model(self, vector_router: bool = False):
        from emperor.parametric.core.routers import VectorRouterConfig
        from emperor.sampler.core.config import RouterConfig

        config_type = VectorRouterConfig if vector_router else RouterConfig
        router_config = self._clone_router_config(config_type)
        router_overrides = RouterConfig(input_dim=self.input_dim)
        return router_config.build(router_overrides)

    def _init_bias_router_model(self):
        from emperor.sampler.core.config import RouterConfig

        if self.bias_mixture_config is None:
            return None
        if isinstance(self.bias_mixture_config, GeneratorBiasMixtureConfig):
            return None
        router_config = self._clone_router_config(RouterConfig)
        router_overrides = RouterConfig(input_dim=self.input_dim)
        return router_config.build(router_overrides)

    def _build_sampler_model(self):
        return self.sampler_config.build()

    def _clone_router_config(self, config_type: type):
        return config_type(
            input_dim=self.router_config.input_dim,
            num_experts=self.router_config.num_experts,
            noisy_topk_flag=self.router_config.noisy_topk_flag,
            model_config=self.router_config.model_config,
        )


class VectorParameterHandler(ParameterHandlerBase):
    def _init_shared_sampler(self) -> tuple:
        ParametricHandlerValidator.validate(self)
        raise ValueError("VectorWeightsMixtureConfig does not support SHARED_ROUTER.")

    def _init_independent_sampler(self) -> tuple:
        weight_router = self._build_weight_router_model(vector_router=True)
        bias_router = self._init_bias_router_model()
        sampler = self._build_sampler_model()
        return weight_router, bias_router, sampler


class MatrixParameterHandler(ParameterHandlerBase):
    def _init_shared_sampler(self) -> tuple:
        router = self._build_weight_router_model()
        sampler = self._build_sampler_model()
        return router, None, sampler

    def _init_independent_sampler(self) -> tuple:
        weights_router = self._build_weight_router_model()
        bias_router = self._init_bias_router_model()
        sampler = self._build_sampler_model()
        return weights_router, bias_router, sampler


class GeneratorParameterHandler(ParameterHandlerBase):
    def _init_shared_sampler(self) -> tuple:
        router = self._build_weight_router_model()
        sampler = self._build_sampler_model()
        return router, None, sampler

    def _init_independent_sampler(self) -> tuple:
        weights_router = None
        bias_router = self._init_bias_router_model()
        sampler = self._build_sampler_model() if bias_router is not None else None
        return weights_router, bias_router, sampler


class ParametricLayerHandler(Layer):
    def __init__(self, cfg, overrides=None):
        super().__init__(cfg, overrides)
        ParametricHandlerValidator.validate(self)

    def _handle_model_processing(
        self,
        main_model_input: Tensor,
        state: LayerState,
    ) -> Tensor:
        ParametricHandlerValidator.validate_state(state)
        skip_mask = getattr(state, "skip_mask", None)
        output, skip_mask, loss = self.model(main_model_input, skip_mask)
        state.skip_mask = skip_mask
        state.loss = loss if state.loss is None else state.loss + loss
        return output
