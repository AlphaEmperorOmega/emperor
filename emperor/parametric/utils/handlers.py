from emperor.base.utils import Module
from emperor.sampler.model import SamplerModel
from emperor.sampler.utils.routers import RouterConfig, RouterModel
from emperor.parametric.utils.routers import VectorRouterModel
from emperor.parametric.utils._validator import _ParametricHandlerValidator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.parametric.utils.config import ParametricLayerConfig, AdaptiveRouterOptions


class ParameterHanlderBase(Module):
    def __init__(
        self,
        cfg: "ParametricLayerConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.adaptive_weight_option = self.cfg.adaptive_weight_option
        self.adaptive_bias_option = self.cfg.adaptive_bias_option
        self.init_sampler_model_option = self.cfg.init_sampler_model_option
        self.router_config = self.cfg.router_config
        self.router_overrides = RouterConfig(input_dim=self.input_dim)
        self.sampler_config = self.cfg.sampler_config
        self.validator = _ParametricHandlerValidator(self)

    def build_sampler_models(
        self,
    ) -> tuple[RouterModel | None, RouterModel | None, SamplerModel | None]:
        from emperor.parametric.utils.config import AdaptiveRouterOptions

        shared_option = AdaptiveRouterOptions.SHARED_ROUTER
        if self.init_sampler_model_option == shared_option:
            return self._init_shared_sampler()
        return self._init_independent_sampler()

    def _init_shared_sampler(self):
        raise NotImplementedError(
            "The method `_init_shared_sampler` must be implemented in the child class."
        )

    def _init_independent_sampler(self):
        raise NotImplementedError(
            "The method `_init_independent_sampler` must be implemented in the child class."
        )

    def _init_bias_router_model(self):
        from emperor.parametric.utils.mixtures.options import AdaptiveBiasOptions

        if self.adaptive_bias_option == AdaptiveBiasOptions.GENERATOR:
            return None
        return RouterModel(self.router_config, self.router_overrides)


class VectorParameterHandler(ParameterHanlderBase):
    def __init__(
        self,
        cfg: "ParametricLayerConfig",
    ):
        super().__init__(cfg)

    def _init_shared_sampler(self) -> None:
        self.validator.ensure_shared_sampler_is_disabled()

    def _init_independent_sampler(
        self,
    ) -> tuple[VectorRouterModel, RouterModel | None, SamplerModel]:
        self.validator.ensure_indepentent_router_for_vector_option()
        weight_router = VectorRouterModel(self.router_config, self.router_overrides)
        bias_router = self._init_bias_router_model()
        sampler = SamplerModel(self.sampler_config)
        return weight_router, bias_router, sampler


class MatrixParameterHandler(ParameterHanlderBase):
    def __init__(
        self,
        cfg: "ParametricLayerConfig",
    ):
        super().__init__(cfg)

    def _init_shared_sampler(self) -> tuple[RouterModel, None, SamplerModel]:
        self.validator.ensure_router_and_sampler_configs_exist()
        router = RouterModel(self.router_config, self.router_overrides)
        sampler = SamplerModel(self.sampler_config)
        return router, None, sampler

    def _init_independent_sampler(
        self,
    ) -> tuple[RouterModel, RouterModel | None, SamplerModel]:
        self.validator.ensure_router_and_sampler_configs_exist()
        weights_router = RouterModel(self.router_config, self.router_overrides)
        bias_router = self._init_bias_router_model()
        sampler = SamplerModel(self.sampler_config)
        return weights_router, bias_router, sampler


class GeneratorParameterHandler(ParameterHanlderBase):
    def __init__(
        self,
        cfg: "ParametricLayerConfig",
    ):
        super().__init__(cfg)

    def _init_shared_sampler(self) -> tuple[RouterModel, None, SamplerModel]:
        self.validator.ensure_router_and_sampler_configs_exist()
        router = RouterModel(self.router_config, self.router_overrides)
        sampler = SamplerModel(self.sampler_config)
        return router, None, sampler

    def _init_independent_sampler(
        self,
    ) -> tuple[RouterModel | None, RouterModel | None, SamplerModel | None]:
        weights_router = None
        bias_router = self._init_bias_router_model()
        sampler = SamplerModel(self.sampler_config, self.router_overrides)
        return weights_router, bias_router, sampler
