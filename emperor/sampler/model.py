from torch import Tensor
from emperor.base.utils import Module
from emperor.sampler.utils.routers import RouterConfig, RouterModel
from emperor.sampler.utils._validator import SamplerModelValidator
from emperor.sampler.utils.samplers import (
    SamplerBase,
    SamplerConfig,
    SamplerFull,
    SamplerSparse,
    SamplerTopk,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class SamplerModel(Module):
    def __init__(
        self,
        cfg: "SamplerConfig | ModelConfig",
        overrides: "SamplerConfig | None" = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.overrides = overrides

        config = getattr(cfg, "sampler_model_config", cfg)
        self.sampler_config: "SamplerConfig" = self._overwrite_config(config, overrides)

        self.num_experts: int = self.sampler_config.num_experts
        self.router_config: "RouterConfig" = self.sampler_config.router_config
        self.validator = SamplerModelValidator(self)
        self.sampler_model = self.__init_sampler_model()
        self.router = self.__maybe_init_router()

    def __init_sampler_model(self) -> SamplerBase:
        if self.sampler_config.top_k == 1:
            return SamplerSparse(self.cfg, self.overrides)
        elif self.sampler_config.top_k == self.num_experts:
            return SamplerFull(self.cfg, self.overrides)
        else:
            return SamplerTopk(self.cfg, self.overrides)

    def __maybe_init_router(self) -> RouterModel | None:
        if self.router_config is None:
            return None
        return RouterModel(self.router_config)

    def sample_probabilities_and_indices(
        self,
        input_matrix: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor]:
        logits = self.__maybe_compute_routing(input_matrix)
        return self.sampler_model.get_probabilities_and_indices(logits, skip_mask)

    def __maybe_compute_routing(self, input_matrix: Tensor) -> Tensor:
        if self.router is not None:
            return self.router.compute_logit_scores(input_matrix)
        return input_matrix

    def get_updated_skip_mask(self) -> Tensor | None:
        return self.sampler_model.updated_skip_mask

    def get_auxiliary_loss(self) -> Tensor:
        return self.sampler_model.auxiliary_loss
