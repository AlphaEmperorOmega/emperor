from typing import TYPE_CHECKING

from torch import Tensor

from emperor.base.module import Module
from emperor.sampler.core._validator import SamplerModelValidator
from emperor.sampler.core.base import SamplerBase
from emperor.sampler.core.config import RouterConfig, SamplerConfig
from emperor.sampler.core.routers import RouterModel
from emperor.sampler.core.tracker import SamplerUsageTrackerManager
from emperor.sampler.core.variants import SamplerFull, SamplerSparse, SamplerTopk

if TYPE_CHECKING:
    from emperor.config import ModelConfig
    from emperor.sampler.core.tracker import SamplerUsageTracker


class SamplerModel(Module):
    VALIDATOR = SamplerModelValidator

    def __init__(
        self,
        cfg: "SamplerConfig | ModelConfig",
        overrides: "SamplerConfig | None" = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.overrides = overrides

        config = getattr(cfg, "sampler_model_config", cfg)
        self.sampler_config: SamplerConfig = self._override_config(config, overrides)

        self.num_experts: int = self.sampler_config.num_experts
        self.router_config: RouterConfig = self.sampler_config.router_config
        self.VALIDATOR.validate(self)
        self.sampler_model = self.__init_sampler_model()
        self.router = self.__maybe_init_router()

    @property
    def usage_tracker(self) -> "SamplerUsageTracker | None":
        return self._modules.get(SamplerUsageTrackerManager.TRACKER_MODULE_NAME)

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
        return self.router_config.build()

    def sample_probabilities_and_indices(
        self,
        input_matrix: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor]:
        self.VALIDATOR.validate_forward_inputs(input_matrix)
        logits = self.__maybe_compute_routing(input_matrix)
        output = self.sampler_model.get_probabilities_and_indices(logits, skip_mask)
        SamplerUsageTrackerManager.maybe_record_sampler_output(self, output)
        return output

    def __maybe_compute_routing(self, input_matrix: Tensor) -> Tensor:
        if self.router is not None:
            return self.router.compute_logit_scores(input_matrix)
        return input_matrix

    def get_updated_skip_mask(self) -> Tensor | None:
        return self.sampler_model.updated_skip_mask

    def get_auxiliary_loss(self) -> Tensor:
        return self.sampler_model.auxiliary_loss
