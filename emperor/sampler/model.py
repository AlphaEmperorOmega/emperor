from torch import Tensor
from emperor.base.utils import Module
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

        self.num_experts = self.sampler_config.num_experts
        self.sampler_model = self._init_sampler_model()

    def _init_sampler_model(self) -> SamplerBase:
        if self.sampler_config.top_k == 1:
            return SamplerSparse(self.cfg, self.overrides)
        elif self.sampler_config.top_k == self.num_experts:
            return SamplerFull(self.cfg, self.overrides)
        else:
            return SamplerTopk(self.cfg, self.overrides)

    def sample_probabilities_and_indices(
        self,
        input_matrix: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor]:
        return self.sampler_model.get_probabilities_and_indices(input_matrix, skip_mask)

    def get_updated_skip_mask(self) -> Tensor | None:
        return self.sampler_model.updated_skip_mask

    def get_auxiliary_loss(self) -> Tensor:
        return self.sampler_model.auxiliary_loss
