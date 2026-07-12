from emperor.base.module import Module
from emperor.parametric.core.mixtures._validator import AdaptiveMixtureValidator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig
    from emperor.parametric.core.mixtures.config import AdaptiveMixtureConfig


class AdaptiveMixtureBase(Module):
    def __init__(
        self,
        cfg: "AdaptiveMixtureConfig | ModelConfig",
        overrides: "AdaptiveMixtureConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "mixture_model_config", cfg)
        self.cfg: "AdaptiveMixtureConfig" = self._override_config(config, overrides)
        self.main_cfg = self._resolve_main_config(self.cfg, cfg)

        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.top_k = self.cfg.top_k
        self.num_experts = self.cfg.num_experts
        self.weighted_parameters_flag = self.cfg.weighted_parameters_flag
        self.clip_parameter_option = self.cfg.clip_parameter_option
        self.clip_range = self.cfg.clip_range

        AdaptiveMixtureValidator.validate(self)
