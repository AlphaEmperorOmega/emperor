from torch import Tensor
from emperor.base.layer import Layer, LayerStackConfig
from emperor.base.utils import Module
from emperor.sampler.core.config import RouterConfig
from emperor.sampler.core._validator import RouterModelValidator


class RouterModel(Module):
    def __init__(
        self,
        cfg: "RouterConfig",
        overrides: "RouterConfig | None" = None,
    ) -> None:
        super().__init__()
        config = getattr(cfg, "router_model_config", cfg)
        self.cfg: "RouterConfig" = self._override_config(config, overrides)

        self.input_dim = self.cfg.input_dim or getattr(cfg, "input_dim", None)
        self.num_experts = self.cfg.num_experts
        self.noisy_topk_flag = self.cfg.noisy_topk_flag
        self.model_config = self.cfg.model_config

        RouterModelValidator.validate(self)
        self.router_output_dim = self.__resolve_router_output_dim()
        self.model = self._init_model()

    def _init_model(self):
        overrides = LayerStackConfig(
            input_dim=self.input_dim, output_dim=self.router_output_dim
        )
        return self.model_config.build(overrides)

    def __resolve_router_output_dim(self) -> int:
        return 2 * self.num_experts if self.noisy_topk_flag else self.num_experts

    def compute_logit_scores(self, input_batch: Tensor) -> Tensor:
        RouterModelValidator.validate_input_batch(self, input_batch)
        return Layer.forward_with_state(self.model, input_batch)
