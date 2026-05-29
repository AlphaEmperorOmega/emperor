import torch

from torch import Tensor
from dataclasses import dataclass
from torch.nn.parameter import Parameter
from emperor.sampler.core.config import RouterConfig
from emperor.sampler.core.routers import RouterModel
from emperor.sampler.core._validator import RouterModelValidator


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


@dataclass
class VectorRouterConfig(RouterConfig):
    def _registry_owner(self) -> type:
        from emperor.parametric.core.routers import VectorRouterModel

        return VectorRouterModel


class VectorRouterModel(RouterModel):
    def __init__(
        self,
        cfg: "RouterConfig | ModelConfig",
        overrides: "RouterConfig | None" = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.parameter_bank = self._create_router_model()

    def _create_router_model(self) -> Parameter:
        parameters = Parameter(
            torch.randn(self.input_dim, self.input_dim, self.num_experts)
        )
        self._initialize_parameters(parameters)
        return parameters

    def compute_logit_scores(
        self,
        input_batch: Tensor,
    ) -> Tensor:
        RouterModelValidator.validate_input_batch(self, input_batch)
        return torch.einsum("bi,ijn->bjn", input_batch, self.parameter_bank)
