import torch

from torch import Tensor
from torch.nn.parameter import Parameter
from emperor.base.utils import matmul
from emperor.sampler.utils.routers import RouterConfig, RouterModel


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


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
        return matmul(input_batch, self.parameter_bank)
