import copy
import torch.nn as nn

from torch import Tensor, overrides
from torch.nn import Linear, Sequential
from torch.nn.parameter import Parameter
from dataclasses import dataclass, field
from Emperor.base.layer import Layer, LayerStack, LayerStackConfig
from Emperor.base.utils import randn, matmul
from Emperor.base.utils import Module, ConfigBase
from Emperor.linears.options import LinearLayerOptions
from Emperor.sampler.utils.routers import RouterModel


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class VectorRouterModel(RouterModel):
    def __init__(
        self,
        cfg: "RouterConfig | ModelConfig",
        overrides: "RouterConfig | None" = None,
        bias_parameters_flag: bool = False,
        bias_output_dim: int | None = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.bias_parameters_flag = bias_parameters_flag
        self.bias_num_experts = bias_output_dim
        self.parameter_bank = self.__generate_parameter_bank()

    def __generate_parameter_bank(self) -> Parameter:
        # This is required for `VectorRouterModel` in case `feature_dim`
        # you find this wierd in the future
        feature_dim = (
            self.bias_num_experts if self.bias_parameters_flag else self.input_dim
        )
        parameters = Parameter(randn(feature_dim, self.input_dim, self.num_experts))
        self._initialize_parameters(parameters)
        return parameters

    def compute_logit_scores(
        self,
        input_batch: Tensor,
    ) -> Tensor:
        return matmul(input_batch, self.parameter_bank)
