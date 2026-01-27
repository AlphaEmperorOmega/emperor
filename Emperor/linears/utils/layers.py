import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Parameter
from dataclasses import dataclass, field
from Emperor.base.utils import ConfigBase, Module
from Emperor.linears.utils._validator import LinearBaseValidator
from Emperor.linears.utils.monitors import DataMonitor, ParameterMonitor
from Emperor.behaviours.model import (
    AdaptiveParameterBehaviour,
    AdaptiveParameterBehaviourConfig,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


@dataclass
class LinearLayerConfig(ConfigBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Input dimension of the linear layer"},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Output dimension of the linera layer"},
    )
    bias_flag: bool | None = field(
        default=None,
        metadata={
            "help": "When true bias will be added to after the matrix multiplication between, the input and output"
        },
    )
    data_monitor: type[DataMonitor] | None = field(
        default=None,
        metadata={"help": ""},
    )
    parameter_monitor: type[ParameterMonitor] | None = field(
        default=None,
        metadata={"help": ""},
    )


class LinearBase(Module):
    def __init__(
        self,
        cfg: "LinearLayerConfig | LinearLayerConfig | ModelConfig",
        overrides: "LinearLayerConfig | LinearLayerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "linear_layer_config", cfg)
        self.cfg: "LinearLayerConfig" = self._overwrite_config(config, overrides)
        self.main_cfg: "AdaptiveParameterBehaviourConfig" = self._resolve_main_config(
            self.cfg, cfg
        )
        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.bias_flag: bool = self.cfg.bias_flag
        self.data_monitor: DataMonitor = self.construct(self.cfg.data_monitor)
        self.weight_params, self.bias_params = self._init_parameters()
        self.parameter_monitor: ParameterMonitor = self.construct(
            self.cfg.parameter_monitor
        )
        self.validator = LinearBaseValidator(self)

    def _init_parameters(self) -> tuple[Parameter, Parameter | None]:
        weight_params = self.__init_weight_parameters()
        bias_params = self.__init_bias_parameters()
        return weight_params, bias_params

    def __init_weight_parameters(self) -> Parameter:
        weight_shape = (self.input_dim, self.output_dim)
        return self._init_parameter_bank(weight_shape)

    def __init_bias_parameters(self) -> Parameter | None:
        if not self.bias_flag:
            return None
        bias_shape = (self.output_dim,)
        return self._init_parameter_bank(bias_shape, nn.init.zeros_)

    def _update_data_monitor(self, X: Tensor, output: Tensor) -> None:
        if self.data_monitor:
            self.data_monitor.update(X, output)

    def _update_parameter_monitor(self) -> None:
        if self.parameter_monitor:
            self.parameter_monitor.update(self.weight_params, self.bias_params)


class LinearLayer(LinearBase):
    def __init__(
        self,
        cfg: "LinearLayerConfig | ModelConfig",
        overrides: "LinearLayerConfig | None" = None,
    ):
        super().__init__(cfg, overrides)

    def forward(self, X: Tensor) -> Tensor:
        output = F.linear(X, self.weight_params.T, self.bias_params)
        self._update_data_monitor(X, output)
        self._update_parameter_monitor()
        return output


class AdaptiveLinearLayer(LinearBase):
    def __init__(
        self,
        cfg: "LinearLayerConfig | ModelConfig",
        overrides: "LinearLayerConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        self.weight_params, self.bias_params = self._init_parameters()
        self.adaptive_behaviour = self.__init_behaviour()

    def __init_behaviour(self):
        overrides = AdaptiveParameterBehaviourConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return AdaptiveParameterBehaviour(self.main_cfg, overrides)

    def forward(self, X: Tensor) -> Tensor:
        return self.adaptive_behaviour.compute_adaptive_parameters(
            self._compute_affine_transformation_callback,
            self.weight_params,
            self.bias_params,
            X,
        )

    def _compute_affine_transformation_callback(
        self, weights: Tensor, bias: Tensor | None, X: Tensor
    ) -> Tensor:
        output = self.__compute_linear_transformation(X, weights)
        return self.__add_bias_parameters(output, bias)

    def __compute_linear_transformation(
        self,
        X: Tensor,
        dynamic_diagonal_weights: Tensor,
    ) -> Tensor:
        if dynamic_diagonal_weights.dim() == 3:
            return torch.einsum("ij,ijk->ik", X, dynamic_diagonal_weights)
        return torch.matmul(X, dynamic_diagonal_weights)

    def __add_bias_parameters(
        self,
        X: Tensor,
        bias_params: Tensor | None = None,
    ) -> Tensor:
        if bias_params is not None:
            return X + bias_params
        return X
