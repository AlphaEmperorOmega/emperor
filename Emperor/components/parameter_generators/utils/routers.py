from torch import Tensor
from Emperor.base.utils import randn, matmul
from torch.nn.parameter import Parameter
from Emperor.base.utils import Module
from torch.nn import Linear, Sequential

from typing import TYPE_CHECKING, Optional, Tuple, Union

if TYPE_CHECKING:
    from Emperor.config import ModelConfig
    from Emperor.config import RouterConfig


class RouterModel(Module):
    def __init__(
        self,
        cfg: "ModelConfig",
        input_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        bias_flag: Optional[bool] = None,
        noisy_topk_flag: Optional[bool] = None,
        residual_flag: Optional[bool] = None,
        activation: Optional[Module] = None,
        num_layers: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.cfg: "RouterConfig" = cfg.router_model_config
        self.input_dim = self._resolve(input_dim, self.cfg.input_dim)
        self.hidden_dim = self._resolve(hidden_dim, self.cfg.hidden_dim)
        self.output_dim = self._resolve(output_dim, self.cfg.output_dim)
        self.residual_flag = self._resolve(residual_flag, self.cfg.residual_flag)
        self.bias_flag = self._resolve(bias_flag, cfg.bias_flag)
        self.noisy_topk_flag = self._resolve(noisy_topk_flag, cfg.noisy_topk_flag)
        self.activation = self._resolve(activation, self.cfg.activation)
        self.num_layers = self._resolve(num_layers, self.cfg.num_layers)
        assert self.output_dim is not None
        self.router_output_dim = (
            2 * self.output_dim if self.noisy_topk_flag else self.output_dim
        )
        assert self.num_layers > 0, (
            "Expected `num_layers` in `RouterModel` to be at least one."
        )
        self._build_models_hook()

    def _build_models_hook(self) -> None:
        self.weight_router_model = self.__build_model()
        self._initialize_parameters(self.weight_router_model)
        if self.bias_flag:
            self.bias_router_model = self.__build_model()
            self._initialize_parameters(self.bias_router_model)

    def __build_model(self) -> Union[Linear, Sequential]:
        if self.num_layers > 1:
            router_model = self.__build_multilayer_router()
        else:
            router_model = Linear(self.input_dim, self.router_output_dim, bias=False)
        self._initialize_parameters(router_model)
        return router_model

    def __build_multilayer_router(self) -> Sequential:
        router_layers = []

        layer_adjustment = 1
        if self.input_dim != self.hidden_dim:
            layer_adjustment = 2
            router_layers.append(Linear(self.input_dim, self.hidden_dim))
            router_layers.append(self.activation)

        for _ in range(self.num_layers - layer_adjustment):
            layer = RouterLayer(self.hidden_dim, self.activation, self.residual_flag)
            router_layers.append(layer)

        router_layers.append(
            Linear(self.hidden_dim, self.router_output_dim, bias=False)
        )
        return Sequential(*router_layers)

    def compute_logit_scores(
        self, input_batch: Tensor, compute_weight_flag: bool = True
    ) -> Tensor:
        if compute_weight_flag:
            return self._compute_weight_logit_scores_hook(input_batch)
        return self._compute_bias_logit_scores_hook(input_batch)

    def _compute_weight_logit_scores_hook(self, input_batch: Tensor) -> Tensor:
        return self.weight_router_model(input_batch)

    def _compute_bias_logit_scores_hook(self, input_batch: Tensor) -> Tensor:
        return self.bias_router_model(input_batch)


class RouterLayer(Module):
    def __init__(
        self,
        hidden_dim: int,
        activation: Module,
        residual_flag: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.residual_flag = residual_flag
        self.layer = Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, input_batch: Tensor):
        output = self.layer(input_batch)
        output = self.activation(output)
        if self.residual_flag:
            output = output + input_batch
        return output


class VectorChoiceRouterModel(RouterModel):
    def __init__(
        self,
        cfg: "ModelConfig",
        input_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        bias_flag: Optional[bool] = None,
        residual_flag: Optional[bool] = None,
        activation: Optional[Module] = None,
        num_layers: Optional[int] = None,
    ) -> None:
        super().__init__(
            cfg,
            input_dim,
            hidden_dim,
            output_dim,
            bias_flag,
            residual_flag,
            activation,
            num_layers,
        )

    def _build_models_hook(self) -> None:
        self.weight_router_model = Parameter(
            randn(self.input_dim, self.input_dim, self.router_output_dim)
        )
        self._initialize_parameters(self.weight_router_model)
        self.bias_router_model = None
        if self.bias_flag:
            self.bias_router_model = Parameter(
                randn(self.output_dim, self.input_dim, self.router_output_dim)
            )
            self._initialize_parameters(self.bias_router_model)

    def _compute_weight_logit_scores_hook(self, input_batch: Tensor) -> Tensor:
        return matmul(input_batch, self.weight_router_model)

    def _compute_bias_logit_scores_hook(self, input_batch: Tensor) -> Tensor:
        return matmul(input_batch, self.bias_router_model)
