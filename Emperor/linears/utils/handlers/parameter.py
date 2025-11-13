from torch import Tensor
from Emperor.base.utils import Module
from Emperor.linears.utils.enums import DynamicParametersOptions
from Emperor.linears.utils.layers import DynamicLinearLayerConfig


class DepthMappingLayer(Module):
    def __init__(self, cfg: "DynamicLinearLayerConfig"):
        self.cfg = cfg
        self.input_dim = cfg.input_dim
        self.depth_dim = cfg.dynamic_generators_depth
        self.is_initial_layer = False
        self.weight_params, self.bias_params = self.__init_parameter_bank()

    def __init_parameter_bank(self):
        input_weight_shape = (self.depth_dim, self.input_dim, self.input_dim)
        weight_bank = self._init_parameter_bank(input_weight_shape)
        bias_shape = (self.depth_dim, self.input_dim)
        bias_bank = self._init_parameter_bank(bias_shape)
        return weight_bank, bias_bank

    def set_initial_layer(self, is_initial_layer: bool) -> None:
        self.is_initial_layer = bool(is_initial_layer)

    def forward(
        self,
        input_batch: Tensor,
    ) -> Tensor:
        operation = self.__get_operation()
        return (
            torch.einsum(operation, input_batch, self.weight_params) + self.bias_params
        )

    def __get_operation(self) -> str:
        if self.is_initial_layer:
            return "bi,kij->bkj"
        return "bik,kij->bkj"


class DepthMappingLayerStack(Module):
    def __init__(
        self,
        cfg: "LayerStackConfig | ModelConfig",
        overrides: "LayerStackConfig | None" = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.overrides = overrides

        cfg = self.__update_config()
        self.model = LayerStack(cfg)

    def __update_config(self) -> LayerStackConfig:
        config = getattr(self.cfg, "layer_block_stack_config", self.cfg)
        updated_config = self._overwrite_config(config, self.overrides)
        overrides = self.__override_config()
        return self._overwrite_config(updated_config, overrides)

    def __override_config(self) -> LayerStackConfig:
        return LayerStackConfig(
            model_type=DepthMappingLayer,
        )
