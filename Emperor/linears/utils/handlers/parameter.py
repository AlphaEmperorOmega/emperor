import copy
import torch

from torch import Tensor
from Emperor.base.utils import Module
from Emperor.base.layer import LayerStack, LayerStackConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig
    from Emperor.linears.utils.layers import DynamicLinearLayerConfig


class DepthMappingLayer(Module):
    def __init__(self, cfg: "ModelConfig"):
        super().__init__()
        self.cfg: "DynamicLinearLayerConfig" = getattr(cfg, "linear_layer_config", cfg)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.generator_depth = self.cfg.generator_depth.value
        self.weight_params, self.bias_params = self.__init_parameter_bank()
        self.__ensure_generator_depth_is_valid()

    def __ensure_generator_depth_is_valid(self):
        if self.generator_depth == 0:
            raise ValueError("generator_depth cannot be 0")

    def __init_parameter_bank(self):
        input_weight_shape = (self.generator_depth, self.input_dim, self.output_dim)
        weight_bank = self._init_parameter_bank(input_weight_shape)
        bias_shape = (self.generator_depth, self.output_dim)
        bias_bank = self._init_parameter_bank(bias_shape)
        return weight_bank, bias_bank

    def forward(
        self,
        input_batch: Tensor,
    ) -> Tensor:
        operation = "bkj,kji->bki"
        output = torch.einsum(operation, input_batch, self.weight_params)
        return output + self.bias_params


class DepthMappingLayerStack(Module):
    def __init__(
        self,
        cfg: "ModelConfig",
        overrides: "LayerStackConfig | None" = None,
    ):
        super().__init__()
        self.cfg = cfg
        print(self.cfg)
        self.identifier = "layer_stack_config"
        updated_config = self.__update_config()
        self.generator_depth = self.cfg.linear_layer_config.generator_depth.value
        self.model = LayerStack(updated_config, overrides).build_model()

    def __update_config(self) -> "ModelConfig | LayerStackConfig":
        config = getattr(self.cfg, self.identifier, self.cfg)
        overrides = self.__override_config()
        updated_config = self._overwrite_config(config, overrides)

        if not hasattr(self.cfg, self.identifier):
            return updated_config

        # TODO: Update the _overwrite_config to handle nested configs
        # in order to remove the copy.deepcopy below
        c = copy.deepcopy(self.cfg)
        c.layer_stack_config = updated_config
        return c

    def __override_config(self) -> LayerStackConfig:
        from Emperor.base.enums import BaseOptions

        class UpdatedLinearLayerOptions(BaseOptions):
            DEPTH_MAPPING = DepthMappingLayer

        return LayerStackConfig(model_type=UpdatedLinearLayerOptions.DEPTH_MAPPING)

    def forward(self, input_batch: Tensor) -> Tensor:
        if not input_batch.dim() == 2:
            raise ValueError("Input batch must be a 2D tensor")

        input_batch = input_batch.unsqueeze(1)
        input_batch = input_batch.repeat(1, self.generator_depth, 1)
        return self.model(input_batch)
