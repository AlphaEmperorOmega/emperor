import torch

from torch import Tensor
from emperor.base.utils import Module
from emperor.base.layer import LayerStack, LayerStackConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.linears.utils.config import LinearLayerConfig
    from emperor.augmentations.adaptive_parameters.model import AdaptiveParameterBehaviourConfig


class DepthMappingLayer(Module):
    def __init__(self, cfg: "LinearLayerConfig"):
        super().__init__()
        self.cfg = cfg
        self.main_cfg = self._resolve_main_config(self.cfg, cfg)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.generator_depth = self.main_cfg.generator_depth.value

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
        output = torch.einsum("bkj,kji->bki", input_batch, self.weight_params)
        return output + self.bias_params


class DepthMappingLayerStack(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterBehaviourConfig",
        overrides: "AdaptiveParameterBehaviourConfig | None" = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.main_cfg = self._resolve_main_config(self.cfg, cfg)
        updated_overrides = self.__override_config(overrides)
        self.generator_depth = cfg.generator_depth.value
        self.model = LayerStack(self.main_cfg, updated_overrides).build_model()

    def __override_config(
        self, overrides: "LayerStackConfig | None" = None
    ) -> LayerStackConfig:
        from emperor.base.enums import BaseOptions

        class UpdatedLinearLayerOptions(BaseOptions):
            DEPTH_MAPPING = DepthMappingLayer

        if overrides is None:
            return LayerStackConfig(model_type=UpdatedLinearLayerOptions.DEPTH_MAPPING)
        overrides.model_type = UpdatedLinearLayerOptions.DEPTH_MAPPING
        return overrides

    def forward(self, input_batch: Tensor) -> Tensor:
        if not input_batch.dim() == 2:
            raise ValueError("Input batch must be a 2D tensor")

        input_batch = input_batch.unsqueeze(1)
        input_batch = input_batch.repeat(1, self.generator_depth, 1)
        return self.model(input_batch)
