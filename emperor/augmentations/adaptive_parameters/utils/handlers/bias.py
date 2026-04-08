import torch

from torch import Tensor
from torch.nn import Sequential
from emperor.base.utils import Module
from emperor.base.layer import Layer, LayerStackConfig
from emperor.augmentations.adaptive_parameters.utils.handlers._validator import (
    BiasHandlerAbstractValidator,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters.config import (
        AdaptiveParameterAugmentationConfig,
    )


class BiasHandlerAbstract(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterAugmentationConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.main_cfg = self._resolve_main_config(self.cfg, cfg)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim

        self.validator = BiasHandlerAbstractValidator(self)

    def _init_model(
        self, overrides: LayerStackConfig | None = None
    ) -> "Layer | Sequential":
        from emperor.linears.utils.stack import LinearLayerStack

        return LinearLayerStack(self.main_cfg, overrides).build_model()


class AffineBiasTransformHandler(BiasHandlerAbstract):
    def __init__(
        self,
        cfg: "AdaptiveParameterAugmentationConfig",
    ):
        super().__init__(cfg)
        overrides = LayerStackConfig(input_dim=self.input_dim, output_dim=2)
        self.scalar_offset_generator = self._init_model(overrides)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        self.validator.ensure_parameters_exist(bias_params)
        parameters = self.scalar_offset_generator(logits)
        bias_scaling_factor, bias_offset = parameters.chunk(2, dim=-1)
        return bias_scaling_factor * bias_params + bias_offset


class ElementwiseBiasHandler(BiasHandlerAbstract):
    def __init__(
        self,
        cfg: "AdaptiveParameterAugmentationConfig",
    ):
        super().__init__(cfg)
        overrides = LayerStackConfig(
            input_dim=self.input_dim, output_dim=self.output_dim
        )
        self.generator_model = self._init_model(overrides)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        self.validator.ensure_parameters_exist(bias_params)
        parameters = self.generator_model(logits)
        return bias_params + parameters


class GatedBiasHandler(BiasHandlerAbstract):
    def __init__(
        self,
        cfg: "AdaptiveParameterAugmentationConfig",
    ):
        super().__init__(cfg)
        overrides = LayerStackConfig(
            input_dim=self.input_dim, output_dim=self.output_dim
        )
        self.gate_generator = self._init_model(overrides)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        self.validator.ensure_parameters_exist(bias_params)
        gate = torch.sigmoid(self.gate_generator(logits))
        return bias_params * gate


class BiasGeneratorHandler(BiasHandlerAbstract):
    def __init__(
        self,
        cfg: "AdaptiveParameterAugmentationConfig",
    ):
        super().__init__(cfg)
        overrides = LayerStackConfig(
            input_dim=self.input_dim, output_dim=self.output_dim
        )
        self.bias_generator = self._init_model(overrides)

    def forward(self, bias_params: None, logits: Tensor) -> Tensor | None:
        return self.bias_generator(logits)


class WeightedBankBiasGeneratorHandler(BiasHandlerAbstract):
    def __init__(
        self,
        cfg: "AdaptiveParameterAugmentationConfig",
    ):
        super().__init__(cfg)
        self.bias_bank_expansion_factor = self.cfg.bias_bank_expansion_factor
        self.weight_bank = self._init_parameter_bank(
            (self.bias_bank_expansion_factor, self.output_dim)
        )
        overrides = LayerStackConfig(input_dim=self.input_dim, output_dim=self.bias_bank_expansion_factor)
        self.distribution_generator = self._init_model(overrides)

    def forward(self, bias_params: None, logits: Tensor) -> Tensor:
        bank_logits = self.distribution_generator(logits)
        bank_distribution = torch.softmax(bank_logits, dim=-1)
        return torch.matmul(bank_distribution, self.weight_bank)
