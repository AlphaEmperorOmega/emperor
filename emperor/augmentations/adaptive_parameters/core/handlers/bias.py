import torch

from typing import cast
from torch import Tensor
from dataclasses import dataclass
from torch.nn import Sequential
from emperor.base.utils import ConfigBase, Module, optional_field
from emperor.base.registry import subclass_registry
from emperor.base.layer import Layer, LayerStackConfig
from emperor.augmentations.adaptive_parameters.options import DynamicBiasOptions
from emperor.augmentations.adaptive_parameters.core.handlers._validator import (
    BiasHandlerAbstractValidator,
)


@dataclass
class BiasHandlerConfig(ConfigBase):
    input_dim: int | None = optional_field(
        "Input dimension of the bias transformation."
    )
    output_dim: int | None = optional_field(
        "Output dimension of the bias transformation."
    )
    bias_flag: bool | None = optional_field(
        "Whether the linear layer has a bias parameter."
    )
    model_type: DynamicBiasOptions | None = optional_field(
        "Input-dependent adjustment of the bias vector."
    )
    bank_expansion_factor: int | None = optional_field(
        "Size of the weight bank for WEIGHTED_BANK bias option."
    )
    model_config: LayerStackConfig | None = optional_field(
        "Layer stack configuration for the internal generator network."
    )

    def build(self, overrides: "ConfigBase | None" = None) -> "BiasHandlerAbstract":
        if self.model_type is None:
            raise ValueError("`model_type` must be set before building the handler")
        handler_cls = BiasHandlerAbstract.resolve(self.model_type)
        return handler_cls(self, cast("BiasHandlerConfig | None", overrides))


@subclass_registry
class BiasHandlerAbstract(Module):
    def __init__(
        self,
        cfg: BiasHandlerConfig,
        overrides: BiasHandlerConfig | None = None,
    ):
        super().__init__()
        self.cfg: BiasHandlerConfig = self._override_config(cfg, overrides)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim

        self.validator = BiasHandlerAbstractValidator(self)

    def _init_model(
        self, overrides: LayerStackConfig | None = None
    ) -> "Layer | Sequential":
        from emperor.linears.core.stack import LinearLayerStack

        return LinearLayerStack(self.cfg.model_config, overrides).build_model()


@BiasHandlerAbstract.register(DynamicBiasOptions.SCALE_AND_OFFSET)
class AffineBiasTransformHandler(BiasHandlerAbstract):
    def __init__(
        self,
        cfg: BiasHandlerConfig,
        overrides: BiasHandlerConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        layer_overrides = LayerStackConfig(input_dim=self.input_dim, output_dim=2)
        self.scalar_offset_generator = self._init_model(layer_overrides)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        self.validator.ensure_parameters_exist(bias_params)
        parameters = self.scalar_offset_generator(logits)
        bias_scaling_factor, bias_offset = parameters.chunk(2, dim=-1)
        return bias_scaling_factor * bias_params + bias_offset


@BiasHandlerAbstract.register(DynamicBiasOptions.ELEMENT_WISE_OFFSET)
class ElementwiseBiasHandler(BiasHandlerAbstract):
    def __init__(
        self,
        cfg: BiasHandlerConfig,
        overrides: BiasHandlerConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        layer_overrides = LayerStackConfig(
            input_dim=self.input_dim, output_dim=self.output_dim
        )
        self.generator_model = self._init_model(layer_overrides)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        self.validator.ensure_parameters_exist(bias_params)
        parameters = self.generator_model(logits)
        return bias_params + parameters


@BiasHandlerAbstract.register(DynamicBiasOptions.GATED)
class GatedBiasHandler(BiasHandlerAbstract):
    def __init__(
        self,
        cfg: BiasHandlerConfig,
        overrides: BiasHandlerConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        layer_overrides = LayerStackConfig(
            input_dim=self.input_dim, output_dim=self.output_dim
        )
        self.gate_generator = self._init_model(layer_overrides)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        self.validator.ensure_parameters_exist(bias_params)
        gate = torch.sigmoid(self.gate_generator(logits))
        return bias_params * gate


@BiasHandlerAbstract.register(DynamicBiasOptions.DYNAMIC_PARAMETERS)
class BiasGeneratorHandler(BiasHandlerAbstract):
    def __init__(
        self,
        cfg: BiasHandlerConfig,
        overrides: BiasHandlerConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        layer_overrides = LayerStackConfig(
            input_dim=self.input_dim, output_dim=self.output_dim
        )
        self.bias_generator = self._init_model(layer_overrides)

    def forward(self, bias_params: None, logits: Tensor) -> Tensor | None:
        return self.bias_generator(logits)


@BiasHandlerAbstract.register(DynamicBiasOptions.WEIGHTED_BANK)
class WeightedBankBiasGeneratorHandler(BiasHandlerAbstract):
    def __init__(
        self,
        cfg: BiasHandlerConfig,
        overrides: BiasHandlerConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.bank_expansion_factor = self.cfg.bank_expansion_factor
        self.weight_bank = self._init_parameter_bank(
            (self.bank_expansion_factor, self.output_dim)
        )
        layer_overrides = LayerStackConfig(
            input_dim=self.input_dim, output_dim=self.bank_expansion_factor
        )
        self.distribution_generator = self._init_model(layer_overrides)

    def forward(self, bias_params: None, logits: Tensor) -> Tensor:
        bank_logits = self.distribution_generator(logits)
        bank_distribution = torch.softmax(bank_logits, dim=-1)
        return torch.matmul(bank_distribution, self.weight_bank)
