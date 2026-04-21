import torch

from torch import Tensor
from dataclasses import dataclass
from torch.nn import Sequential
from emperor.base.utils import ConfigBase, Module, optional_field
from emperor.base.registry import subclass_registry
from emperor.base.layer import Layer, LayerStackConfig
from emperor.augmentations.adaptive_parameters.options import DynamicBiasOptions
from emperor.augmentations.adaptive_parameters.core._validator import (
    DynamicBiasAbstractValidator,
)


@dataclass
class DynamicBiasConfig(ConfigBase):
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

    def _registry_owner(self) -> type:
        return DynamicBiasAbstract


@subclass_registry
class DynamicBiasAbstract(Module):
    def __init__(
        self,
        cfg: DynamicBiasConfig,
        overrides: DynamicBiasConfig | None = None,
    ):
        super().__init__()
        self.cfg: DynamicBiasConfig = self._override_config(cfg, overrides)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim

        self.validator = DynamicBiasAbstractValidator(self)

    def _init_model(
        self, overrides: LayerStackConfig | None = None
    ) -> "Layer | Sequential":
        from emperor.linears.core.stack import LinearLayerStack

        return LinearLayerStack(self.cfg.model_config, overrides).build_model()


@DynamicBiasAbstract.register(DynamicBiasOptions.SCALE_AND_OFFSET)
class AffineTransformDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: DynamicBiasConfig,
        overrides: DynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        layer_overrides = LayerStackConfig(input_dim=self.input_dim, output_dim=2)
        self.scalar_offset_generator = self._init_model(layer_overrides)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        self.validator.ensure_parameters_exist(bias_params)
        parameters = self.scalar_offset_generator(logits)
        bias_scaling_factor, bias_offset = parameters.chunk(2, dim=-1)
        return bias_scaling_factor * bias_params + bias_offset


@DynamicBiasAbstract.register(DynamicBiasOptions.ELEMENT_WISE_OFFSET)
class ElementwiseDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: DynamicBiasConfig,
        overrides: DynamicBiasConfig | None = None,
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


@DynamicBiasAbstract.register(DynamicBiasOptions.GATED)
class GatedDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: DynamicBiasConfig,
        overrides: DynamicBiasConfig | None = None,
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


@DynamicBiasAbstract.register(DynamicBiasOptions.DYNAMIC_PARAMETERS)
class GeneratorDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: DynamicBiasConfig,
        overrides: DynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        layer_overrides = LayerStackConfig(
            input_dim=self.input_dim, output_dim=self.output_dim
        )
        self.bias_generator = self._init_model(layer_overrides)

    def forward(self, bias_params: None, logits: Tensor) -> Tensor | None:
        return self.bias_generator(logits)


@DynamicBiasAbstract.register(DynamicBiasOptions.WEIGHTED_BANK)
class WeightedBankDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: DynamicBiasConfig,
        overrides: DynamicBiasConfig | None = None,
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
