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
        "Input dimensionality of the dynamic bias module."
    )
    output_dim: int | None = optional_field(
        "Output dimensionality of the dynamic bias module."
    )
    bias_flag: bool | None = optional_field(
        "Indicates whether the associated linear layer includes a bias parameter."
    )
    model_type: DynamicBiasOptions | None = optional_field(
        "Dynamic bias strategy used to generate input-dependent bias updates."
    )
    bank_expansion_factor: int | None = optional_field(
        "Number of entries in the bank used by the WEIGHTED_BANK bias strategy."
    )
    model_config: LayerStackConfig | None = optional_field(
        "Configuration for the internal generator network."
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
        self.bias_flag = self.cfg.bias_flag
        self.model_type = self.cfg.model_type
        self.bank_expansion_factor = self.cfg.bank_expansion_factor

        self.validator = DynamicBiasAbstractValidator(self)

    def _init_generator_model(
        self, overrides: LayerStackConfig | None = None
    ) -> "Layer | Sequential":
        from emperor.linears.core.stack import LinearLayerStack

        return LinearLayerStack(self.cfg.model_config, overrides).build_model()

    def _init_output_generator(self, output_dim: int) -> "Layer | Sequential":
        overrides = LayerStackConfig(input_dim=self.input_dim, output_dim=output_dim)
        return self._init_generator_model(overrides)

    def _require_bias_params(self, bias_params: Tensor | None) -> Tensor:
        self.validator.ensure_parameters_exist(bias_params)
        return bias_params


@DynamicBiasAbstract.register(DynamicBiasOptions.SCALE_AND_OFFSET)
class AffineTransformDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: DynamicBiasConfig,
        overrides: DynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        affine_parameter_dim = 2
        self.scalar_offset_generator = self._init_output_generator(
            affine_parameter_dim
        )

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        bias_params = self._require_bias_params(bias_params)
        affine_parameters = self.scalar_offset_generator(logits)
        bias_scale, bias_offset = affine_parameters.chunk(2, dim=-1)
        return bias_scale * bias_params + bias_offset


@DynamicBiasAbstract.register(DynamicBiasOptions.ELEMENT_WISE_OFFSET)
class ElementwiseDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: DynamicBiasConfig,
        overrides: DynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.generator_model = self._init_output_generator(self.output_dim)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        bias_params = self._require_bias_params(bias_params)
        generated_bias_offset = self.generator_model(logits)
        return bias_params + generated_bias_offset


@DynamicBiasAbstract.register(DynamicBiasOptions.GATED)
class GatedDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: DynamicBiasConfig,
        overrides: DynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.gate_generator = self._init_output_generator(self.output_dim)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        bias_params = self._require_bias_params(bias_params)
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
        self.bias_generator = self._init_output_generator(self.output_dim)

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
        self.weight_bank = self._init_parameter_bank(
            (self.bank_expansion_factor, self.output_dim)
        )
        self.distribution_generator = self._init_output_generator(
            self.bank_expansion_factor
        )

    def forward(self, bias_params: None, logits: Tensor) -> Tensor:
        bank_logits = self.distribution_generator(logits)
        bank_distribution = torch.softmax(bank_logits, dim=-1)
        return torch.matmul(bank_distribution, self.weight_bank)
