import torch

from typing import cast
from dataclasses import dataclass, field
from torch import Tensor
from torch.nn import Sequential
from emperor.base.utils import Module, ConfigBase
from emperor.base.layer import Layer, LayerStackConfig
from emperor.augmentations.adaptive_parameters.options import DynamicBiasOptions
from emperor.augmentations.adaptive_parameters.core.handlers._validator import (
    BiasHandlerAbstractValidator,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters.config import (
        AdaptiveParameterAugmentationConfig,
    )


@dataclass
class BiasHandlerConfig(ConfigBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Input dimension of the bias transformation."},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Output dimension of the bias transformation."},
    )
    bias_flag: bool | None = field(
        default=None,
        metadata={"help": "Whether the linear layer has a bias parameter."},
    )
    bias_option: DynamicBiasOptions | None = field(
        default=None,
        metadata={"help": "Input-dependent adjustment of the bias vector."},
    )
    bank_expansion_factor: int | None = field(
        default=None,
        metadata={"help": "Size of the weight bank for WEIGHTED_BANK bias option."},
    )
    model_config: LayerStackConfig | None = field(
        default=None,
        metadata={"help": "Layer stack configuration for the internal generator network."},
    )

    def build(
        self, overrides: "ConfigBase | None" = None
    ) -> "BiasHandlerAbstract":
        if self.bias_option is None:
            raise ValueError("`bias_option` must be set before building the handler")
        handler_cls = BiasHandlerAbstract.resolve(self.bias_option)
        return handler_cls(self, cast("BiasHandlerConfig | None", overrides))


class BiasHandlerAbstract(Module):
    _registry: dict[DynamicBiasOptions, type["BiasHandlerAbstract"]] = {}

    @classmethod
    def register(cls, option: DynamicBiasOptions):
        def decorator(handler_cls: type["BiasHandlerAbstract"]):
            cls._registry[option] = handler_cls
            return handler_cls

        return decorator

    @classmethod
    def resolve(cls, option: DynamicBiasOptions) -> type["BiasHandlerAbstract"]:
        if option not in cls._registry:
            raise ValueError(f"No handler registered for bias option: {option}")
        return cls._registry[option]

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
