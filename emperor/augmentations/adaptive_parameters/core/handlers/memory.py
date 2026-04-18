import torch

from typing import cast
from torch import Tensor
from torch.nn import Sequential
from dataclasses import dataclass, field
from emperor.base.utils import Module, ConfigBase
from emperor.base.layer import Layer, LayerStackConfig
from emperor.augmentations.adaptive_parameters.options import (
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)


@dataclass
class MemoryHandlerConfig(ConfigBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Input dimension of the memory transformation."},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Output dimension of the memory transformation."},
    )
    memory_option: LinearMemoryOptions | None = field(
        default=None,
        metadata={
            "help": "Blends a learned memory representation with the linear layer input or output."
        },
    )
    memory_size_option: LinearMemorySizeOptions | None = field(
        default=None,
        metadata={"help": "Size of the learned memory representation."},
    )
    memory_position_option: LinearMemoryPositionOptions | None = field(
        default=None,
        metadata={"help": "Controls when memory is applied in the computation."},
    )
    model_config: LayerStackConfig | None = field(
        default=None,
        metadata={"help": "Layer stack configuration for the internal generator network."},
    )

    def build(
        self, overrides: "ConfigBase | None" = None
    ) -> "MemoryHandlerAbstract":
        if self.memory_option is None:
            raise ValueError("`memory_option` must be set before building the handler")
        handler_cls = MemoryHandlerAbstract.resolve(self.memory_option)
        return handler_cls(self, cast("MemoryHandlerConfig | None", overrides))


class MemoryHandlerAbstract(Module):
    _registry: dict[LinearMemoryOptions, type["MemoryHandlerAbstract"]] = {}

    @classmethod
    def register(cls, option: LinearMemoryOptions):
        def decorator(handler_cls: type["MemoryHandlerAbstract"]):
            cls._registry[option] = handler_cls
            return handler_cls

        return decorator

    @classmethod
    def resolve(cls, option: LinearMemoryOptions) -> type["MemoryHandlerAbstract"]:
        if option not in cls._registry:
            raise ValueError(f"No handler registered for memory option: {option}")
        return cls._registry[option]

    def __init__(
        self,
        cfg: MemoryHandlerConfig,
        overrides: MemoryHandlerConfig | None = None,
    ):
        super().__init__()
        self.cfg: MemoryHandlerConfig = self._override_config(cfg, overrides)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.memory_size_option = self.cfg.memory_size_option
        self.memory_position_option = self.cfg.memory_position_option

    def _validate_memory_option(self):
        if self.memory_size_option == LinearMemorySizeOptions.DISABLED:
            raise ValueError(
                "The `memory_size_option` cannot be set to LinearMemorySizeOptions.DISABLED if this handler is used."
            )

    def _get_memory_dim(self) -> int:
        if self.memory_position_option == LinearMemoryPositionOptions.BEFORE_AFFINE:
            return self.input_dim
        return self.output_dim

    def _init_model(self, config, overrides) -> "Layer | Sequential":
        from emperor.linears.core.stack import LinearLayerStack

        return LinearLayerStack(config, overrides).build_model()


@MemoryHandlerAbstract.register(LinearMemoryOptions.FUSION)
class MemoryFusionHandler(MemoryHandlerAbstract):
    def __init__(
        self,
        cfg: MemoryHandlerConfig,
        overrides: MemoryHandlerConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.memory_model = self.__init_memory_model()
        self.compression_model = self.__init_compression_model()
        self._validate_memory_option()

    def __init_memory_model(self) -> "Layer | Sequential":
        input_dim = self._get_memory_dim()
        output_dim = self.memory_size_option.value
        layer_overrides = LayerStackConfig(input_dim=input_dim, output_dim=output_dim)
        return self._init_model(self.cfg.model_config, layer_overrides)

    def __init_compression_model(self) -> "Layer | Sequential":
        dim = self._get_memory_dim()
        compression_input_dim = dim + self.memory_size_option.value
        layer_overrides = LayerStackConfig(
            input_dim=compression_input_dim,
            hidden_dim=dim,
            output_dim=dim,
        )
        return self._init_model(self.cfg.model_config, layer_overrides)

    def forward(self, logits: Tensor) -> Tensor:
        memory = self.memory_model(logits)
        combined = torch.cat([logits, memory], dim=-1)
        return self.compression_model(combined)


@MemoryHandlerAbstract.register(LinearMemoryOptions.WEIGHTED)
class WeightedMemoryHandler(MemoryHandlerAbstract):
    def __init__(
        self,
        cfg: MemoryHandlerConfig,
        overrides: MemoryHandlerConfig | None = None,
    ):
        super().__init__(cfg, overrides)

        self.memory_model = self.__init_memory_model()
        self.memory_weight_model = self.__init_weight_model()
        self._validate_memory_option()

    def __init_memory_model(self) -> "Layer | Sequential":
        dim = self._get_memory_dim()
        layer_overrides = LayerStackConfig(input_dim=dim, output_dim=dim)
        return self._init_model(self.cfg.model_config, layer_overrides)

    def __init_weight_model(self) -> "Layer | Sequential":
        dim = self._get_memory_dim() * 2
        layer_overrides = LayerStackConfig(input_dim=dim, hidden_dim=dim, output_dim=2)
        return self._init_model(self.cfg.model_config, layer_overrides)

    def forward(self, logits: Tensor) -> Tensor:
        memory = self.memory_model(logits)
        weights = self.__compute_weights(logits, memory)
        merged_logits_memory = self.__reshape_and_concat(logits, memory)
        return self.__blend_inputs_and_memory(merged_logits_memory, weights)

    def __compute_weights(self, logits: Tensor, memory: Tensor) -> Tensor:
        combined = torch.cat([logits, memory], dim=-1)
        weight_logits = self.memory_weight_model(combined)
        weights = torch.softmax(weight_logits, dim=-1)
        return weights.unsqueeze(-1)

    def __reshape_and_concat(self, logits: Tensor, memory: Tensor) -> Tensor:
        logits = logits.unsqueeze(-2)
        memory = memory.unsqueeze(-2)
        return torch.cat((logits, memory), dim=-2)

    def __blend_inputs_and_memory(
        self, input_and_memory: Tensor, weights: Tensor
    ) -> Tensor:
        weighted_logits = input_and_memory * weights
        return torch.sum(weighted_logits, dim=-2)
