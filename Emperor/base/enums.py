from enum import Enum
import torch.nn.functional as F


class BaseOptions(Enum):
    def __call__(self, x):
        return self.value(x)

    @classmethod
    def has_option(cls, name: str) -> bool:
        return name in cls.__members__

    @classmethod
    def get_option(cls, name: str | None):
        if name is None:
            return None
        if cls.has_option(name):
            return cls[name]
        raise ValueError(f"Option '{name}' does not exist in {cls.__name__}.")

    @classmethod
    def names(cls) -> list[str]:
        return list(cls.__members__.keys())


class ActivationOptions(BaseOptions):
    NONE = "no_activation"
    RELU = F.relu
    GELU = F.gelu
    SIGMOID = F.sigmoid
    TANH = F.tanh
    LEAKY_RELU = F.leaky_relu
    ELU = F.elu
    SELU = F.selu
    SOFTPLUS = F.softplus
    SOFTSIGN = F.softsign
    SILU = F.silu


class LayerNormPositionOptions(BaseOptions):
    NONE = "no_layer_norm_added"
    DEFAULT = "layer_norm_after_model_output"
    BEFORE = "layer_norm_before_model_processing"
    AFTER = "layer_norm_after_residual_connection"
