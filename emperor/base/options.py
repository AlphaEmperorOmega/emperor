from enum import Enum, member
import torch.nn.functional as F


class BaseOptions(Enum):
    def __call__(self, x):
        return self.value(x)

    @classmethod
    def cli_name(cls, name: str) -> str:
        return name.lower().replace("_", "-")

    @classmethod
    def has_option(cls, name: str) -> bool:
        return cls._option_name(name) is not None

    @classmethod
    def _option_name(cls, name: str | None) -> str | None:
        if name is None:
            return None
        normalized_name = cls.cli_name(name)
        for option_name in cls.__members__:
            if normalized_name in {option_name, cls.cli_name(option_name)}:
                return option_name
        return None

    @classmethod
    def get_option(cls, name: str | None):
        option_name = cls._option_name(name)
        if option_name is None:
            if name is None:
                return None
            raise ValueError(f"Option '{name}' does not exist in {cls.__name__}.")
        return cls[option_name]

    @classmethod
    def cli_names(cls) -> list[str]:
        return [cls.cli_name(name) for name in cls.__members__]

    @classmethod
    def names(cls) -> list[str]:
        return list(cls.__members__.keys())

    @classmethod
    def display_names(cls) -> list[str]:
        return cls.cli_names()

    @classmethod
    def legacy_get_option(cls, name: str | None):
        if name is None:
            return None
        if cls.has_option(name):
            return cls.get_option(name)
        raise ValueError(f"Option '{name}' does not exist in {cls.__name__}.")


class ActivationOptions(BaseOptions):
    DISABLED = 0
    RELU = member(F.relu)
    GELU = member(F.gelu)
    SIGMOID = member(F.sigmoid)
    TANH = member(F.tanh)
    LEAKY_RELU = member(F.leaky_relu)
    ELU = member(F.elu)
    SELU = member(F.selu)
    SOFTPLUS = member(F.softplus)
    SOFTSIGN = member(F.softsign)
    SILU = member(F.silu)
    MISH = member(F.mish)


class LayerNormPositionOptions(BaseOptions):
    DISABLED = 0
    DEFAULT = 1
    BEFORE = 2
    AFTER = 3


class LastLayerBiasOptions(BaseOptions):
    DEFAULT = 0
    DISABLED = 1
    ENABLED = 2
