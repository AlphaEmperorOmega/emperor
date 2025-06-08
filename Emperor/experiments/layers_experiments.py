import random
from enum import Enum
from Emperor.components.parameter_generators.layers import (
    DefaultLinearLayer,
    GeneratorParameterLayer,
    MatrixParameterLayer,
    ParameterLayerBase,
    VectorParameterLayer,
)

from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class ParameterLayerPresetFactory:
    def __init__(
        self,
        model: Type["ParameterLayerBase"],
        cfg: "ModelConfig",
    ):
        self.model = model
        self.cfg = cfg

    def __create_model(self) -> ParameterLayerBase:
        return self.model(self.cfg)

    def __set_topk(self, topk: int) -> None:
        self.cfg.sampler_model_config.top_k = topk
        self.cfg.mixture_model_config.top_k = topk

    def __set_noisy_topk_flag(self, flag: bool = False) -> None:
        self.cfg.router_model_config.noisy_topk_flag = flag
        self.cfg.sampler_model_config.noisy_topk_flag = flag

    def create_sparse_layer(self) -> ParameterLayerBase:
        self.__set_topk(1)
        return self.__create_model()

    def create_topk_layer(self, topk: int = 3) -> ParameterLayerBase:
        self.__set_topk(topk)
        return self.__create_model()

    def create_full_mixture_layer(self) -> ParameterLayerBase:
        full_mixture = self.cfg.mixture_model_config.depth_dim
        self.__set_topk(full_mixture)
        return self.__create_model()

    def create_random_topk_layer(self) -> ParameterLayerBase:
        max_k = self.cfg.mixture_model_config.depth_dim
        chosen_k = random.randint(1, max_k)
        if chosen_k == max_k:
            self.cfg.mixture_model_config.weighted_parameters_flag = True

        self.__set_topk(chosen_k)
        return self.__create_model()

    def create_sparse_threshold_layer(
        self, threshold: float = 0.1
    ) -> ParameterLayerBase:
        self.cfg.sampler_model_config.threshold = threshold
        return self.create_sparse_layer()

    def create_topk_threshold_layer(
        self, topk: int = 3, threshold: float = 0.1
    ) -> ParameterLayerBase:
        self.cfg.sampler_model_config.threshold = threshold
        return self.create_topk_layer(topk)

    def create_full_mixture_threshold_layer(
        self, threshold: float = 0.1
    ) -> ParameterLayerBase:
        self.cfg.sampler_model_config.threshold = threshold
        return self.create_full_mixture_layer()

    def create_sparse_noisy_topk_layer(self) -> ParameterLayerBase:
        self.__set_noisy_topk_flag(True)
        return self.create_sparse_layer()

    def create_topk_noisy_topk_layer(self, topk: int = 3) -> ParameterLayerBase:
        self.__set_noisy_topk_flag(True)
        return self.create_topk_layer(topk)

    def create_full_mixture_noisy_topk_layer(self) -> ParameterLayerBase:
        self.__set_noisy_topk_flag(True)
        return self.create_full_mixture_layer()


class ParameterLayerFactory(Enum):
    DEFAULT = DefaultLinearLayer
    VECTOR = VectorParameterLayer
    MATRIX = MatrixParameterLayer
    GENERATOR = GeneratorParameterLayer

    def create(
        self,
        cfg: "ModelConfig",
    ) -> ParameterLayerBase:
        return self.value(cfg)


class ParameterLayerPreset(Enum):
    SPARSE = "create_sparse_layer"
    TOPK = "create_topk_layer"
    FULL_MIXTURE = "create_full_mixture_layer"
    RANDOM_TOPK = "create_random_topk_layer"
    SPARSE_THRESHOLD = "create_sparse_threshold_layer"
    TOPK_THRESHOLD = "create_topk_threshold_layer"
    FULL_MIXTURE_THRESHOLD = "create_full_mixture_threshold_layer"
    SPARSE_NOISY_TOPK = "create_sparse_noisy_topk_layer"
    TOPK_NOISY_TOPK = "create_topk_noisy_topk_layer"
    FULL_MIXTURE_NOISY_TOPK = "create_full_mixture_noisy_topk_layer"

    def create(
        self,
        layer_type: "ParameterLayerBase",
        cfg: "ModelConfig",
    ) -> ParameterLayerBase:
        try:
            preset_factory = ParameterLayerPresetFactory(layer_type.value, cfg)
            return getattr(preset_factory, self.value)()
        except AttributeError:
            raise ValueError(
                f"Method {self.value} is not defined in ParameterLayerFactory."
            )
