import torch.nn as nn
from Emperor.base.utils import Trainer
from Emperor.base.datasets import FashionMNIST
from Emperor.components.parameter_generators.layers import (
    ParameterLayerBase,
    ParameterLayerConfig,
)

from typing import TYPE_CHECKING

from Emperor.components.parameter_generators.utils.mixture import MixtureConfig
from Emperor.components.parameter_generators.utils.routers import RouterConfig
from Emperor.components.parameter_generators.utils.samplers import SamplerConfig

if TYPE_CHECKING:
    from Emperor.config import ModelConfig
    from Emperor.experiments.layers.layers_factories import ParameterLayerOptions


class ParameterLayerPresetFactory:
    def __init__(
        self,
        model: "ParameterLayerOptions",
        cfg: "ModelConfig",
    ):
        self.model = model.value
        self.cfg = cfg

    def __create_model(self) -> "ParameterLayerBase":
        return self.model(self.cfg)

    def __set_topk(self, topk: int) -> None:
        self.cfg.sampler_model_config.top_k = topk
        self.cfg.mixture_model_config.top_k = topk

    def __set_noisy_topk_flag(self, flag: bool = False) -> None:
        self.cfg.router_model_config.noisy_topk_flag = flag
        self.cfg.sampler_model_config.noisy_topk_flag = flag

    def create_sparse_layer(self) -> "ParameterLayerBase":
        self.__set_topk(1)
        return self.__create_model()

    def create_topk_layer(self, topk: int = 3) -> "ParameterLayerBase":
        self.__set_topk(topk)
        return self.__create_model()

    def create_full_mixture_layer(self) -> "ParameterLayerBase":
        full_mixture = self.cfg.mixture_model_config.depth_dim
        self.__set_topk(full_mixture)
        return self.__create_model()

    def create_random_topk_layer(self) -> "ParameterLayerBase":
        max_k = self.cfg.mixture_model_config.depth_dim
        chosen_k = random.randint(1, max_k)
        if chosen_k == max_k:
            self.cfg.mixture_model_config.weighted_parameters_flag = True

        self.__set_topk(chosen_k)
        return self.__create_model()

    def create_sparse_threshold_layer(
        self, threshold: float = 0.1
    ) -> "ParameterLayerBase":
        self.cfg.sampler_model_config.threshold = threshold
        return self.create_sparse_layer()

    def create_topk_threshold_layer(
        self, topk: int = 3, threshold: float = 0.1
    ) -> "ParameterLayerBase":
        self.cfg.sampler_model_config.threshold = threshold
        return self.create_topk_layer(topk)

    def create_full_mixture_threshold_layer(
        self, threshold: float = 0.1
    ) -> "ParameterLayerBase":
        self.cfg.sampler_model_config.threshold = threshold
        return self.create_full_mixture_layer()

    def create_sparse_noisy_topk_layer(self) -> "ParameterLayerBase":
        self.__set_noisy_topk_flag(True)
        return self.create_sparse_layer()

    def create_topk_noisy_topk_layer(self, topk: int = 3) -> "ParameterLayerBase":
        self.__set_noisy_topk_flag(True)
        return self.create_topk_layer(topk)

    def create_full_mixture_noisy_topk_layer(self) -> "ParameterLayerBase":
        self.__set_noisy_topk_flag(True)
        return self.create_full_mixture_layer()

    def set_layer_input_dim(self, input_dim: int) -> "ParatermLayerPresetFactory":
        self.cfg.input_dim = input_dim
        self.cfg.router_model_config.input_dim = input_dim
        self.cfg.mixture_model_config.input_dim = input_dim
        return self

    def set_layer_hidden_dim(self, hidden_dim: int) -> "ParatermLayerPresetFactory":
        self.cfg.hidden_dim = hidden_dim
        self.cfg.router_model_config.hidden_dim = hidden_dim
        self.cfg.mixture_model_config.router_output_dim = hidden_dim
        return self

    def set_layer_output_dim(self, output_dim: int) -> "ParatermLayerPresetFactory":
        self.cfg.output_dim = output_dim
        self.cfg.mixture_model_config.output_dim = output_dim
        return self


class FashionMNISTModelTrainer:
    def __init__(
        self,
        model,
        cfg,
        test_dataset_flag: bool = True,
    ) -> None:
        self.cfg = cfg
        self.data = self.__create_dataset(test_dataset_flag)
        self.model = model
        self.trainer = Trainer(max_epochs=10)

    def __create_dataset(self, test_dataset_flag) -> FashionMNIST:
        return FashionMNIST(
            batch_size=self.cfg.batch_size,
            testDatasetFalg=test_dataset_flag,
        )

    def train(self) -> None:
        self.trainer.fit(self.model, self.data, printLossFlag=True)

    @staticmethod
    def fashion_minist_model_config() -> "ModelConfig":
        # MODEL WISE CONFI
        BATCH_SIZE = 256
        INPUT_DIM = 784
        HIDDEN_DIM = 64
        OUTPUT_DIM = 10
        DEPTH_DIM = 5
        TOP_K = 3
        GATHER_FREQUENCY_FLAG = False

        # AUXILIARY LOSSES OPITONS
        COEFFICIENT_OF_VARIATION_LOSS_WEIGHT: float = 0.0
        SWITCH_LOSS_WEIGHT: float = 0.0
        ZERO_CENTERED_LOSS_WEIGHT: float = 0.0
        MUTUAL_INFORMATION_LOSS_WEIGHT: float = 0.0

        # PARAMETER GENRETOR ROUTER OPITONS
        ROUTER_INPUT_DIM = INPUT_DIM
        ROUTER_HIDDEN_DIM = HIDDEN_DIM
        ROUTER_OUTPUT_DIM = DEPTH_DIM
        ROUTER_NOISY_TOPK_FLAG = False
        ROUTER_ACTIVATION_FUNCTION = nn.ReLU()
        ROUTER_NUM_LAYERNUM_LAYERSS = 3

        # PARAMETER GENRETOR SAMPLER OPITONS
        SAMPLER_TOP_K = TOP_K
        SAMPLER_THRESHOLD = 0.1
        SAMPLER_NUM_TOPK_SAMPLES = 0
        SAMPLER_NORMALIZE_PROBABILITIES_FLAG = False
        SAMPLER_NOISY_TOPK_FLAG = ROUTER_NOISY_TOPK_FLAG
        SAMPLER_ROUTER_OUTPUT_DIM = ROUTER_OUTPUT_DIM

        # PARAMETER GENRETOR MIXTURE OPITONS
        MIXTURE_INPUT_DIM = ROUTER_INPUT_DIM
        MIXTURE_OUTPUT_DIM = OUTPUT_DIM
        MIXTURE_DEPTH_DIM = ROUTER_OUTPUT_DIM
        MIXTURE_TOP_K = TOP_K
        MIXTURE_WEIGHTED_PARAMETERS_FLAG = False
        MIXTURE_BIAS_PARAMETERS_FLAG = False
        MIXTURE_ROUTER_OUTPUT_DIM = ROUTER_OUTPUT_DIM
        MIXTURE_CROSS_DIAGONAL_FLAG = False

        # PARAMETER GENERATOR OPTIONS
        PARAMETER_GENERATOR_BIAS_PARAMETER_FLAG = MIXTURE_WEIGHTED_PARAMETERS_FLAG

        return ModelConfig(
            batch_size=BATCH_SIZE,
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            gather_frequency_flag=GATHER_FREQUENCY_FLAG,
            coefficient_of_variation_loss_weight=COEFFICIENT_OF_VARIATION_LOSS_WEIGHT,
            switch_loss_weight=SWITCH_LOSS_WEIGHT,
            zero_centered_loss_weight=ZERO_CENTERED_LOSS_WEIGHT,
            mutual_information_loss_weight=MUTUAL_INFORMATION_LOSS_WEIGHT,
            router_model_config=RouterConfig(
                input_dim=ROUTER_INPUT_DIM,
                hidden_dim=ROUTER_HIDDEN_DIM,
                output_dim=ROUTER_OUTPUT_DIM,
                noisy_topk_flag=ROUTER_NOISY_TOPK_FLAG,
                activation=ROUTER_ACTIVATION_FUNCTION,
                num_layers=ROUTER_NUM_LAYERNUM_LAYERSS,
            ),
            sampler_model_config=SamplerConfig(
                top_k=SAMPLER_TOP_K,
                threshold=SAMPLER_THRESHOLD,
                num_topk_samples=SAMPLER_NUM_TOPK_SAMPLES,
                normalize_probabilities_flag=SAMPLER_NORMALIZE_PROBABILITIES_FLAG,
                noisy_topk_flag=SAMPLER_NOISY_TOPK_FLAG,
                router_output_dim=SAMPLER_ROUTER_OUTPUT_DIM,
            ),
            mixture_model_config=MixtureConfig(
                input_dim=MIXTURE_INPUT_DIM,
                output_dim=MIXTURE_OUTPUT_DIM,
                depth_dim=MIXTURE_DEPTH_DIM,
                top_k=MIXTURE_TOP_K,
                bias_parameters_flag=MIXTURE_BIAS_PARAMETERS_FLAG,
                weighted_parameters_flag=MIXTURE_WEIGHTED_PARAMETERS_FLAG,
                router_output_dim=MIXTURE_ROUTER_OUTPUT_DIM,
                cross_diagonal_flag=MIXTURE_CROSS_DIAGONAL_FLAG,
            ),
            parameter_generator_model_config=ParameterLayerConfig(
                bias_parameters_flag=PARAMETER_GENERATOR_BIAS_PARAMETER_FLAG,
                time_tracker_flag=False,
            ),
        )
