import os
import copy
import datetime
import torch.nn as nn
from Emperor.base.utils import Trainer
from Emperor.base.datasets import FashionMNIST
from Emperor.config import BIAS_PARAMETER_FLAG
from Emperor.generators.utils.layers import ParameterLayerConfig
from Emperor.generators.utils.mixture import MixtureConfig
from Emperor.generators.utils.routers import RouterConfig
from Emperor.generators.utils.samplers import SamplerConfig
from Emperor.linears.utils.layers import LinearLayerConfig, DynamicLinearLayerConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class PresetBase:
    def __init__(self):
        self.cfg = None
        self.default_presets = {
            "BATCH_SIZE": 256,
            "SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT": 0.1,
            "SAMPLER_SWITCH_WEIGHT": 0.1,
            "SAMPLER_ZERO_CENTRED_WEIGHT": 0.0,
            "SAMPLER_MUTUAL_INFORMATION_WEIGHT": 0.0,
            "PARAMETER_GENERATOR_DYNAMIC_DIAGONAL_PARAMS_FLAG": True,
            "MIXTURE_ANTI_DIAGONAL_FLAG": True,
            "ROUTER_DIAGONAL_LINEAR_MODEL_FLAG": True,
            "BIAS_PARAMETER_FLAG": True,
            "NUM_EXPERTS": 16,
        }

    def get_preset_config(self) -> "ModelConfig":
        return self.cfg

    @classmethod
    def create(cls):
        return cls()


class SparseWithAuxiliaryPreset(PresetBase):
    def __init__(self):
        super().__init__()
        self.cfg = ModelTrainerBase.config_preset(
            TOP_K=1,
            **self.default_presets,
        )


class TopkWithAuxiliaryPreset(PresetBase):
    def __init__(self):
        super().__init__()
        self.cfg = ModelTrainerBase.config_preset(
            TOP_K=3,
            **self.default_presets,
        )


class SparseNoAuxiliaryPreset(PresetBase):
    def __init__(self):
        super().__init__()
        overrides = copy.deepcopy(self.default_presets)
        overrides.pop("SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT")
        overrides.pop("SAMPLER_SWITCH_WEIGHT")
        self.cfg = ModelTrainerBase.config_preset(
            TOP_K=1,
            MIXTURE_WEIGHTED_PARAMETERS_FLAG=True,
            **overrides,
        )


class TopKNoAuxiliaryPreset(PresetBase):
    def __init__(self):
        super().__init__()
        overrides = copy.deepcopy(self.default_presets)
        overrides.pop("SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT")
        overrides.pop("SAMPLER_SWITCH_WEIGHT")
        self.cfg = ModelTrainerBase.config_preset(
            TOP_K=3,
            MIXTURE_WEIGHTED_PARAMETERS_FLAG=True,
            **overrides,
        )


class SparseAuxiliaryAndWeightedParametersPreset(PresetBase):
    def __init__(self):
        super().__init__()
        self.cfg = ModelTrainerBase.config_preset(
            TOP_K=1,
            MIXTURE_WEIGHTED_PARAMETERS_FLAG=True,
            **self.default_presets,
        )


class TopKAuxiliaryAndWeightedParametersPreset(PresetBase):
    def __init__(self):
        super().__init__()
        self.cfg = ModelTrainerBase.config_preset(
            TOP_K=3,
            MIXTURE_WEIGHTED_PARAMETERS_FLAG=True,
            **self.default_presets,
        )


class FullMixturePreset(PresetBase):
    def __init__(self):
        super().__init__()
        overrides = copy.deepcopy(self.default_presets)
        overrides.pop("SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT")
        overrides.pop("SAMPLER_SWITCH_WEIGHT")

        self.cfg = ModelTrainerBase.config_preset(
            TOP_K=16,
            MIXTURE_WEIGHTED_PARAMETERS_FLAG=True,
            **overrides,
        )


class FullMixtureLayerDynamicMaskPreset(PresetBase):
    def __init__(self):
        super().__init__()
        overrides = copy.deepcopy(self.default_presets)
        overrides.pop("SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT")
        overrides.pop("SAMPLER_SWITCH_WEIGHT")

        self.cfg = ModelTrainerBase.config_preset(
            TOP_K=16,
            THRESHOLD=0.1,
            MIXTURE_WEIGHTED_PARAMETERS_FLAG=True,
            **overrides,
        )


class SparseThresholdPreset(PresetBase):
    def __init__(self):
        super().__init__()
        self.cfg = ModelTrainerBase.config_preset(
            TOP_K=1,
            THRESHOLD=0.1,
            **self.default_presets,
        )


class TopKThresholdPreset(PresetBase):
    def __init__(self):
        super().__init__()
        self.cfg = ModelTrainerBase.config_preset(
            TOP_K=3,
            THRESHOLD=0.1,
            **self.default_presets,
        )


class FullMixtureThresholdPreset(PresetBase):
    def __init__(self):
        super().__init__()
        overrides = copy.deepcopy(self.default_presets)
        overrides.pop("SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT")
        overrides.pop("SAMPLER_SWITCH_WEIGHT")
        self.cfg = ModelTrainerBase.config_preset(
            TOP_K=16,
            THRESHOLD=0.1,
            MIXTURE_WEIGHTED_PARAMETERS_FLAG=True,
            **overrides,
        )


class FullMixtureNoWeightSumDepthTwoPreset(PresetBase):
    def __init__(self):
        super().__init__()
        overrides = copy.deepcopy(self.default_presets)
        overrides.pop("NUM_EXPERTS")
        overrides.pop("SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT")
        overrides.pop("SAMPLER_SWITCH_WEIGHT")

        self.cfg = ModelTrainerBase.config_preset(
            NUM_EXPERTS=2,
            TOP_K=2,
            MIXTURE_WEIGHTED_PARAMETERS_FLAG=False,
            **overrides,
        )


class FullMixtureNoWeightSumDepthThreePreset(PresetBase):
    def __init__(self):
        super().__init__()
        overrides = copy.deepcopy(self.default_presets)
        overrides.pop("NUM_EXPERTS")
        overrides.pop("SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT")
        overrides.pop("SAMPLER_SWITCH_WEIGHT")

        self.cfg = ModelTrainerBase.config_preset(
            NUM_EXPERTS=3,
            TOP_K=3,
            MIXTURE_WEIGHTED_PARAMETERS_FLAG=False,
            **overrides,
        )


class FullMixtureNoWeightSumDepthFourPreset(PresetBase):
    def __init__(self):
        super().__init__()
        overrides = copy.deepcopy(self.default_presets)
        overrides.pop("NUM_EXPERTS")
        overrides.pop("SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT")
        overrides.pop("SAMPLER_SWITCH_WEIGHT")

        self.cfg = ModelTrainerBase.config_preset(
            NUM_EXPERTS=4,
            TOP_K=4,
            MIXTURE_WEIGHTED_PARAMETERS_FLAG=False,
            **overrides,
        )


class FullMixtureNoWeightSumDepthFivePreset(PresetBase):
    def __init__(self):
        super().__init__()
        overrides = copy.deepcopy(self.default_presets)
        overrides.pop("NUM_EXPERTS")
        overrides.pop("SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT")
        overrides.pop("SAMPLER_SWITCH_WEIGHT")

        self.cfg = ModelTrainerBase.config_preset(
            NUM_EXPERTS=5,
            TOP_K=5,
            MIXTURE_WEIGHTED_PARAMETERS_FLAG=False,
            **overrides,
        )


class ModelTrainerBase:
    def __init__(self) -> None:
        pass

    @staticmethod
    def config_preset(
        # BASE CONFIGS
        BATCH_SIZE=256,
        INPUT_DIM=784,
        HIDDEN_DIM=64,
        OUTPUT_DIM=10,
        NUM_EXPERTS=32,
        THRESHOLD=0.0,
        NOISY_TOPK_FLAG=False,
        BIAS_PARAMETER_FLAG=True,
        TOP_K=3,
        GATHER_FREQUENCY_FLAG=False,
        # ROUTER CONFIG OPTIONS
        ROUTER_INPUT_DIM=784,
        ROUTER_NOISY_TOPK_FLAG=False,
        ROUTER_NUM_LAYERS=3,
        ROUTER_DIAGONAL_LINEAR_MODEL_FLAG=False,
        # ROUTER CONFIG OPTIONS
        SAMPLER_NUM_TOPK_SAMPLES=0,
        SAMPLER_FILTER_THRESHOLD=False,
        SAMPLER_NORMALIZE_PROBABILITIES_FLAG=False,
        SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT=0.0,
        SAMPLER_SWITCH_WEIGHT=0.0,
        SAMPLER_ZERO_CENTRED_WEIGHT=0.0,
        SAMPLER_MUTUAL_INFORMATION_WEIGHT=0.0,
        # ROUTER CONFIG OPTIONS
        MIXTURE_WEIGHTED_PARAMETERS_FLAG=False,
        MIXTURE_ANTI_DIAGONAL_FLAG=False,
        # ROUTER CONFIG OPTIONS
        PARAMETER_GENERATOR_TRACK_TIME_FLAG=False,
        PARAMETER_GENERATOR_DYNAMIC_DIAGONAL_PARAMS_FLAG=False,
    ) -> "ModelConfig":
        from Emperor.config import ModelConfig

        ROUTER_INPUT_DIM = HIDDEN_DIM if ROUTER_INPUT_DIM is None else ROUTER_INPUT_DIM
        ROUTER_HIDDEN_DIM = HIDDEN_DIM
        ROUTER_OUTPUT_DIM = NUM_EXPERTS
        ROUTER_RESIDUAL_FLAG = True
        ROUTER_NOISY_TOPK_FLAG = NOISY_TOPK_FLAG
        ROUTER_ACTIVATION = nn.ReLU()

        SAMPLER_TOP_K = TOP_K
        SAMPLER_THRESHOLD = THRESHOLD
        SAMPLER_NOISY_TOPK_FLAG = NOISY_TOPK_FLAG
        SAMPLER_ROUTER_OUTPUT_DIM = NUM_EXPERTS

        MIXTURE_INPUT_DIM = INPUT_DIM
        MIXTURE_OUTPUT_DIM = OUTPUT_DIM
        MIXTURE_DEPTH_DIM = NUM_EXPERTS
        MIXTURE_TOP_K = TOP_K
        MIXTURE_BIAS_PARAMETERS_FLAG = BIAS_PARAMETER_FLAG
        MIXTURE_NUM_EXPERTS = NUM_EXPERTS

        # PARAMETER GENERATOR OPTIONS
        PARAMETER_GENERATOR_BIAS_PARAMETER_FLAG = BIAS_PARAMETER_FLAG

        return ModelConfig(
            batch_size=BATCH_SIZE,
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            gather_frequency_flag=GATHER_FREQUENCY_FLAG,
            router_model_config=RouterConfig(
                input_dim=ROUTER_INPUT_DIM,
                hidden_dim=ROUTER_HIDDEN_DIM,
                output_dim=ROUTER_OUTPUT_DIM,
                residual_flag=ROUTER_RESIDUAL_FLAG,
                noisy_topk_flag=ROUTER_NOISY_TOPK_FLAG,
                activation=ROUTER_ACTIVATION,
                num_layers=ROUTER_NUM_LAYERS,
                diagonal_model_type_flag=ROUTER_DIAGONAL_LINEAR_MODEL_FLAG,
            ),
            sampler_model_config=SamplerConfig(
                top_k=SAMPLER_TOP_K,
                threshold=SAMPLER_THRESHOLD,
                filter_above_threshold=SAMPLER_FILTER_THRESHOLD,
                num_topk_samples=SAMPLER_NUM_TOPK_SAMPLES,
                normalize_probabilities_flag=SAMPLER_NORMALIZE_PROBABILITIES_FLAG,
                noisy_topk_flag=SAMPLER_NOISY_TOPK_FLAG,
                num_experts=SAMPLER_ROUTER_OUTPUT_DIM,
                coefficient_of_variation_loss_weight=SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT,
                switch_loss_weight=SAMPLER_SWITCH_WEIGHT,
                zero_centred_loss_weight=SAMPLER_ZERO_CENTRED_WEIGHT,
                mutual_information_loss_weight=SAMPLER_MUTUAL_INFORMATION_WEIGHT,
            ),
            mixture_model_config=MixtureConfig(
                input_dim=MIXTURE_INPUT_DIM,
                output_dim=MIXTURE_OUTPUT_DIM,
                depth_dim=MIXTURE_DEPTH_DIM,
                top_k=MIXTURE_TOP_K,
                weighted_parameters_flag=MIXTURE_WEIGHTED_PARAMETERS_FLAG,
                bias_parameters_flag=MIXTURE_BIAS_PARAMETERS_FLAG,
                num_experts=MIXTURE_NUM_EXPERTS,
                dynamic_diagonal_params_flag=MIXTURE_ANTI_DIAGONAL_FLAG,
            ),
            parameter_generator_model_config=ParameterLayerConfig(
                bias_parameters_flag=PARAMETER_GENERATOR_BIAS_PARAMETER_FLAG,
                time_tracker_flag=PARAMETER_GENERATOR_TRACK_TIME_FLAG,
                dynamic_diagonal_params_flag=PARAMETER_GENERATOR_DYNAMIC_DIAGONAL_PARAMS_FLAG,
            ),
            linear_layer_config=LinearLayerConfig(
                input_dim=INPUT_DIM,
                output_dim=OUTPUT_DIM,
                bias_flag=True,
                anti_diagonal_flag=True,
                dynamic_bias_flag=True,
            ),
        )


class FashionMNISTModelTrainer(ModelTrainerBase):
    def __init__(
        self,
        model,
        cfg,
        test_dataset_flag: bool = True,
        num_epochs: int = 5,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.data = self.__create_dataset(test_dataset_flag)
        self.model = model
        self.__initialize_monitor()

        if test_dataset_flag:
            assert cfg.batch_size < 64, (
                "The entire mini dataset contains 64 samplers, ensure that the `batch_size` is smaller than 64"
            )
        self.trainer = Trainer(max_epochs=num_epochs)

    def __create_dataset(self, test_dataset_flag) -> FashionMNIST:
        return FashionMNIST(
            batch_size=self.cfg.batch_size,
            testDatasetFalg=test_dataset_flag,
        )

    def train(self) -> None:
        self.trainer.fit(self.model, self.data, print_loss_flag=True)

    def __initialize_monitor(self) -> None:
        trainer_name = (
            self.__class__.__name__
            + "_"
            + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        dataset_name = self.data.__class__.__name__
        model_name = self.model.__class__.__name__
        log_dir = os.path.join(
            trainer_name, dataset_name, model_name, str(self.model.lr)
        )
        self.model.initialize_monitor(log_dir=log_dir)
