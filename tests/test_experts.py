import unittest
from Emperor.layers.layers import ParameterLayerConfig
from Emperor.layers.utils.mixture import MixtureConfig
from Emperor.layers.utils.routers import RouterConfig
from Emperor.layers.utils.samplers import SamplerConfig
from Emperor.config import ModelConfig


class TestMixtureOfExperts(unittest.TestCase):
    def setUp(self):
        # MODEL WISE CONFI
        BATCH_SIZE = 2
        INPUT_DIM = 4
        HIDDEN_DIM = 5
        OUTPUT_DIM = 6
        GATHER_FREQUENCY_FLAG = False

        # PARAMETER GENRETOR ROUTER OPITONS
        ROUTER_INPUT_DIM = HIDDEN_DIM
        ROUTER_HIDDEN_DIM = 8
        ROUTER_OUTPUT_DIM = 9
        ROUTER_NOISY_TOPK_FLAG = False
        ROUTER_ACTIVATION_FUNCTION = nn.ReLU()
        ROUTER_NUM_LAYERNUM_LAYERSS = 5
        ROUTER_DYNAMIC_LINEAR_MODEL_FLAG = False

        # PARAMETER GENRETOR SAMPLER OPITONS
        SAMPLER_TOP_K = 3
        SAMPLER_THRESHOLD = 0.0
        SAMPLER_FILTER_THRESHOLD = False
        SAMPLER_DYNAMIC_TOPK_THRESHOLD = (1 / ROUTER_OUTPUT_DIM) * 1e-6
        SAMPLER_NUM_TOPK_SAMPLES = 0
        SAMPLER_NORMALIZE_PROBABILITIES_FLAG = False
        SAMPLER_NOISY_TOPK_FLAG = ROUTER_NOISY_TOPK_FLAG
        SAMPLER_ROUTER_OUTPUT_DIM = ROUTER_OUTPUT_DIM
        SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT = 0.0
        SAMPLER_SWITCH_WEIGHT = 0.0
        SAMPLER_ZERO_CENTRED_WEIGHT = 0.0
        SAMPLER_MUTUAL_INFORMATION_WEIGHT = 0.0

        # PARAMETER GENRETOR MIXTURE OPITONS
        MIXTURE_INPUT_DIM = ROUTER_INPUT_DIM
        MIXTURE_OUTPUT_DIM = ROUTER_OUTPUT_DIM
        MIXTURE_DEPTH_DIM = ROUTER_OUTPUT_DIM
        MIXTURE_TOP_K = SAMPLER_TOP_K
        MIXTURE_WEIGHTED_PARAMETERS_FLAG = False
        MIXTURE_BIAS_PARAMETERS_FLAG = False
        MIXTURE_ROUTER_OUTPUT_DIM = ROUTER_OUTPUT_DIM
        MIXTURE_CROSS_DIAGONAL_FLAG = False

        # PARAMETER GENERATOR OPTIONS
        PARAMETER_GENERATOR_BIAS_PARAMETER_FLAG = MIXTURE_WEIGHTED_PARAMETERS_FLAG

        self.cfg = ModelConfig(
            batch_size=BATCH_SIZE,
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            gather_frequency_flag=GATHER_FREQUENCY_FLAG,
            router_model_config=RouterConfig(
                input_dim=ROUTER_INPUT_DIM,
                hidden_dim=ROUTER_HIDDEN_DIM,
                output_dim=ROUTER_OUTPUT_DIM,
                noisy_topk_flag=ROUTER_NOISY_TOPK_FLAG,
                activation=ROUTER_ACTIVATION_FUNCTION,
                num_layers=ROUTER_NUM_LAYERNUM_LAYERSS,
                diagonal_linear_model_flag=ROUTER_DYNAMIC_LINEAR_MODEL_FLAG,
                residual_flag=False,
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
                bias_parameters_flag=MIXTURE_BIAS_PARAMETERS_FLAG,
                weighted_parameters_flag=MIXTURE_WEIGHTED_PARAMETERS_FLAG,
                num_experts=MIXTURE_ROUTER_OUTPUT_DIM,
                dynamic_diagonal_params_flag=MIXTURE_CROSS_DIAGONAL_FLAG,
            ),
            parameter_generator_model_config=ParameterLayerConfig(
                bias_parameters_flag=PARAMETER_GENERATOR_BIAS_PARAMETER_FLAG,
                time_tracker_flag=False,
                dynamic_diagonal_params_flag=False,
            ),
        )

        self.parameter_generator_cfg = self.cfg.mixture_model_config
