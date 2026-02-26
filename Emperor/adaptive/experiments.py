from Emperor.base.enums import ActivationOptions
from Emperor.linears.options import LinearLayerStackOptions
from Emperor.experiments.utils.base import Experiments
from Emperor.adaptive.utils.layers import AdaptiveRouterOptions
from Emperor.adaptive.utils.presets import AdaptiveParameterLayerPresets
from Emperor.adaptive.utils.mixtures.types.utils.enums import ClipParameterOptions
from Emperor.experts.utils.enums import (
    ExpertWeightingPositionOptions,
    InitSamplerOptions,
)
from Emperor.adaptive.options import (
    AdaptiveLayerStackOptions,
    AdaptiveParameterLayerOptions,
)
from Emperor.adaptive.utils.mixtures.options import (
    AdaptiveBiasOptions,
    AdaptiveWeightOptions,
)
from Emperor.behaviours.utils.enums import (
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    DynamicBiasOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.base.layer import LayerStackConfig


class AdaptiveParameterExperiments(Experiments):
    def __init__(
        self,
        mini_datasetset_flag: bool = True,
    ) -> None:
        super().__init__(mini_datasetset_flag)

    def train_model(self, layer_type: AdaptiveParameterLayerOptions):
        preset = AdaptiveParameterExperimentPresets(layer_type).get_config()
        self._set_model_config(preset)
        self._train_model(
            AdaptiveLayerStackOptions.BASE, print_parameter_count_flag=True
        )

    def train_adaptive_vector_stack_model(self):
        self.train_model(AdaptiveParameterLayerOptions.VECTOR)

    def train_adaptive_matrix_stack_model(self):
        self.train_model(AdaptiveParameterLayerOptions.MATRIX)

    def train_adaptive_generator_stack_model(self):
        self.train_model(AdaptiveParameterLayerOptions.GENERATOR)

    def test_all_types(self):
        for option_type in AdaptiveParameterLayerOptions:
            self.train_model(option_type)


class AdaptiveParameterExperimentPresets:
    def __init__(self, layer_options: AdaptiveParameterLayerOptions) -> None:
        self.layer_options = layer_options

    def get_config(self) -> "LayerStackConfig":
        adaptive_weight_option = None
        match self.layer_options:
            case AdaptiveParameterLayerOptions.VECTOR:
                adaptive_weight_option = AdaptiveWeightOptions.MATRIX
            case AdaptiveParameterLayerOptions.MATRIX:
                adaptive_weight_option = AdaptiveWeightOptions.MATRIX
            case AdaptiveParameterLayerOptions.GENERATOR:
                adaptive_weight_option = AdaptiveWeightOptions.GENERATOR

        return AdaptiveParameterLayerPresets.adaptive_parameter_layer_stack_preset(
            return_model_config_flag=True,
            batch_size=64,
            input_dim=784,
            output_dim=10,
            hidden_dim=0,
            num_layers=1,
            activation=ActivationOptions.RELU,
            residual_flag=False,
            dropout_probability=0.0,
            adaptive_weight_option=adaptive_weight_option,
            adaptive_bias_option=AdaptiveBiasOptions.GENERATOR,
            adaptive_mixture_top_k=3,
            adaptive_mixture_num_experts=9,
            adaptive_mixture_weighted_parameters_flag=False,
            adaptive_mixture_clip_parameter_option=ClipParameterOptions.BEFORE,
            adaptive_mixture_clip_range=5.0,
            adaptive_init_sampler_model_option=AdaptiveRouterOptions.INDEPENTENT_ROUTER,
            adaptive_behaviour_generator_depth=DynamicDepthOptions.DISABLED,
            adaptive_behaviour_diagonal_option=DynamicDiagonalOptions.DIAGONAL,
            adaptive_behaviour_bias_option=DynamicBiasOptions.DISABLED,
            adaptive_behaviour_memory_option=LinearMemoryOptions.DISABLED,
            adaptive_behaviour_memory_size_option=LinearMemorySizeOptions.DISABLED,
            adaptive_behaviour_memory_position_option=LinearMemoryPositionOptions.BEFORE_AFFINE,
            experts_router_model_bias_flag=False,
            experts_router_model_noisy_topk_flag=False,
            experts_router_model_generator_depth=DynamicDepthOptions.DEPTH_OF_TWO,
            experts_router_model_diagonal_option=DynamicDiagonalOptions.DIAGONAL,
            experts_router_model_bias_option=DynamicBiasOptions.DISABLED,
            experts_router_model_memory_option=LinearMemoryOptions.DISABLED,
            experts_router_model_memory_size_option=LinearMemorySizeOptions.DISABLED,
            experts_router_model_memory_position_option=LinearMemoryPositionOptions.BEFORE_AFFINE,
            experts_router_model_layer_stack_option=LinearLayerStackOptions.ADAPTIVE,
            experts_sampler_threshold=0.0,
            experts_sampler_filter_above_threshold=False,
            experts_sampler_num_topk_samples=0,
            experts_sampler_normalize_probabilities_flag=False,
            experts_sampler_switch_loss_weight=0.1,
            experts_sampler_zero_centred_loss_weight=0.1,
            experts_sampler_mutual_information_loss_weight=0.0,
            experts_sampler_coefficient_of_variation_loss_weight=0.1,
            experts_layer_stack_option=LinearLayerStackOptions.ADAPTIVE,
            experts_compute_expert_mixture_flag=True,
            experts_weighting_position_option=ExpertWeightingPositionOptions.AFTER_EXPERTS,
            experts_init_sampler_option=InitSamplerOptions.DISABLED,
            experts_weighted_parameters_flag=True,
            experts_model_bias_flag=False,
            experts_model_generator_depth=DynamicDepthOptions.DEPTH_OF_TWO,
            experts_model_diagonal_option=DynamicDiagonalOptions.DIAGONAL,
            experts_model_bias_option=DynamicBiasOptions.DISABLED,
            experts_model_memory_option=LinearMemoryOptions.DISABLED,
            experts_model_memory_size_option=LinearMemorySizeOptions.DISABLED,
            experts_model_memory_position_option=LinearMemoryPositionOptions.BEFORE_AFFINE,
            stack_bias_flag=False,
            stack_num_layers=1,
            stack_hidden_dim=0,
            stack_activation=ActivationOptions.RELU,
            stack_residual_flag=True,
            stack_dropout_probability=0.0,
        )
