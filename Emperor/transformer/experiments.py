from Emperor.base.enums import ActivationOptions
from Emperor.transformer.options import TransformerOptions
from Emperor.experiments.utils.factories import Experiments
from Emperor.linears.options import LinearLayerStackOptions
from Emperor.transformer.utils.layers import TransformerConfig
from Emperor.adaptive.utils.layers import AdaptiveRouterOptions
from Emperor.transformer.utils.presets import TransformerPresets
from Emperor.adaptive.utils.mixtures.types.utils.enums import ClipParameterOptions
from Emperor.experts.utils.enums import ExpertWeightingPositionOptions, LayerRoleOptions
from Emperor.adaptive.utils.mixtures.options import (
    AdaptiveBiasOptions,
    AdaptiveWeightOptions,
)
from Emperor.behaviours.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)


class AdaptiveParameterExperiments(Experiments):
    def __init__(
        self,
        mini_datasetset_flag: bool = True,
    ) -> None:
        super().__init__(mini_datasetset_flag)

    def train_model(self, layer_type: TransformerOptions):
        preset = AdaptiveParameterExperimentPresets(layer_type).get_config()
        self._set_model_config(preset)
        self._train_model(TransformerOptions.DEFAULT, print_parameter_count_flag=True)

    def train_transformer_encoder_stack_model(self):
        self.train_model(TransformerOptions.ENCODER)

    def train_transformer_decoder_stack_model(self):
        self.train_model(TransformerOptions.DECODER)

    def train_transformer_encoder_layer_model(self):
        self.train_model(TransformerOptions.ENCODER_LAYER)

    def train_transformer_decoder_layer_model(self):
        self.train_model(TransformerOptions.DECODER_LAYER)

    def test_all_types(self):
        for option_type in TransformerOptions:
            self.train_model(option_type)


class AdaptiveParameterExperimentPresets:
    def __init__(self, layer_options: TransformerOptions) -> None:
        self.layer_options = layer_options

    def get_config(self) -> "TransformerConfig":
        return TransformerPresets.transformer_preset(
            input_dim=8,
            hidden_dim=4,
            output_dim=6,
            layer_stack_option=LinearLayerStackOptions.ADAPTIVE,
            num_layers=2,
            embedding_dim=8,
            source_sequence_length=6,
            target_sequence_length=6,
            layer_norm_dim=8,
            causal_attention_mask_flag=False,
            attention_model_type=LinearLayerStackOptions.ADAPTIVE,
            attention_batch_size=8,
            attention_num_heads=4,
            attention_target_sequence_length=18,
            attention_source_sequence_length=20,
            attention_target_dtype=float32,
            attention_dropout_probability=0.0,
            attention_key_value_bias_flag=False,
            attention_zero_attention_flag=False,
            attention_causal_attention_mask_flag=False,
            attention_add_key_value_bias_flag=False,
            attention_average_attention_weights_flag=False,
            attention_return_attention_weights_flag=False,
            attention_adaptive_stack_num_layers=2,
            attention_adaptive_stack_activation=ActivationOptions.RELU,
            attention_adaptive_stack_residual_flag=False,
            attention_adaptive_stack_dropout_probability=0.0,
            attention_adaptive_weight_option=AdaptiveWeightOptions.GENERATOR,
            attention_adaptive_bias_option=AdaptiveBiasOptions.GENERATOR,
            attention_adaptive_mixture_top_k=3,
            attention_adaptive_mixture_num_experts=6,
            attention_adaptive_mixture_weighted_parameters_flag=False,
            attention_adaptive_mixture_clip_parameter_option=ClipParameterOptions.BEFORE,
            attention_adaptive_mixture_clip_range=5.0,
            attention_adaptive_init_sampler_model_option=AdaptiveRouterOptions.SHARED_ROUTER,
            attention_adaptive_time_tracker_flag=False,
            attention_adaptive_behaviour_generator_depth=DynamicDepthOptions.DISABLED,
            attention_adaptive_behaviour_diagonal_option=DynamicDiagonalOptions.DISABLED,
            attention_adaptive_behaviour_bias_option=DynamicBiasOptions.DISABLED,
            attention_adaptive_behaviour_memory_option=LinearMemoryOptions.DISABLED,
            attention_adaptive_behaviour_memory_size_option=LinearMemorySizeOptions.DISABLED,
            attention_adaptive_behaviour_memory_position_option=LinearMemoryPositionOptions.BEFORE_AFFINE,
            attention_router_bias_flag=False,
            attention_router_noisy_topk_flag=False,
            attention_router_generator_depth=DynamicDepthOptions.DISABLED,
            attention_router_diagonal_option=DynamicDiagonalOptions.DISABLED,
            attention_router_bias_option=DynamicBiasOptions.DISABLED,
            attention_router_memory_option=LinearMemoryOptions.DISABLED,
            attention_router_memory_size_option=LinearMemorySizeOptions.DISABLED,
            attention_router_memory_position_option=LinearMemoryPositionOptions.BEFORE_AFFINE,
            attention_router_layer_stack_option=LinearLayerStackOptions.BASE,
            attention_sampler_threshold=0.0,
            attention_sampler_filter_above_threshold=False,
            attention_sampler_num_topk_samples=0,
            attention_sampler_normalize_probabilities_flag=False,
            attention_sampler_switch_loss_weight=0.0,
            attention_sampler_zero_centred_loss_weight=0.0,
            attention_sampler_mutual_information_loss_weight=0.0,
            attention_sampler_coefficient_of_variation_loss_weight=0.0,
            attention_experts_layer_stack_option=LinearLayerStackOptions.BASE,
            attention_experts_compute_expert_mixture_flag=True,
            attention_experts_weighting_position_option=ExpertWeightingPositionOptions.BEFORE_EXPERTS,
            attention_experts_init_sampler_model_flag=False,
            attention_experts_weighted_parameters_flag=False,
            attention_experts_layer_role_option=LayerRoleOptions.GENERAL,
            attention_experts_bias_flag=False,
            attention_experts_generator_depth=DynamicDepthOptions.DISABLED,
            attention_experts_diagonal_option=DynamicDiagonalOptions.DISABLED,
            attention_experts_bias_option=DynamicBiasOptions.DISABLED,
            attention_experts_memory_option=LinearMemoryOptions.DISABLED,
            attention_experts_memory_size_option=LinearMemorySizeOptions.DISABLED,
            attention_experts_memory_position_option=LinearMemoryPositionOptions.BEFORE_AFFINE,
            forward_adaptive_stack_num_layers=2,
            forward_adaptive_stack_activation=ActivationOptions.RELU,
            forward_adaptive_stack_residual_flag=False,
            forward_adaptive_stack_dropout_probability=0.0,
            forward_adaptive_weight_option=AdaptiveWeightOptions.GENERATOR,
            forward_adaptive_bias_option=AdaptiveBiasOptions.GENERATOR,
            forward_adaptive_mixture_top_k=3,
            forward_adaptive_mixture_num_experts=6,
            forward_adaptive_mixture_weighted_parameters_flag=False,
            forward_adaptive_mixture_clip_parameter_option=ClipParameterOptions.BEFORE,
            forward_adaptive_mixture_clip_range=5.0,
            forward_adaptive_init_sampler_model_option=AdaptiveRouterOptions.SHARED_ROUTER,
            forward_adaptive_time_tracker_flag=False,
            forward_adaptive_behaviour_generator_depth=DynamicDepthOptions.DISABLED,
            forward_adaptive_behaviour_diagonal_option=DynamicDiagonalOptions.DISABLED,
            forward_adaptive_behaviour_bias_option=DynamicBiasOptions.DISABLED,
            forward_adaptive_behaviour_memory_option=LinearMemoryOptions.DISABLED,
            forward_adaptive_behaviour_memory_size_option=LinearMemorySizeOptions.DISABLED,
            forward_adaptive_behaviour_memory_position_option=LinearMemoryPositionOptions.BEFORE_AFFINE,
            forward_router_bias_flag=False,
            forward_router_noisy_topk_flag=False,
            forward_router_generator_depth=DynamicDepthOptions.DISABLED,
            forward_router_diagonal_option=DynamicDiagonalOptions.DISABLED,
            forward_router_bias_option=DynamicBiasOptions.DISABLED,
            forward_router_memory_option=LinearMemoryOptions.DISABLED,
            forward_router_memory_size_option=LinearMemorySizeOptions.DISABLED,
            forward_router_memory_position_option=LinearMemoryPositionOptions.BEFORE_AFFINE,
            forward_router_layer_stack_option=LinearLayerStackOptions.BASE,
            forward_sampler_threshold=0.0,
            forward_sampler_filter_above_threshold=False,
            forward_sampler_num_topk_samples=0,
            forward_sampler_normalize_probabilities_flag=False,
            forward_sampler_switch_loss_weight=0.0,
            forward_sampler_zero_centred_loss_weight=0.0,
            forward_sampler_mutual_information_loss_weight=0.0,
            forward_sampler_coefficient_of_variation_loss_weight=0.0,
            forward_experts_layer_stack_option=LinearLayerStackOptions.BASE,
            forward_experts_compute_expert_mixture_flag=True,
            forward_experts_weighting_position_option=ExpertWeightingPositionOptions.BEFORE_EXPERTS,
            forward_experts_init_sampler_model_flag=False,
            forward_experts_weighted_parameters_flag=False,
            forward_experts_layer_role_option=LayerRoleOptions.GENERAL,
            forward_experts_bias_flag=False,
            forward_experts_generator_depth=DynamicDepthOptions.DISABLED,
            forward_experts_diagonal_option=DynamicDiagonalOptions.DISABLED,
            forward_experts_bias_option=DynamicBiasOptions.DISABLED,
            forward_experts_memory_option=LinearMemoryOptions.DISABLED,
            forward_experts_memory_size_option=LinearMemorySizeOptions.DISABLED,
            forward_experts_memory_position_option=LinearMemoryPositionOptions.BEFORE_AFFINE,
            forward_experts_stack_num_layers=2,
            forward_experts_stack_activation=ActivationOptions.RELU,
            forward_experts_stack_residual_flag=False,
            forward_experts_stack_dropout_probability=0.0,
            stack_bias_flag=False,
            stack_num_layers=2,
            stack_hidden_dim=0,
            stack_activation=ActivationOptions.RELU,
            stack_residual_flag=False,
            stack_dropout_probability=0.0,
        )
