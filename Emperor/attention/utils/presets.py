from Emperor.adaptive.options import (
    AdaptiveLayerStackOptions,
    AdaptiveParameterLayerOptions,
)
from Emperor.adaptive.utils.layers import AdaptiveParameterLayer
from Emperor.adaptive.utils.presets import AdaptiveParameterLayerPresets
from Emperor.attention.utils.layer import MultiHeadAttentionConfig
from Emperor.linears.options import LinearLayerStackOptions
from Emperor.linears.utils.presets import LinearPresets


class MultiHeadAttentionPresets:
    @staticmethod
    def adaptive_generator_mixture_preset(
        input_dim=8,
        output_dim=6,
        model_type=LinearLayerStackOptions.ADAPTIVE,
        batch_size=BATCH_SIZE,
        num_heads=NUM_EXPERTS,
        query_key_projection_dim=16,
        value_projection_dim=32,
        embedding_dim=64,
        target_sequence_length=16,
        source_sequence_length=32,
        target_dtype=float32,
        use_separate_projection_weight_flag=False,
        dropout_probability=0.0,
        key_value_bias_flag=False,
        zero_attention_flag=False,
        causal_attention_mask_flag=False,
        add_key_value_bias_flag=False,
        router_model_bias_flag: bool = False,
        router_model_noisy_topk_flag: bool = False,
        router_model_generator_depth: DynamicDepthOptions = DynamicDepthOptions.DISABLED,
        router_model_diagonal_option: DynamicDiagonalOptions = DynamicDiagonalOptions.DISABLED,
        router_model_bias_option: DynamicBiasOptions = DynamicBiasOptions.DISABLED,
        router_model_memory_option: LinearMemoryOptions = LinearMemoryOptions.DISABLED,
        router_model_memory_size_option: LinearMemorySizeOptions = LinearMemorySizeOptions.DISABLED,
        router_model_memory_position_option: LinearMemoryPositionOptions = LinearMemoryPositionOptions.BEFORE_AFFINE,
        router_model_layer_stack_option=LinearLayerStackOptions.BASE,
        sampler_threshold=0.0,
        sampler_filter_above_threshold=False,
        sampler_num_topk_samples=0,
        sampler_normalize_probabilities_flag=False,
        sampler_switch_loss_weight=0.0,
        sampler_zero_centred_loss_weight=0.0,
        sampler_mutual_information_loss_weight=0.0,
        sampler_coefficient_of_variation_loss_weight=0.0,
        experts_top_k=3,
        experts_num_experts=6,
        experts_layer_stack_option=LinearLayerStackOptions.BASE,
        experts_compute_expert_mixture_flag=False,
        experts_weighting_position_option=ExpertWeightingPositionOptions.BEFORE_EXPERTS,
        experts_init_sampler_model_flag=False,
        experts_weighted_parameters_flag=False,
        experts_layer_role_option=LayerRoleOptions.GENERAL,
        experts_model_bias_flag: bool = False,
        experts_model_generator_depth: DynamicDepthOptions = DynamicDepthOptions.DISABLED,
        experts_model_diagonal_option: DynamicDiagonalOptions = DynamicDiagonalOptions.DISABLED,
        experts_model_bias_option: DynamicBiasOptions = DynamicBiasOptions.DISABLED,
        experts_model_memory_option: LinearMemoryOptions = LinearMemoryOptions.DISABLED,
        experts_model_memory_size_option: LinearMemorySizeOptions = LinearMemorySizeOptions.DISABLED,
        experts_model_memory_position_option: LinearMemoryPositionOptions = LinearMemoryPositionOptions.BEFORE_AFFINE,
        experts_stack_num_layers: int = 2,
        experts_stack_activation: ActivationOptions = ActivationOptions.RELU,
        experts_stack_residual_flag: bool = False,
        experts_stack_dropout_probability: float = 0.0,
        stack_num_layers: int = 2,
        stack_hidden_dim: int = 0,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_flag: bool = False,
        stack_dropout_probability: float = 0.0,
    ) -> "MultiHeadAttentionConfig":
        if isinstance(model_type, AdaptiveLayerStackOptions):
            projector_config = (
                AdaptiveParameterLayerPresets.adaptive_generator_mixture_preset()
            )
        elif isinstance(model_type, LinearLayerStackOptions):
            projector_config = LinearPresets.base_linear_layer_stack_preset(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=experts_model_bias_flag,
                data_monitor=None,
                parameter_monitor=None,
                stack_num_layers=stack_num_layers,
                stack_hidden_dim=stack_hidden_dim,
                stack_activation=stack_activation,
                stack_residual_flag=stack_residual_flag,
                stack_dropout_probability=stack_dropout_probability,
            )
            if model_type == LinearLayerStackOptions.ADAPTIVE:
                projector_config = LinearPresets.adaptive_linear_layer_stack_preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    bias_flag=experts_model_bias_flag,
                    generator_depth=experts_model_generator_depth,
                    diagonal_option=experts_model_diagonal_option,
                    bias_option=experts_model_bias_option,
                    memory_option=experts_model_memory_option,
                    memory_size_option=experts_model_memory_size_option,
                    memory_position_option=experts_model_memory_position_option,
                    stack_num_layers=stack_num_layers,
                    stack_hidden_dim=stack_hidden_dim,
                    stack_activation=stack_activation,
                    stack_residual_flag=stack_residual_flag,
                    stack_dropout_probability=stack_dropout_probability,
                )

        return MultiHeadAttentionConfig(
            model_type=model_type,
            batch_size=batch_size,
            num_heads=num_heads,
            query_key_projection_dim=query_key_projection_dim,
            value_projection_dim=value_projection_dim,
            embedding_dim=embedding_dim,
            target_sequence_length=target_sequence_length,
            source_sequence_length=source_sequence_length,
            target_dtype=target_dtype,
            use_separate_projection_weight_flag=use_separate_projection_weight_flag,
            dropout_probability=dropout_probability,
            key_value_bias_flag=key_value_bias_flag,
            zero_attention_flag=zero_attention_flag,
            causal_attention_mask_flag=causal_attention_mask_flag,
            add_key_value_bias_flag=add_key_value_bias_flag,
            override_config=projector_config,
        )
