from Emperor.config import ModelConfig
from Emperor.base.layer import LayerStackConfig
from Emperor.sampler.utils.presets import SamplerPresets
from Emperor.linears.utils.presets import LinearPresets
from Emperor.linears.options import LinearLayerStackOptions
from Emperor.experts.utils.layers import MixtureOfExpertsConfig
from Emperor.base.enums import ActivationOptions, LayerNormPositionOptions
from Emperor.experts.utils.enums import ExpertWeightingPositionOptions, LayerRoleOptions
from Emperor.behaviours.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)


class MixtureOfExpertsPresets:
    @staticmethod
    def experts_preset(
        return_model_config_flag: bool = False,
        batch_size=2,
        input_dim=8,
        output_dim=6,
        top_k=3,
        num_experts=6,
        compute_expert_mixture_flag=True,
        weighted_parameters_flag=True,
        weighting_position_option=ExpertWeightingPositionOptions.BEFORE_EXPERTS,
        init_sampler_model_flag=True,
        layer_role_option: LayerRoleOptions = LayerRoleOptions.GENERAL,
        bias_flag: bool = True,
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DISABLED,
        diagonal_option: DynamicDiagonalOptions = DynamicDiagonalOptions.DISABLED,
        bias_option: DynamicBiasOptions = DynamicBiasOptions.DISABLED,
        memory_option: LinearMemoryOptions = LinearMemoryOptions.DISABLED,
        memory_size_option: LinearMemorySizeOptions = LinearMemorySizeOptions.DISABLED,
        memory_position_option: LinearMemoryPositionOptions = LinearMemoryPositionOptions.BEFORE_AFFINE,
        noisy_topk_flag: bool = False,
        threshold=0.0,
        filter_above_threshold=False,
        num_topk_samples=0,
        normalize_probabilities_flag=False,
        switch_loss_weight=0.0,
        zero_centred_loss_weight=0.0,
        mutual_information_loss_weight=0.0,
        coefficient_of_variation_loss_weight=0.0,
        layer_stack_option=LinearLayerStackOptions.ADAPTIVE,
        router_layer_stack_option=LinearLayerStackOptions.BASE,
        stack_num_layers: int = 2,
        stack_hidden_dim: int = 0,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_flag: bool = False,
        stack_dropout_probability: float = 0.0,
    ) -> "MixtureOfExpertsConfig | ModelConfig":
        _hidden_dim = max(input_dim, output_dim)
        stack_hidden_dim = stack_hidden_dim if stack_hidden_dim > 0 else _hidden_dim

        if layer_stack_option == LinearLayerStackOptions.BASE:
            expert_model_config = LinearPresets.base_linear_layer_stack_preset(
                batch_size=batch_size,
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=bias_flag,
                data_monitor=None,
                parameter_monitor=None,
                stack_num_layers=stack_num_layers,
                stack_hidden_dim=stack_hidden_dim,
                stack_activation=stack_activation,
                stack_residual_flag=stack_residual_flag,
                stack_dropout_probability=stack_dropout_probability,
            )
        elif layer_stack_option == LinearLayerStackOptions.ADAPTIVE:
            expert_model_config = LinearPresets.adaptive_linear_layer_stack_preset(
                batch_size=batch_size,
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=bias_flag,
                generator_depth=generator_depth,
                diagonal_option=diagonal_option,
                bias_option=bias_option,
                memory_option=memory_option,
                memory_size_option=memory_size_option,
                memory_position_option=memory_position_option,
                stack_num_layers=stack_num_layers,
                stack_hidden_dim=stack_hidden_dim,
                stack_activation=stack_activation,
                stack_residual_flag=stack_residual_flag,
                stack_dropout_probability=stack_dropout_probability,
            )

        config = MixtureOfExpertsConfig(
            layer_stack_option=layer_stack_option,
            top_k=top_k,
            num_experts=num_experts,
            compute_expert_mixture_flag=compute_expert_mixture_flag,
            weighted_parameters_flag=weighted_parameters_flag,
            weighting_position_option=weighting_position_option,
            init_sampler_model_flag=init_sampler_model_flag,
            layer_role_option=layer_role_option,
            override_config=expert_model_config,
            router_model_config=SamplerPresets.router_preset(
                input_dim=input_dim,
                num_experts=num_experts,
                bias_flag=bias_flag,
                noisy_topk_flag=noisy_topk_flag,
                layer_stack_option=router_layer_stack_option,
                bias_option=bias_option,
                memory_option=memory_option,
                generator_depth=generator_depth,
                diagonal_option=diagonal_option,
                memory_size_option=memory_size_option,
                memory_position_option=memory_position_option,
                stack_num_layers=stack_num_layers,
                stack_hidden_dim=stack_hidden_dim,
                stack_activation=stack_activation,
                stack_residual_flag=stack_residual_flag,
                stack_dropout_probability=stack_dropout_probability,
            ),
            sampler_model_config=SamplerPresets.sampler_preset(
                num_experts=num_experts,
                top_k=top_k,
                threshold=threshold,
                filter_above_threshold=filter_above_threshold,
                num_topk_samples=num_topk_samples,
                normalize_probabilities_flag=normalize_probabilities_flag,
                noisy_topk_flag=noisy_topk_flag,
                coefficient_of_variation_loss_weight=coefficient_of_variation_loss_weight,
                switch_loss_weight=switch_loss_weight,
                zero_centred_loss_weight=zero_centred_loss_weight,
                mutual_information_loss_weight=mutual_information_loss_weight,
            ),
        )

        if not return_model_config_flag:
            return config
        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            output_dim=output_dim,
            mixture_of_experts_config=config,
        )

    @staticmethod
    def experts_stack_preset(
        return_model_config_flag: bool = False,
        batch_size=8,
        input_dim=8,
        output_dim=6,
        top_k=3,
        num_experts=6,
        compute_expert_mixture_flag=True,
        weighted_parameters_flag=True,
        weighting_position_option=ExpertWeightingPositionOptions.BEFORE_EXPERTS,
        init_sampler_model_flag=True,
        layer_role_option: LayerRoleOptions = LayerRoleOptions.GENERAL,
        bias_flag: bool = True,
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DISABLED,
        diagonal_option: DynamicDiagonalOptions = DynamicDiagonalOptions.DISABLED,
        bias_option: DynamicBiasOptions = DynamicBiasOptions.DISABLED,
        memory_option: LinearMemoryOptions = LinearMemoryOptions.DISABLED,
        memory_size_option: LinearMemorySizeOptions = LinearMemorySizeOptions.DISABLED,
        memory_position_option: LinearMemoryPositionOptions = LinearMemoryPositionOptions.BEFORE_AFFINE,
        noisy_topk_flag: bool = False,
        threshold=0.0,
        filter_above_threshold=False,
        num_topk_samples=0,
        normalize_probabilities_flag=False,
        switch_loss_weight=0.0,
        zero_centred_loss_weight=0.0,
        mutual_information_loss_weight=0.0,
        coefficient_of_variation_loss_weight=0.0,
        layer_stack_option=LinearLayerStackOptions.ADAPTIVE,
        router_layer_stack_option=LinearLayerStackOptions.BASE,
        stack_num_layers: int = 2,
        stack_hidden_dim: int = 0,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_flag: bool = False,
        stack_dropout_probability: float = 0.0,
    ) -> "LayerStackConfig | ModelConfig":
        _hidden_dim = max(input_dim, output_dim)
        stack_hidden_dim = stack_hidden_dim if stack_hidden_dim > 0 else _hidden_dim

        config = LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=stack_hidden_dim,
            output_dim=output_dim,
            num_layers=stack_num_layers,
            activation=stack_activation,
            layer_norm_position=LayerNormPositionOptions.NONE,
            residual_flag=stack_residual_flag,
            adaptive_computation_flag=False,
            dropout_probability=stack_dropout_probability,
            override_config=MixtureOfExpertsPresets.experts_preset(
                input_dim=input_dim,
                output_dim=output_dim,
                top_k=top_k,
                num_experts=num_experts,
                compute_expert_mixture_flag=compute_expert_mixture_flag,
                weighted_parameters_flag=weighted_parameters_flag,
                weighting_position_option=weighting_position_option,
                init_sampler_model_flag=init_sampler_model_flag,
                layer_role_option=layer_role_option,
                bias_flag=bias_flag,
                generator_depth=generator_depth,
                diagonal_option=diagonal_option,
                bias_option=bias_option,
                memory_option=memory_option,
                memory_size_option=memory_size_option,
                memory_position_option=memory_position_option,
                noisy_topk_flag=noisy_topk_flag,
                threshold=threshold,
                filter_above_threshold=filter_above_threshold,
                num_topk_samples=num_topk_samples,
                normalize_probabilities_flag=normalize_probabilities_flag,
                switch_loss_weight=switch_loss_weight,
                zero_centred_loss_weight=zero_centred_loss_weight,
                mutual_information_loss_weight=mutual_information_loss_weight,
                coefficient_of_variation_loss_weight=coefficient_of_variation_loss_weight,
                layer_stack_option=layer_stack_option,
                router_layer_stack_option=router_layer_stack_option,
                stack_num_layers=stack_num_layers,
                stack_hidden_dim=stack_hidden_dim,
                stack_activation=stack_activation,
                stack_residual_flag=stack_residual_flag,
                stack_dropout_probability=stack_dropout_probability,
            ),
        )

        if not return_model_config_flag:
            return config
        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            output_dim=output_dim,
            layer_stack_config=config,
        )
