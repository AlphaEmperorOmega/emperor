from Emperor.config import ModelConfig
from Emperor.base.utils import ConfigUtils
from Emperor.base.layer import LayerStackConfig
from Emperor.linears.utils.config import LinearsConfigs
from Emperor.experts.experts import MixtureOfExpertsConfig
from Emperor.linears.options import LinearLayerStackOptions
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
from Emperor.sampler.utils.config import SamplerConfigs


class MixtureOfExpertsConfigs:
    @staticmethod
    def experts_config(
        input_dim=8,
        hidden_dim=12,
        output_dim=6,
        layer_stack_option=LinearLayerStackOptions.ADAPTIVE,
        router_layer_stack_option=LinearLayerStackOptions.BASE,
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
        activation: ActivationOptions = ActivationOptions.RELU,
        residual_flag: bool = False,
        dropout_probability: float = 0.0,
        stack_num_layers: int = 2,
        noisy_topk_flag: bool = False,
        threshold=0.0,
        filter_above_threshold=False,
        num_topk_samples=0,
        normalize_probabilities_flag=False,
        switch_loss_weight=0.0,
        zero_centred_loss_weight=0.0,
        mutual_information_loss_weight=0.0,
        coefficient_of_variation_loss_weight=0.0,
    ):
        arguments = ConfigUtils.get_method_arguments()
        return MixtureOfExpertsConfig(
            layer_stack_option=layer_stack_option,
            top_k=top_k,
            num_experts=num_experts,
            compute_expert_mixture_flag=compute_expert_mixture_flag,
            weighted_parameters_flag=weighted_parameters_flag,
            weighting_position_option=weighting_position_option,
            init_sampler_model_flag=init_sampler_model_flag,
            layer_role_option=layer_role_option,
            override_config=LayerStackConfig(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=num_experts,
                num_layers=stack_num_layers,
                activation=activation,
                layer_norm_position=LayerNormPositionOptions.NONE,
                residual_flag=residual_flag,
                adaptive_computation_flag=False,
                dropout_probability=dropout_probability,
                override_config=LinearsConfigs.adaptive_linear_layer_config(
                    **arguments
                ),
            ),
            router_model_config=SamplerConfigs.router_config(**arguments),
            sampler_model_config=SamplerConfigs.sampler_config(**arguments),
        )

    @staticmethod
    def linear_adaptive_layer_preset(
        batch_size=2,
        input_dim=8,
        hidden_dim=12,
        output_dim=6,
        layer_stack_option=LinearLayerStackOptions.ADAPTIVE,
        router_layer_stack_option=LinearLayerStackOptions.BASE,
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
        activation: ActivationOptions = ActivationOptions.RELU,
        residual_flag: bool = False,
        dropout_probability: float = 0.0,
        stack_num_layers: int = 2,
        noisy_topk_flag: bool = False,
        threshold=0.0,
        filter_above_threshold=False,
        num_topk_samples=0,
        normalize_probabilities_flag=False,
        switch_loss_weight=0.0,
        zero_centred_loss_weight=0.0,
        mutual_information_loss_weight=0.0,
        coefficient_of_variation_loss_weight=0.0,
    ) -> "ModelConfig":
        arguments = ConfigUtils.get_method_arguments()
        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            output_dim=output_dim,
            mixture_of_experts_config=MixtureOfExpertsConfig(
                layer_stack_option=layer_stack_option,
                top_k=top_k,
                num_experts=num_experts,
                compute_expert_mixture_flag=compute_expert_mixture_flag,
                weighted_parameters_flag=weighted_parameters_flag,
                weighting_position_option=weighting_position_option,
                init_sampler_model_flag=init_sampler_model_flag,
                layer_role_option=layer_role_option,
                override_config=LayerStackConfig(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=num_experts,
                    num_layers=stack_num_layers,
                    activation=activation,
                    layer_norm_position=LayerNormPositionOptions.NONE,
                    residual_flag=residual_flag,
                    adaptive_computation_flag=False,
                    dropout_probability=dropout_probability,
                    override_config=LinearsConfigs.adaptive_linear_layer_config(
                        **arguments
                    ),
                ),
                router_model_config=SamplerConfigs.router_config(**arguments),
                sampler_model_config=SamplerConfigs.sampler_config(**arguments),
            ),
        )
