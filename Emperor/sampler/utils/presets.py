from Emperor.config import ModelConfig
from Emperor.base.utils import ConfigUtils
from Emperor.base.layer import LayerStackConfig
from Emperor.sampler.utils.routers import RouterConfig
from Emperor.linears.utils.presets import LinearPresets
from Emperor.sampler.utils.samplers import SamplerConfig
from Emperor.base.enums import ActivationOptions, LayerNormPositionOptions
from Emperor.linears.options import LinearLayerOptions, LinearLayerStackOptions
from Emperor.behaviours.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)


class SamplerPresets:
    @staticmethod
    def router_config(
        input_dim: int = 12,
        hidden_dim: int = 16,
        num_experts: int = 6,
        bias_flag: bool = True,
        noisy_topk_flag: bool = False,
        residual_flag: bool = False,
        dropout_probability: float = 0.0,
        stack_num_layers: int = 2,
        activation: ActivationOptions = ActivationOptions.RELU,
        layer_stack_option: LinearLayerStackOptions = LinearLayerStackOptions.BASE,
        bias_option: DynamicBiasOptions = DynamicBiasOptions.DISABLED,
        memory_option: LinearMemoryOptions = LinearMemoryOptions.DISABLED,
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DISABLED,
        diagonal_option: DynamicDiagonalOptions = DynamicDiagonalOptions.DISABLED,
        memory_size_option: LinearMemorySizeOptions = LinearMemorySizeOptions.DISABLED,
        memory_position_option: LinearMemoryPositionOptions = LinearMemoryPositionOptions.BEFORE_AFFINE,
        **kwargs,
    ) -> "RouterConfig":
        # arguments = ConfigUtils.get_method_arguments()

        if layer_stack_option == LinearLayerStackOptions.BASE:
            router_model_config = LinearPresets.base_linear_layer_stack_preset()
        elif layer_stack_option == LinearLayerStackOptions.ADAPTIVE:
            router_model_config = LinearPresets.adaptive_linear_layer_preset()

        return RouterConfig(
            layer_stack_option=layer_stack_option,
            num_experts=num_experts,
            noisy_topk_flag=noisy_topk_flag,
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
                override_config=router_model_config,
            ),
        )

    @staticmethod
    def sampler_config(
        num_experts: int = 6,
        top_k: int = 3,
        threshold: float = 0.0,
        filter_above_threshold: bool = False,
        num_topk_samples: int = 0,
        normalize_probabilities_flag: bool = False,
        noisy_topk_flag: bool = False,
        coefficient_of_variation_loss_weight: float = 0.0,
        switch_loss_weight: float = 0.0,
        zero_centred_loss_weight: float = 0.0,
        mutual_information_loss_weight: float = 0.0,
        **kwargs,
    ) -> "SamplerConfig":
        return SamplerConfig(
            top_k=top_k,
            threshold=threshold,
            filter_above_threshold=filter_above_threshold,
            num_topk_samples=num_topk_samples,
            normalize_probabilities_flag=normalize_probabilities_flag,
            noisy_topk_flag=noisy_topk_flag,
            num_experts=num_experts,
            coefficient_of_variation_loss_weight=coefficient_of_variation_loss_weight,
            switch_loss_weight=switch_loss_weight,
            zero_centred_loss_weight=zero_centred_loss_weight,
            mutual_information_loss_weight=mutual_information_loss_weight,
        )

    @staticmethod
    def router_preset(
        batch_size: int = 2,
        input_dim: int = 12,
        hidden_dim: int = 16,
        num_experts: int = 6,
        bias_flag: bool = True,
        noisy_topk_flag: bool = False,
        residual_flag: bool = False,
        dropout_probability: float = 0.0,
        num_layers: int = 2,
        activation: ActivationOptions = ActivationOptions.RELU,
        layer_stack_option: LinearLayerStackOptions = LinearLayerStackOptions.BASE,
        bias_option: DynamicBiasOptions = DynamicBiasOptions.DISABLED,
        memory_option: LinearMemoryOptions = LinearMemoryOptions.DISABLED,
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DISABLED,
        diagonal_option: DynamicDiagonalOptions = DynamicDiagonalOptions.DISABLED,
        memory_size_option: LinearMemorySizeOptions = LinearMemorySizeOptions.DISABLED,
        memory_position_option: LinearMemoryPositionOptions = LinearMemoryPositionOptions.BEFORE_AFFINE,
    ) -> "ModelConfig":
        stored_arguments = ConfigUtils.get_method_arguments()

        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            output_dim=num_experts,
            router_model_config=SamplerPresets.router_config(**stored_arguments),
        )

    @staticmethod
    def sampler_preset(
        batch_size: int = 2,
        input_dim: int = 12,
        hidden_dim: int = 16,
        num_experts: int = 6,
        bias_flag: bool = True,
        residual_flag: bool = False,
        dropout_probability: float = 0.0,
        num_layers: int = 2,
        top_k: int = 3,
        threshold: float = 0.0,
        filter_above_threshold: bool = False,
        num_topk_samples: int = 0,
        normalize_probabilities_flag: bool = False,
        noisy_topk_flag: bool = False,
        coefficient_of_variation_loss_weight: float = 0.0,
        switch_loss_weight: float = 0.0,
        zero_centred_loss_weight: float = 0.0,
        mutual_information_loss_weight: float = 0.0,
        activation: ActivationOptions = ActivationOptions.RELU,
        model_type: LinearLayerOptions = LinearLayerOptions.BASE,
        bias_option: DynamicBiasOptions = DynamicBiasOptions.DISABLED,
        memory_option: LinearMemoryOptions = LinearMemoryOptions.DISABLED,
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DISABLED,
        diagonal_option: DynamicDiagonalOptions = DynamicDiagonalOptions.DISABLED,
        memory_size_option: LinearMemorySizeOptions = LinearMemorySizeOptions.DISABLED,
        memory_position_option: LinearMemoryPositionOptions = LinearMemoryPositionOptions.BEFORE_AFFINE,
    ) -> "ModelConfig":
        stored_arguments = ConfigUtils.get_method_arguments()
        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            output_dim=num_experts,
            sampler_model_config=SamplerPresets.sampler_config(**stored_arguments),
            router_model_config=SamplerPresets.router_config(**stored_arguments),
        )
