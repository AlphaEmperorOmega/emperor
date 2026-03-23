from emperor.config import ModelConfig
from emperor.sampler.utils.routers import RouterConfig
from emperor.linears.utils.presets import LinearPresets
from emperor.sampler.utils.samplers import SamplerConfig
from emperor.base.enums import ActivationOptions
from emperor.linears.options import LinearLayerStackOptions
from emperor.augmentations.options import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)


class SamplerPresets:
    @staticmethod
    def router_preset(
        return_model_config_flag: bool = False,
        batch_size: int = 2,
        input_dim: int = 12,
        num_experts: int = 6,
        bias_flag: bool = True,
        noisy_topk_flag: bool = False,
        layer_stack_option: LinearLayerStackOptions = LinearLayerStackOptions.BASE,
        bias_option: DynamicBiasOptions = DynamicBiasOptions.DISABLED,
        memory_option: LinearMemoryOptions = LinearMemoryOptions.DISABLED,
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DISABLED,
        diagonal_option: DynamicDiagonalOptions = DynamicDiagonalOptions.DISABLED,
        memory_size_option: LinearMemorySizeOptions = LinearMemorySizeOptions.DISABLED,
        memory_position_option: LinearMemoryPositionOptions = LinearMemoryPositionOptions.BEFORE_AFFINE,
        stack_num_layers: int = 2,
        stack_hidden_dim: int = 0,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_flag: bool = False,
        stack_dropout_probability: float = 0.0,
    ) -> "RouterConfig":
        # arguments = ConfigUtils.get_method_arguments()
        _hidden_dim = max(input_dim, num_experts)
        stack_hidden_dim = stack_hidden_dim if stack_hidden_dim > 0 else _hidden_dim

        if layer_stack_option == LinearLayerStackOptions.BASE:
            router_model_config = LinearPresets.base_linear_layer_stack_preset(
                batch_size=batch_size,
                input_dim=input_dim,
                output_dim=num_experts,
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
            router_model_config = LinearPresets.adaptive_linear_layer_stack_preset(
                batch_size=batch_size,
                input_dim=input_dim,
                output_dim=num_experts,
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

        config = RouterConfig(
            input_dim=input_dim,
            layer_stack_option=layer_stack_option,
            num_experts=num_experts,
            noisy_topk_flag=noisy_topk_flag,
            override_config=router_model_config,
        )

        if not return_model_config_flag:
            return config
        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            output_dim=num_experts,
            router_model_config=config,
        )

    @staticmethod
    def sampler_preset(
        return_model_config_flag: bool = False,
        batch_size: int = 2,
        input_dim: int = 12,
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
    ) -> "SamplerConfig":
        config = SamplerConfig(
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

        if not return_model_config_flag:
            return config
        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            output_dim=num_experts,
            sampler_model_config=config,
        )
