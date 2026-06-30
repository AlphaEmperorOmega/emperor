from typing import Any

from emperor.augmentations.adaptive_parameters.config import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.augmentations.adaptive_parameters.core.bias import (
    DynamicBiasConfig,
)
from emperor.augmentations.adaptive_parameters.core.diagonal import (
    DynamicDiagonalConfig,
)
from emperor.augmentations.adaptive_parameters.core.mask import (
    AxisMaskConfig,
)
from emperor.augmentations.adaptive_parameters.core.weight import (
    DynamicWeightConfig,
)
from emperor.base.layer.config import LayerConfig, LayerStackConfig, RecurrentLayerConfig
from emperor.base.layer.gate import GateConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.experts.config import MixtureOfExpertsModelConfig
from emperor.experts.core.config import (
    MixtureOfExpertsConfig,
    MixtureOfExpertsLayerConfig,
)
from emperor.halting.config import StickBreakingConfig
from emperor.linears.core.config import AdaptiveLinearLayerConfig, LinearLayerConfig
from emperor.sampler.core.config import RouterConfig, SamplerConfig
from models.adaptive_parameter_config_factory import (
    build_bias_config,
    build_diagonal_config,
    build_mask_config,
    build_weight_config,
)

from models.experts._builder_options import (
    ExpertsAdaptiveGeneratorStackOptions,
    ExpertsControllerStackOptions,
)
from models.experts.experts_linear_adaptive._controller_stack import (
    build_controller_stack,
)


class ControlConfigFactory:
    def __init__(self, builder: Any) -> None:
        self.builder = builder

    def build(self) -> MixtureOfExpertsModelConfig | RecurrentLayerConfig:
        return self.__maybe_wrap_recurrent(self.__build_main_model_config())

    def build_adaptive_linear_layer_config(
        self,
        bias_flag: bool,
    ) -> AdaptiveLinearLayerConfig:
        return AdaptiveLinearLayerConfig(
            bias_flag=bias_flag,
            adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                weight_config=self.__build_weight_config(),
                bias_config=self.__build_bias_config(),
                diagonal_config=self.__build_diagonal_config(),
                mask_config=self.__build_mask_config(),
                model_config=self.__build_generator_model_config(),
            ),
        )

    def __build_main_model_config(self) -> MixtureOfExpertsModelConfig:
        builder = self.builder
        mixture_options = builder.mixture_options
        return MixtureOfExpertsModelConfig(
            input_dim=builder.hidden_dim,
            output_dim=builder.hidden_dim,
            top_k=mixture_options.top_k,
            routing_initialization_mode=(
                mixture_options.routing_initialization_mode
            ),
            sampler_config=self.__build_sampler_config(),
            stack_config=self.__build_main_stack_config(),
        )

    def __maybe_wrap_recurrent(
        self,
        block_config: MixtureOfExpertsModelConfig,
    ) -> MixtureOfExpertsModelConfig | RecurrentLayerConfig:
        recurrent_options = self.builder.recurrent_controller_options
        if not recurrent_options.recurrent_flag:
            return block_config
        return RecurrentLayerConfig(
            max_steps=recurrent_options.recurrent_max_steps,
            recurrent_layer_norm_position=(
                recurrent_options.recurrent_layer_norm_position
            ),
            block_config=block_config,
            gate_config=self.__build_recurrent_gate_config(),
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            halting_config=self.__build_halting_config(
                enabled=recurrent_options.recurrent_halting_flag
            ),
        )

    def __build_main_stack_config(self) -> LayerStackConfig:
        builder = self.builder
        stack_options = builder.stack_options
        gate_config = self.__build_gate_config()
        self.__validate_shared_gate_config(gate_config)
        return LayerStackConfig(
            input_dim=stack_options.hidden_dim,
            hidden_dim=stack_options.hidden_dim,
            output_dim=stack_options.hidden_dim,
            num_layers=stack_options.num_layers,
            last_layer_bias_option=stack_options.last_layer_bias_option,
            apply_output_pipeline_flag=stack_options.apply_output_pipeline_flag,
            shared_gate_config=(
                builder.layer_controller_options.shared_gate_config
            ),
            layer_config=MixtureOfExpertsLayerConfig(
                activation=stack_options.activation,
                layer_norm_position=stack_options.layer_norm_position,
                residual_connection_option=(
                    stack_options.residual_connection_option
                ),
                dropout_probability=stack_options.dropout_probability,
                gate_config=gate_config,
                halting_config=self.__build_halting_config(),
                layer_model_config=self.__build_mixture_of_experts_config(),
            ),
        )

    def __validate_shared_gate_config(self, gate_config: GateConfig | None) -> None:
        if self.__is_active_gate_config(
            self.builder.shared_gate_config
        ) and self.__is_active_gate_config(gate_config):
            raise ValueError(
                "shared_gate_config cannot be provided when stack_gate_flag "
                "enables per-layer gate_config."
            )

    @staticmethod
    def __is_active_gate_config(gate_config: GateConfig | None) -> bool:
        return gate_config is not None

    def __build_mixture_of_experts_config(self) -> MixtureOfExpertsConfig:
        builder = self.builder
        mixture_options = builder.mixture_options
        return MixtureOfExpertsConfig(
            input_dim=builder.hidden_dim,
            output_dim=builder.hidden_dim,
            top_k=mixture_options.top_k,
            num_experts=mixture_options.num_experts,
            capacity_factor=mixture_options.capacity_factor,
            dropped_token_behavior=mixture_options.dropped_token_behavior,
            compute_expert_mixture_flag=(
                mixture_options.compute_expert_mixture_flag
            ),
            weighted_parameters_flag=mixture_options.weighted_parameters_flag,
            weighting_position_option=mixture_options.weighting_position_option,
            routing_initialization_mode=(
                mixture_options.routing_initialization_mode
            ),
            sampler_config=self.__build_sampler_config(),
            expert_model_config=self.__build_expert_model_config(),
        )

    def __build_gate_config(
        self,
        enabled: bool | None = None,
    ) -> GateConfig | None:
        layer_options = self.builder.layer_controller_options
        if enabled is None:
            enabled = layer_options.stack_gate_flag
        if not enabled:
            return None
        return GateConfig(
            model_config=self.__build_gate_model_config(enabled=enabled),
            option=layer_options.gate_option,
            activation=layer_options.gate_activation,
        )

    def __build_recurrent_gate_config(self) -> GateConfig | None:
        recurrent_options = self.builder.recurrent_controller_options
        if not recurrent_options.recurrent_gate_flag:
            return None
        return GateConfig(
            model_config=self.__build_gate_model_config(
                enabled=recurrent_options.recurrent_gate_flag,
            ),
            option=recurrent_options.recurrent_gate_option,
            activation=recurrent_options.recurrent_gate_activation,
        )

    def __build_gate_model_config(
        self,
        enabled: bool,
    ) -> LayerStackConfig | None:
        if not enabled:
            return None
        options = self.__gate_stack_options()
        return build_controller_stack(
            options,
            layer_model_config=LinearLayerConfig(
                bias_flag=options.bias_flag,
            ),
        )

    def __build_halting_config(
        self,
        enabled: bool | None = None,
    ) -> StickBreakingConfig | None:
        layer_options = self.builder.layer_controller_options
        if enabled is None:
            enabled = layer_options.stack_halting_flag
        if not enabled:
            return None
        options = self.__halting_stack_options()
        return StickBreakingConfig(
            threshold=layer_options.halting_threshold,
            halting_dropout=layer_options.halting_dropout,
            hidden_state_mode=layer_options.halting_hidden_state_mode,
            halting_gate_config=build_controller_stack(
                options,
                hidden_dim=options.hidden_dim or self.builder.output_dim,
                output_dim=layer_options.halting_output_dim,
                layer_model_config=LinearLayerConfig(
                    bias_flag=options.bias_flag,
                ),
            ),
        )

    def __build_expert_model_config(self) -> LayerStackConfig:
        return build_controller_stack(
            self.__expert_stack_options(),
            layer_model_config=self.build_adaptive_linear_layer_config(
                self.__expert_stack_options().bias_flag
            ),
        )

    def __build_sampler_config(self) -> SamplerConfig:
        mixture_options = self.builder.mixture_options
        sampler_options = self.builder.sampler_options
        return SamplerConfig(
            top_k=mixture_options.top_k,
            threshold=sampler_options.threshold,
            filter_above_threshold=sampler_options.filter_above_threshold,
            num_topk_samples=sampler_options.num_topk_samples,
            normalize_probabilities_flag=(
                sampler_options.normalize_probabilities_flag
            ),
            noisy_topk_flag=sampler_options.noisy_topk_flag,
            num_experts=mixture_options.num_experts,
            coefficient_of_variation_loss_weight=(
                sampler_options.coefficient_of_variation_loss_weight
            ),
            switch_loss_weight=sampler_options.switch_loss_weight,
            zero_centred_loss_weight=sampler_options.zero_centred_loss_weight,
            mutual_information_loss_weight=(
                sampler_options.mutual_information_loss_weight
            ),
            router_config=self.__build_router_config(),
        )

    def __build_router_config(self) -> RouterConfig:
        builder = self.builder
        mixture_options = builder.mixture_options
        router_options = builder.router_options
        sampler_stack_options = self.__sampler_stack_options()
        return RouterConfig(
            input_dim=builder.hidden_dim,
            num_experts=mixture_options.num_experts,
            noisy_topk_flag=router_options.noisy_topk_flag,
            model_config=build_controller_stack(
                sampler_stack_options,
                layer_model_config=self.build_adaptive_linear_layer_config(
                    sampler_stack_options.bias_flag
                ),
            ),
        )

    def __build_weight_config(self) -> DynamicWeightConfig | None:
        builder = self.builder
        return build_weight_config(
            builder.weight_option,
            generator_depth=builder.generator_depth,
            decay_schedule=builder.weight_decay_schedule,
            decay_rate=builder.weight_decay_rate,
            decay_warmup_batches=builder.weight_decay_warmup_batches,
            normalization_option=builder.weight_normalization_option,
            normalization_position_option=(
                builder.weight_normalization_position_option
            ),
            bank_expansion_factor=builder.weight_bank_expansion_factor,
        )

    def __build_bias_config(self) -> DynamicBiasConfig | None:
        builder = self.builder
        return build_bias_config(
            builder.bias_option,
            decay_schedule=builder.bias_decay_schedule,
            decay_rate=builder.bias_decay_rate,
            decay_warmup_batches=builder.bias_decay_warmup_batches,
            bank_expansion_factor=builder.bias_bank_expansion_factor,
        )

    def __build_diagonal_config(self) -> DynamicDiagonalConfig | None:
        return build_diagonal_config(
            self.builder.diagonal_option,
        )

    def __build_mask_config(self) -> AxisMaskConfig | None:
        builder = self.builder
        return build_mask_config(
            builder.row_mask_option,
            mask_dimension_option=builder.mask_dimension_option,
            mask_threshold=builder.mask_threshold,
            mask_surrogate_scale=builder.mask_surrogate_scale,
            mask_floor=builder.mask_floor,
            mask_transition_width=builder.mask_transition_width,
        )

    def __build_generator_model_config(self) -> LayerStackConfig:
        builder = self.builder
        options = self.__adaptive_generator_stack_options()
        return build_controller_stack(
            options,
            layer_model_config=LinearLayerConfig(
                bias_flag=builder.bias_flag,
            ),
        )

    def __gate_stack_options(self) -> ExpertsControllerStackOptions:
        return self.builder.layer_controller_options.gate_stack_options

    def __halting_stack_options(self) -> ExpertsControllerStackOptions:
        return self.builder.layer_controller_options.halting_stack_options

    def __expert_stack_options(self) -> ExpertsControllerStackOptions:
        return self.builder.expert_stack_options

    def __sampler_stack_options(self) -> ExpertsControllerStackOptions:
        return self.builder.sampler_stack_options

    def __adaptive_generator_stack_options(
        self,
    ) -> ExpertsAdaptiveGeneratorStackOptions:
        return self.builder.adaptive_generator_stack_options
