from typing import Any

from emperor.augmentations.adaptive_parameters.config import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.augmentations.adaptive_parameters.core.bias import (
    DynamicBiasConfig,
    WeightedBankDynamicBiasConfig,
)
from emperor.augmentations.adaptive_parameters.core.diagonal import (
    DynamicDiagonalConfig,
)
from emperor.augmentations.adaptive_parameters.core.mask import (
    AxisMaskConfig,
    DiagonalAxisMaskConfig,
    PerAxisScoreMaskConfig,
    TopSliceAxisMaskConfig,
    WeightInformedScoreAxisMaskConfig,
)
from emperor.augmentations.adaptive_parameters.core.weight import (
    DualModelDynamicWeightConfig,
    DynamicWeightConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    SingleModelDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
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

from models.experts.experts_linear_adaptive._controller_stack import (
    ControllerStackOptions,
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
        return MixtureOfExpertsModelConfig(
            input_dim=builder.hidden_dim,
            output_dim=builder.hidden_dim,
            top_k=builder.top_k,
            routing_initialization_mode=builder.routing_initialization_mode,
            sampler_config=self.__build_sampler_config(),
            stack_config=self.__build_main_stack_config(),
        )

    def __maybe_wrap_recurrent(
        self,
        block_config: MixtureOfExpertsModelConfig,
    ) -> MixtureOfExpertsModelConfig | RecurrentLayerConfig:
        builder = self.builder
        if not builder.recurrent_flag:
            return block_config
        return RecurrentLayerConfig(
            max_steps=builder.recurrent_max_steps,
            recurrent_layer_norm_position=builder.recurrent_layer_norm_position,
            block_config=block_config,
            gate_config=self.__build_recurrent_gate_config(),
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            halting_config=self.__build_halting_config(
                enabled=builder.recurrent_halting_flag
            ),
        )

    def __build_main_stack_config(self) -> LayerStackConfig:
        builder = self.builder
        gate_config = self.__build_gate_config()
        self.__validate_shared_gate_config(gate_config)
        return LayerStackConfig(
            input_dim=builder.hidden_dim,
            hidden_dim=builder.hidden_dim,
            output_dim=builder.hidden_dim,
            num_layers=builder.stack_num_layers,
            last_layer_bias_option=builder.stack_last_layer_bias_option,
            apply_output_pipeline_flag=builder.stack_apply_output_pipeline_flag,
            shared_gate_config=builder.shared_gate_config,
            layer_config=MixtureOfExpertsLayerConfig(
                activation=builder.stack_activation,
                layer_norm_position=builder.layer_norm_position,
                residual_connection_option=builder.stack_residual_connection_option,
                dropout_probability=builder.stack_dropout_probability,
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
        return MixtureOfExpertsConfig(
            input_dim=builder.hidden_dim,
            output_dim=builder.hidden_dim,
            top_k=builder.top_k,
            num_experts=builder.num_experts,
            capacity_factor=builder.capacity_factor,
            dropped_token_behavior=builder.dropped_token_behavior,
            compute_expert_mixture_flag=builder.compute_expert_mixture_flag,
            weighted_parameters_flag=builder.weighted_parameters_flag,
            weighting_position_option=builder.weighting_position_option,
            routing_initialization_mode=builder.routing_initialization_mode,
            sampler_config=self.__build_sampler_config(),
            expert_model_config=self.__build_expert_model_config(),
        )

    def __build_gate_config(
        self,
        enabled: bool | None = None,
    ) -> GateConfig | None:
        builder = self.builder
        if enabled is None:
            enabled = builder.stack_gate_flag
        if not enabled:
            return None
        return GateConfig(
            model_config=self.__build_gate_model_config(enabled=enabled),
            option=builder.gate_option,
            activation=builder.gate_activation,
        )

    def __build_recurrent_gate_config(self) -> GateConfig | None:
        builder = self.builder
        if not builder.recurrent_gate_flag:
            return None
        return GateConfig(
            model_config=self.__build_gate_model_config(
                enabled=builder.recurrent_gate_flag,
            ),
            option=builder.recurrent_gate_option,
            activation=builder.recurrent_gate_activation,
        )

    def __build_gate_model_config(
        self,
        enabled: bool,
    ) -> LayerStackConfig | None:
        if not enabled:
            return None
        return build_controller_stack(
            self.__gate_stack_options(),
            layer_model_config=LinearLayerConfig(
                bias_flag=self.builder.gate_bias_flag,
            ),
        )

    def __build_halting_config(
        self,
        enabled: bool | None = None,
    ) -> StickBreakingConfig | None:
        builder = self.builder
        if enabled is None:
            enabled = builder.stack_halting_flag
        if not enabled:
            return None
        options = self.__halting_stack_options()
        return StickBreakingConfig(
            threshold=builder.halting_threshold,
            halting_dropout=builder.halting_dropout,
            hidden_state_mode=builder.halting_hidden_state_mode,
            halting_gate_config=build_controller_stack(
                options,
                hidden_dim=options.hidden_dim or builder.output_dim,
                output_dim=builder.halting_output_dim,
                layer_model_config=LinearLayerConfig(
                    bias_flag=builder.halting_bias_flag,
                ),
            ),
        )

    def __build_expert_model_config(self) -> LayerStackConfig:
        return build_controller_stack(
            self.__expert_stack_options(),
            layer_model_config=self.build_adaptive_linear_layer_config(
                self.builder.expert_bias_flag
            ),
        )

    def __build_sampler_config(self) -> SamplerConfig:
        builder = self.builder
        return SamplerConfig(
            top_k=builder.top_k,
            threshold=builder.sampler_threshold,
            filter_above_threshold=builder.sampler_filter_above_threshold,
            num_topk_samples=builder.sampler_num_topk_samples,
            normalize_probabilities_flag=builder.sampler_normalize_probabilities_flag,
            noisy_topk_flag=builder.sampler_noisy_topk_flag,
            num_experts=builder.num_experts,
            coefficient_of_variation_loss_weight=(
                builder.sampler_coefficient_of_variation_loss_weight
            ),
            switch_loss_weight=builder.sampler_switch_loss_weight,
            zero_centred_loss_weight=builder.sampler_zero_centred_loss_weight,
            mutual_information_loss_weight=(
                builder.sampler_mutual_information_loss_weight
            ),
            router_config=self.__build_router_config(),
        )

    def __build_router_config(self) -> RouterConfig:
        builder = self.builder
        return RouterConfig(
            input_dim=builder.hidden_dim,
            num_experts=builder.num_experts,
            noisy_topk_flag=builder.router_noisy_topk_flag,
            model_config=build_controller_stack(
                self.__sampler_stack_options(),
                layer_model_config=self.build_adaptive_linear_layer_config(
                    builder.sampler_bias_flag
                ),
            ),
        )

    def __build_weight_config(self) -> DynamicWeightConfig | None:
        builder = self.builder
        if builder.weight_option is None:
            return None
        kwargs: dict[str, Any] = {
            "generator_depth": builder.generator_depth,
            "decay_schedule": builder.weight_decay_schedule,
            "decay_rate": builder.weight_decay_rate,
            "decay_warmup_batches": builder.weight_decay_warmup_batches,
        }
        if builder.weight_option in {
            SingleModelDynamicWeightConfig,
            DualModelDynamicWeightConfig,
        }:
            kwargs["normalization_option"] = builder.weight_normalization_option
            kwargs["normalization_position_option"] = (
                builder.weight_normalization_position_option
            )
        elif builder.weight_option in {
            LowRankDynamicWeightConfig,
            HypernetworkDynamicWeightConfig,
        }:
            kwargs["normalization_option"] = builder.weight_normalization_option
        elif builder.weight_option in {
            LayeredWeightedBankDynamicWeightConfig,
            SoftWeightedBankDynamicWeightConfig,
        }:
            kwargs["bank_expansion_factor"] = builder.weight_bank_expansion_factor
        return builder.weight_option(**kwargs)

    def __build_bias_config(self) -> DynamicBiasConfig | None:
        builder = self.builder
        if builder.bias_option is None:
            return None
        kwargs: dict[str, Any] = {
            "decay_schedule": builder.bias_decay_schedule,
            "decay_rate": builder.bias_decay_rate,
            "decay_warmup_batches": builder.bias_decay_warmup_batches,
        }
        if builder.bias_option is WeightedBankDynamicBiasConfig:
            kwargs["bank_expansion_factor"] = builder.bias_bank_expansion_factor
        return builder.bias_option(**kwargs)

    def __build_diagonal_config(self) -> DynamicDiagonalConfig | None:
        if self.builder.diagonal_option is None:
            return None
        return self.builder.diagonal_option()

    def __build_mask_config(self) -> AxisMaskConfig | None:
        builder = self.builder
        if builder.row_mask_option is None:
            return None
        kwargs: dict[str, Any] = {
            "mask_threshold": builder.mask_threshold,
            "mask_surrogate_scale": builder.mask_surrogate_scale,
            "mask_floor": builder.mask_floor,
        }
        if builder.row_mask_option in {
            WeightInformedScoreAxisMaskConfig,
            PerAxisScoreMaskConfig,
            TopSliceAxisMaskConfig,
        }:
            kwargs["mask_dimension_option"] = builder.mask_dimension_option
        if builder.row_mask_option in {TopSliceAxisMaskConfig, DiagonalAxisMaskConfig}:
            kwargs["mask_transition_width"] = builder.mask_transition_width
        return builder.row_mask_option(**kwargs)

    def __build_generator_model_config(self) -> LayerStackConfig:
        builder = self.builder
        return build_controller_stack(
            self.__adaptive_generator_stack_options(),
            layer_model_config=LinearLayerConfig(
                bias_flag=builder.bias_flag,
            ),
        )

    def __gate_stack_options(self) -> ControllerStackOptions:
        builder = self.builder
        return ControllerStackOptions(
            hidden_dim=builder.gate_hidden_dim,
            num_layers=builder.gate_stack_num_layers,
            last_layer_bias_option=builder.gate_stack_last_layer_bias_option,
            apply_output_pipeline_flag=builder.gate_stack_apply_output_pipeline_flag,
            activation=builder.gate_stack_activation,
            layer_norm_position=builder.gate_layer_norm_position,
            residual_connection_option=builder.gate_stack_residual_connection_option,
            dropout_probability=builder.gate_stack_dropout_probability,
        )

    def __halting_stack_options(self) -> ControllerStackOptions:
        builder = self.builder
        return ControllerStackOptions(
            hidden_dim=builder.halting_hidden_dim,
            num_layers=builder.halting_stack_num_layers,
            last_layer_bias_option=builder.halting_stack_last_layer_bias_option,
            apply_output_pipeline_flag=builder.halting_stack_apply_output_pipeline_flag,
            activation=builder.halting_stack_activation,
            layer_norm_position=builder.halting_layer_norm_position,
            residual_connection_option=(
                builder.halting_stack_residual_connection_option
            ),
            dropout_probability=builder.halting_stack_dropout_probability,
        )

    def __expert_stack_options(self) -> ControllerStackOptions:
        builder = self.builder
        return ControllerStackOptions(
            hidden_dim=builder.hidden_dim,
            num_layers=builder.expert_stack_num_layers,
            last_layer_bias_option=builder.expert_stack_last_layer_bias_option,
            apply_output_pipeline_flag=(
                builder.expert_stack_apply_output_pipeline_flag
            ),
            activation=builder.expert_stack_activation,
            layer_norm_position=builder.expert_stack_layer_norm_position,
            residual_connection_option=builder.expert_stack_residual_connection_option,
            dropout_probability=builder.expert_stack_dropout_probability,
        )

    def __sampler_stack_options(self) -> ControllerStackOptions:
        builder = self.builder
        return ControllerStackOptions(
            hidden_dim=builder.hidden_dim,
            num_layers=builder.sampler_stack_num_layers,
            last_layer_bias_option=builder.sampler_stack_last_layer_bias_option,
            apply_output_pipeline_flag=(
                builder.sampler_stack_apply_output_pipeline_flag
            ),
            activation=builder.sampler_stack_activation,
            layer_norm_position=builder.sampler_stack_layer_norm_position,
            residual_connection_option=(
                builder.sampler_stack_residual_connection_option
            ),
            dropout_probability=builder.sampler_stack_dropout_probability,
        )

    def __adaptive_generator_stack_options(self) -> ControllerStackOptions:
        builder = self.builder
        return ControllerStackOptions(
            hidden_dim=builder.adaptive_generator_stack_hidden_dim,
            num_layers=builder.adaptive_generator_stack_num_layers,
            last_layer_bias_option=(
                builder.adaptive_generator_stack_last_layer_bias_option
            ),
            apply_output_pipeline_flag=(
                builder.adaptive_generator_stack_apply_output_pipeline_flag
            ),
            activation=builder.adaptive_generator_stack_activation,
            layer_norm_position=builder.adaptive_generator_stack_layer_norm_position,
            residual_connection_option=(
                builder.adaptive_generator_stack_residual_connection_option
            ),
            dropout_probability=builder.adaptive_generator_stack_dropout_probability,
        )
