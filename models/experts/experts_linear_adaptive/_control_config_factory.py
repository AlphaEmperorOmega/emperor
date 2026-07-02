from dataclasses import dataclass

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
from emperor.augmentations.adaptive_parameters.options import (
    BankExpansionFactorOptions,
    DynamicDepthOptions,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
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
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsStackOptions,
)
from models.experts.experts_linear_adaptive._controller_stack import (
    build_controller_stack,
)


@dataclass(frozen=True)
class AdaptiveAugmentationDependencies:
    generator_depth: DynamicDepthOptions
    diagonal_option: type[DynamicDiagonalConfig] | None
    bias_option: type[DynamicBiasConfig] | None
    weight_option: type[DynamicWeightConfig] | None
    weight_normalization_option: WeightNormalizationOptions
    weight_normalization_position_option: WeightNormalizationPositionOptions
    weight_decay_schedule: WeightDecayScheduleOptions
    weight_decay_rate: float
    weight_decay_warmup_batches: int
    weight_bank_expansion_factor: BankExpansionFactorOptions
    bias_decay_schedule: WeightDecayScheduleOptions
    bias_decay_rate: float
    bias_decay_warmup_batches: int
    bias_bank_expansion_factor: BankExpansionFactorOptions
    row_mask_option: type[AxisMaskConfig] | None
    mask_dimension_option: MaskDimensionOptions
    mask_threshold: float
    mask_surrogate_scale: float
    mask_floor: float
    mask_transition_width: float


@dataclass(frozen=True)
class ControlConfigDependencies:
    stack_options: ExpertsStackOptions
    mixture_options: ExpertsMixtureOptions
    expert_stack_options: ExpertsControllerStackOptions
    sampler_options: ExpertsSamplerOptions
    router_options: ExpertsRouterOptions
    sampler_stack_options: ExpertsControllerStackOptions
    layer_controller_options: ExpertsLayerControllerOptions
    recurrent_controller_options: ExpertsRecurrentControllerOptions
    adaptive_generator_stack_options: ExpertsAdaptiveGeneratorStackOptions
    adaptive_augmentation_options: AdaptiveAugmentationDependencies
    hidden_dim: int
    output_dim: int


class ControlConfigFactory:
    def __init__(self, dependencies: ControlConfigDependencies) -> None:
        self.stack_options = dependencies.stack_options
        self.mixture_options = dependencies.mixture_options
        self.expert_stack_options = dependencies.expert_stack_options
        self.sampler_options = dependencies.sampler_options
        self.router_options = dependencies.router_options
        self.sampler_stack_options = dependencies.sampler_stack_options
        self.layer_controller_options = dependencies.layer_controller_options
        self.recurrent_controller_options = dependencies.recurrent_controller_options
        self.adaptive_generator_stack_options = (
            dependencies.adaptive_generator_stack_options
        )
        self.adaptive_augmentation_options = (
            dependencies.adaptive_augmentation_options
        )
        self.hidden_dim = dependencies.hidden_dim
        self.output_dim = dependencies.output_dim

    def build(self) -> MixtureOfExpertsModelConfig | RecurrentLayerConfig:
        return self.__maybe_wrap_recurrent(self.__build_main_model_config())

    def build_adaptive_linear_layer_config(
        self,
        bias_flag: bool,
    ) -> AdaptiveLinearLayerConfig:
        adaptive_augmentation_config = self.__build_adaptive_augmentation_config()
        return AdaptiveLinearLayerConfig(
            bias_flag=bias_flag,
            adaptive_augmentation_config=adaptive_augmentation_config,
        )

    def __build_adaptive_augmentation_config(
        self,
    ) -> AdaptiveParameterAugmentationConfig:
        weight_config = self.__build_weight_config()
        bias_config = self.__build_bias_config()
        diagonal_config = self.__build_diagonal_config()
        mask_config = self.__build_mask_config()
        model_config = self.__build_generator_model_config()
        return AdaptiveParameterAugmentationConfig(
            weight_config=weight_config,
            bias_config=bias_config,
            diagonal_config=diagonal_config,
            mask_config=mask_config,
            model_config=model_config,
        )

    def __build_main_model_config(self) -> MixtureOfExpertsModelConfig:
        mixture_options = self.mixture_options
        return MixtureOfExpertsModelConfig(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
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
        recurrent_options = self.recurrent_controller_options
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
                enabled_flag=recurrent_options.recurrent_halting_flag
            ),
        )

    def __build_main_stack_config(self) -> LayerStackConfig:
        stack_options = self.stack_options
        layer_controller = self.layer_controller_options
        gate_config = self.__build_gate_config()
        halting_config = self.__build_halting_config()
        layer_config = self.__build_layer_config(gate_config, halting_config)
        return LayerStackConfig(
            input_dim=stack_options.hidden_dim,
            hidden_dim=stack_options.hidden_dim,
            output_dim=stack_options.hidden_dim,
            num_layers=stack_options.num_layers,
            last_layer_bias_option=stack_options.last_layer_bias_option,
            apply_output_pipeline_flag=stack_options.apply_output_pipeline_flag,
            shared_gate_config=layer_controller.shared_gate_config,
            layer_config=layer_config,
        )

    def __build_layer_config(
        self,
        gate_config: GateConfig | None,
        halting_config: StickBreakingConfig | None,
    ) -> LayerConfig:
        stack_options = self.stack_options
        return MixtureOfExpertsLayerConfig(
            activation=stack_options.activation,
            layer_norm_position=stack_options.layer_norm_position,
            residual_connection_option=(
                stack_options.residual_connection_option
            ),
            dropout_probability=stack_options.dropout_probability,
            gate_config=gate_config,
            halting_config=halting_config,
            layer_model_config=self.__build_mixture_of_experts_config(),
        )

    def __build_mixture_of_experts_config(self) -> MixtureOfExpertsConfig:
        mixture_options = self.mixture_options
        return MixtureOfExpertsConfig(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
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
        enabled_flag: bool | None = None,
    ) -> GateConfig | None:
        layer_controller = self.layer_controller_options
        if enabled_flag is None:
            enabled_flag = layer_controller.stack_gate_flag
        if not enabled_flag:
            return None
        model_config = self.__build_gate_model_config(enabled_flag=enabled_flag)
        return GateConfig(
            model_config=model_config,
            option=layer_controller.gate_option,
            activation=layer_controller.gate_activation,
        )

    def __build_recurrent_gate_config(self) -> GateConfig | None:
        recurrent_options = self.recurrent_controller_options
        if not recurrent_options.recurrent_gate_flag:
            return None
        model_config = self.__build_gate_model_config(
            enabled_flag=recurrent_options.recurrent_gate_flag,
        )
        return GateConfig(
            model_config=model_config,
            option=recurrent_options.recurrent_gate_option,
            activation=recurrent_options.recurrent_gate_activation,
        )

    def __build_gate_model_config(
        self,
        enabled_flag: bool,
    ) -> LayerStackConfig | None:
        if not enabled_flag:
            return None
        gate_stack_options = self.layer_controller_options.gate_stack_options
        return self.__build_linear_controller_stack(gate_stack_options)

    def __build_halting_config(
        self,
        enabled_flag: bool | None = None,
    ) -> StickBreakingConfig | None:
        layer_controller = self.layer_controller_options
        if enabled_flag is None:
            enabled_flag = layer_controller.stack_halting_flag
        if not enabled_flag:
            return None
        halting_stack_options = layer_controller.halting_stack_options
        halting_gate_config = self.__build_halting_gate_stack(
            halting_stack_options
        )
        return StickBreakingConfig(
            threshold=layer_controller.halting_threshold,
            halting_dropout=layer_controller.halting_dropout,
            hidden_state_mode=layer_controller.halting_hidden_state_mode,
            halting_gate_config=halting_gate_config,
        )

    def __build_expert_model_config(self) -> LayerStackConfig:
        expert_stack_options = self.expert_stack_options
        layer_model_config = self.build_adaptive_linear_layer_config(
            expert_stack_options.bias_flag
        )
        return self.__build_controller_stack(
            expert_stack_options,
            layer_model_config,
        )

    def __build_sampler_config(self) -> SamplerConfig:
        mixture_options = self.mixture_options
        sampler_options = self.sampler_options
        router_config = self.__build_router_config()
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
            router_config=router_config,
        )

    def __build_router_config(self) -> RouterConfig:
        mixture_options = self.mixture_options
        router_options = self.router_options
        sampler_stack_options = self.sampler_stack_options
        layer_model_config = self.build_adaptive_linear_layer_config(
            sampler_stack_options.bias_flag
        )
        model_config = self.__build_controller_stack(
            sampler_stack_options,
            layer_model_config,
        )
        return RouterConfig(
            input_dim=self.hidden_dim,
            num_experts=mixture_options.num_experts,
            noisy_topk_flag=router_options.noisy_topk_flag,
            model_config=model_config,
        )

    def __build_weight_config(self) -> DynamicWeightConfig | None:
        adaptive_options = self.adaptive_augmentation_options
        return build_weight_config(
            adaptive_options.weight_option,
            generator_depth=adaptive_options.generator_depth,
            decay_schedule=adaptive_options.weight_decay_schedule,
            decay_rate=adaptive_options.weight_decay_rate,
            decay_warmup_batches=adaptive_options.weight_decay_warmup_batches,
            normalization_option=adaptive_options.weight_normalization_option,
            normalization_position_option=(
                adaptive_options.weight_normalization_position_option
            ),
            bank_expansion_factor=adaptive_options.weight_bank_expansion_factor,
        )

    def __build_bias_config(self) -> DynamicBiasConfig | None:
        adaptive_options = self.adaptive_augmentation_options
        return build_bias_config(
            adaptive_options.bias_option,
            decay_schedule=adaptive_options.bias_decay_schedule,
            decay_rate=adaptive_options.bias_decay_rate,
            decay_warmup_batches=adaptive_options.bias_decay_warmup_batches,
            bank_expansion_factor=adaptive_options.bias_bank_expansion_factor,
        )

    def __build_diagonal_config(self) -> DynamicDiagonalConfig | None:
        adaptive_options = self.adaptive_augmentation_options
        return build_diagonal_config(
            adaptive_options.diagonal_option,
        )

    def __build_mask_config(self) -> AxisMaskConfig | None:
        adaptive_options = self.adaptive_augmentation_options
        return build_mask_config(
            adaptive_options.row_mask_option,
            mask_dimension_option=adaptive_options.mask_dimension_option,
            mask_threshold=adaptive_options.mask_threshold,
            mask_surrogate_scale=adaptive_options.mask_surrogate_scale,
            mask_floor=adaptive_options.mask_floor,
            mask_transition_width=adaptive_options.mask_transition_width,
        )

    def __build_generator_model_config(self) -> LayerStackConfig:
        adaptive_generator = self.adaptive_generator_stack_options
        layer_model_config = LinearLayerConfig(
            bias_flag=self.stack_options.bias_flag,
        )
        return self.__build_controller_stack(
            adaptive_generator,
            layer_model_config,
        )

    def __build_linear_controller_stack(
        self,
        options: ExpertsControllerStackOptions,
    ) -> LayerStackConfig:
        layer_model_config = LinearLayerConfig(
            bias_flag=options.bias_flag,
        )
        return self.__build_controller_stack(options, layer_model_config)

    def __build_halting_gate_stack(
        self,
        options: ExpertsControllerStackOptions,
    ) -> LayerStackConfig:
        layer_controller = self.layer_controller_options
        layer_model_config = LinearLayerConfig(
            bias_flag=options.bias_flag,
        )
        return build_controller_stack(
            options,
            hidden_dim=options.hidden_dim or self.output_dim,
            output_dim=layer_controller.halting_output_dim,
            layer_model_config=layer_model_config,
        )

    @staticmethod
    def __build_controller_stack(
        options: ExpertsControllerStackOptions | ExpertsAdaptiveGeneratorStackOptions,
        layer_model_config: LinearLayerConfig | AdaptiveLinearLayerConfig,
    ) -> LayerStackConfig:
        return build_controller_stack(
            options,
            layer_model_config=layer_model_config,
        )
