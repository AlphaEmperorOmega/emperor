# ruff: noqa: E501

from dataclasses import dataclass

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AxisMaskConfig,
    DynamicBiasConfig,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
)
from emperor.experts import (
    MixtureOfExpertsConfig,
    MixtureOfExpertsLayerConfig,
    MixtureOfExpertsModelConfig,
)
from emperor.halting import HaltingConfig
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    RecurrentLayerConfig,
    ResidualConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.sampler import RouterConfig, SamplerConfig
from models.neuron.expert_linear_adaptive._hidden._adaptive_generator_stack_config_factory import (
    AdaptiveGeneratorStackConfigFactory,
)
from models.neuron.expert_linear_adaptive._hidden._adaptive_parameter_config_factory import (
    build_bias_config,
    build_diagonal_config,
    build_mask_config,
    build_weight_config,
)
from models.neuron.expert_linear_adaptive._hidden._control_support import (
    ExpertsGateConfigFactory,
    ExpertsHaltingConfigFactory,
    ExpertsMemoryConfigFactory,
    ExpertsRecurrentConfigFactory,
    build_controller_stack,
)
from models.neuron.expert_linear_adaptive._hidden._router_controller_config import (
    RouterControllerModelConfig,
)
from models.neuron.expert_linear_adaptive._hidden.runtime_options import (
    AdaptiveGeneratorStackOptions,
    AdaptiveGeneratorStackSource,
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsStackOptions,
    ExpertsSubmoduleStackOptions,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
)


@dataclass(frozen=True)
class ControlConfigDependencies:
    stack_options: ExpertsStackOptions
    submodule_stack_options: ExpertsSubmoduleStackOptions
    mixture_options: ExpertsMixtureOptions
    expert_stack_options: ExpertsSubmoduleStackOptions
    sampler_options: ExpertsSamplerOptions
    router_options: ExpertsRouterOptions
    router_stack_options: ExpertsSubmoduleStackOptions
    router_layer_controller_options: ExpertsLayerControllerOptions
    router_dynamic_memory_options: ExpertsDynamicMemoryOptions
    router_recurrent_controller_options: ExpertsRecurrentControllerOptions
    layer_controller_options: ExpertsLayerControllerOptions
    dynamic_memory_options: ExpertsDynamicMemoryOptions
    recurrent_controller_options: ExpertsRecurrentControllerOptions
    expert_layer_controller_options: ExpertsLayerControllerOptions
    expert_dynamic_memory_options: ExpertsDynamicMemoryOptions
    expert_recurrent_controller_options: ExpertsRecurrentControllerOptions
    adaptive_generator_stack_options: AdaptiveGeneratorStackOptions
    hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions
    hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions
    hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions
    hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions
    router_adaptive_weight_options: HiddenAdaptiveWeightOptions
    router_adaptive_bias_options: HiddenAdaptiveBiasOptions
    router_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions
    router_adaptive_mask_options: HiddenAdaptiveMaskOptions
    hidden_dim: int
    output_dim: int


class ControlConfigFactory:
    def __init__(self, dependencies: ControlConfigDependencies) -> None:
        self.stack_options = dependencies.stack_options
        self.submodule_stack_options = dependencies.submodule_stack_options
        self.mixture_options = dependencies.mixture_options
        self.expert_stack_options = dependencies.expert_stack_options
        self.sampler_options = dependencies.sampler_options
        self.router_options = dependencies.router_options
        self.router_stack_options = dependencies.router_stack_options
        self.router_layer_controller_options = (
            dependencies.router_layer_controller_options
        )
        self.router_dynamic_memory_options = dependencies.router_dynamic_memory_options
        self.router_recurrent_controller_options = (
            dependencies.router_recurrent_controller_options
        )
        self.layer_controller_options = dependencies.layer_controller_options
        self.dynamic_memory_options = dependencies.dynamic_memory_options
        self.recurrent_controller_options = dependencies.recurrent_controller_options
        self.expert_layer_controller_options = (
            dependencies.expert_layer_controller_options
        )
        self.expert_dynamic_memory_options = dependencies.expert_dynamic_memory_options
        self.expert_recurrent_controller_options = (
            dependencies.expert_recurrent_controller_options
        )
        self.adaptive_generator_stack_options = (
            dependencies.adaptive_generator_stack_options
        )
        self.adaptive_generator_stack_config_factory = (
            AdaptiveGeneratorStackConfigFactory(self.adaptive_generator_stack_options)
        )
        self.hidden_adaptive_weight_options = (
            dependencies.hidden_adaptive_weight_options
        )
        self.hidden_adaptive_bias_options = dependencies.hidden_adaptive_bias_options
        self.hidden_adaptive_diagonal_options = (
            dependencies.hidden_adaptive_diagonal_options
        )
        self.hidden_adaptive_mask_options = dependencies.hidden_adaptive_mask_options
        self.router_adaptive_weight_options = (
            dependencies.router_adaptive_weight_options
        )
        self.router_adaptive_bias_options = dependencies.router_adaptive_bias_options
        self.router_adaptive_diagonal_options = (
            dependencies.router_adaptive_diagonal_options
        )
        self.router_adaptive_mask_options = dependencies.router_adaptive_mask_options
        self.hidden_adaptive_augmentation_config = (
            self.__build_adaptive_augmentation_config(
                weight_options=self.hidden_adaptive_weight_options,
                bias_options=self.hidden_adaptive_bias_options,
                diagonal_options=self.hidden_adaptive_diagonal_options,
                mask_options=self.hidden_adaptive_mask_options,
            )
        )
        self.router_adaptive_augmentation_config = (
            self.__build_adaptive_augmentation_config(
                weight_options=self.router_adaptive_weight_options,
                bias_options=self.router_adaptive_bias_options,
                diagonal_options=self.router_adaptive_diagonal_options,
                mask_options=self.router_adaptive_mask_options,
            )
        )
        self.hidden_dim = dependencies.hidden_dim
        self.output_dim = dependencies.output_dim
        self.gate_config_factory = ExpertsGateConfigFactory(
            layer_controller_options=self.layer_controller_options,
            recurrent_controller_options=self.recurrent_controller_options,
            submodule_stack_options=self.submodule_stack_options,
        )
        self.halting_config_factory = ExpertsHaltingConfigFactory(
            layer_controller_options=self.layer_controller_options,
            recurrent_controller_options=self.recurrent_controller_options,
            submodule_stack_options=self.submodule_stack_options,
            output_dim=self.output_dim,
        )
        self.memory_config_factory = ExpertsMemoryConfigFactory(
            stack_options=self.stack_options,
            dynamic_memory_options=self.dynamic_memory_options,
            submodule_stack_options=self.submodule_stack_options,
        )
        self.recurrent_config_factory = ExpertsRecurrentConfigFactory(
            recurrent_controller_options=self.recurrent_controller_options,
            gate_config_factory=self.gate_config_factory,
            halting_config_factory=self.halting_config_factory,
        )
        self.expert_gate_config_factory = ExpertsGateConfigFactory(
            layer_controller_options=self.expert_layer_controller_options,
            recurrent_controller_options=self.expert_recurrent_controller_options,
            submodule_stack_options=self.expert_stack_options,
            recurrent_stack_inherits_gate_stack=False,
        )
        self.expert_halting_config_factory = ExpertsHaltingConfigFactory(
            layer_controller_options=self.expert_layer_controller_options,
            recurrent_controller_options=self.expert_recurrent_controller_options,
            submodule_stack_options=self.expert_stack_options,
            output_dim=self.expert_stack_options.hidden_dim,
            halting_stack_defaults=self.expert_stack_options,
            recurrent_stack_inherits_halting_stack=False,
        )
        self.expert_memory_config_factory = ExpertsMemoryConfigFactory(
            stack_options=self.expert_stack_options,
            dynamic_memory_options=self.expert_dynamic_memory_options,
            submodule_stack_options=self.expert_stack_options,
        )
        self.expert_recurrent_config_factory = ExpertsRecurrentConfigFactory(
            recurrent_controller_options=self.expert_recurrent_controller_options,
            gate_config_factory=self.expert_gate_config_factory,
            halting_config_factory=self.expert_halting_config_factory,
        )
        self.router_gate_config_factory = ExpertsGateConfigFactory(
            layer_controller_options=self.router_layer_controller_options,
            recurrent_controller_options=self.router_recurrent_controller_options,
            submodule_stack_options=self.submodule_stack_options,
            recurrent_stack_inherits_gate_stack=False,
        )
        self.router_halting_config_factory = ExpertsHaltingConfigFactory(
            layer_controller_options=self.router_layer_controller_options,
            recurrent_controller_options=self.router_recurrent_controller_options,
            submodule_stack_options=self.submodule_stack_options,
            output_dim=self.router_stack_options.hidden_dim,
            recurrent_stack_inherits_halting_stack=False,
        )
        self.router_memory_config_factory = ExpertsMemoryConfigFactory(
            stack_options=self.router_stack_options,
            dynamic_memory_options=self.router_dynamic_memory_options,
            submodule_stack_options=self.submodule_stack_options,
        )
        self.router_recurrent_config_factory = ExpertsRecurrentConfigFactory(
            recurrent_controller_options=self.router_recurrent_controller_options,
            gate_config_factory=self.router_gate_config_factory,
            halting_config_factory=self.router_halting_config_factory,
        )

    def build(self) -> MixtureOfExpertsModelConfig | RecurrentLayerConfig:
        return self.recurrent_config_factory.build_config(
            self.__build_main_model_config()
        )

    def build_hidden_adaptive_linear_layer_config(
        self,
        bias_flag: bool,
    ) -> AdaptiveLinearLayerConfig:
        return AdaptiveLinearLayerConfig(
            bias_flag=bias_flag,
            adaptive_augmentation_config=self.hidden_adaptive_augmentation_config,
        )

    def build_router_adaptive_linear_layer_config(
        self,
        bias_flag: bool,
    ) -> AdaptiveLinearLayerConfig:
        return AdaptiveLinearLayerConfig(
            bias_flag=bias_flag,
            adaptive_augmentation_config=self.router_adaptive_augmentation_config,
        )

    def __build_adaptive_augmentation_config(
        self,
        *,
        weight_options: HiddenAdaptiveWeightOptions,
        bias_options: HiddenAdaptiveBiasOptions,
        diagonal_options: HiddenAdaptiveDiagonalOptions,
        mask_options: HiddenAdaptiveMaskOptions,
    ) -> AdaptiveParameterAugmentationConfig:
        weight_config = self.__build_weight_config(weight_options)
        bias_config = self.__build_bias_config(bias_options)
        diagonal_config = self.__build_diagonal_config(diagonal_options)
        mask_config = self.__build_mask_config(mask_options)
        model_config = self.__build_shared_generator_model_config()
        return AdaptiveParameterAugmentationConfig(
            weight_config=weight_config,
            bias_config=bias_config,
            diagonal_config=diagonal_config,
            mask_config=mask_config,
            model_config=model_config,
        )

    def __resolve_enabled_adaptive_parameter_option(
        self,
        *,
        option_flag: bool,
        option: type | None,
        option_flag_name: str,
        option_name: str,
    ) -> type | None:
        if not option_flag:
            return None
        if option is None:
            raise ValueError(
                f"{option_name} must be set when {option_flag_name} is True."
            )
        return option

    def __build_main_model_config(self) -> MixtureOfExpertsModelConfig:
        mixture_options = self.mixture_options
        return MixtureOfExpertsModelConfig(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            top_k=mixture_options.top_k,
            routing_initialization_mode=(mixture_options.routing_initialization_mode),
            sampler_config=self.__build_sampler_config(),
            stack_config=self.__build_main_stack_config(),
        )

    def __build_main_stack_config(self) -> LayerStackConfig:
        stack_options = self.stack_options
        layer_controller = self.layer_controller_options
        gate_config = self.gate_config_factory.build_gate_config()
        halting_config = self.halting_config_factory.build_halting_config()
        memory_config = self.memory_config_factory.build_memory_config()
        layer_config = self.__build_layer_config(gate_config, halting_config)
        return LayerStackConfig(
            input_dim=stack_options.hidden_dim,
            hidden_dim=stack_options.hidden_dim,
            output_dim=stack_options.hidden_dim,
            num_layers=stack_options.num_layers,
            last_layer_bias_option=stack_options.last_layer_bias_option,
            apply_output_pipeline_flag=stack_options.apply_output_pipeline_flag,
            shared_gate_config=layer_controller.shared_gate_config,
            shared_memory_config=memory_config,
            layer_config=layer_config,
        )

    def __build_layer_config(
        self,
        gate_config: GateConfig | None,
        halting_config: HaltingConfig | None,
    ) -> LayerConfig:
        stack_options = self.stack_options
        return MixtureOfExpertsLayerConfig(
            activation=stack_options.activation,
            layer_norm_position=stack_options.layer_norm_position,
            residual_config=None
            if (stack_options.residual_connection_option) is None
            else ResidualConfig(option=(stack_options.residual_connection_option)),
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
            compute_expert_mixture_flag=(mixture_options.compute_expert_mixture_flag),
            weighted_parameters_flag=mixture_options.weighted_parameters_flag,
            weighting_position_option=mixture_options.weighting_position_option,
            routing_initialization_mode=(mixture_options.routing_initialization_mode),
            sampler_config=self.__build_sampler_config(),
            expert_model_config=self.__build_expert_model_config(),
        )

    def __build_expert_model_config(self) -> LayerStackConfig | RecurrentLayerConfig:
        return self.expert_recurrent_config_factory.build_config(
            self.__build_expert_stack_config()
        )

    def __build_expert_stack_config(self) -> LayerStackConfig:
        expert_stack_options = self.expert_stack_options
        layer_model_config = self.build_hidden_adaptive_linear_layer_config(
            expert_stack_options.bias_flag
        )
        layer_controller = self.expert_layer_controller_options
        gate_config = self.expert_gate_config_factory.build_gate_config()
        halting_config = self.expert_halting_config_factory.build_halting_config()
        memory_config = self.expert_memory_config_factory.build_memory_config()
        return LayerStackConfig(
            hidden_dim=expert_stack_options.hidden_dim,
            num_layers=expert_stack_options.num_layers,
            last_layer_bias_option=expert_stack_options.last_layer_bias_option,
            apply_output_pipeline_flag=(
                expert_stack_options.apply_output_pipeline_flag
            ),
            shared_gate_config=layer_controller.shared_gate_config,
            shared_memory_config=memory_config,
            layer_config=LayerConfig(
                activation=expert_stack_options.activation,
                layer_norm_position=expert_stack_options.layer_norm_position,
                residual_config=None
                if (expert_stack_options.residual_connection_option) is None
                else ResidualConfig(
                    option=(expert_stack_options.residual_connection_option)
                ),
                dropout_probability=expert_stack_options.dropout_probability,
                gate_config=gate_config,
                halting_config=halting_config,
                memory_config=None,
                layer_model_config=layer_model_config,
            ),
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
            normalize_probabilities_flag=(sampler_options.normalize_probabilities_flag),
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
        router_stack_options = self.router_stack_options
        layer_model_config = self.build_router_adaptive_linear_layer_config(
            router_stack_options.bias_flag
        )
        if self.__router_controller_enabled():
            model_config = self.__build_controlled_router_model_config(
                layer_model_config
            )
        else:
            model_config = self.__build_controller_stack(
                router_stack_options,
                layer_model_config,
            )
        return RouterConfig(
            input_dim=self.hidden_dim,
            num_experts=mixture_options.num_experts,
            noisy_topk_flag=router_options.noisy_topk_flag,
            model_config=model_config,
        )

    def __router_controller_enabled(self) -> bool:
        return (
            self.router_layer_controller_options.stack_gate_flag
            or self.router_layer_controller_options.stack_halting_flag
            or self.router_dynamic_memory_options.memory_flag
            or self.router_recurrent_controller_options.recurrent_flag
        )

    def __build_controlled_router_model_config(
        self,
        layer_model_config: AdaptiveLinearLayerConfig,
    ) -> RouterControllerModelConfig:
        trunk_dim = self.router_stack_options.hidden_dim
        trunk_config = self.router_recurrent_config_factory.build_config(
            self.__build_router_trunk_stack_config(layer_model_config)
        )
        return RouterControllerModelConfig(
            input_dim=self.hidden_dim,
            hidden_dim=trunk_dim,
            output_dim=self.__router_output_dim(),
            adapter_config=self.__build_router_projection_layer_config(
                layer_model_config
            ),
            trunk_config=trunk_config,
            head_config=self.__build_router_projection_layer_config(layer_model_config),
        )

    def __router_output_dim(self) -> int:
        num_experts = self.mixture_options.num_experts
        if self.router_options.noisy_topk_flag:
            return 2 * num_experts
        return num_experts

    def __build_router_trunk_stack_config(
        self,
        layer_model_config: AdaptiveLinearLayerConfig,
    ) -> LayerStackConfig:
        router_stack_options = self.router_stack_options
        gate_config = self.router_gate_config_factory.build_gate_config()
        halting_config = self.router_halting_config_factory.build_halting_config()
        memory_config = self.router_memory_config_factory.build_memory_config()
        return LayerStackConfig(
            input_dim=router_stack_options.hidden_dim,
            hidden_dim=router_stack_options.hidden_dim,
            output_dim=router_stack_options.hidden_dim,
            num_layers=router_stack_options.num_layers,
            last_layer_bias_option=router_stack_options.last_layer_bias_option,
            apply_output_pipeline_flag=(
                router_stack_options.apply_output_pipeline_flag
            ),
            shared_memory_config=memory_config,
            layer_config=LayerConfig(
                activation=router_stack_options.activation,
                layer_norm_position=router_stack_options.layer_norm_position,
                residual_config=None
                if (router_stack_options.residual_connection_option) is None
                else ResidualConfig(
                    option=(router_stack_options.residual_connection_option)
                ),
                dropout_probability=router_stack_options.dropout_probability,
                gate_config=gate_config,
                halting_config=halting_config,
                memory_config=None,
                layer_model_config=layer_model_config,
            ),
        )

    @staticmethod
    def __build_router_projection_layer_config(
        layer_model_config: AdaptiveLinearLayerConfig,
    ) -> LayerConfig:
        return LayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=layer_model_config,
        )

    def __build_weight_config(
        self,
        adaptive_options: HiddenAdaptiveWeightOptions,
    ) -> DynamicWeightConfig | None:
        weight_option = self.__resolve_enabled_adaptive_parameter_option(
            option_flag=adaptive_options.option_flag,
            option=adaptive_options.option,
            option_flag_name="weight_option_flag",
            option_name="weight_option",
        )
        if weight_option is None:
            return None
        model_config = self.__build_generator_model_config(
            adaptive_options.generator_stack_source
        )
        return build_weight_config(
            weight_option,
            generator_depth=adaptive_options.generator_depth,
            decay_schedule=adaptive_options.decay_schedule,
            decay_rate=adaptive_options.decay_rate,
            decay_warmup_batches=adaptive_options.decay_warmup_batches,
            normalization_option=adaptive_options.normalization_option,
            normalization_position_option=(
                adaptive_options.normalization_position_option
            ),
            bank_expansion_factor=adaptive_options.bank_expansion_factor,
            model_config=model_config,
        )

    def __build_bias_config(
        self,
        adaptive_options: HiddenAdaptiveBiasOptions,
    ) -> DynamicBiasConfig | None:
        bias_option = self.__resolve_enabled_adaptive_parameter_option(
            option_flag=adaptive_options.option_flag,
            option=adaptive_options.option,
            option_flag_name="bias_option_flag",
            option_name="bias_option",
        )
        if bias_option is None:
            return None
        model_config = self.__build_generator_model_config(
            adaptive_options.generator_stack_source
        )
        return build_bias_config(
            bias_option,
            decay_schedule=adaptive_options.decay_schedule,
            decay_rate=adaptive_options.decay_rate,
            decay_warmup_batches=adaptive_options.decay_warmup_batches,
            bank_expansion_factor=adaptive_options.bank_expansion_factor,
            model_config=model_config,
        )

    def __build_diagonal_config(
        self,
        adaptive_options: HiddenAdaptiveDiagonalOptions,
    ) -> DynamicDiagonalConfig | None:
        diagonal_option = self.__resolve_enabled_adaptive_parameter_option(
            option_flag=adaptive_options.option_flag,
            option=adaptive_options.option,
            option_flag_name="diagonal_option_flag",
            option_name="diagonal_option",
        )
        if diagonal_option is None:
            return None
        model_config = self.__build_generator_model_config(
            adaptive_options.generator_stack_source
        )
        return build_diagonal_config(
            diagonal_option,
            model_config=model_config,
        )

    def __build_mask_config(
        self,
        adaptive_options: HiddenAdaptiveMaskOptions,
    ) -> AxisMaskConfig | None:
        row_mask_option = self.__resolve_enabled_adaptive_parameter_option(
            option_flag=adaptive_options.option_flag,
            option=adaptive_options.row_mask_option,
            option_flag_name="mask_option_flag",
            option_name="row_mask_option",
        )
        if row_mask_option is None:
            return None
        model_config = self.__build_generator_model_config(
            adaptive_options.generator_stack_source
        )
        return build_mask_config(
            row_mask_option,
            mask_dimension_option=adaptive_options.mask_dimension_option,
            mask_threshold=adaptive_options.mask_threshold,
            mask_surrogate_scale=adaptive_options.mask_surrogate_scale,
            mask_floor=adaptive_options.mask_floor,
            mask_transition_width=adaptive_options.mask_transition_width,
            model_config=model_config,
        )

    def __build_generator_model_config(
        self,
        source: AdaptiveGeneratorStackSource,
    ) -> LayerStackConfig | None:
        return self.adaptive_generator_stack_config_factory.build_config_from_source(
            source
        )

    def __build_shared_generator_model_config(self) -> LayerStackConfig:
        return self.adaptive_generator_stack_config_factory.build_shared_config()

    @staticmethod
    def __build_controller_stack(
        options: ExpertsSubmoduleStackOptions,
        layer_model_config: LinearLayerConfig | AdaptiveLinearLayerConfig,
    ) -> LayerStackConfig:
        return build_controller_stack(
            options,
            layer_model_config=layer_model_config,
        )
