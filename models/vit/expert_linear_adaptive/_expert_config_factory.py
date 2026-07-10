from dataclasses import dataclass

import models.vit.expert_linear_adaptive.config as config
from models.vit.expert_linear_adaptive import _config_defaults as config_defaults
from models.vit.expert_linear_adaptive._vit_expert_config_factory import (
    VitExpertAdaptiveConfigDependencies as _ExpertAdaptiveDependencies,
)
from models.vit.expert_linear_adaptive._vit_expert_config_factory import (
    VitExpertAdaptiveConfigFactory,
)
from models.vit.expert_linear_adaptive.runtime_options import (
    AdaptiveGeneratorStackOptions,
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsSubmoduleStackOptions,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
    SubmoduleStackOptions,
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
)


@dataclass(frozen=True)
class ExpertAdaptiveConfigDependencies:
    hidden_dim: int
    encoder_options: TransformerEncoderOptions | None
    attention_options: TransformerAttentionOptions | None
    feed_forward_options: TransformerFeedForwardOptions | None
    mixture_options: ExpertsMixtureOptions | None
    mixture_submodule_stack_options: ExpertsSubmoduleStackOptions | None
    mixture_layer_controller_options: ExpertsLayerControllerOptions | None
    mixture_dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    mixture_recurrent_controller_options: ExpertsRecurrentControllerOptions | None
    expert_stack_options: ExpertsSubmoduleStackOptions | None
    sampler_options: ExpertsSamplerOptions | None
    router_options: ExpertsRouterOptions | None
    router_stack_options: ExpertsSubmoduleStackOptions | None
    router_layer_controller_options: ExpertsLayerControllerOptions | None
    router_dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    router_recurrent_controller_options: ExpertsRecurrentControllerOptions | None
    expert_layer_controller_options: ExpertsLayerControllerOptions | None
    expert_dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    expert_recurrent_controller_options: ExpertsRecurrentControllerOptions | None
    adaptive_generator_stack_options: AdaptiveGeneratorStackOptions | None
    hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions | None
    hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions | None
    hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None
    hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions | None
    router_adaptive_weight_options: HiddenAdaptiveWeightOptions | None
    router_adaptive_bias_options: HiddenAdaptiveBiasOptions | None
    router_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None
    router_adaptive_mask_options: HiddenAdaptiveMaskOptions | None
    expert_attention_use_kv_expert_models_flag: bool


class ExpertAdaptiveConfigFactory:
    def __init__(self, dependencies: ExpertAdaptiveConfigDependencies) -> None:
        self.dependencies = dependencies
        config_module = config
        self.encoder_options = self.__default_encoder_options(
            dependencies.encoder_options,
            config_module,
        )
        self.attention_options = self.__default_attention_options(
            dependencies.attention_options,
            config_module,
        )
        self.feed_forward_options = self.__default_feed_forward_options(
            dependencies.feed_forward_options,
            config_module,
        )
        self.mixture_options = self.__default_mixture_options(
            dependencies.mixture_options,
            config_module,
        )
        self.mixture_submodule_stack_options = (
            self.__default_mixture_submodule_stack_options(
                dependencies.mixture_submodule_stack_options,
                config_module,
            )
        )
        self.mixture_layer_controller_options = (
            self.__default_mixture_layer_controller_options(
                dependencies.mixture_layer_controller_options,
                config_module,
            )
        )
        self.mixture_dynamic_memory_options = (
            self.__default_mixture_dynamic_memory_options(
                dependencies.mixture_dynamic_memory_options,
                config_module,
            )
        )
        self.mixture_recurrent_controller_options = (
            self.__default_mixture_recurrent_controller_options(
                dependencies.mixture_recurrent_controller_options,
                config_module,
            )
        )
        self.expert_stack_options = self.__default_expert_stack_options(
            dependencies.expert_stack_options,
            config_module,
        )
        self.sampler_options = self.__default_sampler_options(
            dependencies.sampler_options,
            config_module,
        )
        self.router_options = self.__default_router_options(
            dependencies.router_options,
            config_module,
        )
        self.router_stack_options = self.__default_router_stack_options(
            dependencies.router_stack_options,
            config_module,
        )
        self.router_layer_controller_options = (
            self.__default_router_layer_controller_options(
                dependencies.router_layer_controller_options,
                config_module,
            )
        )
        self.router_dynamic_memory_options = (
            self.__default_router_dynamic_memory_options(
                dependencies.router_dynamic_memory_options,
                config_module,
            )
        )
        self.router_recurrent_controller_options = (
            self.__default_router_recurrent_controller_options(
                dependencies.router_recurrent_controller_options,
                config_module,
            )
        )
        self.expert_layer_controller_options = (
            self.__default_expert_layer_controller_options(
                dependencies.expert_layer_controller_options,
                config_module,
            )
        )
        self.expert_dynamic_memory_options = (
            self.__default_expert_dynamic_memory_options(
                dependencies.expert_dynamic_memory_options,
                config_module,
            )
        )
        self.expert_recurrent_controller_options = (
            self.__default_expert_recurrent_controller_options(
                dependencies.expert_recurrent_controller_options,
                config_module,
            )
        )
        self.adaptive_generator_stack_options = (
            self.__default_adaptive_generator_stack_options(
                dependencies.adaptive_generator_stack_options,
                config_module,
            )
        )
        self.hidden_adaptive_weight_options = (
            self.__default_hidden_adaptive_weight_options(
                dependencies.hidden_adaptive_weight_options,
                config_module,
            )
        )
        self.hidden_adaptive_bias_options = self.__default_hidden_adaptive_bias_options(
            dependencies.hidden_adaptive_bias_options,
            config_module,
        )
        self.hidden_adaptive_diagonal_options = (
            self.__default_hidden_adaptive_diagonal_options(
                dependencies.hidden_adaptive_diagonal_options,
                config_module,
            )
        )
        self.hidden_adaptive_mask_options = self.__default_hidden_adaptive_mask_options(
            dependencies.hidden_adaptive_mask_options,
            config_module,
        )
        self.router_adaptive_weight_options = (
            self.__default_router_adaptive_weight_options(
                dependencies.router_adaptive_weight_options,
                config_module,
            )
        )
        self.router_adaptive_bias_options = self.__default_router_adaptive_bias_options(
            dependencies.router_adaptive_bias_options,
            config_module,
        )
        self.router_adaptive_diagonal_options = (
            self.__default_router_adaptive_diagonal_options(
                dependencies.router_adaptive_diagonal_options,
                config_module,
            )
        )
        self.router_adaptive_mask_options = self.__default_router_adaptive_mask_options(
            dependencies.router_adaptive_mask_options,
            config_module,
        )

    def build_feed_forward_base_stack_config(
        self,
        feed_forward_stack_options: SubmoduleStackOptions,
    ):
        return self.__expert_config_factory().build_feed_forward_base_stack_config(
            feed_forward_stack_options
        )

    def build_attention_config(
        self,
        *,
        batch_size: int,
        hidden_dim: int,
        sequence_length: int,
        projection_model_config,
    ):
        return self.__expert_config_factory().build_attention_config(
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            sequence_length=sequence_length,
            projection_model_config=projection_model_config,
        )

    def __default_encoder_options(
        self,
        encoder_options: TransformerEncoderOptions | None,
        config_module: object,
    ) -> TransformerEncoderOptions:
        if encoder_options is not None:
            return encoder_options
        return config_defaults.vit_encoder_options(config_module)

    def __default_attention_options(
        self,
        attention_options: TransformerAttentionOptions | None,
        config_module: object,
    ) -> TransformerAttentionOptions:
        if attention_options is not None:
            return attention_options
        return config_defaults.vit_attention_options(config_module)

    def __default_feed_forward_options(
        self,
        feed_forward_options: TransformerFeedForwardOptions | None,
        config_module: object,
    ) -> TransformerFeedForwardOptions:
        if feed_forward_options is not None:
            return feed_forward_options
        return config_defaults.vit_feed_forward_options(config_module)

    def __default_mixture_options(
        self,
        mixture_options: ExpertsMixtureOptions | None,
        config_module: object,
    ) -> ExpertsMixtureOptions:
        if mixture_options is not None:
            return mixture_options
        return config_defaults.experts_mixture_options(config_module)

    def __default_mixture_submodule_stack_options(
        self,
        mixture_submodule_stack_options: ExpertsSubmoduleStackOptions | None,
        config_module: object,
    ) -> ExpertsSubmoduleStackOptions:
        if mixture_submodule_stack_options is not None:
            return mixture_submodule_stack_options
        return config_defaults.experts_submodule_stack_options(
            config_module,
            "SUBMODULE_STACK",
        )

    def __default_mixture_layer_controller_options(
        self,
        mixture_layer_controller_options: ExpertsLayerControllerOptions | None,
        config_module: object,
    ) -> ExpertsLayerControllerOptions:
        if mixture_layer_controller_options is not None:
            return mixture_layer_controller_options
        return config_defaults.experts_layer_controller_options(
            config_module,
            gate_prefix="GATE",
            gate_stack_prefix="GATE_STACK",
            halting_prefix="HALTING",
            halting_stack_prefix="HALTING_STACK",
        )

    def __default_mixture_dynamic_memory_options(
        self,
        mixture_dynamic_memory_options: ExpertsDynamicMemoryOptions | None,
        config_module: object,
    ) -> ExpertsDynamicMemoryOptions:
        if mixture_dynamic_memory_options is not None:
            return mixture_dynamic_memory_options
        return config_defaults.experts_dynamic_memory_options(
            config_module,
            memory_prefix="MEMORY",
            memory_stack_prefix="MEMORY_STACK",
        )

    def __default_mixture_recurrent_controller_options(
        self,
        mixture_recurrent_controller_options: (
            ExpertsRecurrentControllerOptions | None
        ),
        config_module: object,
    ) -> ExpertsRecurrentControllerOptions:
        if mixture_recurrent_controller_options is not None:
            return mixture_recurrent_controller_options
        return config_defaults.experts_recurrent_controller_options(
            config_module,
            recurrent_prefix="RECURRENT",
            gate_stack_prefix="RECURRENT_GATE_STACK",
            halting_stack_prefix="RECURRENT_HALTING_STACK",
        )

    def __default_expert_stack_options(
        self,
        expert_stack_options: ExpertsSubmoduleStackOptions | None,
        config_module: object,
    ) -> ExpertsSubmoduleStackOptions:
        if expert_stack_options is not None:
            return expert_stack_options
        return config_defaults.experts_submodule_stack_options(
            config_module,
            "EXPERT_STACK",
            bias_key="EXPERT_BIAS_FLAG",
        )

    def __default_sampler_options(
        self,
        sampler_options: ExpertsSamplerOptions | None,
        config_module: object,
    ) -> ExpertsSamplerOptions:
        if sampler_options is not None:
            return sampler_options
        return config_defaults.experts_sampler_options(config_module)

    def __default_router_options(
        self,
        router_options: ExpertsRouterOptions | None,
        config_module: object,
    ) -> ExpertsRouterOptions:
        if router_options is not None:
            return router_options
        return config_defaults.experts_router_options(config_module)

    def __default_router_stack_options(
        self,
        router_stack_options: ExpertsSubmoduleStackOptions | None,
        config_module: object,
    ) -> ExpertsSubmoduleStackOptions:
        if router_stack_options is not None:
            return router_stack_options
        return config_defaults.experts_submodule_stack_options(
            config_module,
            "ROUTER_STACK",
            bias_key="ROUTER_BIAS_FLAG",
        )

    def __default_router_layer_controller_options(
        self,
        router_layer_controller_options: ExpertsLayerControllerOptions | None,
        config_module: object,
    ) -> ExpertsLayerControllerOptions:
        if router_layer_controller_options is not None:
            return router_layer_controller_options
        return config_defaults.experts_layer_controller_options(
            config_module,
            gate_prefix="ROUTER_GATE",
            gate_stack_prefix="ROUTER_GATE_STACK",
            halting_prefix="ROUTER_HALTING",
            halting_stack_prefix="ROUTER_HALTING_STACK",
        )

    def __default_router_dynamic_memory_options(
        self,
        router_dynamic_memory_options: ExpertsDynamicMemoryOptions | None,
        config_module: object,
    ) -> ExpertsDynamicMemoryOptions:
        if router_dynamic_memory_options is not None:
            return router_dynamic_memory_options
        return config_defaults.experts_dynamic_memory_options(
            config_module,
            memory_prefix="ROUTER_MEMORY",
            memory_stack_prefix="ROUTER_MEMORY_STACK",
        )

    def __default_router_recurrent_controller_options(
        self,
        router_recurrent_controller_options: ExpertsRecurrentControllerOptions | None,
        config_module: object,
    ) -> ExpertsRecurrentControllerOptions:
        if router_recurrent_controller_options is not None:
            return router_recurrent_controller_options
        return config_defaults.experts_recurrent_controller_options(
            config_module,
            recurrent_prefix="ROUTER_RECURRENT",
            gate_stack_prefix="ROUTER_RECURRENT_GATE_STACK",
            halting_stack_prefix="ROUTER_RECURRENT_HALTING_STACK",
        )

    def __default_expert_layer_controller_options(
        self,
        expert_layer_controller_options: ExpertsLayerControllerOptions | None,
        config_module: object,
    ) -> ExpertsLayerControllerOptions:
        if expert_layer_controller_options is not None:
            return expert_layer_controller_options
        return config_defaults.experts_layer_controller_options(
            config_module,
            gate_prefix="EXPERT_GATE",
            gate_stack_prefix="EXPERT_GATE_STACK",
            halting_prefix="EXPERT_HALTING",
            halting_stack_prefix="EXPERT_HALTING_STACK",
        )

    def __default_expert_dynamic_memory_options(
        self,
        expert_dynamic_memory_options: ExpertsDynamicMemoryOptions | None,
        config_module: object,
    ) -> ExpertsDynamicMemoryOptions:
        if expert_dynamic_memory_options is not None:
            return expert_dynamic_memory_options
        return config_defaults.experts_dynamic_memory_options(
            config_module,
            memory_prefix="EXPERT_MEMORY",
            memory_stack_prefix="EXPERT_MEMORY_STACK",
        )

    def __default_expert_recurrent_controller_options(
        self,
        expert_recurrent_controller_options: (ExpertsRecurrentControllerOptions | None),
        config_module: object,
    ) -> ExpertsRecurrentControllerOptions:
        if expert_recurrent_controller_options is not None:
            return expert_recurrent_controller_options
        return config_defaults.experts_recurrent_controller_options(
            config_module,
            recurrent_prefix="EXPERT_RECURRENT",
            gate_stack_prefix="EXPERT_RECURRENT_GATE_STACK",
            halting_stack_prefix="EXPERT_RECURRENT_HALTING_STACK",
        )

    def __default_adaptive_generator_stack_options(
        self,
        adaptive_generator_stack_options: AdaptiveGeneratorStackOptions | None,
        config_module: object,
    ) -> AdaptiveGeneratorStackOptions:
        if adaptive_generator_stack_options is not None:
            return adaptive_generator_stack_options
        return config_defaults.adaptive_generator_stack_options(config_module)

    def __default_hidden_adaptive_weight_options(
        self,
        hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions | None,
        config_module: object,
    ) -> HiddenAdaptiveWeightOptions:
        if hidden_adaptive_weight_options is not None:
            return hidden_adaptive_weight_options
        return config_defaults.hidden_adaptive_weight_options(config_module)

    def __default_hidden_adaptive_bias_options(
        self,
        hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions | None,
        config_module: object,
    ) -> HiddenAdaptiveBiasOptions:
        if hidden_adaptive_bias_options is not None:
            return hidden_adaptive_bias_options
        return config_defaults.hidden_adaptive_bias_options(config_module)

    def __default_hidden_adaptive_diagonal_options(
        self,
        hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None,
        config_module: object,
    ) -> HiddenAdaptiveDiagonalOptions:
        if hidden_adaptive_diagonal_options is not None:
            return hidden_adaptive_diagonal_options
        return config_defaults.hidden_adaptive_diagonal_options(config_module)

    def __default_hidden_adaptive_mask_options(
        self,
        hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions | None,
        config_module: object,
    ) -> HiddenAdaptiveMaskOptions:
        if hidden_adaptive_mask_options is not None:
            return hidden_adaptive_mask_options
        return config_defaults.hidden_adaptive_mask_options(config_module)

    def __default_router_adaptive_weight_options(
        self,
        router_adaptive_weight_options: HiddenAdaptiveWeightOptions | None,
        config_module: object,
    ) -> HiddenAdaptiveWeightOptions:
        if router_adaptive_weight_options is not None:
            return router_adaptive_weight_options
        return config_defaults.hidden_adaptive_weight_options(
            config_module,
            prefix="ROUTER_",
            stack_prefix="ROUTER_WEIGHT_GENERATOR_STACK",
        )

    def __default_router_adaptive_bias_options(
        self,
        router_adaptive_bias_options: HiddenAdaptiveBiasOptions | None,
        config_module: object,
    ) -> HiddenAdaptiveBiasOptions:
        if router_adaptive_bias_options is not None:
            return router_adaptive_bias_options
        return config_defaults.hidden_adaptive_bias_options(
            config_module,
            prefix="ROUTER_",
            stack_prefix="ROUTER_BIAS_GENERATOR_STACK",
        )

    def __default_router_adaptive_diagonal_options(
        self,
        router_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None,
        config_module: object,
    ) -> HiddenAdaptiveDiagonalOptions:
        if router_adaptive_diagonal_options is not None:
            return router_adaptive_diagonal_options
        return config_defaults.hidden_adaptive_diagonal_options(
            config_module,
            prefix="ROUTER_",
            stack_prefix="ROUTER_DIAGONAL_GENERATOR_STACK",
        )

    def __default_router_adaptive_mask_options(
        self,
        router_adaptive_mask_options: HiddenAdaptiveMaskOptions | None,
        config_module: object,
    ) -> HiddenAdaptiveMaskOptions:
        if router_adaptive_mask_options is not None:
            return router_adaptive_mask_options
        return config_defaults.hidden_adaptive_mask_options(
            config_module,
            prefix="ROUTER_",
            stack_prefix="ROUTER_MASK_GENERATOR_STACK",
        )

    def __expert_config_factory(self) -> VitExpertAdaptiveConfigFactory:
        dependencies = self.dependencies
        return VitExpertAdaptiveConfigFactory(
            _ExpertAdaptiveDependencies(
                hidden_dim=dependencies.hidden_dim,
                encoder_options=self.encoder_options,
                attention_options=self.attention_options,
                feed_forward_options=self.feed_forward_options,
                mixture_options=self.mixture_options,
                expert_stack_options=self.expert_stack_options,
                sampler_options=self.sampler_options,
                router_options=self.router_options,
                router_stack_options=self.router_stack_options,
                expert_layer_controller_options=(self.expert_layer_controller_options),
                expert_dynamic_memory_options=self.expert_dynamic_memory_options,
                expert_recurrent_controller_options=(
                    self.expert_recurrent_controller_options
                ),
                expert_attention_use_kv_expert_models_flag=(
                    dependencies.expert_attention_use_kv_expert_models_flag
                ),
                mixture_submodule_stack_options=(self.mixture_submodule_stack_options),
                mixture_layer_controller_options=(
                    self.mixture_layer_controller_options
                ),
                mixture_dynamic_memory_options=(self.mixture_dynamic_memory_options),
                mixture_recurrent_controller_options=(
                    self.mixture_recurrent_controller_options
                ),
                router_layer_controller_options=(self.router_layer_controller_options),
                router_dynamic_memory_options=self.router_dynamic_memory_options,
                router_recurrent_controller_options=(
                    self.router_recurrent_controller_options
                ),
                adaptive_generator_stack_options=(
                    self.adaptive_generator_stack_options
                ),
                hidden_adaptive_weight_options=self.hidden_adaptive_weight_options,
                hidden_adaptive_bias_options=self.hidden_adaptive_bias_options,
                hidden_adaptive_diagonal_options=(
                    self.hidden_adaptive_diagonal_options
                ),
                hidden_adaptive_mask_options=self.hidden_adaptive_mask_options,
                router_adaptive_weight_options=self.router_adaptive_weight_options,
                router_adaptive_bias_options=self.router_adaptive_bias_options,
                router_adaptive_diagonal_options=(
                    self.router_adaptive_diagonal_options
                ),
                router_adaptive_mask_options=self.router_adaptive_mask_options,
            )
        )
