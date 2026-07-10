from dataclasses import dataclass

import models.gpt.expert_linear.config as config
from models.gpt.expert_linear import _config_defaults as config_defaults
from models.gpt.expert_linear._gpt_expert_config_factory import (
    GptExpertConfigDependencies as _ExpertDependencies,
)
from models.gpt.expert_linear._gpt_expert_config_factory import (
    GptExpertConfigFactory,
)
from models.gpt.expert_linear.runtime_options import (
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsSubmoduleStackOptions,
    SubmoduleStackOptions,
    TransformerAttentionOptions,
    TransformerDecoderOptions,
    TransformerFeedForwardOptions,
)


@dataclass(frozen=True)
class ExpertConfigDependencies:
    hidden_dim: int
    decoder_options: TransformerDecoderOptions | None
    attention_options: TransformerAttentionOptions | None
    feed_forward_options: TransformerFeedForwardOptions | None
    mixture_options: ExpertsMixtureOptions | None
    expert_stack_options: ExpertsSubmoduleStackOptions | None
    sampler_options: ExpertsSamplerOptions | None
    router_options: ExpertsRouterOptions | None
    router_stack_options: ExpertsSubmoduleStackOptions | None
    expert_layer_controller_options: ExpertsLayerControllerOptions | None
    expert_dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    expert_recurrent_controller_options: ExpertsRecurrentControllerOptions | None
    expert_attention_use_kv_expert_models_flag: bool


class ExpertConfigFactory:
    def __init__(self, dependencies: ExpertConfigDependencies) -> None:
        self.dependencies = dependencies
        config_module = config
        self.decoder_options = self.__default_decoder_options(
            dependencies.decoder_options,
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

    def __default_decoder_options(
        self,
        decoder_options: TransformerDecoderOptions | None,
        config_module: object,
    ) -> TransformerDecoderOptions:
        if decoder_options is not None:
            return decoder_options
        return config_defaults.gpt_decoder_options(config_module)

    def __default_attention_options(
        self,
        attention_options: TransformerAttentionOptions | None,
        config_module: object,
    ) -> TransformerAttentionOptions:
        if attention_options is not None:
            return attention_options
        return config_defaults.gpt_attention_options(config_module)

    def __default_feed_forward_options(
        self,
        feed_forward_options: TransformerFeedForwardOptions | None,
        config_module: object,
    ) -> TransformerFeedForwardOptions:
        if feed_forward_options is not None:
            return feed_forward_options
        return config_defaults.gpt_feed_forward_options(config_module)

    def __default_mixture_options(
        self,
        mixture_options: ExpertsMixtureOptions | None,
        config_module: object,
    ) -> ExpertsMixtureOptions:
        if mixture_options is not None:
            return mixture_options
        return config_defaults.experts_mixture_options(config_module)

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

    def __expert_config_factory(self) -> GptExpertConfigFactory:
        dependencies = self.dependencies
        return GptExpertConfigFactory(
            _ExpertDependencies(
                hidden_dim=dependencies.hidden_dim,
                decoder_options=self.decoder_options,
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
            )
        )
