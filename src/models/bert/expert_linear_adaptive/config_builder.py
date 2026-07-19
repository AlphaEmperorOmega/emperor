import torch

import models.bert.expert_linear_adaptive.config as config
from emperor.attention import (
    MixtureOfAttentionHeadsConfig,
)
from emperor.experts import MixtureOfExpertsModelConfig
from models.bert.expert_linear_adaptive._base_config_builder import (
    BertBackendConfigBuilder,
)
from models.bert.expert_linear_adaptive._expert_control_config_factory import (
    ControlConfigDependencies,
    ControlConfigFactory,
)
from models.bert.expert_linear_adaptive.experiment_config import ExperimentConfig
from models.bert.expert_linear_adaptive.runtime_defaults import (
    expert_linear_adaptive_builder_kwargs_from_flat,
)
from models.bert.expert_linear_adaptive.runtime_options import (
    AdaptiveGeneratorStackOptions,
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
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
    TransformerPositionalEmbeddingOptions,
)


class BertExpertLinearAdaptiveConfigBuilder(BertBackendConfigBuilder):
    def __init__(
        self,
        *,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        output_dim: int = config.OUTPUT_DIM,
        sequence_length: int = config.SEQUENCE_LENGTH,
        embedding_dropout_probability: float = config.EMBEDDING_DROPOUT_PROBABILITY,
        encoder_options: TransformerEncoderOptions | None = None,
        positional_embedding_options: (
            TransformerPositionalEmbeddingOptions | None
        ) = None,
        attention_options: TransformerAttentionOptions | None = None,
        feed_forward_options: TransformerFeedForwardOptions | None = None,
        attention_projection_stack_options: (
            ExpertsSubmoduleStackOptions | None
        ) = None,
        attention_projection_layer_controller_options: (
            ExpertsLayerControllerOptions | None
        ) = None,
        attention_projection_dynamic_memory_options: (
            ExpertsDynamicMemoryOptions | None
        ) = None,
        attention_projection_recurrent_controller_options: (
            ExpertsRecurrentControllerOptions | None
        ) = None,
        feed_forward_stack_options: ExpertsSubmoduleStackOptions | None = None,
        feed_forward_layer_controller_options: (
            ExpertsLayerControllerOptions | None
        ) = None,
        feed_forward_dynamic_memory_options: (
            ExpertsDynamicMemoryOptions | None
        ) = None,
        feed_forward_recurrent_controller_options: (
            ExpertsRecurrentControllerOptions | None
        ) = None,
        submodule_stack_options: ExpertsSubmoduleStackOptions | None = None,
        layer_controller_options: ExpertsLayerControllerOptions | None = None,
        dynamic_memory_options: ExpertsDynamicMemoryOptions | None = None,
        recurrent_controller_options: ExpertsRecurrentControllerOptions | None = None,
        mixture_options: ExpertsMixtureOptions | None = None,
        mixture_submodule_stack_options: ExpertsSubmoduleStackOptions | None = None,
        mixture_layer_controller_options: ExpertsLayerControllerOptions | None = None,
        mixture_dynamic_memory_options: ExpertsDynamicMemoryOptions | None = None,
        mixture_recurrent_controller_options: (
            ExpertsRecurrentControllerOptions | None
        ) = None,
        expert_stack_options: ExpertsSubmoduleStackOptions | None = None,
        sampler_options: ExpertsSamplerOptions | None = None,
        router_options: ExpertsRouterOptions | None = None,
        router_stack_options: ExpertsSubmoduleStackOptions | None = None,
        router_layer_controller_options: ExpertsLayerControllerOptions | None = None,
        router_dynamic_memory_options: ExpertsDynamicMemoryOptions | None = None,
        router_recurrent_controller_options: (
            ExpertsRecurrentControllerOptions | None
        ) = None,
        expert_layer_controller_options: ExpertsLayerControllerOptions | None = None,
        expert_dynamic_memory_options: ExpertsDynamicMemoryOptions | None = None,
        expert_recurrent_controller_options: (
            ExpertsRecurrentControllerOptions | None
        ) = None,
        adaptive_generator_stack_options: AdaptiveGeneratorStackOptions | None = None,
        hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions | None = None,
        hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions | None = None,
        hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None = None,
        hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions | None = None,
        router_adaptive_weight_options: HiddenAdaptiveWeightOptions | None = None,
        router_adaptive_bias_options: HiddenAdaptiveBiasOptions | None = None,
        router_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None = None,
        router_adaptive_mask_options: HiddenAdaptiveMaskOptions | None = None,
        expert_attention_use_kv_expert_models_flag: bool = (
            config.EXPERT_ATTENTION_USE_KV_EXPERT_MODELS_FLAG
        ),
    ) -> None:
        defaults = expert_linear_adaptive_builder_kwargs_from_flat({}, config)
        self.adaptive_generator_stack_options = (
            adaptive_generator_stack_options
            or defaults["adaptive_generator_stack_options"]
        )
        self.hidden_adaptive_weight_options = (
            hidden_adaptive_weight_options or defaults["hidden_adaptive_weight_options"]
        )
        self.hidden_adaptive_bias_options = (
            hidden_adaptive_bias_options or defaults["hidden_adaptive_bias_options"]
        )
        self.hidden_adaptive_diagonal_options = (
            hidden_adaptive_diagonal_options
            or defaults["hidden_adaptive_diagonal_options"]
        )
        self.hidden_adaptive_mask_options = (
            hidden_adaptive_mask_options or defaults["hidden_adaptive_mask_options"]
        )
        self.router_layer_controller_options = (
            router_layer_controller_options
            or defaults["router_layer_controller_options"]
        )
        self.router_dynamic_memory_options = (
            router_dynamic_memory_options or defaults["router_dynamic_memory_options"]
        )
        self.router_recurrent_controller_options = (
            router_recurrent_controller_options
            or defaults["router_recurrent_controller_options"]
        )
        self.router_adaptive_weight_options = (
            router_adaptive_weight_options or defaults["router_adaptive_weight_options"]
        )
        self.router_adaptive_bias_options = (
            router_adaptive_bias_options or defaults["router_adaptive_bias_options"]
        )
        self.router_adaptive_diagonal_options = (
            router_adaptive_diagonal_options
            or defaults["router_adaptive_diagonal_options"]
        )
        self.router_adaptive_mask_options = (
            router_adaptive_mask_options or defaults["router_adaptive_mask_options"]
        )
        self.mixture_submodule_stack_options = (
            mixture_submodule_stack_options
            or defaults["mixture_submodule_stack_options"]
        )
        self.mixture_layer_controller_options = (
            mixture_layer_controller_options
            or defaults["mixture_layer_controller_options"]
        )
        self.mixture_dynamic_memory_options = (
            mixture_dynamic_memory_options or defaults["mixture_dynamic_memory_options"]
        )
        self.mixture_recurrent_controller_options = (
            mixture_recurrent_controller_options
            or defaults["mixture_recurrent_controller_options"]
        )
        self.mixture_options = mixture_options or defaults["mixture_options"]
        self.expert_stack_options = (
            expert_stack_options or defaults["expert_stack_options"]
        )
        self.sampler_options = sampler_options or defaults["sampler_options"]
        self.router_options = router_options or defaults["router_options"]
        self.router_stack_options = (
            router_stack_options or defaults["router_stack_options"]
        )
        self.expert_layer_controller_options = (
            expert_layer_controller_options
            or defaults["expert_layer_controller_options"]
        )
        self.expert_dynamic_memory_options = (
            expert_dynamic_memory_options or defaults["expert_dynamic_memory_options"]
        )
        self.expert_recurrent_controller_options = (
            expert_recurrent_controller_options
            or defaults["expert_recurrent_controller_options"]
        )
        self.expert_attention_use_kv_expert_models_flag = (
            expert_attention_use_kv_expert_models_flag
        )
        super().__init__(
            batch_size=batch_size,
            learning_rate=learning_rate,
            input_dim=input_dim,
            output_dim=output_dim,
            sequence_length=sequence_length,
            embedding_dropout_probability=embedding_dropout_probability,
            encoder_options=encoder_options or defaults["encoder_options"],
            positional_embedding_options=(
                positional_embedding_options or defaults["positional_embedding_options"]
            ),
            attention_options=attention_options or defaults["attention_options"],
            feed_forward_options=(
                feed_forward_options or defaults["feed_forward_options"]
            ),
            attention_projection_stack_options=(
                attention_projection_stack_options
                or defaults["attention_projection_stack_options"]
            ),
            attention_projection_layer_controller_options=(
                attention_projection_layer_controller_options
                or defaults["attention_projection_layer_controller_options"]
            ),
            attention_projection_dynamic_memory_options=(
                attention_projection_dynamic_memory_options
                or defaults["attention_projection_dynamic_memory_options"]
            ),
            attention_projection_recurrent_controller_options=(
                attention_projection_recurrent_controller_options
                or defaults["attention_projection_recurrent_controller_options"]
            ),
            feed_forward_stack_options=(
                feed_forward_stack_options or defaults["feed_forward_stack_options"]
            ),
            feed_forward_layer_controller_options=(
                feed_forward_layer_controller_options
                or defaults["feed_forward_layer_controller_options"]
            ),
            feed_forward_dynamic_memory_options=(
                feed_forward_dynamic_memory_options
                or defaults["feed_forward_dynamic_memory_options"]
            ),
            feed_forward_recurrent_controller_options=(
                feed_forward_recurrent_controller_options
                or defaults["feed_forward_recurrent_controller_options"]
            ),
            submodule_stack_options=(
                submodule_stack_options or defaults["submodule_stack_options"]
            ),
            layer_controller_options=(
                layer_controller_options or defaults["layer_controller_options"]
            ),
            dynamic_memory_options=(
                dynamic_memory_options or defaults["dynamic_memory_options"]
            ),
            recurrent_controller_options=(
                recurrent_controller_options or defaults["recurrent_controller_options"]
            ),
            experiment_config_type=ExperimentConfig,
        )

    def _build_feed_forward_base_stack_config(self) -> MixtureOfExpertsModelConfig:
        return self._build_expert_model_config()

    def _build_attention_config(self):
        encoder_options = self.encoder_options
        attention_options = self.attention_options
        return MixtureOfAttentionHeadsConfig(
            batch_size=self.batch_size,
            num_heads=attention_options.num_heads,
            embedding_dim=self.hidden_dim,
            query_key_projection_dim=self.hidden_dim,
            value_projection_dim=self.hidden_dim,
            target_sequence_length=self.sequence_length,
            source_sequence_length=self.sequence_length,
            target_dtype=torch.float32,
            dropout_probability=encoder_options.dropout_probability,
            zero_attention_flag=False,
            causal_attention_mask_flag=encoder_options.causal_attention_mask_flag,
            add_key_value_bias_flag=attention_options.add_key_value_bias_flag,
            average_attention_weights_flag=False,
            return_attention_weights_flag=False,
            batch_first_flag=True,
            projection_model_config=self._build_attention_projection_stack_config(),
            experts_config=self._build_attention_experts_config(),
            use_kv_expert_models_flag=(self.expert_attention_use_kv_expert_models_flag),
        )

    def _build_attention_experts_config(self):
        model_config = self._build_expert_model_config(
            use_feed_forward_stack_options=False,
        )
        return model_config.stack_config.layer_config.layer_model_config

    def _feed_forward_experts_stack_options(self) -> ExpertsStackOptions:
        feed_forward_stack_options = self._effective_feed_forward_stack_options()
        return ExpertsStackOptions(
            hidden_dim=feed_forward_stack_options.hidden_dim,
            bias_flag=feed_forward_stack_options.bias_flag,
            layer_norm_position=feed_forward_stack_options.layer_norm_position,
            num_layers=feed_forward_stack_options.num_layers,
            activation=feed_forward_stack_options.activation,
            residual_connection_option=(
                feed_forward_stack_options.residual_connection_option
            ),
            dropout_probability=feed_forward_stack_options.dropout_probability,
            last_layer_bias_option=feed_forward_stack_options.last_layer_bias_option,
            apply_output_pipeline_flag=(
                feed_forward_stack_options.apply_output_pipeline_flag
            ),
        )

    def _legacy_experts_stack_options(self) -> ExpertsStackOptions:
        return ExpertsStackOptions(
            hidden_dim=self.hidden_dim,
            bias_flag=self.feed_forward_options.bias_flag,
            layer_norm_position=self.encoder_options.layer_norm_position,
            num_layers=self.feed_forward_options.num_layers,
            activation=self.encoder_options.activation,
            residual_connection_option=None,
            dropout_probability=self.encoder_options.dropout_probability,
            last_layer_bias_option=config.LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=True,
        )

    def _build_expert_model_config(
        self,
        *,
        use_feed_forward_stack_options: bool = True,
    ) -> MixtureOfExpertsModelConfig:
        stack_options = (
            self._feed_forward_experts_stack_options()
            if use_feed_forward_stack_options
            else self._legacy_experts_stack_options()
        )
        factory = ControlConfigFactory(
            ControlConfigDependencies(
                stack_options=stack_options,
                submodule_stack_options=self.submodule_stack_options,
                mixture_options=self.mixture_options,
                expert_stack_options=self.expert_stack_options,
                sampler_options=self.sampler_options,
                router_options=self.router_options,
                router_stack_options=self.router_stack_options,
                router_layer_controller_options=self.router_layer_controller_options,
                router_dynamic_memory_options=self.router_dynamic_memory_options,
                router_recurrent_controller_options=(
                    self.router_recurrent_controller_options
                ),
                layer_controller_options=self.layer_controller_options,
                dynamic_memory_options=self.dynamic_memory_options,
                recurrent_controller_options=self.recurrent_controller_options,
                expert_layer_controller_options=self.expert_layer_controller_options,
                expert_dynamic_memory_options=self.expert_dynamic_memory_options,
                expert_recurrent_controller_options=(
                    self.expert_recurrent_controller_options
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
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
            )
        )
        model_config = factory.build()
        if isinstance(model_config, MixtureOfExpertsModelConfig):
            return model_config
        return model_config.block_config
