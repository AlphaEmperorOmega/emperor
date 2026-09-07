import torch

import models.gpt.expert_linear_adaptive.config as config
from emperor.attention import (
    MixtureOfAttentionHeadsConfig,
)
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    GroupingConfig,
)
from emperor.experts import MixtureOfExpertsModelConfig
from models.gpt.expert_linear_adaptive._base_config_builder import (
    GptBackendConfigBuilder,
)
from models.gpt.expert_linear_adaptive._expert_control_config_factory import (
    ControlConfigDependencies,
    ControlConfigFactory,
)
from models.gpt.expert_linear_adaptive.experiment_config import ExperimentConfig
from models.gpt.expert_linear_adaptive.runtime_defaults import DEFAULT_RUNTIME
from models.gpt.expert_linear_adaptive.runtime_options import (
    ExpertsStackOptions,
    RuntimeOptions,
)


class _GptExpertLinearAdaptiveConfigBuilderImplementation(GptBackendConfigBuilder):
    def __init__(self, runtime: RuntimeOptions) -> None:
        options = runtime._construction_options(config)
        defaults = DEFAULT_RUNTIME._construction_options(config)
        self.adaptive_generator_stack_options = (
            options.adaptive_generator_stack_options
            or defaults.adaptive_generator_stack_options
        )
        self.hidden_adaptive_weight_options = (
            options.hidden_adaptive_weight_options
            or defaults.hidden_adaptive_weight_options
        )
        self.grouping_config = options.grouping_config
        self.attention_grouping_config = options.attention_grouping_config
        self.feed_forward_grouping_config = options.feed_forward_grouping_config
        self.hidden_adaptive_bias_options = (
            options.hidden_adaptive_bias_options
            or defaults.hidden_adaptive_bias_options
        )
        self.hidden_adaptive_diagonal_options = (
            options.hidden_adaptive_diagonal_options
            or defaults.hidden_adaptive_diagonal_options
        )
        self.hidden_adaptive_mask_options = (
            options.hidden_adaptive_mask_options
            or defaults.hidden_adaptive_mask_options
        )
        self.router_layer_controller_options = (
            options.router_layer_controller_options
            or defaults.router_layer_controller_options
        )
        self.router_dynamic_memory_options = (
            options.router_dynamic_memory_options
            or defaults.router_dynamic_memory_options
        )
        self.router_recurrent_controller_options = (
            options.router_recurrent_controller_options
            or defaults.router_recurrent_controller_options
        )
        self.router_adaptive_weight_options = (
            options.router_adaptive_weight_options
            or defaults.router_adaptive_weight_options
        )
        self.router_grouping_config = options.router_grouping_config
        self.router_adaptive_bias_options = (
            options.router_adaptive_bias_options
            or defaults.router_adaptive_bias_options
        )
        self.router_adaptive_diagonal_options = (
            options.router_adaptive_diagonal_options
            or defaults.router_adaptive_diagonal_options
        )
        self.router_adaptive_mask_options = (
            options.router_adaptive_mask_options
            or defaults.router_adaptive_mask_options
        )
        self.mixture_submodule_stack_options = (
            options.mixture_submodule_stack_options
            or defaults.mixture_submodule_stack_options
        )
        self.mixture_layer_controller_options = (
            options.mixture_layer_controller_options
            or defaults.mixture_layer_controller_options
        )
        self.mixture_dynamic_memory_options = (
            options.mixture_dynamic_memory_options
            or defaults.mixture_dynamic_memory_options
        )
        self.mixture_recurrent_controller_options = (
            options.mixture_recurrent_controller_options
            or defaults.mixture_recurrent_controller_options
        )
        self.mixture_options = options.mixture_options or defaults.mixture_options
        self.expert_stack_options = (
            options.expert_stack_options or defaults.expert_stack_options
        )
        self.sampler_options = options.sampler_options or defaults.sampler_options
        self.router_options = options.router_options or defaults.router_options
        self.router_stack_options = (
            options.router_stack_options or defaults.router_stack_options
        )
        self.expert_layer_controller_options = (
            options.expert_layer_controller_options
            or defaults.expert_layer_controller_options
        )
        self.expert_dynamic_memory_options = (
            options.expert_dynamic_memory_options
            or defaults.expert_dynamic_memory_options
        )
        self.expert_recurrent_controller_options = (
            options.expert_recurrent_controller_options
            or defaults.expert_recurrent_controller_options
        )
        self.expert_attention_use_kv_expert_models_flag = (
            options.expert_attention_use_kv_expert_models_flag
        )
        super().__init__(
            batch_size=options.batch_size,
            learning_rate=options.learning_rate,
            input_dim=options.input_dim,
            output_dim=options.output_dim,
            sequence_length=options.sequence_length,
            embedding_options=(options.embedding_options or defaults.embedding_options),
            lm_head_options=options.lm_head_options or defaults.lm_head_options,
            decoder_options=options.decoder_options or defaults.decoder_options,
            positional_embedding_options=(
                options.positional_embedding_options
                or defaults.positional_embedding_options
            ),
            attention_options=options.attention_options or defaults.attention_options,
            feed_forward_options=(
                options.feed_forward_options or defaults.feed_forward_options
            ),
            attention_projection_stack_options=(
                options.attention_projection_stack_options
                or defaults.attention_projection_stack_options
            ),
            attention_projection_layer_controller_options=(
                options.attention_projection_layer_controller_options
                or defaults.attention_projection_layer_controller_options
            ),
            attention_projection_dynamic_memory_options=(
                options.attention_projection_dynamic_memory_options
                or defaults.attention_projection_dynamic_memory_options
            ),
            attention_projection_recurrent_controller_options=(
                options.attention_projection_recurrent_controller_options
                or defaults.attention_projection_recurrent_controller_options
            ),
            feed_forward_stack_options=(
                options.feed_forward_stack_options
                or defaults.feed_forward_stack_options
            ),
            feed_forward_layer_controller_options=(
                options.feed_forward_layer_controller_options
                or defaults.feed_forward_layer_controller_options
            ),
            feed_forward_dynamic_memory_options=(
                options.feed_forward_dynamic_memory_options
                or defaults.feed_forward_dynamic_memory_options
            ),
            feed_forward_recurrent_controller_options=(
                options.feed_forward_recurrent_controller_options
                or defaults.feed_forward_recurrent_controller_options
            ),
            submodule_stack_options=(
                options.submodule_stack_options or defaults.submodule_stack_options
            ),
            layer_controller_options=(
                options.layer_controller_options or defaults.layer_controller_options
            ),
            dynamic_memory_options=(
                options.dynamic_memory_options or defaults.dynamic_memory_options
            ),
            recurrent_controller_options=(
                options.recurrent_controller_options
                or defaults.recurrent_controller_options
            ),
            experiment_config_type=ExperimentConfig,
        )

    def _build_feed_forward_base_stack_config(self) -> MixtureOfExpertsModelConfig:
        return self._build_expert_model_config()

    def _build_attention_config(self):
        decoder_options = self.decoder_options
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
            dropout_probability=decoder_options.dropout_probability,
            zero_attention_flag=False,
            causal_attention_mask_flag=True,
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
            residual_model_flag=feed_forward_stack_options.residual_model_flag,
            residual_stack_options=feed_forward_stack_options.residual_stack_options,
            dropout_probability=feed_forward_stack_options.dropout_probability,
            last_layer_bias_option=feed_forward_stack_options.last_layer_bias_option,
            apply_output_postprocessing_flag=(
                feed_forward_stack_options.apply_output_postprocessing_flag
            ),
        )

    def _attention_experts_stack_options(self) -> ExpertsStackOptions:
        return ExpertsStackOptions(
            hidden_dim=self.hidden_dim,
            bias_flag=self.feed_forward_options.bias_flag,
            layer_norm_position=self.decoder_options.layer_norm_position,
            num_layers=self.feed_forward_options.num_layers,
            activation=self.decoder_options.activation,
            residual_connection_option=None,
            residual_model_flag=False,
            residual_stack_options=self.expert_stack_options.residual_stack_options,
            dropout_probability=self.decoder_options.dropout_probability,
            last_layer_bias_option=config.LastLayerBiasOptions.DEFAULT,
            apply_output_postprocessing_flag=True,
        )

    def _build_linear_layer_config(
        self,
        *,
        bias_flag: bool,
    ) -> AdaptiveLinearLayerConfig:
        adaptive_bias_enabled = self.hidden_adaptive_bias_options.option_flag
        return self._control_config_factory(
            self._attention_experts_stack_options(), self.attention_grouping_config
        ).build_hidden_adaptive_linear_layer_config(bias_flag or adaptive_bias_enabled)

    def _build_expert_model_config(
        self,
        *,
        use_feed_forward_stack_options: bool = True,
    ) -> MixtureOfExpertsModelConfig:
        stack_options = (
            self._feed_forward_experts_stack_options()
            if use_feed_forward_stack_options
            else self._attention_experts_stack_options()
        )
        grouping_config = self.grouping_config
        if use_feed_forward_stack_options:
            grouping_config = self.feed_forward_grouping_config
        model_config = self._control_config_factory(
            stack_options, grouping_config
        ).build()
        if isinstance(model_config, MixtureOfExpertsModelConfig):
            return model_config
        return model_config.block_config

    def _control_config_factory(
        self,
        stack_options: ExpertsStackOptions,
        grouping_config: GroupingConfig | None,
    ) -> ControlConfigFactory:
        return ControlConfigFactory(
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
                grouping_config=grouping_config,
                hidden_adaptive_weight_options=self.hidden_adaptive_weight_options,
                hidden_adaptive_bias_options=self.hidden_adaptive_bias_options,
                hidden_adaptive_diagonal_options=(
                    self.hidden_adaptive_diagonal_options
                ),
                hidden_adaptive_mask_options=self.hidden_adaptive_mask_options,
                router_grouping_config=self.router_grouping_config,
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


class GptExpertLinearAdaptiveConfigBuilder(
    _GptExpertLinearAdaptiveConfigBuilderImplementation
):
    def __init__(self, *, runtime: RuntimeOptions = DEFAULT_RUNTIME) -> None:
        if type(runtime) is not RuntimeOptions:
            raise TypeError(
                "models.gpt.expert_linear_adaptive GptExpertLinearAdaptiveConfigBuilder runtime must be RuntimeOptions"
            )
        self.runtime = runtime
        super().__init__(runtime)
