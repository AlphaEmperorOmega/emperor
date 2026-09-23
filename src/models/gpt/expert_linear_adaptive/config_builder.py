import copy
from collections.abc import Iterator
from dataclasses import fields
from typing import TYPE_CHECKING, TypeVar, cast

import torch

import models.gpt.expert_linear_adaptive.config as config
from emperor.attention import (
    MixtureOfAttentionHeadsConfig,
    SelfAttentionConfig,
    SelfAttentionProjectionStrategy,
)
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    DynamicBiasConfig,
    DynamicWeightConfig,
    GroupingConfig,
    WeightDecayScheduleOptions,
)
from emperor.config import ConfigBase
from emperor.embedding.absolute import TextLearnedPositionalEmbeddingConfig
from emperor.embedding.contextual import (
    ByteContextualEmbeddingConfig,
    CausalPrefixKernelConfig,
)
from emperor.embedding.hierarchical import HierarchicalByteEmbeddingConfig
from emperor.embedding.relative import DynamicPositionalBiasConfig
from emperor.experts import (
    ExpertWeightingPositionOptions,
    MixtureOfExpertsConfig,
    MixtureOfExpertsModelConfig,
    RoutingInitializationMode,
)
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.transformer import (
    FeedForwardConfig,
    TransformerConfig,
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayerConfig,
)
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

if TYPE_CHECKING:
    from emperor.config import ModelConfig

_StackConfig = TypeVar("_StackConfig", bound=ConfigBase)


def _nested_configs(config_node: object) -> Iterator[ConfigBase]:
    if isinstance(config_node, ConfigBase):
        yield config_node
        for config_field in fields(config_node):
            yield from _nested_configs(getattr(config_node, config_field.name))


def _isolate_token_rows(stack_config: _StackConfig) -> _StackConfig:
    """Copy a stack so each byte-encoded token stays independent of the batch."""
    isolated = copy.deepcopy(stack_config)
    for child in _nested_configs(isolated):
        # Grouping mixes rows and decay schedules advance once per length group.
        if isinstance(child, AdaptiveParameterAugmentationConfig):
            child.grouping_config = None
        if (
            isinstance(child, (DynamicWeightConfig, DynamicBiasConfig))
            and child.decay_schedule is not None
        ):
            child.decay_schedule = WeightDecayScheduleOptions.DISABLED
    return isolated


class _GptExpertLinearAdaptiveConfigBuilderImplementation(GptBackendConfigBuilder):
    def build(self) -> "ModelConfig":
        model_config = super().build()
        if self.embedding_options.contextual_flag:
            model_config.experiment_config.contextual_embedding_config = (
                self._build_contextual_embedding_config()
            )
        if self.embedding_options.hierarchical_flag:
            model_config.experiment_config.hierarchical_embedding_config = (
                self._build_hierarchical_embedding_config()
            )
        return model_config

    def _build_hierarchical_embedding_config(self) -> HierarchicalByteEmbeddingConfig:
        options = self.embedding_options
        # Each token's bytes follow the leading [W] pooling symbol.
        byte_sequence_length = options.hierarchical_max_token_bytes + 1
        return HierarchicalByteEmbeddingConfig(
            byte_embedding_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            max_token_bytes=options.hierarchical_max_token_bytes,
            byte_position_config=TextLearnedPositionalEmbeddingConfig(
                num_embeddings=byte_sequence_length,
                embedding_dim=self.hidden_dim,
            ),
            encoder_config=TransformerConfig(
                encoder_stack_config=LayerStackConfig(
                    input_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=self.hidden_dim,
                    num_layers=options.hierarchical_encoder_num_layers,
                    last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                    apply_output_postprocessing_flag=True,
                    # The wrapped encoder layer owns its norm, residual, and dropout.
                    layer_config=TransformerEncoderBlockLayerConfig(
                        activation=ActivationOptions.DISABLED,
                        layer_norm_position=LayerNormPositionOptions.DISABLED,
                        dropout_probability=0.0,
                        layer_model_config=self._build_hierarchical_encoder_layer_config(
                            byte_sequence_length
                        ),
                    ),
                )
            ),
            projection_config=_isolate_token_rows(
                self._build_linear_stack_config(
                    input_dim=self.hidden_dim,
                    output_dim=self.hidden_dim,
                    num_layers=1,
                    bias_flag=self.feed_forward_options.bias_flag,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    dropout_probability=0.0,
                    apply_output_postprocessing_flag=False,
                )
            ),
        )

    def _build_hierarchical_encoder_layer_config(
        self, byte_sequence_length: int
    ) -> TransformerEncoderLayerConfig:
        decoder_options = self.decoder_options
        attention_options = self.attention_options
        # The transformer's adaptive projection and feed-forward stacks, without
        # attention heads or feed-forward experts.
        projection_model_config = _isolate_token_rows(
            self._build_attention_projection_base_stack_config()
        )
        feed_forward_stack_config = _isolate_token_rows(
            cast(LayerStackConfig, super()._build_feed_forward_base_stack_config())
        )
        return TransformerEncoderLayerConfig(
            embedding_dim=self.hidden_dim,
            layer_norm_position=decoder_options.layer_norm_position,
            normalization=decoder_options.normalization,
            dropout_probability=decoder_options.dropout_probability,
            residual_config=AdditiveResidualConfig(),
            attention_config=SelfAttentionConfig(
                # Bounds the largest equal-length token group in one batch.
                batch_size=self.batch_size * self.sequence_length,
                num_heads=attention_options.num_heads,
                embedding_dim=self.hidden_dim,
                query_key_projection_dim=self.hidden_dim,
                value_projection_dim=self.hidden_dim,
                target_sequence_length=byte_sequence_length,
                source_sequence_length=byte_sequence_length,
                target_dtype=torch.float32,
                dropout_probability=decoder_options.dropout_probability,
                zero_attention_flag=False,
                causal_attention_mask_flag=False,
                add_key_value_bias_flag=attention_options.add_key_value_bias_flag,
                average_attention_weights_flag=False,
                return_attention_weights_flag=False,
                batch_first_flag=True,
                projection_model_config=projection_model_config,
                projection_strategy=SelfAttentionProjectionStrategy.SEPARATE,
            ),
            feed_forward_config=FeedForwardConfig(
                input_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                stack_config=feed_forward_stack_config,
            ),
        )

    def _build_contextual_embedding_config(self) -> ByteContextualEmbeddingConfig:
        options = self.embedding_options
        kernel_dim = options.kernel_dim
        return ByteContextualEmbeddingConfig(
            max_token_bytes=options.max_token_bytes,
            hidden_dim=self.hidden_dim,
            byte_moe_config=self._build_embedding_moe_config(
                options.max_token_bytes * 9
            ),
            positional_embedding_config=self._build_positional_embedding_config(),
            prefix_kernel_config=CausalPrefixKernelConfig(
                hidden_dim=self.hidden_dim,
                kernel_dim=kernel_dim,
                relative_position_config=DynamicPositionalBiasConfig(
                    num_heads=1,
                    embedding_dim=kernel_dim,
                    max_positions=max(1, self.sequence_length - 1),
                ),
            ),
            context_moe_config=self._build_embedding_moe_config(self.hidden_dim * 2),
            residual_scale_initial_value=options.residual_scale_initial_value,
        )

    def _build_embedding_moe_config(self, input_dim: int) -> MixtureOfExpertsConfig:
        # Copy the transformer's expert and router architecture, not their weights.
        model_config = self._build_expert_model_config(
            use_feed_forward_stack_options=False
        )
        mixture = copy.deepcopy(
            model_config.stack_config.layer_config.layer_model_config
        )
        mixture.input_dim = input_dim
        mixture.output_dim = self.hidden_dim
        # These are the contextual component's causal/batch-isolation constraints.
        mixture.capacity_factor = 0.0
        mixture.routing_initialization_mode = RoutingInitializationMode.LAYER
        mixture.compute_expert_mixture_flag = True
        mixture.weighted_parameters_flag = True
        mixture.weighting_position_option = ExpertWeightingPositionOptions.AFTER_EXPERTS
        mixture.sampler_config.normalize_probabilities_flag = True
        mixture.sampler_config.router_config.input_dim = input_dim
        return mixture

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
            normalization=feed_forward_stack_options.normalization,
            num_layers=feed_forward_stack_options.num_layers,
            activation=feed_forward_stack_options.activation,
            residual_connection_option=(
                feed_forward_stack_options.residual_connection_option
            ),
            residual_block_size=feed_forward_stack_options.residual_block_size,
            residual_rms_norm_epsilon=feed_forward_stack_options.residual_rms_norm_epsilon,
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
            normalization=self.decoder_options.normalization,
            num_layers=self.feed_forward_options.num_layers,
            activation=self.decoder_options.activation,
            residual_connection_option=None,
            residual_block_size=None,
            residual_rms_norm_epsilon=None,
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
