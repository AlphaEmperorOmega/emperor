from typing import TYPE_CHECKING

import torch
from emperor.attention.core.variants.self_attention.config import SelfAttentionConfig
from emperor.base.layer import LayerStackConfig, RecurrentLayerConfig
from emperor.base.layer.config import LayerConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.embedding.absolute.core.config import AbsolutePositionalEmbeddingConfig
from emperor.linears.core.config import LinearLayerConfig
from emperor.transformer.core.config import (
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayerConfig,
)
from emperor.transformer.feed_forward import FeedForwardConfig

from models.bert._base_experiment_config import ExperimentConfig
from models.bert._gate_config_factory import BertGateConfigFactory
from models.bert._halting_config_factory import BertHaltingConfigFactory
from models.bert._memory_config_factory import BertMemoryConfigFactory
from models.bert._recurrent_config_factory import BertRecurrentConfigFactory
from models.experts._builder_options import (
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsStackOptions,
    ExpertsSubmoduleStackOptions,
)
from models.transformer._builder_options import (
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
    TransformerPositionalEmbeddingOptions,
)

if TYPE_CHECKING:
    from emperor.config import ModelConfig
    from emperor.base.utils import ConfigBase


class BertBackendConfigBuilder:
    def __init__(
        self,
        *,
        batch_size: int,
        learning_rate: float,
        input_dim: int,
        output_dim: int,
        sequence_length: int,
        embedding_dropout_probability: float,
        encoder_options: TransformerEncoderOptions,
        positional_embedding_options: TransformerPositionalEmbeddingOptions,
        attention_options: TransformerAttentionOptions,
        feed_forward_options: TransformerFeedForwardOptions,
        submodule_stack_options: ExpertsSubmoduleStackOptions,
        layer_controller_options: ExpertsLayerControllerOptions,
        dynamic_memory_options: ExpertsDynamicMemoryOptions,
        recurrent_controller_options: ExpertsRecurrentControllerOptions,
        experiment_config_type: type[ExperimentConfig] = ExperimentConfig,
    ) -> None:
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.embedding_dropout_probability = embedding_dropout_probability
        self.encoder_options = encoder_options
        self.hidden_dim = encoder_options.hidden_dim
        self.stack_num_layers = encoder_options.num_layers
        self.stack_activation = encoder_options.activation
        self.stack_dropout_probability = encoder_options.dropout_probability
        self.layer_norm_position = encoder_options.layer_norm_position
        self.causal_attention_mask_flag = encoder_options.causal_attention_mask_flag
        self.positional_embedding_options = positional_embedding_options
        self.positional_embedding_option = positional_embedding_options.option
        self.positional_embedding_padding_idx = positional_embedding_options.padding_idx
        self.positional_embedding_auto_expand_flag = (
            positional_embedding_options.auto_expand_flag
        )
        self.attention_options = attention_options
        self.attn_num_heads = attention_options.num_heads
        self.attn_num_layers = attention_options.num_layers
        self.attn_bias_flag = attention_options.bias_flag
        self.attn_add_key_value_bias_flag = (
            attention_options.add_key_value_bias_flag
        )
        self.feed_forward_options = feed_forward_options
        self.ff_num_layers = feed_forward_options.num_layers
        self.ff_bias_flag = feed_forward_options.bias_flag
        self.submodule_stack_options = submodule_stack_options
        self.layer_controller_options = layer_controller_options
        self.dynamic_memory_options = dynamic_memory_options
        self.recurrent_controller_options = recurrent_controller_options
        self.experiment_config_type = experiment_config_type

    def build(self) -> "ModelConfig":
        from emperor.config import ModelConfig

        positional_embedding_config = self._build_positional_embedding_config()
        encoder_config = self._build_encoder_config()
        return ModelConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            sequence_length=self.sequence_length,
            experiment_config=self.experiment_config_type(
                positional_embedding_config=positional_embedding_config,
                embedding_dropout_probability=self.embedding_dropout_probability,
                encoder_config=encoder_config,
            ),
        )

    def _build_positional_embedding_config(
        self,
    ) -> AbsolutePositionalEmbeddingConfig:
        options = self.positional_embedding_options
        positional_embedding_config = options.option
        return positional_embedding_config(
            num_embeddings=self.sequence_length,
            embedding_dim=self.hidden_dim,
            init_size=self.sequence_length,
            padding_idx=options.padding_idx,
            auto_expand_flag=options.auto_expand_flag,
        )

    def _build_encoder_config(self) -> "ConfigBase":
        layer_model_config = self._build_encoder_layer_config()
        gate_factory = self._gate_config_factory()
        halting_factory = self._halting_config_factory()
        memory_factory = self._memory_config_factory()
        layer_config = TransformerEncoderBlockLayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            gate_config=gate_factory.build_gate_config(),
            halting_config=halting_factory.build_halting_config(),
            layer_model_config=layer_model_config,
        )
        stack_config = LayerStackConfig(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=self.encoder_options.num_layers,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=True,
            shared_gate_config=self.layer_controller_options.shared_gate_config,
            shared_memory_config=memory_factory.build_memory_config(),
            # The wrapped TransformerEncoderLayer owns its own norm, residual, and
            # dropout, so the generic Layer pipeline is neutralized to a pass-through.
            layer_config=layer_config,
        )
        recurrent_factory = BertRecurrentConfigFactory(
            recurrent_controller_options=self.recurrent_controller_options,
            gate_config_factory=gate_factory,
            halting_config_factory=halting_factory,
        )
        encoder_config = recurrent_factory.build_config(stack_config)
        if isinstance(encoder_config, RecurrentLayerConfig):
            encoder_config.input_dim = self.hidden_dim
            encoder_config.output_dim = self.hidden_dim
        return encoder_config

    def _build_encoder_layer_config(self) -> TransformerEncoderLayerConfig:
        options = self.encoder_options
        attention_config = self._build_attention_config()
        feed_forward_config = self._build_feed_forward_config()
        return TransformerEncoderLayerConfig(
            embedding_dim=self.hidden_dim,
            layer_norm_position=options.layer_norm_position,
            dropout_probability=options.dropout_probability,
            residual_connection_option=ResidualConnectionOptions.RESIDUAL,
            causal_attention_mask_flag=options.causal_attention_mask_flag,
            attention_config=attention_config,
            feed_forward_config=feed_forward_config,
        )

    def _build_attention_config(self) -> SelfAttentionConfig:
        encoder_options = self.encoder_options
        attention_options = self.attention_options
        projection_model_config = self._build_attention_projection_stack_config()
        return SelfAttentionConfig(
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
            projection_model_config=projection_model_config,
        )

    def _build_feed_forward_config(self) -> FeedForwardConfig:
        stack_config = self._build_feed_forward_stack_config()
        return FeedForwardConfig(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            stack_config=stack_config,
        )

    def _build_attention_projection_stack_config(self) -> LayerStackConfig:
        attention_options = self.attention_options
        return self._build_linear_stack_config(
            num_layers=attention_options.num_layers,
            bias_flag=attention_options.bias_flag,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            dropout_probability=0.0,
        )

    def _build_feed_forward_stack_config(self) -> LayerStackConfig:
        encoder_options = self.encoder_options
        feed_forward_options = self.feed_forward_options
        return self._build_linear_stack_config(
            num_layers=feed_forward_options.num_layers,
            bias_flag=feed_forward_options.bias_flag,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            dropout_probability=encoder_options.dropout_probability,
        )

    def _build_linear_layer_config(self, *, bias_flag: bool) -> LinearLayerConfig:
        return LinearLayerConfig(
            bias_flag=bias_flag,
        )

    def _build_linear_stack_config(
        self,
        *,
        num_layers: int,
        bias_flag: bool,
        layer_norm_position: LayerNormPositionOptions,
        dropout_probability: float,
        input_dim: int | None = None,
        output_dim: int | None = None,
        apply_output_pipeline_flag: bool = True,
    ) -> LayerStackConfig:
        layer_model_config = self._build_linear_layer_config(bias_flag=bias_flag)
        layer_config = LayerConfig(
            activation=self.encoder_options.activation,
            layer_norm_position=layer_norm_position,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=dropout_probability,
            gate_config=None,
            halting_config=None,
            layer_model_config=layer_model_config,
        )
        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=apply_output_pipeline_flag,
            layer_config=layer_config,
        )

    def _gate_config_factory(self) -> BertGateConfigFactory:
        return BertGateConfigFactory(
            layer_controller_options=self.layer_controller_options,
            recurrent_controller_options=self.recurrent_controller_options,
            submodule_stack_options=self.submodule_stack_options,
        )

    def _halting_config_factory(self) -> BertHaltingConfigFactory:
        return BertHaltingConfigFactory(
            layer_controller_options=self.layer_controller_options,
            recurrent_controller_options=self.recurrent_controller_options,
            submodule_stack_options=self.submodule_stack_options,
            output_dim=self.hidden_dim,
        )

    def _memory_config_factory(self) -> BertMemoryConfigFactory:
        stack_options = ExpertsStackOptions(
            hidden_dim=self.hidden_dim,
            bias_flag=True,
            layer_norm_position=self.encoder_options.layer_norm_position,
            num_layers=self.encoder_options.num_layers,
            activation=self.encoder_options.activation,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=self.encoder_options.dropout_probability,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=True,
        )
        return BertMemoryConfigFactory(
            stack_options=stack_options,
            dynamic_memory_options=self.dynamic_memory_options,
            submodule_stack_options=self.submodule_stack_options,
        )
