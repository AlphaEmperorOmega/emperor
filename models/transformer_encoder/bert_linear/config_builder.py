from emperor.base.layer.residual import ResidualConnectionOptions
import torch

import models.transformer_encoder.bert_linear.config as config

from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.base.layer import LayerStackConfig
from emperor.base.layer.config import LayerConfig
from emperor.linears.core.config import LinearLayerConfig
from emperor.attention.core.variants.self_attention.config import SelfAttentionConfig
from emperor.transformer.core.config import (
    TransformerEncoderLayerConfig,
    TransformerEncoderBlockLayerConfig,
)
from emperor.transformer.feed_forward import FeedForwardConfig
from models.transformer_encoder._builder_options import (
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
    TransformerPositionalEmbeddingOptions,
)
from emperor.embedding.absolute.core.config import AbsolutePositionalEmbeddingConfig
from models.transformer_encoder.bert_linear.experiment_config import ExperimentConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class BertLinearConfigBuilder:
    def __init__(
        self,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        stack_hidden_dim: int = config.STACK_HIDDEN_DIM,
        output_dim: int = config.OUTPUT_DIM,
        sequence_length: int = config.SEQUENCE_LENGTH,
        stack_num_layers: int = config.STACK_NUM_LAYERS,
        stack_activation: ActivationOptions = config.STACK_ACTIVATION,
        stack_dropout_probability: float = config.STACK_DROPOUT_PROBABILITY,
        layer_norm_position: LayerNormPositionOptions = config.LAYER_NORM_POSITION,
        causal_attention_mask_flag: bool = config.CAUSAL_ATTENTION_MASK_FLAG,
        positional_embedding_option: type[
            AbsolutePositionalEmbeddingConfig
        ] = config.POSITIONAL_EMBEDDING_OPTION,
        positional_embedding_padding_idx: int = config.POSITIONAL_EMBEDDING_PADDING_IDX,
        positional_embedding_auto_expand_flag: bool = config.POSITIONAL_EMBEDDING_AUTO_EXPAND_FLAG,
        embedding_dropout_probability: float = config.EMBEDDING_DROPOUT_PROBABILITY,
        attn_num_heads: int = config.ATTN_NUM_HEADS,
        attn_num_layers: int = config.ATTN_NUM_LAYERS,
        attn_bias_flag: bool = config.ATTN_BIAS_FLAG,
        attn_add_key_value_bias_flag: bool = config.ATTN_ADD_KEY_VALUE_BIAS_FLAG,
        ff_num_layers: int = config.FF_NUM_LAYERS,
        ff_bias_flag: bool = config.FF_BIAS_FLAG,
        encoder_options: TransformerEncoderOptions | None = None,
        positional_embedding_options: (
            TransformerPositionalEmbeddingOptions | None
        ) = None,
        attention_options: TransformerAttentionOptions | None = None,
        feed_forward_options: TransformerFeedForwardOptions | None = None,
    ) -> None:
        encoder_options = encoder_options or TransformerEncoderOptions(
            hidden_dim=stack_hidden_dim,
            num_layers=stack_num_layers,
            activation=stack_activation,
            dropout_probability=stack_dropout_probability,
            layer_norm_position=layer_norm_position,
            causal_attention_mask_flag=causal_attention_mask_flag,
        )
        positional_embedding_options = (
            positional_embedding_options
            or TransformerPositionalEmbeddingOptions(
                option=positional_embedding_option,
                padding_idx=positional_embedding_padding_idx,
                auto_expand_flag=positional_embedding_auto_expand_flag,
            )
        )
        attention_options = attention_options or TransformerAttentionOptions(
            num_heads=attn_num_heads,
            num_layers=attn_num_layers,
            bias_flag=attn_bias_flag,
            add_key_value_bias_flag=attn_add_key_value_bias_flag,
        )
        feed_forward_options = (
            feed_forward_options
            or TransformerFeedForwardOptions(
                num_layers=ff_num_layers,
                bias_flag=ff_bias_flag,
            )
        )
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.encoder_options = encoder_options
        self.hidden_dim = encoder_options.hidden_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
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
        self.embedding_dropout_probability = embedding_dropout_probability
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

    def build(self) -> "ModelConfig":
        from emperor.config import ModelConfig

        positional_embedding_config = self.__build_positional_embedding_config()
        encoder_config = self.__build_encoder_config()
        return ModelConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            sequence_length=self.sequence_length,
            experiment_config=ExperimentConfig(
                positional_embedding_config=positional_embedding_config,
                embedding_dropout_probability=self.embedding_dropout_probability,
                encoder_config=encoder_config,
            ),
        )

    def __build_positional_embedding_config(
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

    def __build_encoder_config(self) -> LayerStackConfig:
        options = self.encoder_options
        layer_model_config = self.__build_encoder_layer_config()
        layer_config = TransformerEncoderBlockLayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            layer_model_config=layer_model_config,
        )
        return LayerStackConfig(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=options.num_layers,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=True,
            # The wrapped TransformerEncoderLayer owns its own norm, residual, and
            # dropout, so the generic Layer pipeline is neutralized to a pass-through.
            layer_config=layer_config,
        )

    def __build_encoder_layer_config(self) -> TransformerEncoderLayerConfig:
        options = self.encoder_options
        attention_config = self.__build_attention_config()
        feed_forward_config = self.__build_feed_forward_config()
        return TransformerEncoderLayerConfig(
            embedding_dim=self.hidden_dim,
            layer_norm_position=options.layer_norm_position,
            dropout_probability=options.dropout_probability,
            residual_connection_option=ResidualConnectionOptions.RESIDUAL,
            causal_attention_mask_flag=options.causal_attention_mask_flag,
            attention_config=attention_config,
            feed_forward_config=feed_forward_config,
        )

    def __build_attention_config(self) -> SelfAttentionConfig:
        encoder_options = self.encoder_options
        attention_options = self.attention_options
        projection_model_config = self.__build_linear_stack_config(
            num_layers=attention_options.num_layers,
            bias_flag=attention_options.bias_flag,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            dropout_probability=0.0,
        )
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

    def __build_feed_forward_config(self) -> FeedForwardConfig:
        encoder_options = self.encoder_options
        feed_forward_options = self.feed_forward_options
        stack_config = self.__build_linear_stack_config(
            num_layers=feed_forward_options.num_layers,
            bias_flag=feed_forward_options.bias_flag,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            dropout_probability=encoder_options.dropout_probability,
        )
        return FeedForwardConfig(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            stack_config=stack_config,
        )

    def __build_linear_stack_config(
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
        layer_model_config = LinearLayerConfig(
            bias_flag=bias_flag,
        )
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
