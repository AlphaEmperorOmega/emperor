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
        hidden_dim: int = config.HIDDEN_DIM,
        output_dim: int = config.OUTPUT_DIM,
        sequence_length: int = config.SEQUENCE_LENGTH,
        bias_flag: bool = config.BIAS_FLAG,
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
    ) -> None:
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.bias_flag = bias_flag
        self.stack_num_layers = stack_num_layers
        self.stack_activation = stack_activation
        self.stack_dropout_probability = stack_dropout_probability
        self.layer_norm_position = layer_norm_position
        self.causal_attention_mask_flag = causal_attention_mask_flag
        self.positional_embedding_option = positional_embedding_option
        self.positional_embedding_padding_idx = positional_embedding_padding_idx
        self.positional_embedding_auto_expand_flag = (
            positional_embedding_auto_expand_flag
        )
        self.embedding_dropout_probability = embedding_dropout_probability
        self.attn_num_heads = attn_num_heads
        self.attn_num_layers = attn_num_layers
        self.attn_bias_flag = attn_bias_flag
        self.attn_add_key_value_bias_flag = attn_add_key_value_bias_flag
        self.ff_num_layers = ff_num_layers
        self.ff_bias_flag = ff_bias_flag

    def build(self) -> "ModelConfig":
        from emperor.config import ModelConfig

        return ModelConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            sequence_length=self.sequence_length,
            experiment_config=ExperimentConfig(
                positional_embedding_config=self._build_positional_embedding_config(),
                embedding_dropout_probability=self.embedding_dropout_probability,
                encoder_config=self._build_encoder_config(),
            ),
        )

    def _build_positional_embedding_config(self) -> AbsolutePositionalEmbeddingConfig:
        return self.positional_embedding_option(
            num_embeddings=self.sequence_length,
            embedding_dim=self.hidden_dim,
            init_size=self.sequence_length,
            padding_idx=self.positional_embedding_padding_idx,
            auto_expand_flag=self.positional_embedding_auto_expand_flag,
        )

    def _build_encoder_config(self) -> LayerStackConfig:
        return LayerStackConfig(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=self.stack_num_layers,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=True,
            # The wrapped TransformerEncoderLayer owns its own norm, residual, and
            # dropout, so the generic Layer pipeline is neutralized to a pass-through.
            layer_config=TransformerEncoderBlockLayerConfig(
                activation=ActivationOptions.DISABLED,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                residual_connection_option=ResidualConnectionOptions.DISABLED,
                dropout_probability=0.0,
                gate_config=None,
                halting_config=None,
                layer_model_config=self._build_encoder_layer_config(),
            ),
        )

    def _build_encoder_layer_config(self) -> TransformerEncoderLayerConfig:
        return TransformerEncoderLayerConfig(
            embedding_dim=self.hidden_dim,
            layer_norm_position=self.layer_norm_position,
            dropout_probability=self.stack_dropout_probability,
            residual_connection_option=ResidualConnectionOptions.RESIDUAL,
            causal_attention_mask_flag=self.causal_attention_mask_flag,
            attention_config=self._build_attention_config(),
            feed_forward_config=self._build_feed_forward_config(),
        )

    def _build_attention_config(self) -> SelfAttentionConfig:
        return SelfAttentionConfig(
            batch_size=self.batch_size,
            num_heads=self.attn_num_heads,
            embedding_dim=self.hidden_dim,
            query_key_projection_dim=self.hidden_dim,
            value_projection_dim=self.hidden_dim,
            target_sequence_length=self.sequence_length,
            source_sequence_length=self.sequence_length,
            target_dtype=torch.float32,
            dropout_probability=self.stack_dropout_probability,
            zero_attention_flag=False,
            causal_attention_mask_flag=self.causal_attention_mask_flag,
            add_key_value_bias_flag=self.attn_add_key_value_bias_flag,
            average_attention_weights_flag=False,
            return_attention_weights_flag=False,
            projection_model_config=self._build_linear_stack_config(
                num_layers=self.attn_num_layers,
                bias_flag=self.attn_bias_flag,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                dropout_probability=0.0,
            ),
        )

    def _build_feed_forward_config(self) -> FeedForwardConfig:
        return FeedForwardConfig(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            stack_config=self._build_linear_stack_config(
                num_layers=self.ff_num_layers,
                bias_flag=self.ff_bias_flag,
                layer_norm_position=LayerNormPositionOptions.BEFORE,
                dropout_probability=self.stack_dropout_probability,
            ),
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
        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=apply_output_pipeline_flag,
            layer_config=LayerConfig(
                activation=self.stack_activation,
                layer_norm_position=layer_norm_position,
                residual_connection_option=ResidualConnectionOptions.DISABLED,
                dropout_probability=dropout_probability,
                gate_config=None,
                halting_config=None,
                layer_model_config=LinearLayerConfig(
                    bias_flag=bias_flag,
                ),
            ),
        )
