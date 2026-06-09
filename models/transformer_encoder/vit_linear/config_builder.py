import torch

import models.transformer_encoder.vit_linear.config as config

from emperor.attention.self_attention.config import SelfAttentionConfig
from emperor.base.layer import LayerConfig, LayerStackConfig
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.config import ModelConfig
from emperor.embedding.absolute.core.config import AbsolutePositionalEmbeddingConfig
from emperor.linears.core.config import LinearLayerConfig
from emperor.patch import LinearPatchEmbeddingConfig
from emperor.transformer.core.config import (
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayerConfig,
)
from emperor.transformer.feed_forward import FeedForwardConfig
from models.transformer_encoder.vit_linear.experiment_config import ExperimentConfig


class VitLinearConfigBuilder:
    def __init__(
        self,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        output_dim: int = config.OUTPUT_DIM,
        image_patch_size: int = config.IMAGE_PATCH_SIZE,
        input_channels: int = config.INPUT_CHANNELS,
        image_height: int = config.IMAGE_HEIGHT,
        patch_dropout_probability: float = config.PATCH_DROPOUT_PROBABILITY,
        patch_bias_flag: bool = config.PATCH_BIAS_FLAG,
        transformer_num_layers: int = config.TRANSFORMER_NUM_LAYERS,
        activation_function: ActivationOptions = config.ACTIVATION_FUNCTION,
        dropout_probability: float = config.DROPOUT_PROBABILITY,
        layer_norm_position: LayerNormPositionOptions = config.LAYER_NORM_POSITION,
        positional_embedding_option: type[
            AbsolutePositionalEmbeddingConfig
        ] = config.POSITIONAL_EMBEDDING_OPTION,
        positional_embedding_padding_idx: (
            int | None
        ) = config.POSITIONAL_EMBEDDING_PADDING_IDX,
        positional_embedding_auto_expand_flag: bool = config.POSITIONAL_EMBEDDING_AUTO_EXPAND_FLAG,
        attn_num_heads: int = config.ATTN_NUM_HEADS,
        attn_num_layers: int = config.ATTN_NUM_LAYERS,
        attn_bias_flag: bool = config.ATTN_BIAS_FLAG,
        attn_add_key_value_bias_flag: bool = config.ATTN_ADD_KEY_VALUE_BIAS_FLAG,
        ff_num_layers: int = config.FF_NUM_LAYERS,
        ff_bias_flag: bool = config.FF_BIAS_FLAG,
        output_num_layers: int = config.OUTPUT_NUM_LAYERS,
        output_bias_flag: bool = config.OUTPUT_BIAS_FLAG,
    ) -> None:
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.image_patch_size = image_patch_size
        self.input_channels = input_channels
        self.image_height = image_height
        self.patch_dropout_probability = patch_dropout_probability
        self.patch_bias_flag = patch_bias_flag
        self.transformer_num_layers = transformer_num_layers
        self.activation_function = activation_function
        self.dropout_probability = dropout_probability
        self.layer_norm_position = layer_norm_position
        self.positional_embedding_option = positional_embedding_option
        self.positional_embedding_padding_idx = positional_embedding_padding_idx
        self.positional_embedding_auto_expand_flag = (
            positional_embedding_auto_expand_flag
        )
        self.attn_num_heads = attn_num_heads
        self.attn_num_layers = attn_num_layers
        self.attn_bias_flag = attn_bias_flag
        self.attn_add_key_value_bias_flag = attn_add_key_value_bias_flag
        self.ff_num_layers = ff_num_layers
        self.ff_bias_flag = ff_bias_flag
        self.output_num_layers = output_num_layers
        self.output_bias_flag = output_bias_flag
        self.sequence_length = self._sequence_length()

    def build(self) -> ModelConfig:
        return ModelConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            sequence_length=self.sequence_length,
            experiment_config=ExperimentConfig(
                patch_config=self._build_patch_config(),
                positional_embedding_config=self._build_positional_embedding_config(),
                encoder_config=self._build_encoder_config(),
                output_config=self._build_output_config(),
            ),
        )

    def _sequence_length(self) -> int:
        if self.image_patch_size <= 0:
            raise ValueError(
                "image_patch_size must be positive, "
                f"received {self.image_patch_size}."
            )
        if self.image_height % self.image_patch_size != 0:
            raise ValueError(
                "image_height must be divisible by image_patch_size, "
                f"received image_height={self.image_height} and "
                f"image_patch_size={self.image_patch_size}."
            )
        patches_per_axis = self.image_height // self.image_patch_size
        return patches_per_axis * patches_per_axis + 1

    def _build_patch_config(self) -> LinearPatchEmbeddingConfig:
        return LinearPatchEmbeddingConfig(
            embedding_dim=self.hidden_dim,
            num_input_channels=self.input_channels,
            patch_size=self.image_patch_size,
            stride=self.image_patch_size,
            padding=0,
            dropout_probability=self.patch_dropout_probability,
            embedding_stack_config=self._build_linear_stack_config(
                input_dim=self.input_channels * self.image_patch_size**2,
                output_dim=self.hidden_dim,
                num_layers=1,
                bias_flag=self.patch_bias_flag,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                dropout_probability=self.dropout_probability,
                apply_output_pipeline_flag=False,
            ),
        )

    def _build_positional_embedding_config(self) -> AbsolutePositionalEmbeddingConfig:
        return self.positional_embedding_option(
            num_embeddings=self.sequence_length - 1,
            embedding_dim=self.hidden_dim,
            init_size=self.sequence_length,
            padding_idx=self.positional_embedding_padding_idx,
            auto_expand_flag=self.positional_embedding_auto_expand_flag,
            class_token_flag=True,
        )

    def _build_encoder_config(self) -> LayerStackConfig:
        return LayerStackConfig(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=self.transformer_num_layers,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=True,
            layer_config=TransformerEncoderBlockLayerConfig(
                activation=ActivationOptions.DISABLED,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                residual_flag=False,
                dropout_probability=0.0,
                gate_config=None,
                halting_config=None,
                shared_halting_flag=False,
                layer_model_config=self._build_encoder_layer_config(),
            ),
        )

    def _build_encoder_layer_config(self) -> TransformerEncoderLayerConfig:
        return TransformerEncoderLayerConfig(
            embedding_dim=self.hidden_dim,
            layer_norm_position=self.layer_norm_position,
            dropout_probability=self.dropout_probability,
            causal_attention_mask_flag=False,
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
            dropout_probability=self.dropout_probability,
            zero_attention_flag=False,
            causal_attention_mask_flag=False,
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
                dropout_probability=self.dropout_probability,
            ),
        )

    def _build_output_config(self) -> LayerStackConfig:
        return self._build_linear_stack_config(
            input_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=self.output_num_layers,
            bias_flag=self.output_bias_flag,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            dropout_probability=self.dropout_probability,
            apply_output_pipeline_flag=False,
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
                activation=self.activation_function,
                layer_norm_position=layer_norm_position,
                residual_flag=False,
                dropout_probability=dropout_probability,
                gate_config=None,
                halting_config=None,
                shared_halting_flag=False,
                layer_model_config=LinearLayerConfig(
                    bias_flag=bias_flag,
                ),
            ),
        )
