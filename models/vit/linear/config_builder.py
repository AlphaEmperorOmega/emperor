from emperor.base.options import ActivationOptions, LayerNormPositionOptions
from emperor.embedding.absolute.core.config import AbsolutePositionalEmbeddingConfig

import models.vit.linear.config as config
from models.transformer._builder_options import (
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
    TransformerPositionalEmbeddingOptions,
    VitOutputOptions,
    VitPatchOptions,
)
from models.vit._base_config_builder import VitBackendConfigBuilder
from models.vit.linear.experiment_config import ExperimentConfig


class VitLinearConfigBuilder(VitBackendConfigBuilder):
    def __init__(
        self,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        stack_hidden_dim: int = config.STACK_HIDDEN_DIM,
        output_dim: int = config.OUTPUT_DIM,
        image_patch_size: int = config.IMAGE_PATCH_SIZE,
        input_channels: int = config.INPUT_CHANNELS,
        image_height: int = config.IMAGE_HEIGHT,
        patch_dropout_probability: float = config.PATCH_DROPOUT_PROBABILITY,
        patch_bias_flag: bool = config.PATCH_BIAS_FLAG,
        stack_num_layers: int = config.STACK_NUM_LAYERS,
        stack_activation: ActivationOptions = config.STACK_ACTIVATION,
        stack_dropout_probability: float = config.STACK_DROPOUT_PROBABILITY,
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
        patch_options: VitPatchOptions | None = None,
        encoder_options: TransformerEncoderOptions | None = None,
        positional_embedding_options: (
            TransformerPositionalEmbeddingOptions | None
        ) = None,
        attention_options: TransformerAttentionOptions | None = None,
        feed_forward_options: TransformerFeedForwardOptions | None = None,
        output_options: VitOutputOptions | None = None,
    ) -> None:
        patch_options = patch_options or VitPatchOptions(
            patch_size=image_patch_size,
            input_channels=input_channels,
            image_height=image_height,
            dropout_probability=patch_dropout_probability,
            bias_flag=patch_bias_flag,
        )
        encoder_options = encoder_options or TransformerEncoderOptions(
            hidden_dim=stack_hidden_dim,
            num_layers=stack_num_layers,
            activation=stack_activation,
            dropout_probability=stack_dropout_probability,
            layer_norm_position=layer_norm_position,
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
        output_options = output_options or VitOutputOptions(
            num_layers=output_num_layers,
            bias_flag=output_bias_flag,
        )
        super().__init__(
            batch_size=batch_size,
            learning_rate=learning_rate,
            input_dim=input_dim,
            output_dim=output_dim,
            patch_options=patch_options,
            encoder_options=encoder_options,
            positional_embedding_options=positional_embedding_options,
            attention_options=attention_options,
            feed_forward_options=feed_forward_options,
            output_options=output_options,
            experiment_config_type=ExperimentConfig,
        )
