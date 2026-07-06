from typing import TYPE_CHECKING

import torch
from emperor.attention.core.variants.self_attention.config import SelfAttentionConfig
from emperor.base.layer import LayerConfig, LayerStackConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.linears.core.config import LinearLayerConfig
from emperor.patch import LinearPatchEmbeddingConfig
from emperor.transformer.core.config import (
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayerConfig,
)
from emperor.transformer.feed_forward import FeedForwardConfig

from models.transformer._builder_options import (
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
    TransformerPositionalEmbeddingOptions,
    VitOutputOptions,
    VitPatchOptions,
)
from models.vit._base_experiment_config import ExperimentConfig

if TYPE_CHECKING:
    from emperor.config import ModelConfig
    from emperor.base.utils import ConfigBase


class VitBackendConfigBuilder:
    def __init__(
        self,
        *,
        batch_size: int,
        learning_rate: float,
        input_dim: int,
        output_dim: int,
        patch_options: VitPatchOptions,
        encoder_options: TransformerEncoderOptions,
        positional_embedding_options: TransformerPositionalEmbeddingOptions,
        attention_options: TransformerAttentionOptions,
        feed_forward_options: TransformerFeedForwardOptions,
        output_options: VitOutputOptions,
        experiment_config_type: type[ExperimentConfig] = ExperimentConfig,
    ) -> None:
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.patch_options = patch_options
        self.image_patch_size = patch_options.patch_size
        self.input_channels = patch_options.input_channels
        self.image_height = patch_options.image_height
        self.patch_dropout_probability = patch_options.dropout_probability
        self.patch_bias_flag = patch_options.bias_flag
        self.encoder_options = encoder_options
        self.hidden_dim = encoder_options.hidden_dim
        self.stack_num_layers = encoder_options.num_layers
        self.stack_activation = encoder_options.activation
        self.stack_dropout_probability = encoder_options.dropout_probability
        self.layer_norm_position = encoder_options.layer_norm_position
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
        self.output_options = output_options
        self.output_num_layers = output_options.num_layers
        self.output_bias_flag = output_options.bias_flag
        self.experiment_config_type = experiment_config_type
        self.sequence_length = self._sequence_length()

    def build(self) -> "ModelConfig":
        from emperor.config import ModelConfig

        patch_config = self._build_patch_config()
        positional_embedding_config = self._build_positional_embedding_config()
        encoder_config = self._build_encoder_config()
        output_config = self._build_output_config()
        return ModelConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            sequence_length=self.sequence_length,
            experiment_config=self.experiment_config_type(
                patch_config=patch_config,
                positional_embedding_config=positional_embedding_config,
                encoder_config=encoder_config,
                output_config=output_config,
            ),
        )

    def _sequence_length(self) -> int:
        if self.image_patch_size <= 0:
            raise ValueError(
                f"image_patch_size must be positive, received {self.image_patch_size}."
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
        options = self.patch_options
        embedding_stack_config = self._build_plain_linear_stack_config(
            input_dim=options.input_channels * options.patch_size**2,
            output_dim=self.hidden_dim,
            num_layers=1,
            bias_flag=options.bias_flag,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            dropout_probability=self.encoder_options.dropout_probability,
            apply_output_pipeline_flag=False,
        )
        return LinearPatchEmbeddingConfig(
            embedding_dim=self.hidden_dim,
            num_input_channels=options.input_channels,
            patch_size=options.patch_size,
            stride=options.patch_size,
            padding=0,
            dropout_probability=options.dropout_probability,
            embedding_stack_config=embedding_stack_config,
        )

    def _build_positional_embedding_config(self):
        options = self.positional_embedding_options
        positional_embedding_config = options.option
        return positional_embedding_config(
            num_embeddings=self.sequence_length - 1,
            embedding_dim=self.hidden_dim,
            init_size=self.sequence_length,
            padding_idx=options.padding_idx,
            auto_expand_flag=options.auto_expand_flag,
            class_token_flag=True,
        )

    def _build_encoder_config(self) -> "ConfigBase":
        options = self.encoder_options
        layer_model_config = self._build_encoder_layer_config()
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
            layer_config=layer_config,
        )

    def _build_encoder_layer_config(self) -> TransformerEncoderLayerConfig:
        options = self.encoder_options
        attention_config = self._build_attention_config()
        feed_forward_config = self._build_feed_forward_config()
        return TransformerEncoderLayerConfig(
            embedding_dim=self.hidden_dim,
            layer_norm_position=options.layer_norm_position,
            dropout_probability=options.dropout_probability,
            residual_connection_option=ResidualConnectionOptions.RESIDUAL,
            causal_attention_mask_flag=False,
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
            causal_attention_mask_flag=False,
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
        return self._build_backend_linear_stack_config(
            num_layers=attention_options.num_layers,
            bias_flag=attention_options.bias_flag,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            dropout_probability=0.0,
        )

    def _build_feed_forward_stack_config(self) -> "ConfigBase":
        encoder_options = self.encoder_options
        feed_forward_options = self.feed_forward_options
        return self._build_backend_linear_stack_config(
            num_layers=feed_forward_options.num_layers,
            bias_flag=feed_forward_options.bias_flag,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            dropout_probability=encoder_options.dropout_probability,
        )

    def _build_output_config(self) -> LayerStackConfig:
        options = self.output_options
        return self._build_plain_linear_stack_config(
            input_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=options.num_layers,
            bias_flag=options.bias_flag,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            dropout_probability=self.encoder_options.dropout_probability,
            apply_output_pipeline_flag=False,
        )

    def _build_backend_linear_layer_config(
        self,
        *,
        bias_flag: bool,
    ) -> LinearLayerConfig:
        return self._build_plain_linear_layer_config(bias_flag=bias_flag)

    def _build_plain_linear_layer_config(
        self,
        *,
        bias_flag: bool,
    ) -> LinearLayerConfig:
        return LinearLayerConfig(bias_flag=bias_flag)

    def _build_backend_linear_stack_config(
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
        return self._build_linear_stack_config(
            layer_model_config=self._build_backend_linear_layer_config(
                bias_flag=bias_flag,
            ),
            num_layers=num_layers,
            layer_norm_position=layer_norm_position,
            dropout_probability=dropout_probability,
            input_dim=input_dim,
            output_dim=output_dim,
            apply_output_pipeline_flag=apply_output_pipeline_flag,
        )

    def _build_plain_linear_stack_config(
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
        return self._build_linear_stack_config(
            layer_model_config=self._build_plain_linear_layer_config(
                bias_flag=bias_flag,
            ),
            num_layers=num_layers,
            layer_norm_position=layer_norm_position,
            dropout_probability=dropout_probability,
            input_dim=input_dim,
            output_dim=output_dim,
            apply_output_pipeline_flag=apply_output_pipeline_flag,
        )

    def _build_linear_stack_config(
        self,
        *,
        layer_model_config,
        num_layers: int,
        layer_norm_position: LayerNormPositionOptions,
        dropout_probability: float,
        input_dim: int | None = None,
        output_dim: int | None = None,
        apply_output_pipeline_flag: bool = True,
    ) -> LayerStackConfig:
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
