from emperor.base.options import ActivationOptions, LayerNormPositionOptions
from emperor.embedding.absolute.core.config import AbsolutePositionalEmbeddingConfig

import models.bert.linear.config as config
from models.bert._base_config_builder import BertBackendConfigBuilder
from models.bert._builder_adapter import linear_builder_kwargs_from_flat
from models.bert.linear.experiment_config import ExperimentConfig
from models.experts._builder_options import (
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsSubmoduleStackOptions,
)
from models.transformer._builder_options import (
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
    TransformerPositionalEmbeddingOptions,
)


class BertLinearConfigBuilder(BertBackendConfigBuilder):
    def __init__(
        self,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
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
        submodule_stack_options: ExpertsSubmoduleStackOptions | None = None,
        layer_controller_options: ExpertsLayerControllerOptions | None = None,
        dynamic_memory_options: ExpertsDynamicMemoryOptions | None = None,
        recurrent_controller_options: ExpertsRecurrentControllerOptions | None = None,
    ) -> None:
        defaults = linear_builder_kwargs_from_flat({}, config)
        encoder_options = encoder_options or TransformerEncoderOptions(
            hidden_dim=hidden_dim,
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
        super().__init__(
            batch_size=batch_size,
            learning_rate=learning_rate,
            input_dim=input_dim,
            output_dim=output_dim,
            sequence_length=sequence_length,
            embedding_dropout_probability=embedding_dropout_probability,
            encoder_options=encoder_options,
            positional_embedding_options=positional_embedding_options,
            attention_options=attention_options,
            feed_forward_options=feed_forward_options,
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
                recurrent_controller_options
                or defaults["recurrent_controller_options"]
            ),
            experiment_config_type=ExperimentConfig,
        )
