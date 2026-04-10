import torch

from emperor.base.enums import BaseOptions, ActivationOptions, LayerNormPositionOptions
from emperor.datasets.image.classification.mnist import Mnist
from emperor.linears.core.config import LinearLayerConfig
from emperor.base.layer import LayerStackConfig
from emperor.transformer.utils.layers import TransformerConfig
from emperor.transformer.utils.presets import TransformerPresets
from emperor.transformer.utils.patch.selector import PatchOptions
from emperor.transformer.utils.patch.options.base import PatchConfig
from emperor.attention.utils.layer import MultiHeadAttentionConfig
from emperor.transformer.utils.feed_forward import FeedForwardConfig
from emperor.linears.options import LinearLayerOptions, LinearLayerStackOptions
from emperor.embedding.options import AbsolutePositionalEmbeddingOptions
from emperor.embedding.absolute.config import AbsolutePositionalEmbeddingConfig
from emperor.experiments.base import ExperimentBase, ExperimentPresetsBase
from emperor.augmentations.adaptive_parameters.options import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
)
import models.vit.config as config
from models.vit.config import ExperimentConfig
from models.vit.model import Model
from emperor.experiments.base import SearchMode

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExperimentOptions(BaseOptions):
    PRESET = 0
    CONFIG = 1
    ADAPTIVE = 2


class ExperimentPresets(ExperimentPresetsBase):
    def get_config(
        self,
        model_config_options: ExperimentOptions = ExperimentOptions.PRESET,
        dataset: type = Mnist,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
    ) -> list["ModelConfig"]:
        match model_config_options:
            case ExperimentOptions.PRESET:
                return self._create_default_preset_configs(dataset)
            case ExperimentOptions.CONFIG:
                return self._create_default_search_space_configs(dataset, search_mode, log_folder)
            case ExperimentOptions.ADAPTIVE:
                return [self.__adaptive_preset(**self._dataset_config(dataset))]
            case _:
                raise ValueError(
                    "The specified option is not supported. Please choose a valid `ExperimentOptions`."
                )

    def _preset(
        self,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        output_dim: int = config.OUTPUT_DIM,
        dropout_probability: float = config.DROPOUT_PROBABILITY,
        activation_function: ActivationOptions = config.ACTIVATION_FUNCTION,
        output_num_layers: int = config.OUTPUT_NUM_LAYERS,
        transformer_num_layers: int = config.TRANSFORMER_NUM_LAYERS,
        attn_bias_flag: bool = config.ATTN_BIAS_FLAG,
        attn_num_heads: int = config.ATTN_NUM_HEADS,
        attn_model_type: LinearLayerStackOptions = config.ATTN_MODEL_TYPE,
        attn_num_layers: int = config.ATTN_NUM_LAYERS,
        ff_bias_flag: bool = config.FF_BIAS_FLAG,
        ff_model_type: LinearLayerStackOptions = config.FF_MODEL_TYPE,
        ff_num_layers: int = config.FF_NUM_LAYERS,
        output_bias_flag: bool = config.OUTPUT_BIAS_FLAG,
        image_patch_size: int = config.IMAGE_PATCH_SIZE,
        input_channels: int = config.INPUT_CHANNELS,
        image_height: int = config.IMAGE_HEIGHT,
    ) -> "ModelConfig":
        from emperor.config import ModelConfig

        class_token_length = 1
        padding_size = 0
        dilatation_size = 0
        stride = image_patch_size
        h_out = (
            image_height + 2 * padding_size - dilatation_size * (image_patch_size - 1) - 1
        ) // stride + 1
        w_out = (
            image_height + 2 * padding_size - dilatation_size * (image_patch_size - 1) - 1
        ) // stride + 1
        sequence_length = h_out * w_out + 1
        num_embeddings = sequence_length - class_token_length

        return ModelConfig(
            batch_size=batch_size,
            learning_rate=learning_rate,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            override_config=ExperimentConfig(
                patch_config=PatchConfig(
                    patch_option=PatchOptions.CONV,
                    embedding_dim=hidden_dim,
                    num_input_channels=input_channels,
                    patch_size=image_patch_size,
                    stride=image_patch_size,
                    padding=0,
                    dropout_probability=0.0,
                    override_config=LayerStackConfig(
                        model_type=LinearLayerOptions.BASE,
                        input_dim=hidden_dim,
                        hidden_dim=hidden_dim,
                        output_dim=hidden_dim,
                        num_layers=1,
                        activation=activation_function,
                        layer_norm_position=LayerNormPositionOptions.DEFAULT,
                        residual_flag=False,
                        adaptive_computation_flag=False,
                        dropout_probability=dropout_probability,
                        override_config=LinearLayerConfig(
                            input_dim=hidden_dim,
                            output_dim=hidden_dim,
                            bias_flag=True,
                            data_monitor=None,
                            parameter_monitor=None,
                        ),
                    ),
                ),
                positional_embedding_config=AbsolutePositionalEmbeddingConfig(
                    text_processing_flag=False,
                    positional_embedding_option=AbsolutePositionalEmbeddingOptions.LEARNED,
                    num_embeddings=num_embeddings,
                    embedding_dim=hidden_dim,
                    padding_idx=0,
                    init_size=1024,
                    auto_expand_flag=False,
                ),
                encoder_config=TransformerConfig(
                    num_layers=transformer_num_layers,
                    source_sequence_length=sequence_length,
                    target_sequence_length=sequence_length,
                    embedding_dim=hidden_dim,
                    layer_norm_position=LayerNormPositionOptions.DEFAULT,
                    dropout_probability=dropout_probability,
                    causal_attention_mask_flag=False,
                    attention_config=MultiHeadAttentionConfig(
                        batch_size=batch_size,
                        num_heads=attn_num_heads,
                        model_type=attn_model_type,
                        query_key_projection_dim=hidden_dim,
                        value_projection_dim=hidden_dim,
                        embedding_dim=hidden_dim,
                        target_sequence_length=sequence_length,
                        source_sequence_length=sequence_length,
                        target_dtype=torch.float32,
                        attention_option=True,
                        dropout_probability=dropout_probability,
                        key_value_bias_flag=False,
                        zero_attention_flag=False,
                        causal_attention_mask_flag=False,
                        add_key_value_bias_flag=False,
                        average_attention_weights_flag=False,
                        return_attention_weights_flag=False,
                        override_config=LayerStackConfig(
                            input_dim=hidden_dim,
                            hidden_dim=hidden_dim,
                            output_dim=hidden_dim,
                            num_layers=attn_num_layers,
                            activation=activation_function,
                            layer_norm_position=LayerNormPositionOptions.BEFORE,
                            residual_flag=False,
                            adaptive_computation_flag=False,
                            dropout_probability=dropout_probability,
                            override_config=LinearLayerConfig(
                                input_dim=hidden_dim,
                                output_dim=hidden_dim,
                                bias_flag=attn_bias_flag,
                                data_monitor=None,
                                parameter_monitor=None,
                            ),
                        ),
                    ),
                    feed_forward_config=FeedForwardConfig(
                        layer_stack_option=ff_model_type,
                        input_dim=hidden_dim,
                        output_dim=hidden_dim,
                        num_layers=ff_num_layers,
                        override_config=LayerStackConfig(
                            input_dim=hidden_dim,
                            hidden_dim=hidden_dim,
                            output_dim=hidden_dim,
                            num_layers=ff_num_layers,
                            activation=activation_function,
                            layer_norm_position=LayerNormPositionOptions.BEFORE,
                            residual_flag=False,
                            adaptive_computation_flag=False,
                            dropout_probability=dropout_probability,
                            override_config=LinearLayerConfig(
                                input_dim=hidden_dim,
                                output_dim=hidden_dim,
                                bias_flag=ff_bias_flag,
                                data_monitor=None,
                                parameter_monitor=None,
                            ),
                        ),
                    ),
                ),
                output_config=LayerStackConfig(
                    model_type=LinearLayerOptions.BASE,
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_layers=output_num_layers,
                    activation=activation_function,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_flag=False,
                    adaptive_computation_flag=False,
                    dropout_probability=dropout_probability,
                    override_config=LinearLayerConfig(
                        input_dim=hidden_dim,
                        output_dim=output_dim,
                        bias_flag=output_bias_flag,
                        data_monitor=None,
                        parameter_monitor=None,
                    ),
                ),
            ),
        )

    def __adaptive_preset(
        self,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        output_dim: int = config.OUTPUT_DIM,
        dropout_probability: float = config.DROPOUT_PROBABILITY,
        activation_function: ActivationOptions = config.ACTIVATION_FUNCTION,
        transformer_num_layers: int = config.TRANSFORMER_NUM_LAYERS,
        attn_bias_flag: bool = config.ADAPTIVE_ATTN_BIAS_FLAG,
        attn_num_layers: int = config.ATTN_NUM_LAYERS,
        attn_generator_depth: DynamicDepthOptions = config.ADAPTIVE_ATTN_GENERATOR_DEPTH,
        attn_diagonal_option: DynamicDiagonalOptions = config.ADAPTIVE_ATTN_DIAGONAL_OPTION,
        attn_bias_option: DynamicBiasOptions = config.ADAPTIVE_ATTN_BIAS_OPTION,
        attn_behaviour_stack_num_layers: int = config.ADAPTIVE_ATTN_BEHAVIOUR_STACK_NUM_LAYERS,
        ff_bias_flag: bool = config.FF_BIAS_FLAG,
        ff_num_layers: int = config.FF_NUM_LAYERS,
        ff_generator_depth: DynamicDepthOptions = config.ADAPTIVE_FF_GENERATOR_DEPTH,
        ff_diagonal_option: DynamicDiagonalOptions = config.ADAPTIVE_FF_DIAGONAL_OPTION,
        ff_bias_option: DynamicBiasOptions = config.ADAPTIVE_FF_BIAS_OPTION,
        ff_behaviour_stack_num_layers: int = config.ADAPTIVE_FF_BEHAVIOUR_STACK_NUM_LAYERS,
        output_bias_flag: bool = config.OUTPUT_BIAS_FLAG,
        output_num_layers: int = config.ADAPTIVE_OUTPUT_NUM_LAYERS,
        output_generator_depth: DynamicDepthOptions = config.ADAPTIVE_OUTPUT_GENERATOR_DEPTH,
        output_diagonal_option: DynamicDiagonalOptions = config.ADAPTIVE_OUTPUT_DIAGONAL_OPTION,
        output_bias_option: DynamicBiasOptions = config.ADAPTIVE_OUTPUT_BIAS_OPTION,
        output_behaviour_stack_num_layers: int = config.ADAPTIVE_OUTPUT_BEHAVIOUR_STACK_NUM_LAYERS,
        image_patch_size: int = config.IMAGE_PATCH_SIZE,
        input_channels: int = config.INPUT_CHANNELS,
        image_height: int = config.IMAGE_HEIGHT,
    ) -> "ModelConfig":
        from emperor.config import ModelConfig
        from emperor.linears.core.presets import LinearPresets

        class_token_length = 1
        padding_size = 0
        dilatation_size = 0
        stride = image_patch_size
        h_out = (
            image_height + 2 * padding_size - dilatation_size * (image_patch_size - 1) - 1
        ) // stride + 1
        w_out = (
            image_height + 2 * padding_size - dilatation_size * (image_patch_size - 1) - 1
        ) // stride + 1
        sequence_length = h_out * w_out + 1
        num_embeddings = sequence_length - class_token_length

        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            override_config=ExperimentConfig(
                patch_config=PatchConfig(
                    patch_option=PatchOptions.CONV,
                    num_input_channels=input_channels,
                    embedding_dim=hidden_dim,
                    patch_size=image_patch_size,
                    stride=image_patch_size,
                    padding=0,
                    dropout_probability=0.0,
                    override_config=LayerStackConfig(
                        model_type=LinearLayerOptions.BASE,
                        input_dim=hidden_dim,
                        hidden_dim=hidden_dim,
                        output_dim=hidden_dim,
                        num_layers=1,
                        activation=activation_function,
                        layer_norm_position=LayerNormPositionOptions.DEFAULT,
                        residual_flag=False,
                        adaptive_computation_flag=False,
                        dropout_probability=dropout_probability,
                        override_config=LinearLayerConfig(
                            input_dim=hidden_dim,
                            output_dim=hidden_dim,
                            bias_flag=True,
                            data_monitor=None,
                            parameter_monitor=None,
                        ),
                    ),
                ),
                positional_embedding_config=AbsolutePositionalEmbeddingConfig(
                    text_processing_flag=False,
                    positional_embedding_option=AbsolutePositionalEmbeddingOptions.LEARNED,
                    num_embeddings=num_embeddings,
                    embedding_dim=hidden_dim,
                    padding_idx=0,
                    init_size=1024,
                    auto_expand_flag=False,
                ),
                encoder_config=TransformerPresets.transformer_linear_adaptive_preset(
                    batch_size=batch_size,
                    num_layers=transformer_num_layers,
                    layer_norm_position=LayerNormPositionOptions.DEFAULT,
                    num_heads=4,
                    embedding_dim=hidden_dim,
                    query_key_projection_dim=hidden_dim,
                    value_projection_dim=hidden_dim,
                    target_sequence_length=sequence_length,
                    source_sequence_length=sequence_length,
                    target_dtype=torch.float32,
                    attention_option=False,
                    dropout_probability=dropout_probability,
                    key_value_bias_flag=False,
                    zero_attention_flag=False,
                    causal_attention_mask_flag=False,
                    add_key_value_bias_flag=False,
                    average_attention_weights_flag=False,
                    return_attention_weights_flag=False,
                    attn_stack_num_layers=attn_num_layers,
                    attn_bias_flag=attn_bias_flag,
                    attn_generator_depth=attn_generator_depth,
                    attn_diagonal_option=attn_diagonal_option,
                    attn_bias_option=attn_bias_option,
                    attn_behaviour_stack_num_layers=attn_behaviour_stack_num_layers,
                    ff_stack_num_layers=ff_num_layers,
                    ff_bias_flag=ff_bias_flag,
                    ff_generator_depth=ff_generator_depth,
                    ff_diagonal_option=ff_diagonal_option,
                    ff_bias_option=ff_bias_option,
                    ff_behaviour_stack_num_layers=ff_behaviour_stack_num_layers,
                    stack_activation=activation_function,
                    stack_residual_flag=False,
                ),
                output_config=LinearPresets.adaptive_linear_layer_stack_preset(
                    batch_size=batch_size,
                    input_dim=hidden_dim,
                    output_dim=output_dim,
                    bias_flag=output_bias_flag,
                    generator_depth=output_generator_depth,
                    diagonal_option=output_diagonal_option,
                    bias_option=output_bias_option,
                    stack_num_layers=output_num_layers,
                    stack_hidden_dim=hidden_dim,
                    stack_activation=activation_function,
                    stack_residual_flag=False,
                    stack_dropout_probability=dropout_probability,
                    adaptive_behaviour_stack_num_layers=output_behaviour_stack_num_layers,
                ),
            ),
        )


class Experiment(ExperimentBase):
    def __init__(
        self,
        experiment_option: ExperimentOptions | None = None,
    ) -> None:
        super().__init__(experiment_option)

    def _num_epochs(self) -> int:
        return config.NUM_EPOCHS

    def _dataset_options(self) -> list:
        return config.DATASET_OPTIONS

    def _model_type(self) -> type:
        return Model

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        return ExperimentPresets()

    def _experiment_enumeration(self) -> type[BaseOptions]:
        return ExperimentOptions
