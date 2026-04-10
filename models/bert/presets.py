import torch

from emperor.base.enums import BaseOptions, ActivationOptions, LayerNormPositionOptions
from emperor.datasets.text.language_modeling.penn_treebank import PennTreebank
from emperor.datasets.text.language_modeling.wiki_text_2 import WikiText2
from emperor.linears.core.config import LinearLayerConfig
from emperor.base.layer import LayerStackConfig
from emperor.transformer.utils.layers import TransformerConfig
from emperor.attention.utils.layer import MultiHeadAttentionConfig
from emperor.transformer.utils.feed_forward import FeedForwardConfig
from emperor.linears.options import LinearLayerOptions, LinearLayerStackOptions
from emperor.embedding.options import AbsolutePositionalEmbeddingOptions
from emperor.embedding.absolute.config import AbsolutePositionalEmbeddingConfig
from emperor.experiments.base import ExperimentBase, ExperimentPresetsBase
import models.bert.config as config
from models.bert.config import ExperimentConfig
from models.bert.model import Model
from emperor.experiments.base import SearchMode

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExperimentOptions(BaseOptions):
    PRESET = 0
    CONFIG = 1


class ExperimentPresets(ExperimentPresetsBase):
    def get_config(
        self,
        model_config_options: ExperimentOptions = ExperimentOptions.PRESET,
        dataset: type = PennTreebank,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
    ) -> list["ModelConfig"]:
        match model_config_options:
            case ExperimentOptions.PRESET:
                return self._create_default_preset_configs(dataset)
            case ExperimentOptions.CONFIG:
                return self._create_default_search_space_configs(dataset, search_mode, log_folder)
            case _:
                raise ValueError(
                    "The specified option is not supported. Please choose a valid `ExperimentOptions`."
                )

    def _dataset_config(self, dataset: type) -> dict:
        return {
            **super()._dataset_config(dataset),
            "sequence_length": dataset.sequence_length,
        }

    def _preset(
        self,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = PennTreebank.flattened_input_dim,
        hidden_dim: int = config.HIDDEN_DIM,
        output_dim: int = PennTreebank.num_classes,
        sequence_length: int = config.SEQUENCE_LENGTH,
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
    ) -> "ModelConfig":
        from emperor.config import ModelConfig

        return ModelConfig(
            batch_size=batch_size,
            learning_rate=learning_rate,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            override_config=ExperimentConfig(
                positional_embedding_config=AbsolutePositionalEmbeddingConfig(
                    text_processing_flag=True,
                    positional_embedding_option=AbsolutePositionalEmbeddingOptions.LEARNED,
                    num_embeddings=sequence_length,
                    embedding_dim=hidden_dim,
                    padding_idx=0,
                    init_size=sequence_length,
                    auto_expand_flag=False,
                    class_token_flag=False,
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
