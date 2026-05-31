import torch

import models.bert.config as config

from emperor.attention.self_attention.config import SelfAttentionConfig
from emperor.base.layer import LayerConfig, LayerStackConfig
from emperor.base.options import (
    ActivationOptions,
    BaseOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.config import ModelConfig
from emperor.datasets.text.language_modeling.penn_treebank import PennTreebank
from emperor.embedding.absolute.core.config import TextLearnedPositionalEmbeddingConfig
from emperor.experiments.base import ExperimentBase, ExperimentPresetsBase, SearchMode
from emperor.linears.core.config import LinearLayerConfig
from emperor.transformer.core.config import (
    TransformerEncoderLayerConfig,
    TransformerEncoderStackConfig,
)
from emperor.transformer.feed_forward import FeedForwardConfig
from models.bert.config import ExperimentConfig
from models.bert.model import Model

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig as ModelConfigType


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
        search_keys: list[str] | None = None,
        config_overrides: dict | None = None,
        search_overrides: dict | None = None,
    ) -> list["ModelConfigType"]:
        match model_config_options:
            case ExperimentOptions.PRESET:
                return self._create_default_preset_configs(
                    dataset,
                    config_overrides=config_overrides,
                    search_overrides=search_overrides,
                )
            case ExperimentOptions.CONFIG:
                return self._create_preset_search_space_configs(
                    dataset,
                    search_mode,
                    search_keys=search_keys,
                    config_overrides=config_overrides,
                    search_overrides=search_overrides,
                )
            case _:
                raise ValueError(
                    "The specified option is not supported. Please choose a valid "
                    "`ExperimentOptions`."
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
        attn_num_layers: int = config.ATTN_NUM_LAYERS,
        ff_bias_flag: bool = config.FF_BIAS_FLAG,
        ff_num_layers: int = config.FF_NUM_LAYERS,
        output_bias_flag: bool = config.OUTPUT_BIAS_FLAG,
    ) -> "ModelConfigType":
        return ModelConfig(
            batch_size=batch_size,
            learning_rate=learning_rate,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            sequence_length=sequence_length,
            experiment_config=ExperimentConfig(
                positional_embedding_config=TextLearnedPositionalEmbeddingConfig(
                    num_embeddings=sequence_length,
                    embedding_dim=hidden_dim,
                    padding_idx=0,
                    init_size=sequence_length,
                    auto_expand_flag=False,
                ),
                encoder_config=self._encoder_config(
                    batch_size=batch_size,
                    hidden_dim=hidden_dim,
                    sequence_length=sequence_length,
                    num_layers=transformer_num_layers,
                    dropout_probability=dropout_probability,
                    activation_function=activation_function,
                    attn_num_heads=attn_num_heads,
                    attn_num_layers=attn_num_layers,
                    attn_bias_flag=attn_bias_flag,
                    ff_num_layers=ff_num_layers,
                    ff_bias_flag=ff_bias_flag,
                ),
                output_config=self._linear_stack_config(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_layers=output_num_layers,
                    activation=activation_function,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_flag=False,
                    dropout_probability=dropout_probability,
                    bias_flag=output_bias_flag,
                    apply_output_pipeline_flag=False,
                ),
            ),
        )

    def _encoder_config(
        self,
        *,
        batch_size: int,
        hidden_dim: int,
        sequence_length: int,
        num_layers: int,
        dropout_probability: float,
        activation_function: ActivationOptions,
        attn_num_heads: int,
        attn_num_layers: int,
        attn_bias_flag: bool,
        ff_num_layers: int,
        ff_bias_flag: bool,
    ) -> TransformerEncoderStackConfig:
        layer_config = TransformerEncoderLayerConfig(
            embedding_dim=hidden_dim,
            layer_norm_position=LayerNormPositionOptions.DEFAULT,
            dropout_probability=dropout_probability,
            causal_attention_mask_flag=False,
            attention_config=SelfAttentionConfig(
                batch_size=batch_size,
                num_heads=attn_num_heads,
                embedding_dim=hidden_dim,
                query_key_projection_dim=hidden_dim,
                value_projection_dim=hidden_dim,
                target_sequence_length=sequence_length,
                source_sequence_length=sequence_length,
                target_dtype=torch.float32,
                dropout_probability=dropout_probability,
                zero_attention_flag=False,
                causal_attention_mask_flag=False,
                add_key_value_bias_flag=False,
                average_attention_weights_flag=False,
                return_attention_weights_flag=False,
                projection_model_config=self._linear_stack_config(
                    hidden_dim=hidden_dim,
                    num_layers=attn_num_layers,
                    activation=activation_function,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_flag=False,
                    dropout_probability=0.0,
                    bias_flag=attn_bias_flag,
                ),
            ),
            feed_forward_config=FeedForwardConfig(
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                stack_config=self._linear_stack_config(
                    hidden_dim=hidden_dim,
                    num_layers=ff_num_layers,
                    activation=activation_function,
                    layer_norm_position=LayerNormPositionOptions.BEFORE,
                    residual_flag=False,
                    dropout_probability=dropout_probability,
                    bias_flag=ff_bias_flag,
                ),
            ),
        )
        return TransformerEncoderStackConfig(
            num_layers=num_layers,
            embedding_dim=hidden_dim,
            source_sequence_length=sequence_length,
            target_sequence_length=sequence_length,
            causal_attention_mask_flag=False,
            layer_config=layer_config,
        )

    def _linear_stack_config(
        self,
        *,
        hidden_dim: int,
        num_layers: int,
        activation: ActivationOptions,
        layer_norm_position: LayerNormPositionOptions,
        residual_flag: bool,
        dropout_probability: float,
        bias_flag: bool,
        input_dim: int | None = None,
        output_dim: int | None = None,
        apply_output_pipeline_flag: bool = True,
    ) -> LayerStackConfig:
        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=apply_output_pipeline_flag,
            layer_config=LayerConfig(
                activation=activation,
                residual_flag=residual_flag,
                dropout_probability=dropout_probability,
                layer_norm_position=layer_norm_position,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                shared_halting_flag=False,
                layer_model_config=LinearLayerConfig(
                    bias_flag=bias_flag,
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
