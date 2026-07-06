from emperor.base.options import BaseOptions, LayerNormPositionOptions
from emperor.datasets.text.bert_pretraining import PennTreebankBertPretraining
from emperor.embedding.absolute.core.config import (
    TextSinusoidalPositionalEmbeddingConfig,
)
from emperor.experiments.base import (
    BuilderBackedExperimentPresetsBase,
    ExperimentBase,
    ExperimentPresetsBase,
    PresetDefinition,
)

import models.transformer_encoder.bert_linear.config as config
from models.transformer_encoder.bert_linear.config_builder import (
    BertLinearConfigBuilder,
)
from models.transformer_encoder.bert_linear.model import Model


class ExperimentPreset(BaseOptions):
    BASELINE = 1
    PRE_NORM = 2
    POST_NORM = 3
    SINUSOIDAL = 4
    CAUSAL = 5
    ATTENTION_BIAS = 6


_PRESET_DEFINITIONS = {
    ExperimentPreset.BASELINE: PresetDefinition(
        preset_values={},
        description="Default config: a BERT-style pretraining encoder with linear "
        "attention and feed-forward sub-stacks, learned positional "
        "embeddings, and bidirectional attention.",
    ),
    ExperimentPreset.PRE_NORM: PresetDefinition(
        preset_values={
            "layer_norm_position": LayerNormPositionOptions.BEFORE,
        },
        description="Default config with layer normalization applied before each encoder "
        "sub-block.",
    ),
    ExperimentPreset.POST_NORM: PresetDefinition(
        preset_values={
            "layer_norm_position": LayerNormPositionOptions.AFTER,
        },
        description="Default config with layer normalization applied after each encoder "
        "sub-block.",
    ),
    ExperimentPreset.SINUSOIDAL: PresetDefinition(
        preset_values={
            "positional_embedding_option": TextSinusoidalPositionalEmbeddingConfig,
        },
        description="Default config with fixed sinusoidal positional embeddings.",
    ),
    ExperimentPreset.CAUSAL: PresetDefinition(
        preset_values={
            "causal_attention_mask_flag": True,
        },
        description="Default config with causal attention masking enabled for "
        "autoregressive modeling.",
    ),
    ExperimentPreset.ATTENTION_BIAS: PresetDefinition(
        preset_values={
            "attn_bias_flag": True,
            "attn_add_key_value_bias_flag": True,
        },
        description="Default config with attention projection bias and key/value bias "
        "enabled.",
    ),
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=BertLinearConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
            default_dataset=PennTreebankBertPretraining,
        )

    def _dataset_config(self, dataset: type) -> dict:
        return {
            **super()._dataset_config(dataset),
            "sequence_length": dataset.sequence_length,
        }

class Experiment(ExperimentBase):
    def __init__(
        self,
        experiment_preset: ExperimentPreset | None = None,
    ) -> None:
        super().__init__(experiment_preset)

    def _num_epochs(self) -> int:
        return config.NUM_EPOCHS

    def _dataset_options(self) -> list:
        return config.DATASET_OPTIONS

    def _model_type(self) -> type:
        return Model

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        return ExperimentPresets()

    def _experiment_preset_enum(self) -> type[BaseOptions]:
        return ExperimentPreset
