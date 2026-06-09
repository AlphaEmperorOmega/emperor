import models.transformer_encoder.bert_linear.config as config

from models.transformer_encoder.bert_linear.config_builder import BertLinearConfigBuilder
from models.transformer_encoder.bert_linear.model import Model
from emperor.experiments.base import SearchMode
from emperor.datasets.text.bert_pretraining import PennTreebankBertPretraining
from emperor.experiments.base import ExperimentBase, ExperimentPresetsBase, PresetLock
from emperor.base.options import BaseOptions, LayerNormPositionOptions
from emperor.embedding.absolute.core.config import (
    TextSinusoidalPositionalEmbeddingConfig,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExperimentOptions(BaseOptions):
    BASELINE = (
        "Transformer encoder language model with linear attention and feed-forward "
        "sub-stacks, learned positional embeddings, and bidirectional attention."
    )
    PRE_NORM = (
        "Transformer encoder with layer normalization applied before each sub-block."
    )
    POST_NORM = (
        "Transformer encoder with layer normalization applied after each sub-block."
    )
    SINUSOIDAL = (
        "Transformer encoder with fixed sinusoidal positional embeddings."
    )
    CAUSAL = (
        "Transformer encoder with a causal attention mask for autoregressive modeling."
    )
    ATTENTION_BIAS = (
        "Transformer encoder with biased attention projections and key/value bias."
    )


def _lock(option, value, behavior: str) -> PresetLock:
    return PresetLock(
        value=value,
        reason=(
            f"Locked by the {option.name} preset because this preset enables "
            f"{behavior}."
        ),
    )


class ExperimentPresets(ExperimentPresetsBase):
    PRESET_LOCKS = {
        ExperimentOptions.PRE_NORM: {
            "layer_norm_position": _lock(
                ExperimentOptions.PRE_NORM,
                LayerNormPositionOptions.BEFORE,
                "pre-layer normalization",
            ),
        },
        ExperimentOptions.POST_NORM: {
            "layer_norm_position": _lock(
                ExperimentOptions.POST_NORM,
                LayerNormPositionOptions.AFTER,
                "post-layer normalization",
            ),
        },
        ExperimentOptions.SINUSOIDAL: {
            "positional_embedding_option": _lock(
                ExperimentOptions.SINUSOIDAL,
                TextSinusoidalPositionalEmbeddingConfig,
                "fixed sinusoidal positional embeddings",
            ),
        },
        ExperimentOptions.CAUSAL: {
            "causal_attention_mask_flag": _lock(
                ExperimentOptions.CAUSAL,
                True,
                "causal attention masking",
            ),
        },
        ExperimentOptions.ATTENTION_BIAS: {
            "attn_bias_flag": _lock(
                ExperimentOptions.ATTENTION_BIAS,
                True,
                "attention projection bias",
            ),
            "attn_add_key_value_bias_flag": _lock(
                ExperimentOptions.ATTENTION_BIAS,
                True,
                "attention key/value bias",
            ),
        },
    }

    def __init__(self) -> None:
        super().__init__()

    def get_config(
        self,
        model_config_options: ExperimentOptions = ExperimentOptions.BASELINE,
        dataset: type = PennTreebankBertPretraining,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
        search_keys: list[str] | None = None,
        config_overrides: dict | None = None,
        search_overrides: dict | None = None,
    ) -> list["ModelConfig"]:
        preset_callback = self._preset_callback_for_option(model_config_options)
        return self._create_preset_search_space_configs(
            dataset,
            search_mode,
            preset_callback,
            search_keys,
            config_overrides=config_overrides,
            search_overrides=search_overrides,
        )

    def _dataset_config(self, dataset: type) -> dict:
        return {
            **super()._dataset_config(dataset),
            "sequence_length": dataset.sequence_length,
        }

    def _preset_callback_for_option(self, option: ExperimentOptions):
        callbacks = {
            ExperimentOptions.BASELINE: self._baseline_preset,
            ExperimentOptions.PRE_NORM: self._pre_norm_preset,
            ExperimentOptions.POST_NORM: self._post_norm_preset,
            ExperimentOptions.SINUSOIDAL: self._sinusoidal_preset,
            ExperimentOptions.CAUSAL: self._causal_preset,
            ExperimentOptions.ATTENTION_BIAS: self._attention_bias_preset,
        }
        if option not in callbacks:
            raise ValueError(
                "The specified option is not supported. Please choose a valid `ExperimentOptions`."
            )
        return callbacks[option]

    def _baseline_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**kwargs)

    def _pre_norm_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{"layer_norm_position": LayerNormPositionOptions.BEFORE, **kwargs}
        )

    def _post_norm_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{"layer_norm_position": LayerNormPositionOptions.AFTER, **kwargs}
        )

    def _sinusoidal_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "positional_embedding_option": TextSinusoidalPositionalEmbeddingConfig,
                **kwargs,
            }
        )

    def _causal_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"causal_attention_mask_flag": True, **kwargs})

    def _attention_bias_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "attn_bias_flag": True,
                "attn_add_key_value_bias_flag": True,
                **kwargs,
            }
        )

    def _preset(self, **kwargs) -> "ModelConfig":
        return BertLinearConfigBuilder(**kwargs).build()


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
