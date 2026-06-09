import models.transformer_encoder.vit_linear.config as config

from emperor.base.options import BaseOptions, LayerNormPositionOptions
from emperor.datasets.image.classification.mnist import Mnist
from emperor.embedding.absolute.core.config import (
    ImageSinusoidalPositionalEmbeddingConfig,
)
from emperor.experiments.base import (
    ExperimentBase,
    ExperimentPresetsBase,
    PresetLock,
    SearchMode,
)
from models.transformer_encoder.vit_linear.config_builder import VitLinearConfigBuilder
from models.transformer_encoder.vit_linear.model import Model

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExperimentOptions(BaseOptions):
    BASELINE = (
        "Vision Transformer classifier with linear patch embeddings, a trainable "
        "class token, learned image positional embeddings, and a pre-norm "
        "bidirectional encoder."
    )
    POST_NORM = "Vision Transformer classifier with post-norm encoder sub-blocks."
    SINUSOIDAL = "Vision Transformer classifier with fixed sinusoidal image positions."
    ATTENTION_BIAS = "Vision Transformer classifier with biased attention projections."


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
                ImageSinusoidalPositionalEmbeddingConfig,
                "fixed sinusoidal image positions",
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

    def get_config(
        self,
        model_config_options: ExperimentOptions = ExperimentOptions.BASELINE,
        dataset: type = Mnist,
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
            "input_channels": dataset.num_channels,
            "image_height": dataset.default_height,
            "output_dim": dataset.num_classes,
        }

    def _preset_callback_for_option(self, option: ExperimentOptions):
        callbacks = {
            ExperimentOptions.BASELINE: self._baseline_preset,
            ExperimentOptions.POST_NORM: self._post_norm_preset,
            ExperimentOptions.SINUSOIDAL: self._sinusoidal_preset,
            ExperimentOptions.ATTENTION_BIAS: self._attention_bias_preset,
        }
        if option not in callbacks:
            raise ValueError(
                "The specified option is not supported. Please choose a valid "
                "`ExperimentOptions`."
            )
        return callbacks[option]

    def _baseline_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**kwargs)

    def _post_norm_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{"layer_norm_position": LayerNormPositionOptions.AFTER, **kwargs}
        )

    def _sinusoidal_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "positional_embedding_option": ImageSinusoidalPositionalEmbeddingConfig,
                **kwargs,
            }
        )

    def _attention_bias_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "attn_bias_flag": True,
                "attn_add_key_value_bias_flag": True,
                **kwargs,
            }
        )

    def _preset(self, **kwargs) -> "ModelConfig":
        return VitLinearConfigBuilder(**kwargs).build()


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
