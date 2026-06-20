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


class ExperimentPreset(BaseOptions):
    BASELINE = (
        "Vision Transformer classifier with linear patch embeddings, a trainable "
        "class token, learned image positional embeddings, and a pre-norm "
        "bidirectional encoder."
    )
    POST_NORM = "Vision Transformer classifier with post-norm encoder sub-blocks."
    SINUSOIDAL = "Vision Transformer classifier with fixed sinusoidal image positions."
    ATTENTION_BIAS = "Vision Transformer classifier with biased attention projections."


def _lock(preset, value, behavior: str) -> PresetLock:
    return PresetLock(
        value=value,
        reason=(
            f"Locked by the {preset.name} preset because this preset enables "
            f"{behavior}."
        ),
    )


def _preset_locks(
    preset_overrides: dict[ExperimentPreset, dict[str, object]],
) -> dict[ExperimentPreset, dict[str, PresetLock]]:
    return {
        preset: {
            field: _lock(preset, value, _PRESET_LOCK_BEHAVIORS[field])
            for field, value in overrides.items()
        }
        for preset, overrides in preset_overrides.items()
        if overrides
    }


_PRESET_LOCK_BEHAVIORS = {
    "layer_norm_position": "post-layer normalization",
    "positional_embedding_option": "fixed sinusoidal image positions",
    "attn_bias_flag": "attention projection bias",
    "attn_add_key_value_bias_flag": "attention key/value bias",
}

_PRESET_OVERRIDES = {
    ExperimentPreset.BASELINE: {},
    ExperimentPreset.POST_NORM: {
        "layer_norm_position": LayerNormPositionOptions.AFTER,
    },
    ExperimentPreset.SINUSOIDAL: {
        "positional_embedding_option": ImageSinusoidalPositionalEmbeddingConfig,
    },
    ExperimentPreset.ATTENTION_BIAS: {
        "attn_bias_flag": True,
        "attn_add_key_value_bias_flag": True,
    },
}


class ExperimentPresets(ExperimentPresetsBase):
    PRESET_OVERRIDES = _PRESET_OVERRIDES
    PRESET_LOCKS = _preset_locks(PRESET_OVERRIDES)

    def get_config(
        self,
        model_config_preset: ExperimentPreset = ExperimentPreset.BASELINE,
        dataset: type = Mnist,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
        search_keys: list[str] | None = None,
        config_overrides: dict | None = None,
        search_overrides: dict | None = None,
    ) -> list["ModelConfig"]:
        preset_callback = self._preset_callback_for_preset(model_config_preset)
        return self._create_preset_search_space_configs(
            dataset,
            search_mode,
            preset_callback,
            search_keys,
            config_overrides=config_overrides,
            search_overrides=search_overrides,
            model_config_preset=model_config_preset,
        )

    def _dataset_config(self, dataset: type) -> dict:
        return {
            **super()._dataset_config(dataset),
            "input_channels": dataset.num_channels,
            "image_height": dataset.default_height,
            "output_dim": dataset.num_classes,
        }

    def _preset_callback_for_preset(self, preset: ExperimentPreset):
        if preset not in self.PRESET_OVERRIDES:
            raise ValueError(
                "The specified preset is not supported. Please choose a valid "
                "`ExperimentPreset`."
            )
        return lambda **kwargs: self._preset_for_preset(preset, **kwargs)

    def _preset_for_preset(
        self,
        preset: ExperimentPreset,
        **kwargs,
    ) -> "ModelConfig":
        preset_overrides = self.PRESET_OVERRIDES[preset]
        return self._preset(**{**kwargs, **preset_overrides})

    def _preset(self, **kwargs) -> "ModelConfig":
        return VitLinearConfigBuilder(**kwargs).build()


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
