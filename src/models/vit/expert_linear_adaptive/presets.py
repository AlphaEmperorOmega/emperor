import models.vit.expert_linear_adaptive.config as config
import models.vit.expert_linear_adaptive.dataset_options as dataset_options
from emperor.config import BaseOptions
from emperor.embedding.absolute import (
    ImageSinusoidalPositionalEmbeddingConfig,
)
from emperor.layers import LayerNormPositionOptions
from model_runtime.packages import (
    BuilderBackedExperimentPresetsBase,
    ExperimentPresetsBase,
    PresetDefinition,
)
from model_runtime.runs import ExperimentBase
from models.vit.expert_linear_adaptive._builder_adapter import (
    expert_linear_adaptive_builder_kwargs_from_flat,
)
from models.vit.expert_linear_adaptive.config_builder import (
    VitExpertLinearAdaptiveConfigBuilder,
)
from models.vit.expert_linear_adaptive.model import Model


def default_patch_size_for_dataset(dataset: type) -> int:
    image_height = dataset.default_height
    if (
        image_height >= config.IMAGE_HEIGHT
        and image_height % config.IMAGE_PATCH_SIZE == 0
    ):
        return config.IMAGE_PATCH_SIZE
    for patch_size in (4, 2, 1):
        if image_height % patch_size == 0:
            return patch_size
    return 1


class ExperimentPreset(BaseOptions):
    BASELINE = 1
    POST_NORM = 2
    SINUSOIDAL = 3
    ATTENTION_BIAS = 4
    TOP1_SWITCH_AUX = 5
    LOW_RANK_EXPERT_WEIGHT = 6


_PRESET_DEFINITIONS = {
    ExperimentPreset.BASELINE: PresetDefinition(
        preset_values={},
        description=(
            "Default config: a Vision Transformer classifier with adaptive expert "
            "linear patch embeddings, a trainable class token, learned image "
            "positions, an adaptive Mixture of Attention Heads, and a pre-norm "
            "bidirectional encoder."
        ),
    ),
    ExperimentPreset.POST_NORM: PresetDefinition(
        preset_values={
            "layer_norm_position": LayerNormPositionOptions.AFTER,
        },
        description=(
            "Default config with layer normalization after each encoder sub-block."
        ),
    ),
    ExperimentPreset.SINUSOIDAL: PresetDefinition(
        preset_values={
            "positional_embedding_option": ImageSinusoidalPositionalEmbeddingConfig,
        },
        description=(
            "Default config with fixed sinusoidal image positional embeddings."
        ),
    ),
    ExperimentPreset.ATTENTION_BIAS: PresetDefinition(
        preset_values={
            "attn_bias_flag": True,
            "attn_add_key_value_bias_flag": True,
        },
        description=(
            "Default config with attention projection bias and key/value bias enabled."
        ),
    ),
    ExperimentPreset.TOP1_SWITCH_AUX: PresetDefinition(
        preset_values={
            "top_k": 1,
            "sampler_normalize_probabilities_flag": False,
            "sampler_switch_loss_weight": 0.1,
        },
        description=(
            "Default config with top-1 expert routing and switch auxiliary loss."
        ),
    ),
}
_PRESET_DEFINITIONS[ExperimentPreset.LOW_RANK_EXPERT_WEIGHT] = PresetDefinition(
    preset_values={
        "weight_option_flag": True,
        "weight_option": config.LowRankDynamicWeightConfig,
    },
    description="Default config with adaptive low-rank dynamic weights inside expert "
    "feed-forward internals.",
)


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=VitExpertLinearAdaptiveConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
        )

    def _dataset_config(self, dataset: type) -> dict:
        return {
            **super()._dataset_config(dataset),
            "image_patch_size": default_patch_size_for_dataset(dataset),
            "input_channels": dataset.num_channels,
            "image_height": dataset.default_height,
            "output_dim": dataset.num_classes,
        }

    def _preset(self, **kwargs):
        builder_kwargs = expert_linear_adaptive_builder_kwargs_from_flat(
            kwargs,
            config,
        )
        return self._builder_type(**builder_kwargs).build()


class Experiment(ExperimentBase):
    def __init__(
        self,
        experiment_preset: ExperimentPreset | None = None,
        experiment_task=None,
        *,
        model_package=None,
        run_artifacts=None,
    ) -> None:
        super().__init__(
            experiment_preset,
            experiment_task=experiment_task,
            model_package=model_package,
            run_artifacts=run_artifacts,
        )

    def _num_epochs(self) -> int:
        return config.NUM_EPOCHS

    def _dataset_options(self) -> list:
        return dataset_options.DATASET_OPTIONS_BY_TASK[
            dataset_options.DEFAULT_EXPERIMENT_TASK
        ]

    def _model_type(self) -> type:
        return Model

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        return ExperimentPresets()

    def _experiment_preset_enum(self) -> type[BaseOptions]:
        return ExperimentPreset
