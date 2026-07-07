from emperor.base.options import BaseOptions
from emperor.experiments.base import (
    BuilderBackedExperimentPresetsBase,
    ExperimentBase,
    ExperimentPresetsBase,
    PresetDefinition,
)

import models.linears.linear_adaptive.config as adaptive_defaults
import models.vit.linear_adaptive.config as config
from models.linears._builder_adapter import (
    linear_adaptive_builder_kwargs_from_flat,
)
from models.vit._builder_adapter import linear_builder_kwargs_from_flat
from models.vit.linear.presets import (
    _PRESET_DEFINITIONS as _LINEAR_PRESET_DEFINITIONS,
)
from models.vit.linear_adaptive.config_builder import VitLinearAdaptiveConfigBuilder
from models.vit.linear_adaptive.model import Model

import models.vit.linear_adaptive.dataset_options as dataset_options
_BUILDER_KEYS = {
    "batch_size",
    "learning_rate",
    "input_dim",
    "output_dim",
    "patch_options",
    "encoder_options",
    "positional_embedding_options",
    "attention_options",
    "feed_forward_options",
    "output_options",
}


class ExperimentPreset(BaseOptions):
    BASELINE = 1
    POST_NORM = 2
    SINUSOIDAL = 3
    ATTENTION_BIAS = 4
    LOW_RANK_WEIGHT = 5


_PRESET_DEFINITIONS = {
    preset: PresetDefinition(
        preset_values=dict(definition.preset_values),
        description=definition.description.replace("linear", "adaptive linear"),
    )
    for preset, definition in zip(
        ExperimentPreset,
        _LINEAR_PRESET_DEFINITIONS.values(),
        strict=False,
    )
}
_PRESET_DEFINITIONS[ExperimentPreset.LOW_RANK_WEIGHT] = PresetDefinition(
    preset_values={
        "weight_option_flag": True,
        "weight_option": config.LowRankDynamicWeightConfig,
    },
    description="Default config with adaptive low-rank dynamic weights on eligible "
    "encoder attention and feed-forward linear stacks.",
)


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=VitLinearAdaptiveConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
        )

    def _dataset_config(self, dataset: type) -> dict:
        return {
            **super()._dataset_config(dataset),
            "input_channels": dataset.num_channels,
            "image_height": dataset.default_height,
            "output_dim": dataset.num_classes,
        }

    def _preset(self, **kwargs):
        builder_kwargs = linear_builder_kwargs_from_flat(kwargs, config)
        builder_kwargs = {
            key: value for key, value in builder_kwargs.items() if key in _BUILDER_KEYS
        }
        adaptive_kwargs = linear_adaptive_builder_kwargs_from_flat(
            kwargs,
            adaptive_defaults,
        )
        for key in (
            "adaptive_generator_stack_options",
            "hidden_adaptive_weight_options",
            "hidden_adaptive_bias_options",
            "hidden_adaptive_diagonal_options",
            "hidden_adaptive_mask_options",
        ):
            builder_kwargs[key] = adaptive_kwargs[key]
        return self._builder_type(**builder_kwargs).build()


class Experiment(ExperimentBase):
    def __init__(
        self,
        experiment_preset: ExperimentPreset | None = None,
        experiment_task=None,
    ) -> None:
        super().__init__(experiment_preset, experiment_task=experiment_task)

    def _num_epochs(self) -> int:
        return config.NUM_EPOCHS

    def _dataset_options(self) -> list:
        return dataset_options.DATASET_OPTIONS_BY_TASK[dataset_options.DEFAULT_EXPERIMENT_TASK]

    def _model_type(self) -> type:
        return Model

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        return ExperimentPresets()

    def _experiment_preset_enum(self) -> type[BaseOptions]:
        return ExperimentPreset
