from emperor.base.options import BaseOptions
from emperor.datasets.text.bert_pretraining import PennTreebankBertPretraining
from emperor.experiments.base import (
    BuilderBackedExperimentPresetsBase,
    ExperimentBase,
    ExperimentPresetsBase,
    PresetDefinition,
)

import models.bert.linear_adaptive.config as config
import models.linears.linear_adaptive.config as adaptive_defaults
from models.bert._builder_adapter import linear_builder_kwargs_from_flat
from models.bert.linear.presets import (
    _PRESET_DEFINITIONS as _LINEAR_PRESET_DEFINITIONS,
)
from models.bert.linear_adaptive.config_builder import (
    BertLinearAdaptiveConfigBuilder,
)
from models.bert.linear_adaptive.model import Model
from models.linears._builder_adapter import (
    linear_adaptive_builder_kwargs_from_flat,
)

_BUILDER_KEYS = {
    "batch_size",
    "learning_rate",
    "input_dim",
    "output_dim",
    "sequence_length",
    "embedding_dropout_probability",
    "encoder_options",
    "positional_embedding_options",
    "attention_options",
    "feed_forward_options",
    "submodule_stack_options",
    "layer_controller_options",
    "dynamic_memory_options",
    "recurrent_controller_options",
}


class ExperimentPreset(BaseOptions):
    BASELINE = 1
    PRE_NORM = 2
    POST_NORM = 3
    SINUSOIDAL = 4
    CAUSAL = 5
    ATTENTION_BIAS = 6
    GATING = 7
    HALTING = 8
    GATING_HALTING = 9
    MEMORY = 10
    GATING_MEMORY = 11
    HALTING_MEMORY = 12
    GATING_HALTING_MEMORY = 13
    RECURRENT = 14
    RECURRENT_GATING = 15
    RECURRENT_HALTING = 16
    RECURRENT_MEMORY = 17
    RECURRENT_GATING_HALTING = 18
    RECURRENT_GATING_MEMORY = 19
    RECURRENT_HALTING_MEMORY = 20
    RECURRENT_GATING_HALTING_MEMORY = 21
    LOW_RANK_WEIGHT = 22


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
    "attention and feed-forward linear stacks.",
)


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=BertLinearAdaptiveConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
            default_dataset=PennTreebankBertPretraining,
        )

    def _dataset_config(self, dataset: type) -> dict:
        return {
            **super()._dataset_config(dataset),
            "sequence_length": dataset.sequence_length,
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
