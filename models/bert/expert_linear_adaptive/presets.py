from emperor.base.options import BaseOptions
from emperor.datasets.text.bert_pretraining import PennTreebankBertPretraining
from emperor.experiments.base import (
    BuilderBackedExperimentPresetsBase,
    ExperimentBase,
    ExperimentPresetsBase,
    PresetDefinition,
)

import models.bert.expert_linear_adaptive.config as config
import models.experts.linear_adaptive.config as adaptive_expert_defaults
from models.bert._builder_adapter import linear_builder_kwargs_from_flat
from models.bert.expert_linear.presets import (
    _PRESET_DEFINITIONS as _EXPERT_PRESET_DEFINITIONS,
)
from models.bert.expert_linear_adaptive.config_builder import (
    BertExpertLinearAdaptiveConfigBuilder,
)
from models.bert.expert_linear_adaptive.model import Model
from models.experts._builder_adapter import (
    linear_adaptive_builder_kwargs_from_flat as expert_adaptive_kwargs_from_flat,
)


import models.bert.expert_linear_adaptive.dataset_options as dataset_options
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
    TOP1_SWITCH_AUX = 22
    EXPERT_ATTENTION = 23
    LOW_RANK_EXPERT_WEIGHT = 24


_PRESET_DEFINITIONS = {
    preset: PresetDefinition(
        preset_values=dict(definition.preset_values),
        description=definition.description.replace("expert linear", "adaptive expert linear"),
    )
    for preset, definition in zip(
        ExperimentPreset,
        _EXPERT_PRESET_DEFINITIONS.values(),
        strict=False,
    )
}
_PRESET_DEFINITIONS[ExperimentPreset.LOW_RANK_EXPERT_WEIGHT] = PresetDefinition(
    preset_values={
        "weight_option_flag": True,
        "weight_option": config.LowRankDynamicWeightConfig,
    },
    description="Default config with adaptive low-rank dynamic weights inside expert "
    "feed-forward internals.",
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


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=BertExpertLinearAdaptiveConfigBuilder,
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
        expert_kwargs = expert_adaptive_kwargs_from_flat(
            kwargs,
            adaptive_expert_defaults,
        )
        for key in (
            "mixture_options",
            "expert_stack_options",
            "sampler_options",
            "router_options",
            "router_stack_options",
            "expert_layer_controller_options",
            "expert_dynamic_memory_options",
            "expert_recurrent_controller_options",
            "adaptive_generator_stack_options",
            "hidden_adaptive_weight_options",
            "hidden_adaptive_bias_options",
            "hidden_adaptive_diagonal_options",
            "hidden_adaptive_mask_options",
            "router_layer_controller_options",
            "router_dynamic_memory_options",
            "router_recurrent_controller_options",
            "router_adaptive_weight_options",
            "router_adaptive_bias_options",
            "router_adaptive_diagonal_options",
            "router_adaptive_mask_options",
        ):
            builder_kwargs[key] = expert_kwargs[key]
        if "expert_attention_flag" in kwargs:
            builder_kwargs["expert_attention_flag"] = kwargs["expert_attention_flag"]
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
