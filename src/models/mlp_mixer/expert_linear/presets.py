from __future__ import annotations

import models.mlp_mixer.expert_linear.config as config
import models.mlp_mixer.expert_linear.dataset_options as dataset_options
from emperor.config import BaseOptions
from emperor.layers import LayerNormPositionOptions
from model_runtime.packages import (
    BuilderBackedExperimentPresetsBase,
    ExperimentPresetsBase,
    PresetDefinition,
)
from model_runtime.runs import ExperimentBase

from .config_builder import MlpMixerExpertLinearConfigBuilder
from .model import Model
from .runtime_defaults import runtime_from_flat


def default_patch_size_for_dataset(dataset: type) -> int:
    image_height = dataset.default_height
    if (
        image_height >= config.IMAGE_HEIGHT
        and image_height % config.IMAGE_PATCH_SIZE == 0
    ):
        return config.IMAGE_PATCH_SIZE
    for patch_size in (16, 8, 4, 2, 1):
        if image_height % patch_size == 0:
            return patch_size
    return 1


class ExperimentPreset(BaseOptions):
    BASELINE = 1
    POST_NORM = 2
    RECURRENT = 3
    RECURRENT_CONTROLLER = 4
    GATING = 5
    HALTING = 6
    MEMORY = 7
    GATING_HALTING_MEMORY = 8
    TOP_1_EXPERT = 9
    EXPERT_AUXILIARY_LOSS = 10


_PRESET_DEFINITIONS = {
    ExperimentPreset.BASELINE: PresetDefinition(
        preset_values={},
        description=(
            "Eight pre-normalized MLP-Mixer blocks with routed linear experts in "
            "both mixer branches, mean pooling, and no class token or positional "
            "embedding."
        ),
    ),
    ExperimentPreset.POST_NORM: PresetDefinition(
        preset_values={"layer_norm_position": LayerNormPositionOptions.AFTER},
        description="MLP-Mixer blocks with post-normalized residual branches.",
    ),
    ExperimentPreset.RECURRENT: PresetDefinition(
        preset_values={"recurrent_flag": True},
        description="Reuses the Mixer block stack for recurrent refinement.",
    ),
    ExperimentPreset.RECURRENT_CONTROLLER: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_stack_gate_flag": True,
        },
        description="Recurrent Mixer refinement with a learned step controller.",
    ),
    ExperimentPreset.GATING: PresetDefinition(
        preset_values={"stack_gate_flag": True},
        description="Mixer blocks with learned output gates.",
    ),
    ExperimentPreset.HALTING: PresetDefinition(
        preset_values={"stack_halting_flag": True},
        description="Mixer blocks with adaptive-computation halting.",
    ),
    ExperimentPreset.MEMORY: PresetDefinition(
        preset_values={"memory_flag": True},
        description="Mixer blocks with shared dynamic memory.",
    ),
    ExperimentPreset.GATING_HALTING_MEMORY: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
            "stack_halting_flag": True,
            "memory_flag": True,
        },
        description="Mixer blocks with gates, halting, and shared dynamic memory.",
    ),
    ExperimentPreset.TOP_1_EXPERT: PresetDefinition(
        preset_values={
            "top_k": 1,
            "sampler_normalize_probabilities_flag": False,
        },
        description="Top-1 routing in both token and channel mixer branches.",
    ),
    ExperimentPreset.EXPERT_AUXILIARY_LOSS: PresetDefinition(
        preset_values={"sampler_switch_loss_weight": 0.01},
        description=(
            "Expert token and channel mixers with switch load-balancing loss."
        ),
    ),
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=MlpMixerExpertLinearConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
        )

    def _dataset_config(self, dataset: type) -> dict:
        dataset_config = super()._dataset_config(dataset)
        return {
            **dataset_config,
            "image_patch_size": default_patch_size_for_dataset(dataset),
            "input_channels": dataset.num_channels,
            "image_height": dataset.default_height,
            "output_dim": dataset.num_classes,
        }

    def _preset(self, **kwargs):
        runtime = runtime_from_flat(kwargs, config)
        return self._builder_type(runtime=runtime).build()


class Experiment(ExperimentBase):
    def __init__(
        self,
        experiment_preset: ExperimentPreset | None = None,
        experiment_task=None,
        *,
        model_package,
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
        default_task = dataset_options.DEFAULT_EXPERIMENT_TASK
        return dataset_options.DATASET_OPTIONS_BY_TASK[default_task]

    def _model_type(self) -> type:
        return Model

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        return ExperimentPresets()

    def _experiment_preset_enum(self) -> type[BaseOptions]:
        return ExperimentPreset


__all__ = ["Experiment", "ExperimentPreset", "ExperimentPresets"]
