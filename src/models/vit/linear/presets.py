# ruff: noqa: E501

from dataclasses import replace

import models.vit.linear.config as config
import models.vit.linear.dataset_options as dataset_options
from emperor.config import BaseOptions
from emperor.embedding.absolute import (
    ImageSinusoidalPositionalEmbeddingConfig,
)
from emperor.layers import LayerNormPositionOptions, ResidualConnectionOptions
from model_runtime.packages import (
    BuilderBackedExperimentPresetsBase,
    ExperimentPresetsBase,
    PresetDefinition,
)
from model_runtime.runs import ExperimentBase
from models.vit.linear.config_builder import VitLinearConfigBuilder
from models.vit.linear.model import Model
from models.vit.linear.runtime_defaults import (
    patch_options_from_config,
    runtime_from_flat,
)


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
    GATING = 5
    HALTING = 6
    MEMORY = 7
    GATING_HALTING = 8
    GATING_MEMORY = 9
    HALTING_MEMORY = 10
    GATING_HALTING_MEMORY = 11
    RESIDUAL = 12
    RESIDUAL_POST_NORM = 13
    RESIDUAL_GATING = 14
    RESIDUAL_HALTING = 15
    RESIDUAL_MEMORY = 16
    RECURRENT = 17
    RECURRENT_GATING = 18
    RECURRENT_HALTING = 19
    RECURRENT_MEMORY = 20
    RECURRENT_GATING_HALTING = 21
    RECURRENT_GATING_MEMORY = 22
    RECURRENT_HALTING_MEMORY = 23
    RECURRENT_GATING_HALTING_MEMORY = 24
    RECURRENT_RESIDUAL = 25
    RECURRENT_POST_NORM = 26


_PRESET_DEFINITIONS = {
    ExperimentPreset.BASELINE: PresetDefinition(
        preset_values={},
        description="Default config: a Vision Transformer classifier with linear patch "
        "embeddings, a trainable class token, learned image positions, and a "
        "pre-norm bidirectional encoder.",
    ),
    ExperimentPreset.POST_NORM: PresetDefinition(
        preset_values={
            "layer_norm_position": LayerNormPositionOptions.AFTER,
        },
        description="Default config with layer normalization after each encoder "
        "sub-block.",
    ),
    ExperimentPreset.SINUSOIDAL: PresetDefinition(
        preset_values={
            "positional_embedding_option": ImageSinusoidalPositionalEmbeddingConfig,
        },
        description="Default config with fixed sinusoidal image positional embeddings.",
    ),
    ExperimentPreset.ATTENTION_BIAS: PresetDefinition(
        preset_values={
            "attn_bias_flag": True,
            "attn_add_key_value_bias_flag": True,
        },
        description="Default config with attention projection bias and key/value bias "
        "enabled.",
    ),
    ExperimentPreset.GATING: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
        },
        description="Default config with per-layer gating enabled on encoder blocks.",
    ),
    ExperimentPreset.HALTING: PresetDefinition(
        preset_values={
            "stack_halting_flag": True,
        },
        description="Default config with encoder stack halting enabled, so examples can "
        "stop early as they move through the encoder.",
    ),
    ExperimentPreset.MEMORY: PresetDefinition(
        preset_values={
            "memory_flag": True,
        },
        description="Default config with shared encoder stack memory enabled.",
    ),
    ExperimentPreset.GATING_HALTING: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
            "stack_halting_flag": True,
        },
        description="Default config with both per-layer gating and encoder stack "
        "halting enabled.",
    ),
    ExperimentPreset.GATING_MEMORY: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
            "memory_flag": True,
        },
        description="Default config with both per-layer gating and shared encoder stack "
        "memory enabled.",
    ),
    ExperimentPreset.HALTING_MEMORY: PresetDefinition(
        preset_values={
            "stack_halting_flag": True,
            "memory_flag": True,
        },
        description="Default config with both encoder stack halting and shared memory "
        "enabled.",
    ),
    ExperimentPreset.GATING_HALTING_MEMORY: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
            "stack_halting_flag": True,
            "memory_flag": True,
        },
        description="Default config with per-layer gating, encoder stack halting, and "
        "shared memory enabled.",
    ),
    ExperimentPreset.RESIDUAL: PresetDefinition(
        preset_values={
            "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
        },
        description="Default config with residual skip connections enabled between "
        "same-width encoder stack layers.",
    ),
    ExperimentPreset.RESIDUAL_POST_NORM: PresetDefinition(
        preset_values={
            "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
            "layer_norm_position": LayerNormPositionOptions.AFTER,
        },
        description="Default config with residual skip connections and post-layer "
        "normalization enabled.",
    ),
    ExperimentPreset.RESIDUAL_GATING: PresetDefinition(
        preset_values={
            "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
            "stack_gate_flag": True,
        },
        description="Default config with residual skip connections and per-layer gating "
        "enabled.",
    ),
    ExperimentPreset.RESIDUAL_HALTING: PresetDefinition(
        preset_values={
            "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
            "stack_halting_flag": True,
        },
        description="Default config with residual skip connections and encoder stack "
        "halting enabled.",
    ),
    ExperimentPreset.RESIDUAL_MEMORY: PresetDefinition(
        preset_values={
            "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
            "memory_flag": True,
        },
        description="Default config with residual skip connections and shared encoder "
        "stack memory enabled.",
    ),
    ExperimentPreset.RECURRENT: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
        },
        description="Default config wrapped in fixed-step recurrence, reusing the "
        "encoder stack for each recurrent step.",
    ),
    ExperimentPreset.RECURRENT_GATING: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_gate_flag": True,
        },
        description="Default recurrent config with step-level gating enabled after each "
        "recurrent encoder update.",
    ),
    ExperimentPreset.RECURRENT_HALTING: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_halting_flag": True,
        },
        description="Default recurrent config with recurrent halting enabled, allowing "
        "early stopping before the max step count.",
    ),
    ExperimentPreset.RECURRENT_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "memory_flag": True,
        },
        description="Default recurrent config whose reused encoder stack has shared "
        "memory enabled.",
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_gate_flag": True,
            "recurrent_halting_flag": True,
        },
        description="Default recurrent config with both step-level gating and recurrent "
        "halting enabled.",
    ),
    ExperimentPreset.RECURRENT_GATING_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_gate_flag": True,
            "memory_flag": True,
        },
        description="Default recurrent config with step-level gating and shared memory "
        "in the reused encoder stack.",
    ),
    ExperimentPreset.RECURRENT_HALTING_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_halting_flag": True,
            "memory_flag": True,
        },
        description="Default recurrent config with recurrent halting and shared memory "
        "in the reused encoder stack.",
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_gate_flag": True,
            "recurrent_halting_flag": True,
            "memory_flag": True,
        },
        description="Default recurrent config with step-level gating, recurrent "
        "halting, and shared memory in the reused encoder stack.",
    ),
    ExperimentPreset.RECURRENT_RESIDUAL: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
        },
        description="Default recurrent config using a residual encoder stack at each "
        "recurrent step.",
    ),
    ExperimentPreset.RECURRENT_POST_NORM: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "layer_norm_position": LayerNormPositionOptions.AFTER,
        },
        description="Default recurrent config using a post-normalized encoder stack at "
        "each recurrent step.",
    ),
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=VitLinearConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
        )

    def _dataset_config(self, dataset: type) -> dict:
        dataset_config = super()._dataset_config(dataset)
        dataset_patch_size = default_patch_size_for_dataset(dataset)
        dataset_patch_options = replace(
            patch_options_from_config(config),
            patch_size=dataset_patch_size,
            input_channels=dataset.num_channels,
            image_height=dataset.default_height,
        )

        return {
            **dataset_config,
            "patch_options": dataset_patch_options,
            "output_dim": dataset.num_classes,
        }

    def _preset(self, **kwargs):
        return self._builder_type(runtime=runtime_from_flat(kwargs, config)).build()


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
        default_task = dataset_options.DEFAULT_EXPERIMENT_TASK
        return dataset_options.DATASET_OPTIONS_BY_TASK[default_task]

    def _model_type(self) -> type:
        return Model

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        return ExperimentPresets()

    def _experiment_preset_enum(self) -> type[BaseOptions]:
        return ExperimentPreset
