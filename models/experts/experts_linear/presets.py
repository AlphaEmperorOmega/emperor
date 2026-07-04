from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import BaseOptions, LayerNormPositionOptions
from emperor.experiments.base import (
    BuilderBackedExperimentPresetsBase,
    ExperimentBase,
    ExperimentPresetsBase,
    PresetDefinition,
)
from emperor.experts.core.options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)

import models.experts.experts_linear.config as config
from models.experts.experts_linear.config_builder import ExpertsLinearConfigBuilder
from models.experts.experts_linear.model import Model


class ExperimentPreset(BaseOptions):
    BASELINE = 1
    GATING = 2
    HALTING = 3
    GATING_HALTING = 4
    RECURRENT = 5
    RECURRENT_GATING = 6
    RECURRENT_HALTING = 7
    RECURRENT_GATING_HALTING = 8
    SHARED_ROUTER_AFTER_WEIGHT = 9
    TOP1_SWITCH_AUX = 10
    TOP2_BALANCED_AUX = 11
    CAPACITY_TOP1_ZERO = 12
    CAPACITY_TOP1_IDENTITY = 13
    NOISY_SHARED_ROUTER = 14
    RESIDUAL_SHARED_ROUTER = 15
    POST_NORM_AFTER_WEIGHT = 16
    MEMORY = 17
    GATING_MEMORY = 18
    HALTING_MEMORY = 19
    GATING_HALTING_MEMORY = 20
    RECURRENT_MEMORY = 21
    RECURRENT_GATING_MEMORY = 22
    RECURRENT_HALTING_MEMORY = 23
    RECURRENT_GATING_HALTING_MEMORY = 24


_PRESET_LOCK_BEHAVIORS = {
    "stack_gate_flag": "stack gating",
    "stack_halting_flag": "adaptive stack halting",
    "memory_flag": "shared stack memory",
    "recurrent_flag": "recurrent execution",
    "recurrent_gate_flag": "recurrent gating",
    "recurrent_halting_flag": "adaptive recurrent halting",
    "routing_initialization_mode": "shared expert routing",
    "weighting_position_option": "expert weighting after experts",
    "top_k": "expert routing",
    "sampler_normalize_probabilities_flag": "switch-style routing probabilities",
    "sampler_switch_loss_weight": "switch auxiliary loss",
    "sampler_coefficient_of_variation_loss_weight": "balance auxiliary loss",
    "capacity_factor": "expert capacity limiting",
    "dropped_token_behavior": "dropped-token behavior",
    "sampler_noisy_topk_flag": "noisy sampler top-k routing",
    "router_noisy_topk_flag": "noisy router top-k routing",
    "stack_residual_connection_option": "stack residuals",
    "layer_norm_position": "post-layer normalization",
}

_PRESET_DEFINITIONS = {
    ExperimentPreset.BASELINE: PresetDefinition(
        preset_values={},
        description="Default config: a mixture-of-experts classifier with linear "
        "expert and sampler stacks.",
    ),
    ExperimentPreset.GATING: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
        },
        description="Default config with per-layer gating enabled in the expert stack.",
    ),
    ExperimentPreset.HALTING: PresetDefinition(
        preset_values={
            "stack_halting_flag": True,
        },
        description="Default config with stack halting enabled in the expert stack.",
    ),
    ExperimentPreset.GATING_HALTING: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
            "stack_halting_flag": True,
        },
        description="Default config with both per-layer gating and stack halting "
        "enabled in the expert stack.",
    ),
    ExperimentPreset.MEMORY: PresetDefinition(
        preset_values={
            "memory_flag": True,
        },
        description="Default config with shared stack memory enabled in the expert "
        "stack.",
    ),
    ExperimentPreset.GATING_MEMORY: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
            "memory_flag": True,
        },
        description="Default config with both per-layer gating and shared stack "
        "memory enabled in the expert stack.",
    ),
    ExperimentPreset.HALTING_MEMORY: PresetDefinition(
        preset_values={
            "stack_halting_flag": True,
            "memory_flag": True,
        },
        description="Default config with both stack halting and shared stack memory "
        "enabled in the expert stack.",
    ),
    ExperimentPreset.GATING_HALTING_MEMORY: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
            "stack_halting_flag": True,
            "memory_flag": True,
        },
        description="Default config with per-layer gating, stack halting, and shared "
        "stack memory enabled in the expert stack.",
    ),
    ExperimentPreset.RECURRENT: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
        },
        description="Default config wrapped in fixed-step recurrence, reusing the expert "
        "stack for each recurrent step.",
    ),
    ExperimentPreset.RECURRENT_GATING: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_gate_flag": True,
        },
        description="Default recurrent config with step-level gating enabled after each "
        "recurrent update.",
    ),
    ExperimentPreset.RECURRENT_HALTING: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_halting_flag": True,
        },
        description="Default recurrent config with recurrent halting enabled, allowing "
        "early stopping before the max step count.",
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
    ExperimentPreset.RECURRENT_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "memory_flag": True,
        },
        description="Default recurrent config whose reused expert stack has shared "
        "memory enabled.",
    ),
    ExperimentPreset.RECURRENT_GATING_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_gate_flag": True,
            "memory_flag": True,
        },
        description="Default recurrent config with step-level gating and shared memory "
        "in the reused expert stack.",
    ),
    ExperimentPreset.RECURRENT_HALTING_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_halting_flag": True,
            "memory_flag": True,
        },
        description="Default recurrent config with recurrent halting and shared memory "
        "in the reused expert stack.",
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_gate_flag": True,
            "recurrent_halting_flag": True,
            "memory_flag": True,
        },
        description="Default recurrent config with step-level gating, recurrent "
        "halting, and shared memory in the reused expert stack.",
    ),
    ExperimentPreset.SHARED_ROUTER_AFTER_WEIGHT: PresetDefinition(
        preset_values={
            "routing_initialization_mode": RoutingInitializationMode.SHARED,
            "weighting_position_option": (ExpertWeightingPositionOptions.AFTER_EXPERTS),
        },
        description="Default config with shared expert routing and expert weighting after "
        "expert outputs.",
    ),
    ExperimentPreset.TOP1_SWITCH_AUX: PresetDefinition(
        preset_values={
            "top_k": 1,
            "sampler_normalize_probabilities_flag": False,
            "sampler_switch_loss_weight": 0.1,
        },
        description="Default config with top-1 switch routing and switch auxiliary loss "
        "enabled.",
    ),
    ExperimentPreset.TOP2_BALANCED_AUX: PresetDefinition(
        preset_values={
            "top_k": 2,
            "sampler_coefficient_of_variation_loss_weight": 0.1,
        },
        description="Default config with top-2 routing and balance auxiliary loss enabled.",
    ),
    ExperimentPreset.CAPACITY_TOP1_ZERO: PresetDefinition(
        preset_values={
            "top_k": 1,
            "capacity_factor": 1.0,
            "dropped_token_behavior": DroppedTokenOptions.ZEROS,
            "sampler_normalize_probabilities_flag": False,
        },
        description="Default config with top-1 capacity limiting and dropped tokens zeroed.",
    ),
    ExperimentPreset.CAPACITY_TOP1_IDENTITY: PresetDefinition(
        preset_values={
            "top_k": 1,
            "capacity_factor": 1.0,
            "dropped_token_behavior": DroppedTokenOptions.IDENTITY,
            "sampler_normalize_probabilities_flag": False,
        },
        description="Default config with top-1 capacity limiting and dropped tokens "
        "preserved by identity.",
    ),
    ExperimentPreset.NOISY_SHARED_ROUTER: PresetDefinition(
        preset_values={
            "routing_initialization_mode": RoutingInitializationMode.SHARED,
            "sampler_noisy_topk_flag": True,
            "router_noisy_topk_flag": True,
        },
        description="Default config with shared noisy top-k routing enabled for "
        "sampler and router.",
    ),
    ExperimentPreset.RESIDUAL_SHARED_ROUTER: PresetDefinition(
        preset_values={
            "routing_initialization_mode": RoutingInitializationMode.SHARED,
            "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
        },
        description="Default config with residual expert stack connections and shared "
        "expert routing.",
    ),
    ExperimentPreset.POST_NORM_AFTER_WEIGHT: PresetDefinition(
        preset_values={
            "layer_norm_position": LayerNormPositionOptions.AFTER,
            "weighting_position_option": (ExpertWeightingPositionOptions.AFTER_EXPERTS),
        },
        description="Default config with post-layer normalization and expert weighting "
        "after expert outputs.",
    ),
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=ExpertsLinearConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
        )

    def _preset_lock_reason(self, preset: ExperimentPreset, field: str) -> str:
        behavior = _PRESET_LOCK_BEHAVIORS[field]
        return (
            f"Locked by the {preset.name} preset because this preset enables "
            f"{behavior}."
        )


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
