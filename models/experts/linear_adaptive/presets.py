from emperor.augmentations.adaptive_parameters.core.bias import (
    AdditiveDynamicBiasConfig,
)
from emperor.augmentations.adaptive_parameters.core.diagonal import (
    CombinedDynamicDiagonalConfig,
)
from emperor.augmentations.adaptive_parameters.core.mask import (
    WeightInformedScoreAxisMaskConfig,
)
from emperor.augmentations.adaptive_parameters.core.weight import (
    DualModelDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
)
from emperor.base.options import BaseOptions
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

import models.experts.linear_adaptive.config as config
from models.experts._builder_adapter import (
    linear_adaptive_builder_kwargs_from_flat,
)
from models.experts.linear_adaptive.config_builder import (
    LinearAdaptiveConfigBuilder,
)
from models.experts.linear_adaptive.model import Model


class ExperimentPreset(BaseOptions):
    BASELINE = 1
    GATING = 2
    HALTING = 3
    GATING_HALTING = 4
    RECURRENT = 5
    RECURRENT_GATING = 6
    RECURRENT_HALTING = 7
    RECURRENT_GATING_HALTING = 8
    ADAPTIVE_SHARED_ROUTER = 9
    ADAPTIVE_AFTER_WEIGHT = 10
    ADAPTIVE_TOP1_SWITCH = 11
    ADAPTIVE_FULL_SHARED = 12
    ADAPTIVE_FULL_CAPACITY = 13
    ADAPTIVE_BANK_ROUTER = 14
    MEMORY = 15
    GATING_MEMORY = 16
    HALTING_MEMORY = 17
    GATING_HALTING_MEMORY = 18
    RECURRENT_MEMORY = 19
    RECURRENT_GATING_MEMORY = 20
    RECURRENT_HALTING_MEMORY = 21
    RECURRENT_GATING_HALTING_MEMORY = 22


_ADAPTIVE_OPTION_FLAGS = {
    "weight_option": "weight_option_flag",
    "bias_option": "bias_option_flag",
    "diagonal_option": "diagonal_option_flag",
    "row_mask_option": "mask_option_flag",
    "router_weight_option": "router_weight_option_flag",
    "router_bias_option": "router_bias_option_flag",
    "router_diagonal_option": "router_diagonal_option_flag",
    "router_row_mask_option": "router_mask_option_flag",
}


def _with_adaptive_option_flags(overrides: dict[str, object]) -> dict[str, object]:
    option_flags = {
        flag: True
        for option, flag in _ADAPTIVE_OPTION_FLAGS.items()
        if option in overrides and overrides[option] is not None
    }
    return {**option_flags, **overrides}


def _full_adaptive_values() -> dict[str, object]:
    return {
        "weight_option": DualModelDynamicWeightConfig,
        "bias_option": AdditiveDynamicBiasConfig,
        "diagonal_option": CombinedDynamicDiagonalConfig,
        "row_mask_option": WeightInformedScoreAxisMaskConfig,
    }


_PRESET_DEFINITIONS = {
    ExperimentPreset.BASELINE: PresetDefinition(
        preset_values={},
        description="Default config: a mixture-of-experts classifier with adaptive linear "
        "input, output, expert, and sampler stacks.",
    ),
    ExperimentPreset.GATING: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
        },
        description="Default config with per-layer gating enabled in the adaptive expert "
        "stack.",
    ),
    ExperimentPreset.HALTING: PresetDefinition(
        preset_values={
            "stack_halting_flag": True,
        },
        description="Default config with stack halting enabled in the adaptive expert "
        "stack.",
    ),
    ExperimentPreset.GATING_HALTING: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
            "stack_halting_flag": True,
        },
        description="Default config with both per-layer gating and stack halting "
        "enabled in the adaptive expert stack.",
    ),
    ExperimentPreset.MEMORY: PresetDefinition(
        preset_values={
            "memory_flag": True,
        },
        description="Default config with shared stack memory enabled in the adaptive "
        "expert stack.",
    ),
    ExperimentPreset.GATING_MEMORY: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
            "memory_flag": True,
        },
        description="Default config with both per-layer gating and shared stack "
        "memory enabled in the adaptive expert stack.",
    ),
    ExperimentPreset.HALTING_MEMORY: PresetDefinition(
        preset_values={
            "stack_halting_flag": True,
            "memory_flag": True,
        },
        description="Default config with both stack halting and shared stack memory "
        "enabled in the adaptive expert stack.",
    ),
    ExperimentPreset.GATING_HALTING_MEMORY: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
            "stack_halting_flag": True,
            "memory_flag": True,
        },
        description="Default config with per-layer gating, stack halting, and shared "
        "stack memory enabled in the adaptive expert stack.",
    ),
    ExperimentPreset.RECURRENT: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
        },
        description="Default config wrapped in fixed-step recurrence, reusing the adaptive "
        "expert stack for each recurrent step.",
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
        description="Default recurrent config whose reused adaptive expert stack has "
        "shared memory enabled.",
    ),
    ExperimentPreset.RECURRENT_GATING_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_gate_flag": True,
            "memory_flag": True,
        },
        description="Default recurrent config with step-level gating and shared memory "
        "in the reused adaptive expert stack.",
    ),
    ExperimentPreset.RECURRENT_HALTING_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_halting_flag": True,
            "memory_flag": True,
        },
        description="Default recurrent config with recurrent halting and shared memory "
        "in the reused adaptive expert stack.",
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_gate_flag": True,
            "recurrent_halting_flag": True,
            "memory_flag": True,
        },
        description="Default recurrent config with step-level gating, recurrent "
        "halting, and shared memory in the reused adaptive expert stack.",
    ),
    ExperimentPreset.ADAPTIVE_SHARED_ROUTER: PresetDefinition(
        preset_values=_with_adaptive_option_flags({
            "weight_option": DualModelDynamicWeightConfig,
            "routing_initialization_mode": RoutingInitializationMode.SHARED,
        }),
        description="Default adaptive config with dual-model dynamic weights and shared "
        "expert routing.",
    ),
    ExperimentPreset.ADAPTIVE_AFTER_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags({
            "weight_option": DualModelDynamicWeightConfig,
            "weighting_position_option": (ExpertWeightingPositionOptions.AFTER_EXPERTS),
        }),
        description="Default adaptive config with dual-model dynamic weights and expert "
        "weighting after expert outputs.",
    ),
    ExperimentPreset.ADAPTIVE_TOP1_SWITCH: PresetDefinition(
        preset_values=_with_adaptive_option_flags({
            "weight_option": DualModelDynamicWeightConfig,
            "top_k": 1,
            "sampler_normalize_probabilities_flag": False,
            "sampler_switch_loss_weight": 0.1,
        }),
        description="Default adaptive config with dual-model dynamic weights, top-1 switch "
        "routing, and switch auxiliary loss enabled.",
    ),
    ExperimentPreset.ADAPTIVE_FULL_SHARED: PresetDefinition(
        preset_values=_with_adaptive_option_flags({
            **_full_adaptive_values(),
            "routing_initialization_mode": RoutingInitializationMode.SHARED,
        }),
        description="Default adaptive config with dynamic weights, bias, diagonal, row "
        "mask, and shared expert routing.",
    ),
    ExperimentPreset.ADAPTIVE_FULL_CAPACITY: PresetDefinition(
        preset_values=_with_adaptive_option_flags({
            **_full_adaptive_values(),
            "top_k": 1,
            "capacity_factor": 1.0,
            "dropped_token_behavior": DroppedTokenOptions.ZEROS,
            "sampler_normalize_probabilities_flag": False,
        }),
        description="Default adaptive config with dynamic weights, bias, diagonal, row "
        "mask, top-1 capacity limiting, and dropped tokens zeroed.",
    ),
    ExperimentPreset.ADAPTIVE_BANK_ROUTER: PresetDefinition(
        preset_values=_with_adaptive_option_flags({
            "router_weight_option": LayeredWeightedBankDynamicWeightConfig,
            "routing_initialization_mode": RoutingInitializationMode.SHARED,
        }),
        description="Default adaptive config with layered weighted-bank dynamic "
        "weights and shared expert routing.",
    ),
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=LinearAdaptiveConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
        )

    def _preset(self, **kwargs):
        builder_kwargs = linear_adaptive_builder_kwargs_from_flat(
            kwargs,
            config,
        )
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
