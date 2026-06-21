from typing import TYPE_CHECKING

from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import BaseOptions, LayerNormPositionOptions
from emperor.datasets.image.classification.mnist import Mnist
from emperor.experiments.base import (
    ExperimentBase,
    ExperimentPresetsBase,
    PresetLock,
    SearchMode,
)
from emperor.experts.core.options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)

import models.experts.experts_linear.config as config
from models.experts.experts_linear.config_builder import ExpertsLinearConfigBuilder
from models.experts.experts_linear.model import Model

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExperimentPreset(BaseOptions):
    BASELINE = (
        "Default config: a mixture-of-experts classifier with linear expert and "
        "sampler stacks."
    )
    GATING = "Default config with per-layer gating enabled in the expert stack."
    HALTING = "Default config with stack halting enabled in the expert stack."
    GATING_HALTING = (
        "Default config with both per-layer gating and stack halting enabled in "
        "the expert stack."
    )
    RECURRENT = (
        "Default config wrapped in fixed-step recurrence, reusing the expert "
        "stack for each recurrent step."
    )
    RECURRENT_GATING = (
        "Default recurrent config with step-level gating enabled after each "
        "recurrent update."
    )
    RECURRENT_HALTING = (
        "Default recurrent config with recurrent halting enabled, allowing early "
        "stopping before the max step count."
    )
    RECURRENT_GATING_HALTING = (
        "Default recurrent config with both step-level gating and recurrent "
        "halting enabled."
    )
    SHARED_ROUTER_AFTER_WEIGHT = (
        "Default config with shared expert routing and expert weighting after "
        "expert outputs."
    )
    TOP1_SWITCH_AUX = (
        "Default config with top-1 switch routing and switch auxiliary loss enabled."
    )
    TOP2_BALANCED_AUX = (
        "Default config with top-2 routing and balance auxiliary loss enabled."
    )
    CAPACITY_TOP1_ZERO = (
        "Default config with top-1 capacity limiting and dropped tokens zeroed."
    )
    CAPACITY_TOP1_IDENTITY = (
        "Default config with top-1 capacity limiting and dropped tokens preserved "
        "by identity."
    )
    NOISY_SHARED_ROUTER = (
        "Default config with shared noisy top-k routing enabled for sampler and router."
    )
    RESIDUAL_SHARED_ROUTER = (
        "Default config with residual expert stack connections and shared expert "
        "routing."
    )
    POST_NORM_AFTER_WEIGHT = (
        "Default config with post-layer normalization and expert weighting after "
        "expert outputs."
    )


def _lock(preset, value, behavior: str) -> PresetLock:
    return PresetLock(
        value=value,
        reason=(
            f"Locked by the {preset.name} preset because this preset enables "
            f"{behavior}."
        ),
    )


def _preset_locks(
    preset_overrides: dict["ExperimentPreset", dict[str, object]],
) -> dict["ExperimentPreset", dict[str, PresetLock]]:
    return {
        preset: {
            field: _lock(preset, value, _PRESET_LOCK_BEHAVIORS[field])
            for field, value in overrides.items()
        }
        for preset, overrides in preset_overrides.items()
        if overrides
    }


_PRESET_LOCK_BEHAVIORS = {
    "stack_gate_flag": "stack gating",
    "stack_halting_flag": "adaptive stack halting",
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


_PRESET_OVERRIDES = {
    ExperimentPreset.BASELINE: {},
    ExperimentPreset.GATING: {
        "stack_gate_flag": True,
    },
    ExperimentPreset.HALTING: {
        "stack_halting_flag": True,
    },
    ExperimentPreset.GATING_HALTING: {
        "stack_gate_flag": True,
        "stack_halting_flag": True,
    },
    ExperimentPreset.RECURRENT: {
        "recurrent_flag": True,
    },
    ExperimentPreset.RECURRENT_GATING: {
        "recurrent_flag": True,
        "recurrent_gate_flag": True,
    },
    ExperimentPreset.RECURRENT_HALTING: {
        "recurrent_flag": True,
        "recurrent_halting_flag": True,
    },
    ExperimentPreset.RECURRENT_GATING_HALTING: {
        "recurrent_flag": True,
        "recurrent_gate_flag": True,
        "recurrent_halting_flag": True,
    },
    ExperimentPreset.SHARED_ROUTER_AFTER_WEIGHT: {
        "routing_initialization_mode": RoutingInitializationMode.SHARED,
        "weighting_position_option": ExpertWeightingPositionOptions.AFTER_EXPERTS,
    },
    ExperimentPreset.TOP1_SWITCH_AUX: {
        "top_k": 1,
        "sampler_normalize_probabilities_flag": False,
        "sampler_switch_loss_weight": 0.1,
    },
    ExperimentPreset.TOP2_BALANCED_AUX: {
        "top_k": 2,
        "sampler_coefficient_of_variation_loss_weight": 0.1,
    },
    ExperimentPreset.CAPACITY_TOP1_ZERO: {
        "top_k": 1,
        "capacity_factor": 1.0,
        "dropped_token_behavior": DroppedTokenOptions.ZEROS,
        "sampler_normalize_probabilities_flag": False,
    },
    ExperimentPreset.CAPACITY_TOP1_IDENTITY: {
        "top_k": 1,
        "capacity_factor": 1.0,
        "dropped_token_behavior": DroppedTokenOptions.IDENTITY,
        "sampler_normalize_probabilities_flag": False,
    },
    ExperimentPreset.NOISY_SHARED_ROUTER: {
        "routing_initialization_mode": RoutingInitializationMode.SHARED,
        "sampler_noisy_topk_flag": True,
        "router_noisy_topk_flag": True,
    },
    ExperimentPreset.RESIDUAL_SHARED_ROUTER: {
        "routing_initialization_mode": RoutingInitializationMode.SHARED,
        "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
    },
    ExperimentPreset.POST_NORM_AFTER_WEIGHT: {
        "layer_norm_position": LayerNormPositionOptions.AFTER,
        "weighting_position_option": ExpertWeightingPositionOptions.AFTER_EXPERTS,
    },
}


class ExperimentPresets(ExperimentPresetsBase):
    PRESET_OVERRIDES = _PRESET_OVERRIDES
    PRESET_LOCKS = _preset_locks(PRESET_OVERRIDES)

    def __init__(self) -> None:
        super().__init__()

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
        return ExpertsLinearConfigBuilder(**kwargs).build()


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
