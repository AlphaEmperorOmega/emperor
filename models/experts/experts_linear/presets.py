import models.experts.experts_linear.config as config

from models.experts.experts_linear.config_builder import ExpertsLinearConfigBuilder
from models.experts.experts_linear.model import Model
from emperor.experiments.base import SearchMode
from emperor.datasets.image.classification.mnist import Mnist
from emperor.experiments.base import ExperimentBase, ExperimentPresetsBase, PresetLock
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import BaseOptions, LayerNormPositionOptions
from emperor.experts.core.options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExperimentPreset(BaseOptions):
    BASELINE = "Baseline mixture-of-experts with linear expert and sampler stacks."
    GATING = "Mixture-of-experts stack with per-layer gating enabled."
    HALTING = "Mixture-of-experts stack with per-layer halting enabled."
    GATING_HALTING = (
        "Mixture-of-experts stack with both per-layer gating and halting enabled."
    )
    RECURRENT = (
        "Mixture-of-experts model applied recurrently for a fixed number of steps."
    )
    RECURRENT_GATING = (
        "Mixture-of-experts model applied recurrently with a learned recurrent gate."
    )
    RECURRENT_HALTING = (
        "Mixture-of-experts model applied recurrently with adaptive recurrent halting."
    )
    RECURRENT_GATING_HALTING = (
        "Mixture-of-experts model applied recurrently with both learned recurrent "
        "gating and adaptive recurrent halting."
    )
    SHARED_ROUTER_AFTER_WEIGHT = "Mixture-of-experts model with shared routing and expert weighting after experts."
    TOP1_SWITCH_AUX = (
        "Top-1 mixture-of-experts routing with switch auxiliary loss enabled."
    )
    TOP2_BALANCED_AUX = (
        "Top-2 mixture-of-experts routing with balance auxiliary loss enabled."
    )
    CAPACITY_TOP1_ZERO = "Top-1 mixture-of-experts routing with capacity limiting and zeroed dropped tokens."
    CAPACITY_TOP1_IDENTITY = "Top-1 mixture-of-experts routing with capacity limiting and identity dropped tokens."
    NOISY_SHARED_ROUTER = "Mixture-of-experts model with shared noisy top-k routing."
    RESIDUAL_SHARED_ROUTER = (
        "Mixture-of-experts model with residual outer layers and shared routing."
    )
    POST_NORM_AFTER_WEIGHT = "Mixture-of-experts model with post-layer norm and expert weighting after experts."


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
                "The specified preset is not supported. Please choose a valid `ExperimentPreset`."
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
