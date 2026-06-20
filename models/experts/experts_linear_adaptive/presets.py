import models.experts.experts_linear_adaptive.config as config

from models.experts.experts_linear_adaptive.config_builder import ExpertsLinearAdaptiveConfigBuilder
from models.experts.experts_linear_adaptive.model import Model
from emperor.experiments.base import SearchMode
from emperor.datasets.image.classification.mnist import Mnist
from emperor.experiments.base import ExperimentBase, ExperimentPresetsBase, PresetLock
from emperor.base.options import BaseOptions
from emperor.experts.core.options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExperimentPreset(BaseOptions):
    BASELINE = "Baseline mixture-of-experts with adaptive linear input, output, expert, and sampler stacks."
    GATING = "Mixture-of-experts stack with per-layer gating enabled."
    HALTING = "Mixture-of-experts stack with per-layer halting enabled."
    GATING_HALTING = "Mixture-of-experts stack with both per-layer gating and halting enabled."
    RECURRENT = "Mixture-of-experts model applied recurrently for a fixed number of steps."
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
    ADAPTIVE_SHARED_ROUTER = (
        "Adaptive mixture-of-experts model with dual-model dynamic weights and shared routing."
    )
    ADAPTIVE_AFTER_WEIGHT = (
        "Adaptive mixture-of-experts model with dual-model dynamic weights and weighting after experts."
    )
    ADAPTIVE_TOP1_SWITCH = (
        "Adaptive mixture-of-experts model with top-1 routing and switch auxiliary loss."
    )
    ADAPTIVE_FULL_SHARED = (
        "Adaptive mixture-of-experts model with full adaptive augmentation and shared routing."
    )
    ADAPTIVE_FULL_CAPACITY = (
        "Adaptive mixture-of-experts model with full adaptive augmentation and top-1 capacity limiting."
    )
    ADAPTIVE_BANK_ROUTER = (
        "Adaptive mixture-of-experts model with weighted-bank dynamic weights and shared routing."
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


_FULL_ADAPTIVE_KWARGS = {
    "weight_option": DualModelDynamicWeightConfig,
    "bias_option": AdditiveDynamicBiasConfig,
    "diagonal_option": CombinedDynamicDiagonalConfig,
    "row_mask_option": WeightInformedScoreAxisMaskConfig,
}


_PRESET_LOCK_BEHAVIORS = {
    "stack_gate_flag": "stack gating",
    "stack_halting_flag": "adaptive stack halting",
    "recurrent_flag": "recurrent execution",
    "recurrent_gate_flag": "recurrent gating",
    "recurrent_halting_flag": "adaptive recurrent halting",
    "weight_option": "dynamic weights",
    "bias_option": "dynamic bias",
    "diagonal_option": "dynamic diagonal",
    "row_mask_option": "adaptive row masking",
    "routing_initialization_mode": "shared expert routing",
    "weighting_position_option": "expert weighting after experts",
    "top_k": "expert routing",
    "sampler_normalize_probabilities_flag": "switch-style routing probabilities",
    "sampler_switch_loss_weight": "switch auxiliary loss",
    "capacity_factor": "expert capacity limiting",
    "dropped_token_behavior": "dropped-token behavior",
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
    ExperimentPreset.ADAPTIVE_SHARED_ROUTER: {
        "weight_option": DualModelDynamicWeightConfig,
        "routing_initialization_mode": RoutingInitializationMode.SHARED,
    },
    ExperimentPreset.ADAPTIVE_AFTER_WEIGHT: {
        "weight_option": DualModelDynamicWeightConfig,
        "weighting_position_option": ExpertWeightingPositionOptions.AFTER_EXPERTS,
    },
    ExperimentPreset.ADAPTIVE_TOP1_SWITCH: {
        "weight_option": DualModelDynamicWeightConfig,
        "top_k": 1,
        "sampler_normalize_probabilities_flag": False,
        "sampler_switch_loss_weight": 0.1,
    },
    ExperimentPreset.ADAPTIVE_FULL_SHARED: {
        **_FULL_ADAPTIVE_KWARGS,
        "routing_initialization_mode": RoutingInitializationMode.SHARED,
    },
    ExperimentPreset.ADAPTIVE_FULL_CAPACITY: {
        **_FULL_ADAPTIVE_KWARGS,
        "top_k": 1,
        "capacity_factor": 1.0,
        "dropped_token_behavior": DroppedTokenOptions.ZEROS,
        "sampler_normalize_probabilities_flag": False,
    },
    ExperimentPreset.ADAPTIVE_BANK_ROUTER: {
        "weight_option": LayeredWeightedBankDynamicWeightConfig,
        "routing_initialization_mode": RoutingInitializationMode.SHARED,
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
        return ExpertsLinearAdaptiveConfigBuilder(**kwargs).build()


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
