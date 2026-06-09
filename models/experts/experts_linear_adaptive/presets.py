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


class ExperimentOptions(BaseOptions):
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


def _lock(option, value, behavior: str) -> PresetLock:
    return PresetLock(
        value=value,
        reason=(
            f"Locked by the {option.name} preset because this preset enables "
            f"{behavior}."
        ),
    )


class ExperimentPresets(ExperimentPresetsBase):
    PRESET_LOCKS = {
        ExperimentOptions.GATING: {
            "stack_gate_flag": _lock(ExperimentOptions.GATING, True, "stack gating"),
        },
        ExperimentOptions.HALTING: {
            "stack_halting_flag": _lock(ExperimentOptions.HALTING, True, "adaptive stack halting"),
        },
        ExperimentOptions.GATING_HALTING: {
            "stack_gate_flag": _lock(ExperimentOptions.GATING_HALTING, True, "stack gating"),
            "stack_halting_flag": _lock(
                ExperimentOptions.GATING_HALTING,
                True,
                "adaptive stack halting",
            ),
        },
        ExperimentOptions.RECURRENT: {
            "recurrent_flag": _lock(ExperimentOptions.RECURRENT, True, "recurrent execution"),
        },
        ExperimentOptions.RECURRENT_GATING: {
            "recurrent_flag": _lock(
                ExperimentOptions.RECURRENT_GATING,
                True,
                "recurrent execution",
            ),
            "recurrent_gate_flag": _lock(
                ExperimentOptions.RECURRENT_GATING,
                True,
                "recurrent gating",
            ),
        },
        ExperimentOptions.RECURRENT_HALTING: {
            "recurrent_flag": _lock(
                ExperimentOptions.RECURRENT_HALTING,
                True,
                "recurrent execution",
            ),
            "recurrent_halting_flag": _lock(
                ExperimentOptions.RECURRENT_HALTING,
                True,
                "adaptive recurrent halting",
            ),
        },
        ExperimentOptions.RECURRENT_GATING_HALTING: {
            "recurrent_flag": _lock(
                ExperimentOptions.RECURRENT_GATING_HALTING,
                True,
                "recurrent execution",
            ),
            "recurrent_gate_flag": _lock(
                ExperimentOptions.RECURRENT_GATING_HALTING,
                True,
                "recurrent gating",
            ),
            "recurrent_halting_flag": _lock(
                ExperimentOptions.RECURRENT_GATING_HALTING,
                True,
                "adaptive recurrent halting",
            ),
        },
        ExperimentOptions.ADAPTIVE_SHARED_ROUTER: {
            "weight_option": _lock(
                ExperimentOptions.ADAPTIVE_SHARED_ROUTER,
                DualModelDynamicWeightConfig,
                "dual-model dynamic weights",
            ),
            "routing_initialization_mode": _lock(
                ExperimentOptions.ADAPTIVE_SHARED_ROUTER,
                RoutingInitializationMode.SHARED,
                "shared expert routing",
            ),
        },
        ExperimentOptions.ADAPTIVE_AFTER_WEIGHT: {
            "weight_option": _lock(
                ExperimentOptions.ADAPTIVE_AFTER_WEIGHT,
                DualModelDynamicWeightConfig,
                "dual-model dynamic weights",
            ),
            "weighting_position_option": _lock(
                ExperimentOptions.ADAPTIVE_AFTER_WEIGHT,
                ExpertWeightingPositionOptions.AFTER_EXPERTS,
                "expert weighting after experts",
            ),
        },
        ExperimentOptions.ADAPTIVE_TOP1_SWITCH: {
            "weight_option": _lock(
                ExperimentOptions.ADAPTIVE_TOP1_SWITCH,
                DualModelDynamicWeightConfig,
                "dual-model dynamic weights",
            ),
            "top_k": _lock(
                ExperimentOptions.ADAPTIVE_TOP1_SWITCH,
                1,
                "top-1 expert routing",
            ),
            "sampler_normalize_probabilities_flag": _lock(
                ExperimentOptions.ADAPTIVE_TOP1_SWITCH,
                False,
                "switch-style routing probabilities",
            ),
            "sampler_switch_loss_weight": _lock(
                ExperimentOptions.ADAPTIVE_TOP1_SWITCH,
                0.1,
                "switch auxiliary loss",
            ),
        },
        ExperimentOptions.ADAPTIVE_FULL_SHARED: {
            "weight_option": _lock(
                ExperimentOptions.ADAPTIVE_FULL_SHARED,
                DualModelDynamicWeightConfig,
                "dual-model dynamic weights",
            ),
            "bias_option": _lock(
                ExperimentOptions.ADAPTIVE_FULL_SHARED,
                AdditiveDynamicBiasConfig,
                "additive dynamic bias",
            ),
            "diagonal_option": _lock(
                ExperimentOptions.ADAPTIVE_FULL_SHARED,
                CombinedDynamicDiagonalConfig,
                "combined dynamic diagonal",
            ),
            "row_mask_option": _lock(
                ExperimentOptions.ADAPTIVE_FULL_SHARED,
                WeightInformedScoreAxisMaskConfig,
                "weight-informed row masking",
            ),
            "routing_initialization_mode": _lock(
                ExperimentOptions.ADAPTIVE_FULL_SHARED,
                RoutingInitializationMode.SHARED,
                "shared expert routing",
            ),
        },
        ExperimentOptions.ADAPTIVE_FULL_CAPACITY: {
            "weight_option": _lock(
                ExperimentOptions.ADAPTIVE_FULL_CAPACITY,
                DualModelDynamicWeightConfig,
                "dual-model dynamic weights",
            ),
            "bias_option": _lock(
                ExperimentOptions.ADAPTIVE_FULL_CAPACITY,
                AdditiveDynamicBiasConfig,
                "additive dynamic bias",
            ),
            "diagonal_option": _lock(
                ExperimentOptions.ADAPTIVE_FULL_CAPACITY,
                CombinedDynamicDiagonalConfig,
                "combined dynamic diagonal",
            ),
            "row_mask_option": _lock(
                ExperimentOptions.ADAPTIVE_FULL_CAPACITY,
                WeightInformedScoreAxisMaskConfig,
                "weight-informed row masking",
            ),
            "top_k": _lock(
                ExperimentOptions.ADAPTIVE_FULL_CAPACITY,
                1,
                "top-1 capacity routing",
            ),
            "capacity_factor": _lock(
                ExperimentOptions.ADAPTIVE_FULL_CAPACITY,
                1.0,
                "expert capacity limiting",
            ),
            "dropped_token_behavior": _lock(
                ExperimentOptions.ADAPTIVE_FULL_CAPACITY,
                DroppedTokenOptions.ZEROS,
                "zeroed dropped tokens",
            ),
            "sampler_normalize_probabilities_flag": _lock(
                ExperimentOptions.ADAPTIVE_FULL_CAPACITY,
                False,
                "switch-style routing probabilities",
            ),
        },
        ExperimentOptions.ADAPTIVE_BANK_ROUTER: {
            "weight_option": _lock(
                ExperimentOptions.ADAPTIVE_BANK_ROUTER,
                LayeredWeightedBankDynamicWeightConfig,
                "layered weighted-bank dynamic weights",
            ),
            "routing_initialization_mode": _lock(
                ExperimentOptions.ADAPTIVE_BANK_ROUTER,
                RoutingInitializationMode.SHARED,
                "shared expert routing",
            ),
        },
    }

    def __init__(self) -> None:
        super().__init__()

    def get_config(
        self,
        model_config_options: ExperimentOptions = ExperimentOptions.BASELINE,
        dataset: type = Mnist,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
        search_keys: list[str] | None = None,
        config_overrides: dict | None = None,
        search_overrides: dict | None = None,
    ) -> list["ModelConfig"]:
        preset_callback = self._preset_callback_for_option(model_config_options)
        return self._create_preset_search_space_configs(
            dataset,
            search_mode,
            preset_callback,
            search_keys,
            config_overrides=config_overrides,
            search_overrides=search_overrides,
        )

    def _preset_callback_for_option(self, option: ExperimentOptions):
        callbacks = {
            ExperimentOptions.BASELINE: self._baseline_preset,
            ExperimentOptions.GATING: self._gating_preset,
            ExperimentOptions.HALTING: self._halting_preset,
            ExperimentOptions.GATING_HALTING: self._gating_halting_preset,
            ExperimentOptions.RECURRENT: self._recurrent_preset,
            ExperimentOptions.RECURRENT_GATING: self._recurrent_gating_preset,
            ExperimentOptions.RECURRENT_HALTING: self._recurrent_halting_preset,
            ExperimentOptions.RECURRENT_GATING_HALTING: self._recurrent_gating_halting_preset,
            ExperimentOptions.ADAPTIVE_SHARED_ROUTER: self._adaptive_shared_router_preset,
            ExperimentOptions.ADAPTIVE_AFTER_WEIGHT: self._adaptive_after_weight_preset,
            ExperimentOptions.ADAPTIVE_TOP1_SWITCH: self._adaptive_top1_switch_preset,
            ExperimentOptions.ADAPTIVE_FULL_SHARED: self._adaptive_full_shared_preset,
            ExperimentOptions.ADAPTIVE_FULL_CAPACITY: self._adaptive_full_capacity_preset,
            ExperimentOptions.ADAPTIVE_BANK_ROUTER: self._adaptive_bank_router_preset,
        }
        if option not in callbacks:
            raise ValueError(
                "The specified option is not supported. Please choose a valid `ExperimentOptions`."
            )
        return callbacks[option]

    def _baseline_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**kwargs)

    def _gating_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"stack_gate_flag": True, **kwargs})

    def _halting_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"stack_halting_flag": True, **kwargs})

    def _gating_halting_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{"stack_gate_flag": True, "stack_halting_flag": True, **kwargs}
        )

    def _recurrent_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"recurrent_flag": True, **kwargs})

    def _recurrent_gating_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "recurrent_flag": True,
                "recurrent_gate_flag": True,
                **kwargs,
            },
        )

    def _recurrent_halting_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "recurrent_flag": True,
                "recurrent_halting_flag": True,
                **kwargs,
            },
        )

    def _recurrent_gating_halting_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "recurrent_flag": True,
                "recurrent_gate_flag": True,
                "recurrent_halting_flag": True,
                **kwargs,
            },
        )

    def _full_adaptive_kwargs(self) -> dict:
        return {
            "weight_option": DualModelDynamicWeightConfig,
            "bias_option": AdditiveDynamicBiasConfig,
            "diagonal_option": CombinedDynamicDiagonalConfig,
            "row_mask_option": WeightInformedScoreAxisMaskConfig,
        }

    def _adaptive_shared_router_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "weight_option": DualModelDynamicWeightConfig,
                "routing_initialization_mode": RoutingInitializationMode.SHARED,
                **kwargs,
            },
        )

    def _adaptive_after_weight_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "weight_option": DualModelDynamicWeightConfig,
                "weighting_position_option": ExpertWeightingPositionOptions.AFTER_EXPERTS,
                **kwargs,
            },
        )

    def _adaptive_top1_switch_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "weight_option": DualModelDynamicWeightConfig,
                "top_k": 1,
                "sampler_normalize_probabilities_flag": False,
                "sampler_switch_loss_weight": 0.1,
                **kwargs,
            },
        )

    def _adaptive_full_shared_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                **self._full_adaptive_kwargs(),
                "routing_initialization_mode": RoutingInitializationMode.SHARED,
                **kwargs,
            },
        )

    def _adaptive_full_capacity_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                **self._full_adaptive_kwargs(),
                "top_k": 1,
                "capacity_factor": 1.0,
                "dropped_token_behavior": DroppedTokenOptions.ZEROS,
                "sampler_normalize_probabilities_flag": False,
                **kwargs,
            },
        )

    def _adaptive_bank_router_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "weight_option": LayeredWeightedBankDynamicWeightConfig,
                "routing_initialization_mode": RoutingInitializationMode.SHARED,
                **kwargs,
            },
        )

    def _preset(self, **kwargs) -> "ModelConfig":
        return ExpertsLinearAdaptiveConfigBuilder(**kwargs).build()


class Experiment(ExperimentBase):
    def __init__(
        self,
        experiment_option: ExperimentOptions | None = None,
    ) -> None:
        super().__init__(experiment_option)

    def _num_epochs(self) -> int:
        return config.NUM_EPOCHS

    def _dataset_options(self) -> list:
        return config.DATASET_OPTIONS

    def _model_type(self) -> type:
        return Model

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        return ExperimentPresets()

    def _experiment_enumeration(self) -> type[BaseOptions]:
        return ExperimentOptions
