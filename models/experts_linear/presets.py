import models.experts_linear.config as config

from models.experts_linear.config_builder import ExpertsLinearConfigBuilder
from models.experts_linear.model import Model
from emperor.experiments.base import SearchMode
from emperor.datasets.image.classification.mnist import Mnist
from emperor.experiments.base import ExperimentBase, ExperimentPresetsBase
from emperor.base.options import BaseOptions, LayerNormPositionOptions
from emperor.experts.core.options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExperimentOptions(BaseOptions):
    BASELINE = "Baseline mixture-of-experts with linear expert and sampler stacks."
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
    SHARED_ROUTER_AFTER_WEIGHT = (
        "Mixture-of-experts model with shared routing and expert weighting after experts."
    )
    TOP1_SWITCH_AUX = (
        "Top-1 mixture-of-experts routing with switch auxiliary loss enabled."
    )
    TOP2_BALANCED_AUX = (
        "Top-2 mixture-of-experts routing with balance auxiliary loss enabled."
    )
    CAPACITY_TOP1_ZERO = (
        "Top-1 mixture-of-experts routing with capacity limiting and zeroed dropped tokens."
    )
    CAPACITY_TOP1_IDENTITY = (
        "Top-1 mixture-of-experts routing with capacity limiting and identity dropped tokens."
    )
    NOISY_SHARED_ROUTER = "Mixture-of-experts model with shared noisy top-k routing."
    RESIDUAL_SHARED_ROUTER = (
        "Mixture-of-experts model with residual outer layers and shared routing."
    )
    POST_NORM_AFTER_WEIGHT = (
        "Mixture-of-experts model with post-layer norm and expert weighting after experts."
    )


class ExperimentPresets(ExperimentPresetsBase):
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
            ExperimentOptions.SHARED_ROUTER_AFTER_WEIGHT: self._shared_router_after_weight_preset,
            ExperimentOptions.TOP1_SWITCH_AUX: self._top1_switch_aux_preset,
            ExperimentOptions.TOP2_BALANCED_AUX: self._top2_balanced_aux_preset,
            ExperimentOptions.CAPACITY_TOP1_ZERO: self._capacity_top1_zero_preset,
            ExperimentOptions.CAPACITY_TOP1_IDENTITY: self._capacity_top1_identity_preset,
            ExperimentOptions.NOISY_SHARED_ROUTER: self._noisy_shared_router_preset,
            ExperimentOptions.RESIDUAL_SHARED_ROUTER: self._residual_shared_router_preset,
            ExperimentOptions.POST_NORM_AFTER_WEIGHT: self._post_norm_after_weight_preset,
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

    def _shared_router_after_weight_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "routing_initialization_mode": RoutingInitializationMode.SHARED,
                "weighting_position_option": ExpertWeightingPositionOptions.AFTER_EXPERTS,
                **kwargs,
            },
        )

    def _top1_switch_aux_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "top_k": 1,
                "sampler_normalize_probabilities_flag": False,
                "sampler_switch_loss_weight": 0.1,
                **kwargs,
            },
        )

    def _top2_balanced_aux_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "top_k": 2,
                "sampler_coefficient_of_variation_loss_weight": 0.1,
                **kwargs,
            },
        )

    def _capacity_top1_zero_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "top_k": 1,
                "capacity_factor": 1.0,
                "dropped_token_behavior": DroppedTokenOptions.ZEROS,
                "sampler_normalize_probabilities_flag": False,
                **kwargs,
            },
        )

    def _capacity_top1_identity_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "top_k": 1,
                "capacity_factor": 1.0,
                "dropped_token_behavior": DroppedTokenOptions.IDENTITY,
                "sampler_normalize_probabilities_flag": False,
                **kwargs,
            },
        )

    def _noisy_shared_router_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "routing_initialization_mode": RoutingInitializationMode.SHARED,
                "sampler_noisy_topk_flag": True,
                "router_noisy_topk_flag": True,
                **kwargs,
            },
        )

    def _residual_shared_router_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "routing_initialization_mode": RoutingInitializationMode.SHARED,
                "stack_residual_flag": True,
                **kwargs,
            },
        )

    def _post_norm_after_weight_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "layer_norm_position": LayerNormPositionOptions.AFTER,
                "weighting_position_option": ExpertWeightingPositionOptions.AFTER_EXPERTS,
                **kwargs,
            },
        )

    def _preset(self, **kwargs) -> "ModelConfig":
        return ExpertsLinearConfigBuilder(**kwargs).build()


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
