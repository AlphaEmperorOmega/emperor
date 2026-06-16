import models.linears.linear.config as config

from models.linears.linear.config_builder import LinearConfigBuilder
from models.linears.linear.model import Model
from emperor.experiments.base import SearchMode
from emperor.datasets.image.classification.mnist import Mnist
from emperor.experiments.base import ExperimentBase, ExperimentPresetsBase, PresetLock
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import BaseOptions, LayerNormPositionOptions

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


def _lock(option, value, behavior: str) -> PresetLock:
    return PresetLock(
        value=value,
        reason=(
            f"Locked by the {option.name} preset because this preset enables "
            f"{behavior}."
        ),
    )


class ExperimentOptions(BaseOptions):
    BASELINE = "Baseline linear stack preset; supports search-space flags."
    GATING = "Linear stack with a learned gate applied to hidden-layer outputs."
    HALTING = "Linear stack with adaptive computation halting enabled."
    GATING_HALTING = (
        "Linear stack with both learned gating and adaptive computation halting."
    )
    RESIDUAL = "Linear stack with residual (skip) connections on hidden layers."
    POST_NORM = "Linear stack with layer norm applied after each layer (post-norm)."
    RESIDUAL_POST_NORM = (
        "Linear stack with residual connections and post-layer normalization."
    )
    RESIDUAL_GATING = "Linear stack with residual connections and learned gating."
    RESIDUAL_HALTING = (
        "Linear stack with residual connections and adaptive computation halting."
    )
    RECURRENT = "Linear stack applied recurrently for a fixed number of steps."
    RECURRENT_GATING = "Linear stack applied recurrently with a learned recurrent gate."
    RECURRENT_HALTING = (
        "Linear stack applied recurrently with adaptive recurrent halting."
    )
    RECURRENT_GATING_HALTING = (
        "Linear stack applied recurrently with both learned recurrent gating and "
        "adaptive recurrent halting."
    )
    RECURRENT_RESIDUAL = "Residual linear stack applied recurrently."
    RECURRENT_POST_NORM = "Post-normalized linear stack applied recurrently."


class ExperimentPresets(ExperimentPresetsBase):
    PRESET_LOCKS = {
        ExperimentOptions.GATING: {
            "stack_gate_flag": _lock(ExperimentOptions.GATING, True, "stack gating"),
        },
        ExperimentOptions.HALTING: {
            "stack_halting_flag": _lock(
                ExperimentOptions.HALTING, True, "adaptive stack halting"
            ),
        },
        ExperimentOptions.GATING_HALTING: {
            "stack_gate_flag": _lock(
                ExperimentOptions.GATING_HALTING, True, "stack gating"
            ),
            "stack_halting_flag": _lock(
                ExperimentOptions.GATING_HALTING,
                True,
                "adaptive stack halting",
            ),
        },
        ExperimentOptions.RESIDUAL: {
            "stack_residual_connection_option": _lock(
                ExperimentOptions.RESIDUAL,
                ResidualConnectionOptions.RESIDUAL,
                "stack residuals",
            ),
        },
        ExperimentOptions.POST_NORM: {
            "layer_norm_position": _lock(
                ExperimentOptions.POST_NORM,
                LayerNormPositionOptions.AFTER,
                "post-layer normalization",
            ),
        },
        ExperimentOptions.RESIDUAL_POST_NORM: {
            "stack_residual_connection_option": _lock(
                ExperimentOptions.RESIDUAL_POST_NORM,
                ResidualConnectionOptions.RESIDUAL,
                "stack residuals",
            ),
            "layer_norm_position": _lock(
                ExperimentOptions.RESIDUAL_POST_NORM,
                LayerNormPositionOptions.AFTER,
                "post-layer normalization",
            ),
        },
        ExperimentOptions.RESIDUAL_GATING: {
            "stack_residual_connection_option": _lock(
                ExperimentOptions.RESIDUAL_GATING,
                ResidualConnectionOptions.RESIDUAL,
                "stack residuals",
            ),
            "stack_gate_flag": _lock(
                ExperimentOptions.RESIDUAL_GATING,
                True,
                "stack gating",
            ),
        },
        ExperimentOptions.RESIDUAL_HALTING: {
            "stack_residual_connection_option": _lock(
                ExperimentOptions.RESIDUAL_HALTING,
                ResidualConnectionOptions.RESIDUAL,
                "stack residuals",
            ),
            "stack_halting_flag": _lock(
                ExperimentOptions.RESIDUAL_HALTING,
                True,
                "adaptive stack halting",
            ),
        },
        ExperimentOptions.RECURRENT: {
            "recurrent_flag": _lock(
                ExperimentOptions.RECURRENT, True, "recurrent execution"
            ),
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
        ExperimentOptions.RECURRENT_RESIDUAL: {
            "recurrent_flag": _lock(
                ExperimentOptions.RECURRENT_RESIDUAL,
                True,
                "recurrent execution",
            ),
            "stack_residual_connection_option": _lock(
                ExperimentOptions.RECURRENT_RESIDUAL,
                ResidualConnectionOptions.RESIDUAL,
                "stack residuals",
            ),
        },
        ExperimentOptions.RECURRENT_POST_NORM: {
            "recurrent_flag": _lock(
                ExperimentOptions.RECURRENT_POST_NORM,
                True,
                "recurrent execution",
            ),
            "layer_norm_position": _lock(
                ExperimentOptions.RECURRENT_POST_NORM,
                LayerNormPositionOptions.AFTER,
                "post-layer normalization",
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
            ExperimentOptions.RESIDUAL: self._residual_preset,
            ExperimentOptions.POST_NORM: self._post_norm_preset,
            ExperimentOptions.RESIDUAL_POST_NORM: self._residual_post_norm_preset,
            ExperimentOptions.RESIDUAL_GATING: self._residual_gating_preset,
            ExperimentOptions.RESIDUAL_HALTING: self._residual_halting_preset,
            ExperimentOptions.RECURRENT: self._recurrent_preset,
            ExperimentOptions.RECURRENT_GATING: self._recurrent_gating_preset,
            ExperimentOptions.RECURRENT_HALTING: self._recurrent_halting_preset,
            ExperimentOptions.RECURRENT_GATING_HALTING: self._recurrent_gating_halting_preset,
            ExperimentOptions.RECURRENT_RESIDUAL: self._recurrent_residual_preset,
            ExperimentOptions.RECURRENT_POST_NORM: self._recurrent_post_norm_preset,
        }
        if option not in callbacks:
            raise ValueError(
                "The specified option is not supported. Please choose a valid `ExperimentOptions`."
            )
        return callbacks[option]

    def _baseline_preset(
        self,
        **kwargs,
    ) -> "ModelConfig":
        return self._preset(**kwargs)

    def _gating_preset(
        self,
        **kwargs,
    ) -> "ModelConfig":
        return self._preset(**{"stack_gate_flag": True, **kwargs})

    def _halting_preset(
        self,
        **kwargs,
    ) -> "ModelConfig":
        return self._preset(**{"stack_halting_flag": True, **kwargs})

    def _gating_halting_preset(
        self,
        **kwargs,
    ) -> "ModelConfig":
        return self._preset(
            **{
                "stack_gate_flag": True,
                "stack_halting_flag": True,
                **kwargs,
            },
        )

    def _residual_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
                **kwargs,
            }
        )

    def _post_norm_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{"layer_norm_position": LayerNormPositionOptions.AFTER, **kwargs}
        )

    def _residual_post_norm_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
                "layer_norm_position": LayerNormPositionOptions.AFTER,
                **kwargs,
            },
        )

    def _residual_gating_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
                "stack_gate_flag": True,
                **kwargs,
            },
        )

    def _residual_halting_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
                "stack_halting_flag": True,
                **kwargs,
            },
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

    def _recurrent_residual_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "recurrent_flag": True,
                "stack_residual_connection_option": ResidualConnectionOptions.RESIDUAL,
                **kwargs,
            },
        )

    def _recurrent_post_norm_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "recurrent_flag": True,
                "layer_norm_position": LayerNormPositionOptions.AFTER,
                **kwargs,
            },
        )

    def _preset(self, **kwargs) -> "ModelConfig":
        return LinearConfigBuilder(**kwargs).build()


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
