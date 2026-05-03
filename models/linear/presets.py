import models.linear.config as config

from models.linear.config_builder import LinearConfigBuilder
from models.linear.model import Model
from emperor.experiments.base import SearchMode
from emperor.datasets.image.classification.mnist import Mnist
from emperor.experiments.base import ExperimentBase, ExperimentPresetsBase
from emperor.base.enums import (
    BaseOptions,
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExperimentOptions(BaseOptions):
    PRESET = 0
    CONFIG = 1
    GATING = 2
    HALTING = 3
    GATING_HALTING = 4


class ExperimentPresets(ExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__()

    def get_config(
        self,
        model_config_options: ExperimentOptions = ExperimentOptions.PRESET,
        dataset: type = Mnist,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
    ) -> list["ModelConfig"]:
        match model_config_options:
            case ExperimentOptions.PRESET:
                return self._create_named_preset_configs(dataset, self._baseline_preset)
            case ExperimentOptions.CONFIG:
                return self._create_default_search_space_configs(
                    dataset, search_mode, log_folder
                )
            case ExperimentOptions.GATING:
                return self._create_named_preset_configs(dataset, self._gating_preset)
            case ExperimentOptions.HALTING:
                return self._create_named_preset_configs(dataset, self._halting_preset)
            case ExperimentOptions.GATING_HALTING:
                return self._create_named_preset_configs(
                    dataset, self._gating_halting_preset
                )
            case _:
                raise ValueError(
                    "The specified option is not supported. Please choose a valid `LinearExperimentOptions`."
                )

    def _create_named_preset_configs(
        self,
        dataset: type,
        preset_callback: Callable[..., "ModelConfig"],
    ) -> list["ModelConfig"]:
        return [preset_callback(**self._dataset_config(dataset))]

    def _baseline_preset(
        self,
        **kwargs,
    ) -> "ModelConfig":
        return self._preset(**kwargs)

    def _gating_preset(
        self,
        **kwargs,
    ) -> "ModelConfig":
        return self._preset(**kwargs, stack_gate_flag=True)

    def _halting_preset(
        self,
        **kwargs,
    ) -> "ModelConfig":
        return self._preset(**kwargs, stack_halting_flag=True)

    def _gating_halting_preset(
        self,
        **kwargs,
    ) -> "ModelConfig":
        return self._preset(
            **kwargs,
            stack_gate_flag=True,
            stack_halting_flag=True,
        )

    def _preset(
        self,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        output_dim: int = config.OUTPUT_DIM,
        bias_flag: bool = config.BIAS_FLAG,
        layer_norm_position: LayerNormPositionOptions = config.LAYER_NORM_POSITION,
        stack_num_layers: int = config.STACK_NUM_LAYERS,
        stack_activation: ActivationOptions = config.STACK_ACTIVATION,
        stack_residual_flag: bool = config.STACK_RESIDUAL_FLAG,
        stack_dropout_probability: float = config.STACK_DROPOUT_PROBABILITY,
        stack_last_layer_bias_option: LastLayerBiasOptions = config.STACK_LAST_LAYER_BIAS_OPTION,
        stack_apply_output_pipeline_flag: bool = config.STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        stack_gate_flag: bool = config.GATE_FLAG,
        stack_halting_flag: bool = config.HALTING_FLAG,
    ) -> "ModelConfig":
        return LinearConfigBuilder(
            batch_size=batch_size,
            learning_rate=learning_rate,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            bias_flag=bias_flag,
            layer_norm_position=layer_norm_position,
            stack_num_layers=stack_num_layers,
            stack_activation=stack_activation,
            stack_residual_flag=stack_residual_flag,
            stack_dropout_probability=stack_dropout_probability,
            stack_last_layer_bias_option=stack_last_layer_bias_option,
            stack_apply_output_pipeline_flag=stack_apply_output_pipeline_flag,
            stack_gate_flag=stack_gate_flag,
            stack_halting_flag=stack_halting_flag,
        ).build()


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
