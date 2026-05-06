import models.linear.config as config

from models.linear.config_builder import LinearConfigBuilder
from models.linear.model import Model
from emperor.experiments.base import SearchMode, create_search_space
from emperor.datasets.image.classification.mnist import Mnist
from emperor.experiments.base import ExperimentBase, ExperimentPresetsBase
from emperor.base.enums import (
    BaseOptions,
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.halting.options import HaltingHiddenStateModeOptions

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExperimentOptions(BaseOptions):
    PRESET = "Baseline linear stack preset; supports search-space flags."
    GATING = "Linear stack with a learned gate applied to hidden-layer outputs."
    HALTING = "Linear stack with adaptive computation halting enabled."
    GATING_HALTING = (
        "Linear stack with both learned gating and adaptive computation halting."
    )


class ExperimentPresets(ExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__()

    def get_config(
        self,
        model_config_options: ExperimentOptions = ExperimentOptions.PRESET,
        dataset: type = Mnist,
        search_mode: SearchMode = None,
    ) -> list["ModelConfig"]:
        match model_config_options:
            case ExperimentOptions.PRESET:
                return self._create_default_search_space_configs(dataset, search_mode)
            case ExperimentOptions.GATING:
                return self._create_search_space_configs(
                    dataset, search_mode, self._gating_preset
                )
            case ExperimentOptions.HALTING:
                return self._create_search_space_configs(
                    dataset, search_mode, self._halting_preset
                )
            case ExperimentOptions.GATING_HALTING:
                return self._create_search_space_configs(
                    dataset, search_mode, self._gating_halting_preset
                )
            case _:
                raise ValueError(
                    "The specified option is not supported. Please choose a valid `LinearExperimentOptions`."
                )

    def _create_search_space_configs(
        self,
        dataset: type,
        search_mode: SearchMode,
        preset_callback,
    ) -> list["ModelConfig"]:
        return create_search_space(
            preset_callback,
            self._dataset_config(dataset),
            self._extract_search_space_from_config(search_mode),
            search_mode,
        )

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
        gate_hidden_dim: int = config.GATE_HIDDEN_DIM,
        gate_layer_norm_position: LayerNormPositionOptions = config.GATE_LAYER_NORM_POSITION,
        gate_stack_num_layers: int = config.GATE_STACK_NUM_LAYERS,
        gate_stack_activation: ActivationOptions = config.GATE_STACK_ACTIVATION,
        gate_stack_residual_flag: bool = config.GATE_STACK_RESIDUAL_FLAG,
        gate_stack_dropout_probability: float = config.GATE_STACK_DROPOUT_PROBABILITY,
        gate_stack_last_layer_bias_option: LastLayerBiasOptions = config.GATE_STACK_LAST_LAYER_BIAS_OPTION,
        gate_stack_apply_output_pipeline_flag: bool = config.GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        gate_bias_flag: bool = config.GATE_BIAS_FLAG,
        stack_halting_flag: bool = config.HALTING_FLAG,
        halting_threshold: float = config.HALTING_THRESHOLD,
        halting_dropout: float = config.HALTING_DROPOUT,
        halting_hidden_state_mode: HaltingHiddenStateModeOptions = config.HALTING_HIDDEN_STATE_MODE,
        halting_gate_hidden_dim: int = config.HALTING_GATE_HIDDEN_DIM,
        halting_gate_output_dim: int = config.HALTING_GATE_OUTPUT_DIM,
        halting_gate_layer_norm_position: LayerNormPositionOptions = config.HALTING_GATE_LAYER_NORM_POSITION,
        halting_gate_stack_num_layers: int = config.HALTING_GATE_STACK_NUM_LAYERS,
        halting_gate_stack_activation: ActivationOptions = config.HALTING_GATE_STACK_ACTIVATION,
        halting_gate_stack_residual_flag: bool = config.HALTING_GATE_STACK_RESIDUAL_FLAG,
        halting_gate_stack_dropout_probability: float = config.HALTING_GATE_STACK_DROPOUT_PROBABILITY,
        halting_gate_stack_last_layer_bias_option: LastLayerBiasOptions = config.HALTING_GATE_STACK_LAST_LAYER_BIAS_OPTION,
        halting_gate_stack_apply_output_pipeline_flag: bool = config.HALTING_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        halting_gate_bias_flag: bool = config.HALTING_GATE_BIAS_FLAG,
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
            gate_hidden_dim=gate_hidden_dim,
            gate_layer_norm_position=gate_layer_norm_position,
            gate_stack_num_layers=gate_stack_num_layers,
            gate_stack_activation=gate_stack_activation,
            gate_stack_residual_flag=gate_stack_residual_flag,
            gate_stack_dropout_probability=gate_stack_dropout_probability,
            gate_stack_last_layer_bias_option=gate_stack_last_layer_bias_option,
            gate_stack_apply_output_pipeline_flag=gate_stack_apply_output_pipeline_flag,
            gate_bias_flag=gate_bias_flag,
            stack_halting_flag=stack_halting_flag,
            halting_threshold=halting_threshold,
            halting_dropout=halting_dropout,
            halting_hidden_state_mode=halting_hidden_state_mode,
            halting_gate_hidden_dim=halting_gate_hidden_dim,
            halting_gate_output_dim=halting_gate_output_dim,
            halting_gate_layer_norm_position=halting_gate_layer_norm_position,
            halting_gate_stack_num_layers=halting_gate_stack_num_layers,
            halting_gate_stack_activation=halting_gate_stack_activation,
            halting_gate_stack_residual_flag=halting_gate_stack_residual_flag,
            halting_gate_stack_dropout_probability=halting_gate_stack_dropout_probability,
            halting_gate_stack_last_layer_bias_option=halting_gate_stack_last_layer_bias_option,
            halting_gate_stack_apply_output_pipeline_flag=halting_gate_stack_apply_output_pipeline_flag,
            halting_gate_bias_flag=halting_gate_bias_flag,
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
