import models.linear.config as config

from models.linear.model import Model
from emperor.experiments.base import SearchMode
from emperor.base.layer import LayerStackConfig
from models.linear.config import ExperimentConfig
from emperor.base.layer.config import LayerConfig
from emperor.halting.config import StickBreakingConfig
from emperor.linears.core.config import LinearLayerConfig
from emperor.datasets.image.classification.mnist import Mnist
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.experiments.base import ExperimentBase, ExperimentPresetsBase
from emperor.base.enums import (
    BaseOptions,
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExperimentOptions(BaseOptions):
    PRESET = 0
    CONFIG = 1


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
                return self._create_default_preset_configs(dataset)
            case ExperimentOptions.CONFIG:
                return self._create_default_search_space_configs(
                    dataset, search_mode, log_folder
                )
            case _:
                raise ValueError(
                    "The specified option is not supported. Please choose a valid `LinearExperimentOptions`."
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
    ) -> "ModelConfig":
        from emperor.config import ModelConfig

        gate_config = LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=stack_num_layers,
            last_layer_bias_option=stack_last_layer_bias_option,
            apply_output_pipeline_flag=stack_apply_output_pipeline_flag,
            layer_config=LayerConfig(
                activation=stack_activation,
                layer_norm_position=layer_norm_position,
                residual_flag=stack_residual_flag,
                dropout_probability=stack_dropout_probability,
                halting_config=None,
                shared_halting_flag=False,
                gate_config=None,
                layer_model_config=LinearLayerConfig(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    bias_flag=bias_flag,
                ),
            ),
        )

        halting_config = StickBreakingConfig(
            input_dim=output_dim,
            threshold=0.99,
            halting_dropout=0.0,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            halting_gate_config=LayerStackConfig(
                input_dim=output_dim,
                hidden_dim=output_dim,
                output_dim=2,
                num_layers=stack_num_layers,
                last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                apply_output_pipeline_flag=False,
                layer_config=LayerConfig(
                    activation=ActivationOptions.DISABLED,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_flag=stack_residual_flag,
                    dropout_probability=stack_dropout_probability,
                    halting_config=None,
                    shared_halting_flag=False,
                    gate_config=None,
                    layer_model_config=LinearLayerConfig(
                        input_dim=output_dim,
                        output_dim=output_dim,
                        bias_flag=True,
                    ),
                ),
            ),
        )

        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            model_config=ExperimentConfig(
                model_config=LayerStackConfig(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_layers=stack_num_layers,
                    last_layer_bias_option=stack_last_layer_bias_option,
                    apply_output_pipeline_flag=stack_apply_output_pipeline_flag,
                    layer_config=LayerConfig(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        activation=stack_activation,
                        layer_norm_position=layer_norm_position,
                        residual_flag=stack_residual_flag,
                        dropout_probability=stack_dropout_probability,
                        gate_config=gate_config,
                        halting_config=halting_config,
                        shared_halting_flag=False,
                        layer_model_config=LinearLayerConfig(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            bias_flag=bias_flag,
                        ),
                    ),
                )
            ),
        )


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
