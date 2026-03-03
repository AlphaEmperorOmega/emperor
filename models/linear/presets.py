from emperor.base.enums import BaseOptions, ActivationOptions, LayerNormPositionOptions
from emperor.datasets.image.mnist import Mnist
from emperor.linears.utils.layers import LinearLayerConfig
from emperor.base.layer import LayerStackConfig
from emperor.experiments.base import ExperimentPresetsBase
from models.linear.config import (
    ExperimentConfig,
    BATCH_SIZE,
    LEARNING_RATE,
    INPUT_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
    BIAS_FLAG,
    LAYER_NORM_POSITION,
    STACK_NUM_LAYERS,
    STACK_ACTIVATION,
    STACK_RESIDUAL_FLAG,
    STACK_DROPOUT_PROBABILITY,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExperimentOptions(BaseOptions):
    DEFAULT = 0
    BASE = 1


class ExperimentPresets(ExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__()

    def get_config(
        self,
        model_config_options: ExperimentOptions = ExperimentOptions.DEFAULT,
        dataset: type = Mnist,
        num_samples: int | None = None,
    ) -> list["ModelConfig"]:
        match model_config_options:
            case ExperimentOptions.DEFAULT:
                return self._default_config(dataset)
            case ExperimentOptions.BASE:
                return self._create_search_space_configs(dataset, num_samples)
            case _:
                raise ValueError(
                    "The specified option is not supported. Please choose a valid `LinearExperimentOptions`."
                )

    def _preset(
        self,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        output_dim: int = OUTPUT_DIM,
        bias_flag: bool = BIAS_FLAG,
        layer_norm_position: LayerNormPositionOptions = LAYER_NORM_POSITION,
        stack_num_layers: int = STACK_NUM_LAYERS,
        stack_activation: ActivationOptions = STACK_ACTIVATION,
        stack_residual_flag: bool = STACK_RESIDUAL_FLAG,
        stack_dropout_probability: float = STACK_DROPOUT_PROBABILITY,
    ) -> "ModelConfig":
        from emperor.config import ModelConfig
        from emperor.linears.options import LinearLayerOptions

        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            override_config=ExperimentConfig(
                model_config=LayerStackConfig(
                    model_type=LinearLayerOptions.BASE,
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_layers=stack_num_layers,
                    activation=stack_activation,
                    layer_norm_position=layer_norm_position,
                    residual_flag=stack_residual_flag,
                    adaptive_computation_flag=False,
                    dropout_probability=stack_dropout_probability,
                    override_config=LinearLayerConfig(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        bias_flag=bias_flag,
                    ),
                )
            ),
        )
