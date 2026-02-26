import torch

from torch import Tensor
from models.parser import get_parser
from dataclasses import dataclass, field
from Emperor.base.enums import BaseOptions
from Emperor.datasets.image.mnist import Mnist
from Emperor.base.utils import ConfigBase
from Emperor.linears.utils.layers import LinearLayerConfig
from Emperor.base.layer import LayerStack, LayerStackConfig
from Emperor.experiments.utils.classifier import ClassifierExperiment
from Emperor.base.enums import ActivationOptions, LayerNormPositionOptions
from Emperor.experiments.utils.base import (
    ExperimentBase,
    ExperimentPresetsBase,
    create_search_space,
)
from Emperor.behaviours.model import AdaptiveParameterBehaviourConfig
from Emperor.behaviours.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


@dataclass
class ExperimentConfig(ConfigBase):
    model_config: "LayerStackConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )


class Model(ClassifierExperiment):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__(cfg)
        self.main_cfg: ExperimentConfig = self._resolve_main_config(self.cfg, cfg)
        self.model_config: LayerStackConfig = self.main_cfg.model_config
        self.model = LayerStack(self.model_config).build_model()

    def _resolve_main_config(
        self, sub_config: "ConfigBase", main_cfg: "ConfigBase"
    ) -> None:
        if sub_config.override_config is not None:
            return sub_config.override_config
        return main_cfg

    def forward(
        self,
        X: Tensor,
    ) -> Tensor:
        X = X.to(self.device)
        X = torch.flatten(X, start_dim=1)
        X = self.model(X)
        return X


class ExperimentOptions(BaseOptions):
    BASE = 0


class Experiment(ExperimentBase):
    def __init__(
        self,
        experiment_option: ExperimentOptions | None = None,
    ) -> None:
        super().__init__(experiment_option)
        self.accelerator = "cpu"

    def _num_epochs(self) -> int:
        return 20

    def _model_type(self) -> type:
        return Model

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        return ExperimentPresets()

    def _experiment_enumeration(self) -> type[BaseOptions]:
        return ExperimentOptions


class ExperimentPresets(ExperimentPresetsBase):
    def get_config(
        self,
        model_config_options: ExperimentOptions = ExperimentOptions.BASE,
        dataset: type = Mnist,
    ) -> list["ModelConfig"]:
        match model_config_options:
            case ExperimentOptions.BASE:
                return self.__base_grid_search_config(dataset)
            case _:
                raise ValueError(
                    "The specified option is not supported. Please choose a valid `ExperimentOptions`."
                )

    def __base_grid_search_config(
        self,
        dataset: type = Mnist,
        num_random_search_samples: int | None = None,
    ) -> list["ModelConfig"]:
        base_config = {
            "input_dim": dataset.flattened_input_dim,
            "output_dim": dataset.num_classes,
        }

        search_space = {
            "hidden_dim": [128, 256],
            "stack_num_layers": [3, 6],
            "stack_dropout_probability": [0.0, 0.1],
            "stack_activation": [ActivationOptions.RELU, ActivationOptions.SILU],
        }

        return create_search_space(
            self.__preset, base_config, search_space, num_random_search_samples
        )

    def __preset(
        self,
        batch_size: int = 64,
        input_dim: int = 28**2,
        hidden_dim: int = 256,
        output_dim: int = 10,
        bias_flag: bool = True,
        layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DEFAULT,
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_TWO,
        diagonal_option: DynamicDiagonalOptions = DynamicDiagonalOptions.DIAGONAL,
        bias_option: DynamicBiasOptions = DynamicBiasOptions.DYNAMIC_PARAMETERS,
        memory_option: LinearMemoryOptions = LinearMemoryOptions.DISABLED,
        memory_size_option: LinearMemorySizeOptions = LinearMemorySizeOptions.DISABLED,
        memory_position_option: LinearMemoryPositionOptions = LinearMemoryPositionOptions.BEFORE_AFFINE,
        stack_num_layers: int = 3,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_flag: bool = False,
        stack_dropout_probability: float = 0.0,
        adaptive_generator_stack_num_layers: int = 2,
        adaptive_generator_stack_hidden_dim: int = 256,
        adaptive_generator_stack_activation: ActivationOptions = ActivationOptions.RELU,
        adaptive_generator_stack_residual_flag: bool = False,
        adaptive_generator_stack_dropout_probability: float = 0.0,
    ) -> "ModelConfig":
        from Emperor.config import ModelConfig
        from Emperor.linears.options import LinearLayerOptions

        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            override_config=ExperimentConfig(
                model_config=LayerStackConfig(
                    model_type=LinearLayerOptions.ADAPTIVE,
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
                        data_monitor=None,
                        parameter_monitor=None,
                        override_config=AdaptiveParameterBehaviourConfig(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            generator_depth=generator_depth,
                            diagonal_option=diagonal_option,
                            bias_option=bias_option,
                            memory_option=memory_option,
                            memory_size_option=memory_size_option,
                            memory_position_option=memory_position_option,
                            override_config=LayerStackConfig(
                                model_type=LinearLayerOptions.BASE,
                                input_dim=input_dim,
                                hidden_dim=adaptive_generator_stack_hidden_dim,
                                output_dim=output_dim,
                                num_layers=adaptive_generator_stack_num_layers,
                                activation=adaptive_generator_stack_activation,
                                layer_norm_position=layer_norm_position,
                                residual_flag=adaptive_generator_stack_residual_flag,
                                adaptive_computation_flag=False,
                                dropout_probability=adaptive_generator_stack_dropout_probability,
                                override_config=LinearLayerConfig(
                                    input_dim=input_dim,
                                    output_dim=output_dim,
                                    bias_flag=bias_flag,
                                    data_monitor=None,
                                    parameter_monitor=None,
                                    override_config=AdaptiveParameterBehaviourConfig(
                                        generator_depth=generator_depth,
                                    ),
                                ),
                            ),
                        ),
                    ),
                )
            ),
        )


if __name__ == "__main__":
    parser = get_parser(ExperimentOptions.names())
    args = parser.parse_args()
    config_option = ExperimentOptions.get_option(args.config_name)

    experiment = Experiment(config_option)
    experiment.train_model()
