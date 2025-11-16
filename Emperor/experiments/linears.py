from Emperor.linears.options import LinearLayerOptions
from Emperor.experiments.utils.factories import Experiments
from Emperor.linears.utils.layers import DynamicLinearLayerConfig, LinearLayerConfig
from Emperor.linears.utils.monitors import DataMonitor, ParameterMonitor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class LinearsExperiments(Experiments):
    def __init__(
        self,
        mini_datasetset_flag: bool = True,
    ) -> None:
        super().__init__(mini_datasetset_flag)

    def train_model(self, layer_type: LinearLayerOptions):
        preset = LinearsBasePreset(layer_type).get_config()
        self._set_model_config(preset)
        self._train_model(layer_type)

    def train_base_model(self):
        self.train_model(LinearLayerOptions.BASE)

    def train_dynamic_model(self):
        self.train_model(LinearLayerOptions.DYNAMIC)

    def test_all_linear_types(self):
        for layer_type in LinearLayerOptions:
            self.train_model(layer_type)


class LinearsBasePreset:
    def __init__(
        self,
        linear_layer_options: LinearLayerOptions,
    ) -> None:
        self.linear_layer_options = linear_layer_options

    def get_config(self) -> "ModelConfig":
        match self.linear_layer_options:
            case LinearLayerOptions.BASE:
                return self.base_preset()
            case LinearLayerOptions.DYNAMIC:
                return self.dynamic_preset()

    def base_preset(
        self,
        batch_size=64,
        input_dim=784,
        output_dim=10,
        bias_flag=True,
    ) -> "ModelConfig":
        from Emperor.config import ModelConfig

        return ModelConfig(
            batch_size=batch_size,
            linear_layer_config=LinearLayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=bias_flag,
                data_monitor=DataMonitor,
                parameter_monitor=ParameterMonitor,
            ),
        )

    def dynamic_preset(
        self,
        batch_size=64,
        input_dim=784,
        output_dim=10,
        bias_flag=True,
    ) -> "ModelConfig":
        from Emperor.config import ModelConfig

        return ModelConfig(
            batch_size=batch_size,
            linear_layer_config=DynamicLinearLayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=bias_flag,
                data_monitor=DataMonitor,
                parameter_monitor=ParameterMonitor,
            ),
        )
