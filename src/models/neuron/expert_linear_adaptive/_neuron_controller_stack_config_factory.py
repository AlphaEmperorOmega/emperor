from emperor.base.layer import LayerConfig, LayerStackConfig
from emperor.linears.core.config import LinearLayerConfig

from models.neuron.expert_linear_adaptive.runtime_options import (
    NeuronSubmoduleStackOptions,
)


class NeuronControllerStackConfigFactory:
    def build_config(
        self,
        options: NeuronSubmoduleStackOptions,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> LayerStackConfig:
        layer_config = self.__build_layer_config(options)
        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=options.num_layers,
            last_layer_bias_option=options.last_layer_bias_option,
            apply_output_pipeline_flag=options.apply_output_pipeline_flag,
            layer_config=layer_config,
        )

    @staticmethod
    def __build_layer_config(
        options: NeuronSubmoduleStackOptions,
    ) -> LayerConfig:
        return LayerConfig(
            activation=options.activation,
            residual_connection_option=options.residual_connection_option,
            dropout_probability=options.dropout_probability,
            layer_norm_position=options.layer_norm_position,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                bias_flag=options.bias_flag,
            ),
        )
