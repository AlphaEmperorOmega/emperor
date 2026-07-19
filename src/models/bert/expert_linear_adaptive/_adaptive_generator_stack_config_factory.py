from emperor.layers import LayerConfig, LayerStackConfig, ResidualConfig
from emperor.linears import LinearLayerConfig
from models.bert.expert_linear_adaptive.runtime_options import (
    AdaptiveGeneratorStackOptions,
    AdaptiveGeneratorStackSource,
)


class AdaptiveGeneratorStackConfigFactory:
    def __init__(self, shared_options: AdaptiveGeneratorStackOptions) -> None:
        self.shared_options = shared_options

    def build_shared_config(self) -> LayerStackConfig:
        return self.__build_config_from_options(self.shared_options)

    def build_config_from_source(
        self,
        source: AdaptiveGeneratorStackSource,
    ) -> LayerStackConfig | None:
        options = self.__resolve_options(source)
        return None if options is None else self.__build_config_from_options(options)

    def __resolve_options(
        self,
        source: AdaptiveGeneratorStackSource,
    ) -> AdaptiveGeneratorStackOptions | None:
        if not source.independent_flag:
            return None
        defaults = self.shared_options
        return AdaptiveGeneratorStackOptions(
            hidden_dim=self.__resolve(source.hidden_dim, defaults.hidden_dim),
            layer_norm_position=self.__resolve(
                source.layer_norm_position, defaults.layer_norm_position
            ),
            num_layers=self.__resolve(source.num_layers, defaults.num_layers),
            activation=self.__resolve(source.activation, defaults.activation),
            residual_connection_option=self.__resolve(
                source.residual_connection_option,
                defaults.residual_connection_option,
            ),
            dropout_probability=self.__resolve(
                source.dropout_probability, defaults.dropout_probability
            ),
            last_layer_bias_option=self.__resolve(
                source.last_layer_bias_option, defaults.last_layer_bias_option
            ),
            apply_output_pipeline_flag=self.__resolve(
                source.apply_output_pipeline_flag,
                defaults.apply_output_pipeline_flag,
            ),
            bias_flag=self.__resolve(source.bias_flag, defaults.bias_flag),
        )

    @staticmethod
    def __resolve(override, shared_default):
        return shared_default if override is None else override

    @staticmethod
    def __build_config_from_options(
        options: AdaptiveGeneratorStackOptions,
    ) -> LayerStackConfig:
        return LayerStackConfig(
            hidden_dim=options.hidden_dim,
            num_layers=options.num_layers,
            last_layer_bias_option=options.last_layer_bias_option,
            apply_output_pipeline_flag=options.apply_output_pipeline_flag,
            layer_config=LayerConfig(
                activation=options.activation,
                layer_norm_position=options.layer_norm_position,
                residual_config=None
                if options.residual_connection_option is None
                else ResidualConfig(option=options.residual_connection_option),
                dropout_probability=options.dropout_probability,
                gate_config=None,
                halting_config=None,
                layer_model_config=LinearLayerConfig(bias_flag=options.bias_flag),
            ),
        )
