from emperor.base.layer.config import RecurrentLayerConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.utils import ConfigBase

from models.experts._builder_options import ExpertsRecurrentControllerOptions
from models.experts._gate_config_factory import ExpertsGateConfigFactory
from models.experts._halting_config_factory import ExpertsHaltingConfigFactory


class ExpertsRecurrentConfigFactory:
    def __init__(
        self,
        *,
        recurrent_controller_options: ExpertsRecurrentControllerOptions,
        gate_config_factory: ExpertsGateConfigFactory,
        halting_config_factory: ExpertsHaltingConfigFactory,
    ) -> None:
        self.recurrent_controller_options = recurrent_controller_options
        self.gate_config_factory = gate_config_factory
        self.halting_config_factory = halting_config_factory

    def build_config(
        self,
        block_config: ConfigBase,
    ) -> ConfigBase | RecurrentLayerConfig:
        if not self.recurrent_controller_options.recurrent_flag:
            return block_config
        return RecurrentLayerConfig(
            max_steps=self.recurrent_controller_options.recurrent_max_steps,
            recurrent_layer_norm_position=(
                self.recurrent_controller_options.recurrent_layer_norm_position
            ),
            block_config=block_config,
            gate_config=self.gate_config_factory.build_recurrent_gate_config(),
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            halting_config=(
                self.halting_config_factory.build_recurrent_halting_config()
            ),
        )
