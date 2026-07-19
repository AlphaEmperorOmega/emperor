from emperor.layers import (
    LayerStackConfig,
    RecurrentLayerConfig,
)
from models.bert.expert_linear._gate_config_factory import GateConfigFactory
from models.bert.expert_linear._halting_config_factory import HaltingConfigFactory
from models.bert.expert_linear.runtime_options import RecurrentControllerOptions


class RecurrentConfigFactory:
    def __init__(
        self,
        *,
        recurrent_controller_options: RecurrentControllerOptions,
        gate_config_factory: GateConfigFactory,
        halting_config_factory: HaltingConfigFactory,
    ) -> None:
        self.recurrent_controller_options = recurrent_controller_options
        self.gate_config_factory = gate_config_factory
        self.halting_config_factory = halting_config_factory

    def build_config(
        self,
        block_config: LayerStackConfig,
        *,
        input_dim: int | None = None,
        output_dim: int | None = None,
    ) -> LayerStackConfig | RecurrentLayerConfig:
        if not self.recurrent_controller_options.recurrent_flag:
            return block_config
        gate_config = self.gate_config_factory.build_recurrent_gate_config()
        halting_config = self.halting_config_factory.build_recurrent_halting_config()
        return RecurrentLayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            max_steps=self.recurrent_controller_options.recurrent_max_steps,
            recurrent_layer_norm_position=(
                self.recurrent_controller_options.recurrent_layer_norm_position
            ),
            block_config=block_config,
            gate_config=gate_config,
            residual_config=None,
            halting_config=halting_config,
        )
