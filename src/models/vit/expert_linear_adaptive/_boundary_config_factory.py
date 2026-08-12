from dataclasses import dataclass

import models.vit.expert_linear_adaptive.config as config
from emperor.linears import LinearLayerConfig
from models.vit.expert_linear_adaptive import _config_defaults as config_defaults
from models.vit.expert_linear_adaptive.runtime_options import VitOutputOptions


@dataclass(frozen=True)
class BoundaryConfigDependencies:
    hidden_dim: int
    output_dim: int
    output_options: VitOutputOptions | None


class BoundaryConfigFactory:
    def __init__(self, dependencies: BoundaryConfigDependencies) -> None:
        self.hidden_dim = dependencies.hidden_dim
        self.output_dim = dependencies.output_dim
        self.output_options = (
            config_defaults.vit_output_options(config)
            if dependencies.output_options is None
            else dependencies.output_options
        )

    def build_output_config(self) -> LinearLayerConfig:
        return LinearLayerConfig(
            input_dim=self.hidden_dim,
            output_dim=self.output_dim,
            bias_flag=self.output_options.bias_flag,
        )
