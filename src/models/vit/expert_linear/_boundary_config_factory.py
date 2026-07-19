from dataclasses import dataclass

from emperor.linears.core.config import LinearLayerConfig

import models.vit.expert_linear.config as config
from models.vit.expert_linear.runtime_options import VitOutputOptions


@dataclass(frozen=True)
class BoundaryConfigDependencies:
    hidden_dim: int
    output_dim: int
    output_options: VitOutputOptions | None


class BoundaryConfigFactory:
    def __init__(self, dependencies: BoundaryConfigDependencies) -> None:
        self.hidden_dim = dependencies.hidden_dim
        self.output_dim = dependencies.output_dim
        self.output_options = self.__default_output_options(dependencies.output_options)

    def __default_output_options(
        self,
        output_options: VitOutputOptions | None,
    ) -> VitOutputOptions:
        if output_options is not None:
            return output_options
        return VitOutputOptions(bias_flag=config.OUTPUT_BIAS_FLAG)

    def build_output_config(self) -> LinearLayerConfig:
        return LinearLayerConfig(
            input_dim=self.hidden_dim,
            output_dim=self.output_dim,
            bias_flag=self.output_options.bias_flag,
        )
