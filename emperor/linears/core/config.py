from typing import cast
from dataclasses import dataclass
from emperor.base.utils import ConfigBase, optional_field
from emperor.linears.options import LinearOptions

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.base.utils import Module
    from emperor.augmentations.adaptive_parameters.config import (
        AdaptiveParameterAugmentationConfig,
    )


@dataclass
class LinearLayerConfig(ConfigBase):
    input_dim: int | None = optional_field(
        "Number of input features for the linear transformation"
    )
    output_dim: int | None = optional_field(
        "Number of output features produced by the linear transformation"
    )
    bias_flag: bool | None = optional_field(
        "When true a learnable bias vector is added to the output after the linear transformation"
    )
    model_type: LinearOptions | None = optional_field(
        "Selects the linear layer variant for registry-based dispatch"
    )
    adaptive_augmentation_config: "AdaptiveParameterAugmentationConfig | None" = optional_field(
        "Config for input-dependent parameter augmentations applied to the linear layer"
    )

    def build(
        self,
        overrides: "LinearLayerConfig | None" = None,
    ) -> "Module":
        from emperor.linears.core.layers import LinearAbstract

        if self.model_type is None:
            raise ValueError("`model_type` must be set before building the layer")
        layer_cls = LinearAbstract.resolve(self.model_type)
        return layer_cls(self, cast("LinearLayerConfig | None", overrides))
