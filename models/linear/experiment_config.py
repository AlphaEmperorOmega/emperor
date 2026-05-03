from dataclasses import dataclass, field

from emperor.base.utils import ConfigBase
from emperor.base.layer import LayerStackConfig
from emperor.base.layer.config import LayerConfig


@dataclass
class ExperimentConfig(ConfigBase):
    model_config: "LayerStackConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )
    output_model_config: "LayerConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )
