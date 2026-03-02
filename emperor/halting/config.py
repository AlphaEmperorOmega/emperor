from dataclasses import dataclass, field
from emperor.base.utils import ConfigBase
from emperor.halting.options import HaltingOptions


@dataclass
class HaltingConfig(ConfigBase):
    halting_option: HaltingOptions | None = field(
        default=None,
        metadata={"help": "Selects the halting strategy to use"},
    )
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Hidden dimension used to build the halting gate network"},
    )
    threshold: float | None = field(
        default=None,
        metadata={"help": "Halting probability threshold; tokens above this stop computing"},
    )
    halting_dropout: float | None = field(
        default=None,
        metadata={"help": "Dropout probability applied inside the soft halting gate network"},
    )
