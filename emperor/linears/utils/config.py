from dataclasses import dataclass, field
from emperor.base.utils import ConfigBase
from emperor.linears.utils._monitors import TensorMonitor, StatisticsMonitor


@dataclass
class LinearLayerConfig(ConfigBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Input dimension of the linear layer"},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Output dimension of the linear layer"},
    )
    bias_flag: bool | None = field(
        default=None,
        metadata={
            "help": "When true bias will be added to after the matrix multiplication between, the input and output"
        },
    )
    data_monitor: type[TensorMonitor] | None = field(
        default=None,
        metadata={
            "help": "Optional monitor class that tracks input/output statistics and logs to TensorBoard."
        },
    )
    parameter_monitor: type[StatisticsMonitor] | None = field(
        default=None,
        metadata={
            "help": "Optional monitor class that tracks parameter statistics (mean/var/norm) and logs to TensorBoard."
        },
    )
