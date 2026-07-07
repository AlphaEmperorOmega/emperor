from emperor.augmentations.adaptive_parameters.core.bank_monitor import (
    WeightBankUtilizationMonitorCallback,
)
from emperor.augmentations.adaptive_parameters.core.monitor import (
    AdaptiveParameterMonitorCallback,
)
from emperor.experiments.monitors import MonitorOption
from models.vit.linear.monitor_options import MONITOR_OPTIONS as BASE_MONITOR_OPTIONS

MONITOR_OPTIONS = [
    *BASE_MONITOR_OPTIONS,
    MonitorOption(
        name="adaptive",
        label="Adaptive parameters",
        description=(
            "Logs adaptive weight, bias, diagonal, and mask controller statistics "
            "for adaptive linear layers."
        ),
        kinds=["scalar", "histogram"],
        callback_factory=lambda: AdaptiveParameterMonitorCallback(
            log_every_n_steps=100
        ),
    ),
    MonitorOption(
        name="weight-bank",
        label="Weight banks",
        description=(
            "Logs adaptive parameter bank usage and utilization for bank-backed "
            "dynamic weights and biases."
        ),
        kinds=["scalar", "histogram"],
        callback_factory=lambda: WeightBankUtilizationMonitorCallback(
            log_every_n_steps=100
        ),
    ),
]
