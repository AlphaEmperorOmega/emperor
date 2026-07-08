from emperor.augmentations.adaptive_parameters.core.bank_monitor import (
    WeightBankUtilizationMonitorCallback,
)
from emperor.augmentations.adaptive_parameters.core.monitor import (
    AdaptiveParameterMonitorCallback,
)
from emperor.experiments.monitors import MonitorOption

from models.bert.expert_linear.monitor_options import (
    MONITOR_OPTIONS as BASE_MONITOR_OPTIONS,
)

MONITOR_OPTIONS = [
    *BASE_MONITOR_OPTIONS,
    MonitorOption(
        name="adaptive",
        label="Adaptive parameters",
        description=(
            "Logs adaptive weight, bias, diagonal, and mask controller statistics "
            "for adaptive expert linear layers."
        ),
        kinds=["scalar", "histogram"],
        callback_factory=lambda settings: AdaptiveParameterMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
    MonitorOption(
        name="weight-bank",
        label="Weight banks",
        description=(
            "Logs adaptive parameter bank usage and utilization for bank-backed "
            "dynamic expert weights and biases."
        ),
        kinds=["scalar", "histogram"],
        callback_factory=lambda settings: WeightBankUtilizationMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
]
