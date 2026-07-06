from emperor.augmentations.adaptive_parameters import (
    LowRankDynamicWeightConfig,  # noqa: F401
)
from emperor.augmentations.adaptive_parameters.core.bank_monitor import (
    WeightBankUtilizationMonitorCallback,
)
from emperor.augmentations.adaptive_parameters.core.monitor import (
    AdaptiveParameterMonitorCallback,
)
from emperor.experiments.monitors import MonitorOption

from models.bert.expert_linear.config import *  # noqa: F401,F403
import models.experts.linear_adaptive.config as adaptive_expert_defaults

_ADAPTIVE_PREFIXES = (
    "ADAPTIVE_",
    "WEIGHT_",
    "BIAS_",
    "DIAGONAL_",
    "MASK_",
    "ROW_MASK_",
    "ROUTER_WEIGHT_",
    "ROUTER_BIAS_",
    "ROUTER_DIAGONAL_",
    "ROUTER_MASK_",
)
for _name in dir(adaptive_expert_defaults):
    if _name.startswith(_ADAPTIVE_PREFIXES):
        globals()[_name] = getattr(adaptive_expert_defaults, _name)

MONITOR_OPTIONS = [
    *MONITOR_OPTIONS,
    MonitorOption(
        name="adaptive",
        label="Adaptive parameters",
        description=(
            "Logs adaptive weight, bias, diagonal, and mask controller statistics "
            "for adaptive expert linear layers."
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
            "dynamic expert weights and biases."
        ),
        kinds=["scalar", "histogram"],
        callback_factory=lambda: WeightBankUtilizationMonitorCallback(
            log_every_n_steps=100
        ),
    ),
]
