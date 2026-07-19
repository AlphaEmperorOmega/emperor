from emperor.attention.core.monitor import AttentionMonitorCallback
from emperor.augmentations.adaptive_parameters.core.bank_monitor import (
    WeightBankUtilizationMonitorCallback,
)
from emperor.augmentations.adaptive_parameters.core.monitor import (
    AdaptiveParameterMonitorCallback,
)
from emperor.base.layer.monitor import (
    LayerControllerMonitorCallback,
    RecurrentLayerMonitorCallback,
)
from emperor.experiments.monitors import MonitorOption
from emperor.memory.core.monitor import MemoryMonitorCallback

MONITOR_OPTIONS = [
    MonitorOption(
        "attention",
        "Attention",
        "Logs all Transformer attention paths.",
        ["scalar", "histogram", "image"],
        lambda settings: AttentionMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
    MonitorOption(
        "adaptive",
        "Adaptive parameters",
        "Logs active adaptive parameter generators.",
        ["scalar", "histogram"],
        lambda settings: AdaptiveParameterMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
    MonitorOption(
        "weight-bank",
        "Weight banks",
        "Logs adaptive weight-bank utilization.",
        ["scalar", "histogram"],
        lambda settings: WeightBankUtilizationMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
    MonitorOption(
        "recurrent-layer",
        "Recurrent layers",
        "Logs recurrent stack state.",
        ["scalar", "histogram", "image"],
        lambda settings: RecurrentLayerMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
    MonitorOption(
        "layer-controller",
        "Layer controllers",
        "Logs gates, halting, and residuals.",
        ["scalar"],
        lambda settings: LayerControllerMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
    MonitorOption(
        "memory",
        "Memory modules",
        "Logs optional dynamic memory.",
        ["scalar"],
        lambda settings: MemoryMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
]
