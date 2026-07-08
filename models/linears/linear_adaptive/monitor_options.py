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
from emperor.halting.core.monitor import HaltingMonitorCallback
from emperor.linears.core.monitor import LinearMonitorCallback
from emperor.memory.core.monitor import MemoryMonitorCallback

MONITOR_OPTIONS: list[MonitorOption] = [
    MonitorOption(
        name="linear",
        label="Linear layers",
        description=(
            "Logs activation, parameter, gradient, weight-conditioning "
            "(spectral norm / condition number / effective rank), and dead-feature "
            "stats for Emperor linear layers."
        ),
        kinds=["scalar"],
        callback_factory=lambda settings: LinearMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
    MonitorOption(
        name="recurrent-layer",
        label="Recurrent layers",
        description=(
            "Logs recurrent step count, hidden-state convergence, recurrent gate "
            "openness, halted-state preservation, and step-delta visual summaries."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda settings: RecurrentLayerMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
    MonitorOption(
        name="layer-controller",
        label="Layer controllers",
        description=(
            "Logs Layer gate, residual, dropout, layer-norm, and activation "
            "controller statistics without duplicating memory metrics."
        ),
        kinds=["scalar"],
        callback_factory=lambda settings: LayerControllerMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
    MonitorOption(
        name="halting",
        label="Halting (adaptive compute)",
        description=(
            "Logs recurrence depth, halting fraction, max-steps saturation, ponder "
            "loss, plus survival heatmap and ponder-cost histogram for "
            "stick-breaking / soft halting modules."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda settings: HaltingMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
    MonitorOption(
        name="memory",
        label="Memory modules",
        description=(
            "Logs gating, blend-weight, and state statistics for Emperor memory "
            "modules. Inactive until a memory config is enabled."
        ),
        kinds=["scalar"],
        callback_factory=lambda settings: MemoryMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
    MonitorOption(
        name="adaptive",
        label="Adaptive parameters",
        description=(
            "Logs dynamic weight, bias, diagonal, and mask parameter statistics, "
            "plus input-adaptivity (cross-sample variation / collapse detection)."
        ),
        kinds=["scalar", "histogram"],
        callback_factory=lambda settings: AdaptiveParameterMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps,
            log_histograms=True,
            log_internal_stats=True,
        ),
    ),
    MonitorOption(
        name="weight-bank",
        label="Weight bank utilization",
        description=(
            "Logs bank-slot selection entropy, utilization, and routing heatmaps "
            "for weighted-bank dynamic params."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda settings: WeightBankUtilizationMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps,
        ),
    ),
]
