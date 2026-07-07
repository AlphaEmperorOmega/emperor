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
from emperor.linears.core.monitor import LinearMonitorCallback
from emperor.memory.core.monitor import MemoryMonitorCallback
from emperor.sampler.core.monitor import SamplerMonitorCallback

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
        callback_factory=lambda: LinearMonitorCallback(log_every_n_steps=100),
    ),
    MonitorOption(
        name="recurrent-layer",
        label="Recurrent layers",
        description=(
            "Logs recurrent step count, hidden-state convergence, recurrent gate "
            "openness, halted-state preservation, and step-delta visual summaries."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda: RecurrentLayerMonitorCallback(log_every_n_steps=100),
    ),
    MonitorOption(
        name="layer-controller",
        label="Layer controllers",
        description=(
            "Logs Layer gate, residual, dropout, layer-norm, and activation "
            "controller statistics without duplicating memory metrics."
        ),
        kinds=["scalar"],
        callback_factory=lambda: LayerControllerMonitorCallback(log_every_n_steps=100),
    ),
    MonitorOption(
        name="adaptive",
        label="Adaptive parameters",
        description=(
            "Logs dynamic weight, bias, diagonal, and mask parameter statistics, "
            "plus input-adaptivity (cross-sample variation / collapse detection)."
        ),
        kinds=["scalar", "histogram"],
        callback_factory=lambda: AdaptiveParameterMonitorCallback(
            log_every_n_steps=100,
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
        callback_factory=lambda: WeightBankUtilizationMonitorCallback(
            log_every_n_steps=100,
        ),
    ),
    MonitorOption(
        name="sampler",
        label="Sampler usage",
        description=(
            "Logs expert routing balance, capacity drop fraction, auxiliary "
            "load-balancing loss, usage histograms, and routing heatmaps."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda: SamplerMonitorCallback(log_every_n_steps=100),
    ),
    MonitorOption(
        name="memory",
        label="Memory modules",
        description=(
            "Logs gating, blend-weight, and state statistics for Emperor memory "
            "modules. Inactive until a memory config is enabled."
        ),
        kinds=["scalar"],
        callback_factory=lambda: MemoryMonitorCallback(log_every_n_steps=100),
    ),
]
