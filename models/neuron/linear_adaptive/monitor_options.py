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
from emperor.neuron.core.monitor import NeuronClusterMonitorCallback
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
        name="halting",
        label="Halting (adaptive compute)",
        description=(
            "Logs recurrence depth, halting fraction, max-steps saturation, ponder "
            "loss, plus survival heatmap and ponder-cost histogram for "
            "stick-breaking / soft halting modules."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda: HaltingMonitorCallback(log_every_n_steps=100),
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
]

_neuron_monitor_options = [
    MonitorOption(
        name="neuron_cluster",
        label="Neuron cluster growth",
        description=(
            "Logs cluster growth (count, capacity, fill, growth pressure) plus "
            "routing dynamics: route depth, escape/halt fractions, entry-routing "
            "entropy, survival curve, and per-neuron utilization heatmap."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda: NeuronClusterMonitorCallback(log_every_n_steps=100),
        default_enabled=True,
    ),
    MonitorOption(
        name="sampler",
        label="Routing samplers",
        description=(
            "Logs router/sampler internals for the cluster entry sampler and "
            "per-neuron terminal samplers: probability distributions, per-expert "
            "utilization, and auxiliary load-balancing loss components."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda: SamplerMonitorCallback(log_every_n_steps=100),
    ),
]

_existing_monitor_names = {option.name for option in MONITOR_OPTIONS}

MONITOR_OPTIONS = [
    *MONITOR_OPTIONS,
    *[
        option
        for option in _neuron_monitor_options
        if option.name not in _existing_monitor_names
    ],
]
