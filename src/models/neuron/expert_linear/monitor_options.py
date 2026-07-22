from emperor.layers import (
    LayerControllerMonitorCallback,
    RecurrentLayerMonitorCallback,
)
from emperor.linears import LinearMonitorCallback
from emperor.memory import MemoryMonitorCallback
from emperor.monitoring import MonitorOption
from emperor.neuron import NeuronClusterMonitorCallback
from emperor.sampler import SamplerMonitorCallback

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
        name="sampler",
        label="Sampler usage",
        description=(
            "Logs expert routing balance, capacity drop fraction, auxiliary "
            "load-balancing loss, usage histograms, and routing heatmaps."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda settings: SamplerMonitorCallback(
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
        callback_factory=lambda settings: NeuronClusterMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
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
        callback_factory=lambda settings: SamplerMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
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
