from emperor.attention.core.monitor import AttentionMonitorCallback
from emperor.base.layer.monitor import (
    LayerControllerMonitorCallback,
    RecurrentLayerMonitorCallback,
)
from emperor.experiments.monitors import MonitorOption
from emperor.memory.core.monitor import MemoryMonitorCallback
from emperor.sampler.core.monitor import SamplerMonitorCallback

MONITOR_OPTIONS: list[MonitorOption] = [
    MonitorOption(
        name="attention",
        label="Attention",
        description=(
            "Logs Q/K/V norms, attention entropy, max probability, dropout and "
            "mask coverage, auxiliary loss, and attention head visual summaries."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda settings: AttentionMonitorCallback(
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
        name="sampler",
        label="Expert routing",
        description=(
            "Logs top-k routing, load-balance losses, dropped-token behavior, "
            "and expert utilization for MoE-backed encoder sub-stacks."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda settings: SamplerMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
]
