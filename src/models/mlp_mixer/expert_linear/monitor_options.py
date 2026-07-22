from emperor.attention import AttentionMonitorCallback
from emperor.layers import LayerControllerMonitorCallback
from emperor.monitoring import MonitorOption
from emperor.sampler import SamplerMonitorCallback

MONITOR_OPTIONS: list[MonitorOption] = [
    MonitorOption(
        name="mixer",
        label="Token mixer",
        description=(
            "Logs mixer outputs and auxiliary-loss diagnostics without inventing "
            "attention heads or Q/K/V projections."
        ),
        kinds=["scalar", "histogram"],
        callback_factory=lambda settings: AttentionMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
    MonitorOption(
        name="layer-controller",
        label="Layer controllers",
        description="Logs Mixer block gates, residuals, normalization, and dropout.",
        kinds=["scalar"],
        callback_factory=lambda settings: LayerControllerMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
    MonitorOption(
        name="sampler",
        label="Expert routing",
        description="Logs top-k routing, expert utilization, capacity, and losses.",
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda settings: SamplerMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
]
