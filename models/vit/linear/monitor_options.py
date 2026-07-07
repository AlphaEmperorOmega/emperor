from emperor.attention.core.monitor import AttentionMonitorCallback
from emperor.base.layer.monitor import LayerControllerMonitorCallback
from emperor.experiments.monitors import MonitorOption

MONITOR_OPTIONS: list[MonitorOption] = [
    MonitorOption(
        name="attention",
        label="Attention",
        description=(
            "Logs Q/K/V norms, attention entropy, max probability, dropout and "
            "mask coverage, auxiliary loss, and attention head visual summaries."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda: AttentionMonitorCallback(log_every_n_steps=100),
    ),
    MonitorOption(
        name="layer-controller",
        label="Layer controllers",
        description=(
            "Logs Layer gate, residual, dropout, layer-norm, and activation "
            "controller statistics without duplicating memory metrics."
        ),
        kinds=["scalar"],
        callback_factory=lambda: LayerControllerMonitorCallback(
            log_every_n_steps=100
        ),
    ),
]
