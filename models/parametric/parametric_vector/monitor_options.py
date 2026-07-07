from emperor.base.layer.monitor import LayerControllerMonitorCallback
from emperor.experiments.monitors import MonitorOption
from emperor.parametric.core.monitor import ParametricLayerMonitorCallback

MONITOR_OPTIONS: list[MonitorOption] = [
    MonitorOption(
        name="parametric",
        label="Parametric layers",
        description=(
            "Logs generated parameter norms, affine deltas, router entropy, "
            "mixture utilization, skip/drop fraction, auxiliary loss, and "
            "utilization visual summaries."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda: ParametricLayerMonitorCallback(log_every_n_steps=100),
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
]
