from emperor.layers import LayerControllerMonitorCallback
from emperor.monitoring import MonitorOption
from emperor.parametric import ParametricLayerMonitorCallback

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
        callback_factory=lambda settings: ParametricLayerMonitorCallback(
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
]
