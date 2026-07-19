from emperor.attention.monitoring import AttentionMonitorCallback
from emperor.layers import (
    LayerControllerMonitorCallback,
    RecurrentLayerMonitorCallback,
)
from emperor.memory.monitoring import MemoryMonitorCallback
from emperor.monitoring import MonitorOption

MONITOR_OPTIONS = [
    MonitorOption(
        name="attention",
        label="Attention",
        description="Logs encoder, decoder self-attention, and cross-attention health.",
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda settings: AttentionMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
    MonitorOption(
        name="recurrent-layer",
        label="Recurrent layers",
        description="Logs recurrent encoder and decoder controller state.",
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda settings: RecurrentLayerMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
    MonitorOption(
        name="layer-controller",
        label="Layer controllers",
        description="Logs gates, residual joins, halting, and normalization.",
        kinds=["scalar"],
        callback_factory=lambda settings: LayerControllerMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
    MonitorOption(
        name="memory",
        label="Memory modules",
        description="Logs optional encoder and decoder dynamic memory.",
        kinds=["scalar"],
        callback_factory=lambda settings: MemoryMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
]
