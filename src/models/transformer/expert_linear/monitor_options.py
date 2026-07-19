from emperor.attention.monitoring import AttentionMonitorCallback
from emperor.layers import (
    LayerControllerMonitorCallback,
    RecurrentLayerMonitorCallback,
)
from emperor.memory.monitoring import MemoryMonitorCallback
from emperor.monitoring import MonitorOption
from emperor.sampler import SamplerMonitorCallback

MONITOR_OPTIONS = [
    MonitorOption(
        "attention",
        "Attention",
        "Logs expert attention health.",
        ["scalar", "histogram", "image"],
        lambda settings: AttentionMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
    MonitorOption(
        "sampler",
        "Expert routing",
        "Logs attention and feed-forward expert routing.",
        ["scalar", "histogram", "image"],
        lambda settings: SamplerMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
    MonitorOption(
        "recurrent-layer",
        "Recurrent layers",
        "Logs recurrent stack state.",
        ["scalar", "histogram", "image"],
        lambda settings: RecurrentLayerMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
    MonitorOption(
        "layer-controller",
        "Layer controllers",
        "Logs gates and halting.",
        ["scalar"],
        lambda settings: LayerControllerMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
    MonitorOption(
        "memory",
        "Memory modules",
        "Logs optional dynamic memory.",
        ["scalar"],
        lambda settings: MemoryMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
]
