from emperor.attention.core.monitor import AttentionMonitorCallback
from emperor.base.layer.monitor import (
    LayerControllerMonitorCallback,
    RecurrentLayerMonitorCallback,
)
from emperor.experiments.monitors import MonitorOption
from emperor.memory.core.monitor import MemoryMonitorCallback
from emperor.sampler.core.monitor import SamplerMonitorCallback

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
