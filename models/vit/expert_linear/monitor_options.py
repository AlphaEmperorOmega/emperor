from emperor.experiments.monitors import MonitorOption
from emperor.sampler.core.monitor import SamplerMonitorCallback

from models.vit.linear.monitor_options import MONITOR_OPTIONS as BASE_MONITOR_OPTIONS

MONITOR_OPTIONS = [
    *BASE_MONITOR_OPTIONS,
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
