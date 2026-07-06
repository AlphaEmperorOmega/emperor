from emperor.experiments.monitors import MonitorOption
from emperor.sampler.core.monitor import SamplerMonitorCallback

from models.bert.linear.config import *  # noqa: F401,F403
import models.experts.linear.config as expert_defaults

_EXPERT_PREFIXES = (
    "EXPERT_",
    "SAMPLER_",
    "ROUTER_",
)
for _name in dir(expert_defaults):
    if _name.startswith(_EXPERT_PREFIXES):
        globals()[_name] = getattr(expert_defaults, _name)

EXPERT_ATTENTION_FLAG: bool = False
EXPERT_ATTENTION_USE_KV_EXPERT_MODELS_FLAG: bool = False

MONITOR_OPTIONS = [
    *MONITOR_OPTIONS,
    MonitorOption(
        name="sampler",
        label="Expert routing",
        description=(
            "Logs top-k routing, load-balance losses, dropped-token behavior, "
            "and expert utilization for MoE-backed encoder sub-stacks."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda: SamplerMonitorCallback(log_every_n_steps=100),
    ),
]
