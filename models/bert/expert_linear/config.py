
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

# Mixture Of Experts Model Options
EXPERT_ATTENTION_FLAG: bool = False
EXPERT_ATTENTION_USE_KV_EXPERT_MODELS_FLAG: bool = False
