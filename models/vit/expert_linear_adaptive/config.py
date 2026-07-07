from emperor.augmentations.adaptive_parameters import (
    LowRankDynamicWeightConfig,  # noqa: F401
)

from models.vit.expert_linear.config import *  # noqa: F401,F403
import models.experts.linear_adaptive.config as adaptive_expert_defaults

_ADAPTIVE_PREFIXES = (
    "ADAPTIVE_",
    "WEIGHT_",
    "BIAS_",
    "DIAGONAL_",
    "MASK_",
    "ROW_MASK_",
    "ROUTER_WEIGHT_",
    "ROUTER_BIAS_",
    "ROUTER_DIAGONAL_",
    "ROUTER_MASK_",
)
for _name in dir(adaptive_expert_defaults):
    if _name.startswith(_ADAPTIVE_PREFIXES):
        globals()[_name] = getattr(adaptive_expert_defaults, _name)
