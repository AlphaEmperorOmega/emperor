import copy
import torch
import torch.nn as nn
import unittest
from Emperor.base.utils import randn
from Emperor.components.parameter_generators.utils.routers import (
    RouterModel,
    RouterLayer,
    VectorRouterModel,
    RouterConfig,
)
from Emperor.config import ROUTER_NUM_LAYERNUM_LAYERSS, ModelConfig
