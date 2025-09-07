from enum import Enum
import torch.nn.functional as F


class ActivationOptions(Enum):
    RELU = F.relu
    GELU = F.gelu
    SIGMOID = F.sigmoid
    TANH = F.tanh
    LEAKY_RELU = F.leaky_relu
    ELU = F.elu
    SELU = F.selu
    SOFTPLUS = F.softplus
    SOFTSIGN = F.softsign


class LayerNormPositionOptions(Enum):
    NONE = "no_layer_norm_added"
    DEFAULT = "layer_norm_after_model_output"
    BEFORE = "layer_norm_before_model_processing"
    AFTER = "layer_norm_after_residual_connection"
