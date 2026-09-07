"""Learned elementwise replacements for last-dimension normalization."""

from math import expm1, log, sqrt

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from emperor.layers._layer.validation import ElementwiseNormalizationValidator


class ElementwiseNormalization(nn.Module):
    """Share feature validation, affine parameters, and precision handling."""

    VALIDATOR = ElementwiseNormalizationValidator

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.weight = nn.Parameter(torch.ones(dimension))
        self.bias = nn.Parameter(torch.zeros(dimension))

    def forward(self, hidden: Tensor) -> Tensor:
        self.VALIDATOR.validate_forward_input(hidden, self.dimension)
        input_dtype = hidden.dtype
        if input_dtype in (torch.float16, torch.bfloat16):
            hidden = hidden.float()
        transformed = self._transform(hidden)
        output = transformed * self.weight.to(dtype=hidden.dtype)
        output = output + self.bias.to(dtype=hidden.dtype)
        return output.to(dtype=input_dtype)

    def _transform(self, hidden: Tensor) -> Tensor:
        raise NotImplementedError


class DynamicTanh(ElementwiseNormalization):
    """DyT: weight * tanh(alpha * x) + bias, with scalar alpha initially 0.5.

    Reference: https://arxiv.org/abs/2503.10622
    """

    def __init__(self, dimension: int) -> None:
        super().__init__(dimension)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def _transform(self, hidden: Tensor) -> Tensor:
        return torch.tanh(self.alpha.to(dtype=hidden.dtype) * hidden)


class DynamicErf(ElementwiseNormalization):
    """Derf: weight * erf(alpha * x + shift) + bias.

    Scalar alpha starts at 0.5 and scalar shift at zero, following
    https://arxiv.org/abs/2512.10938, Section 5.
    """

    def __init__(self, dimension: int) -> None:
        super().__init__(dimension)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.shift = nn.Parameter(torch.tensor(0.0))

    def _transform(self, hidden: Tensor) -> Tensor:
        scaled = self.alpha.to(dtype=hidden.dtype) * hidden
        return torch.erf(scaled + self.shift.to(dtype=hidden.dtype))


class DynamicISRU(ElementwiseNormalization):
    """Experimental DyISRU: sqrt(C) * x / sqrt(beta + x**2), then affine.

    Uses Equation 16 and Appendix D of
    https://aclanthology.org/2026.eacl-short.48/. The positive scalar beta
    starts at 4 and is learned through softplus, with a numerical floor of
    1e-5. Per-feature weight and bias adapt the paper's transformation to
    this layer interface. No activation statistics are computed.
    """

    def __init__(self, dimension: int) -> None:
        super().__init__(dimension)
        self.raw_beta = nn.Parameter(torch.tensor(log(expm1(4.0))))
        self.output_scale = sqrt(dimension)

    def _transform(self, hidden: Tensor) -> Tensor:
        beta = F.softplus(self.raw_beta.to(dtype=hidden.dtype)).clamp_min(1e-5)
        denominator = torch.hypot(hidden, beta.sqrt())
        return (hidden / denominator) * self.output_scale
