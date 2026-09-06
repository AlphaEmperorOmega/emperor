import math

import torch
from torch import Tensor

from emperor.augmentations.adaptive_parameters._grouping.base import (
    GrouperAbstract,
)
from emperor.augmentations.adaptive_parameters._grouping.config import (
    RMSGroupingConfig,
)


class RMSGrouper(GrouperAbstract):
    def __init__(
        self,
        cfg: RMSGroupingConfig,
        overrides: RMSGroupingConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.normalizer = self._init_normalizer()

    def _reduce(self, X: Tensor, valid_members: Tensor | None = None) -> Tensor:
        grouped_tokens = X
        if valid_members is not None:
            accumulator = self._accumulation_input(grouped_tokens)
            masked = accumulator.masked_fill(~valid_members.unsqueeze(-1), 0)
            counts = valid_members.sum(dim=1, keepdim=True).to(accumulator.dtype)
            return torch.linalg.vector_norm(masked, dim=1) / counts.sqrt()
        members_per_chunk = grouped_tokens.size(1)
        return torch.linalg.vector_norm(
            self._accumulation_input(grouped_tokens), dim=1
        ) / math.sqrt(members_per_chunk)
