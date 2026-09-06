from torch import Tensor

from emperor.augmentations.adaptive_parameters._grouping.base import (
    GrouperAbstract,
)
from emperor.augmentations.adaptive_parameters._grouping.config import (
    MeanGroupingConfig,
)


class MeanGrouper(GrouperAbstract):
    def __init__(
        self,
        cfg: MeanGroupingConfig,
        overrides: MeanGroupingConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.normalizer = self._init_normalizer()

    def _reduce(self, X: Tensor, valid_members: Tensor | None = None) -> Tensor:
        grouped_tokens = X
        if valid_members is not None:
            accumulator = self._accumulation_input(grouped_tokens)
            masked = accumulator.masked_fill(~valid_members.unsqueeze(-1), 0)
            return masked.sum(dim=1) / valid_members.sum(dim=1, keepdim=True)
        return self._accumulation_input(grouped_tokens).mean(dim=1)
