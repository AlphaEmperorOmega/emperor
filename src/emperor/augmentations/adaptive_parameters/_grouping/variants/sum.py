from torch import Tensor

from emperor.augmentations.adaptive_parameters._grouping.base import (
    GrouperAbstract,
)
from emperor.augmentations.adaptive_parameters._grouping.config import (
    SumGroupingConfig,
)


class SumGrouper(GrouperAbstract):
    def __init__(
        self,
        cfg: SumGroupingConfig,
        overrides: SumGroupingConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.normalizer = self._init_normalizer()

    def _reduce(self, X: Tensor, valid_members: Tensor | None = None) -> Tensor:
        grouped_tokens = X
        if valid_members is not None:
            invalid_members_mask = ~valid_members.unsqueeze(-1)
            grouped_tokens = grouped_tokens.masked_fill(invalid_members_mask, 0)
        return grouped_tokens.sum(dim=1)
