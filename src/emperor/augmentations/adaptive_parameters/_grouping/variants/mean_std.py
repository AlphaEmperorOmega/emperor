import torch
from torch import Tensor, nn

from emperor.augmentations.adaptive_parameters._grouping.base import (
    GrouperAbstract,
)
from emperor.augmentations.adaptive_parameters._grouping.config import (
    MeanStdGroupingConfig,
)
from emperor.layers import Layer


class MeanStdGrouper(GrouperAbstract):
    def __init__(
        self,
        cfg: MeanStdGroupingConfig,
        overrides: MeanStdGroupingConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        assert isinstance(self.cfg, MeanStdGroupingConfig)
        self.model_config = self.cfg.model_config
        self.projection = self.__init_projection()
        self.normalizer = self._init_normalizer()

    def __init_projection(self) -> nn.Module:
        return self._init_model(
            self.model_config,
            input_dim=2 * self.feature_dim,
            output_dim=self.feature_dim,
        )

    def _reduce(self, X: Tensor, valid_members: Tensor | None = None) -> Tensor:
        grouped_tokens = X
        accumulator = self._accumulation_input(grouped_tokens)
        statistics = self.__compute_statistics(accumulator, valid_members)
        return self.__project_statistics(statistics, grouped_tokens.dtype)

    def __compute_statistics(
        self, accumulator: Tensor, valid_members: Tensor | None
    ) -> Tensor:
        if valid_members is None:
            deviation, mean = torch.std_mean(accumulator, dim=1, correction=0)
        else:
            mean, deviation = self.__compute_masked_statistics(
                accumulator, valid_members
            )
        return torch.cat((mean, deviation), dim=-1)

    def __compute_masked_statistics(
        self, accumulator: Tensor, valid_members: Tensor
    ) -> tuple[Tensor, Tensor]:
        invalid_members_mask = ~valid_members.unsqueeze(-1)
        masked_members = accumulator.masked_fill(invalid_members_mask, 0)
        member_counts = valid_members.sum(dim=1, keepdim=True).to(accumulator.dtype)
        mean = masked_members.sum(dim=1) / member_counts
        centered_members = masked_members - mean.unsqueeze(1)
        centered_members = centered_members.masked_fill(invalid_members_mask, 0)
        deviation = (
            torch.linalg.vector_norm(centered_members, dim=1) / member_counts.sqrt()
        )
        return mean, deviation

    def __project_statistics(
        self, statistics: Tensor, input_dtype: torch.dtype
    ) -> Tensor:
        projection_input = statistics.to(dtype=input_dtype)
        state = Layer.run_model_from_hidden(self.projection, projection_input)
        self.VALIDATOR.validate_model_output(state, statistics, self.feature_dim)
        return state.hidden
