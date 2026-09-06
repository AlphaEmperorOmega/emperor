import torch
from torch import Tensor, nn

from emperor.augmentations.adaptive_parameters._grouping.base import (
    GrouperAbstract,
)
from emperor.augmentations.adaptive_parameters._grouping.config import (
    AttentionGroupingConfig,
)
from emperor.layers import Layer


class AttentionGrouper(GrouperAbstract):
    def __init__(
        self,
        cfg: AttentionGroupingConfig,
        overrides: AttentionGroupingConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        assert isinstance(self.cfg, AttentionGroupingConfig)
        self.model_config = self.cfg.model_config
        self.scorer = self.__init_scorer()
        self.normalizer = self._init_normalizer()

    def __init_scorer(self) -> nn.Module:
        return self._init_model(
            self.model_config,
            input_dim=self.feature_dim,
            output_dim=1,
        )

    def _reduce(self, X: Tensor, valid_members: Tensor | None = None) -> Tensor:
        grouped_tokens = X
        reduction_tokens = self._accumulation_input(grouped_tokens)
        valid_token_scores = self.__score_valid_members(grouped_tokens, valid_members)
        grouped_attention_scores = self.__restore_group_scores(
            valid_token_scores, reduction_tokens, valid_members
        )
        return self.__compute_weighted_summary(
            grouped_attention_scores, reduction_tokens, valid_members
        )

    def __score_valid_members(
        self, grouped_tokens: Tensor, valid_members: Tensor | None
    ) -> Tensor:
        flat_tokens = grouped_tokens.reshape(-1, grouped_tokens.size(-1))
        if valid_members is not None:
            flat_valid_members = valid_members.reshape(-1)
            flat_tokens = flat_tokens[flat_valid_members]
        state = Layer.run_model_from_hidden(self.scorer, flat_tokens)
        self.VALIDATOR.validate_model_output(state, flat_tokens, 1)
        return state.hidden

    def __restore_group_scores(
        self,
        token_scores: Tensor,
        accumulator: Tensor,
        valid_members: Tensor | None,
    ) -> Tensor:
        group_score_shape = (*accumulator.shape[:2], 1)
        if valid_members is None:
            scores = token_scores.reshape(group_score_shape)
            return scores.to(dtype=accumulator.dtype)
        scores = torch.full(
            group_score_shape,
            -torch.inf,
            dtype=accumulator.dtype,
            device=accumulator.device,
        )
        valid_score_positions = valid_members.unsqueeze(-1)
        token_scores = token_scores.to(dtype=accumulator.dtype)
        return scores.masked_scatter(valid_score_positions, token_scores)

    def __compute_weighted_summary(
        self,
        scores: Tensor,
        accumulator: Tensor,
        valid_members: Tensor | None,
    ) -> Tensor:
        if valid_members is not None:
            invalid_members_mask = ~valid_members.unsqueeze(-1)
            accumulator = accumulator.masked_fill(invalid_members_mask, 0)
        probabilities = scores.softmax(dim=1)
        weighted_members = probabilities * accumulator
        return weighted_members.sum(dim=1)
