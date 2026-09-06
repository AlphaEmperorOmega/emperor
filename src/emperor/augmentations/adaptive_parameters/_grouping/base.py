from dataclasses import replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from emperor.augmentations.adaptive_parameters._grouping.config import (
    GroupingConfig,
)
from emperor.augmentations.adaptive_parameters._grouping.options import (
    SummaryNormalizationOptions,
)
from emperor.augmentations.adaptive_parameters._grouping.plan import GroupPlan
from emperor.augmentations.adaptive_parameters._grouping.splitter import GroupSplitter
from emperor.augmentations.adaptive_parameters._grouping.validation import (
    GroupingValidator,
)
from emperor.config import ConfigBase
from emperor.nn import Module


class GrouperAbstract(Module):
    """Coordinate token splitting and the selected group reduction."""

    VALIDATOR = GroupingValidator

    def __init__(
        self,
        cfg: GroupingConfig,
        overrides: GroupingConfig | None = None,
    ):
        super().__init__()
        overridden = self._override_config(cfg, overrides)
        self.VALIDATOR.validate_grouping_value(overridden)
        self.cfg = self.__resolve_config(overridden)
        feature_dim = self.cfg.feature_dim
        self.VALIDATOR.validate_feature_dim(feature_dim)
        self.feature_dim = feature_dim
        self.splitter = GroupSplitter(self.cfg)

    def __resolve_config(self, cfg: GroupingConfig) -> GroupingConfig:
        normalization = cfg.summary_normalization
        if normalization is None:
            normalization = SummaryNormalizationOptions.DISABLED
        epsilon = cfg.rms_norm_epsilon
        if normalization is SummaryNormalizationOptions.RMS_NORM and epsilon is None:
            epsilon = 1e-6
        return replace(
            cfg, summary_normalization=normalization, rms_norm_epsilon=epsilon
        )

    def _init_model(
        self,
        model_config: ConfigBase | None,
        input_dim: int,
        output_dim: int,
    ) -> nn.Module:
        self.VALIDATOR.validate_grouping_model_config(model_config)
        model = self._build_from_config(
            model_config, input_dim=input_dim, output_dim=output_dim
        )
        self.VALIDATOR.validate_model(model)
        return model

    def _init_normalizer(self) -> nn.Module:
        if self.cfg.summary_normalization is SummaryNormalizationOptions.RMS_NORM:
            return nn.RMSNorm(
                self.feature_dim,
                eps=self.cfg.rms_norm_epsilon,
                elementwise_affine=True,
            )
        return nn.Identity()

    def forward(self, input_rows: Tensor) -> tuple[Tensor, GroupPlan]:
        group_plan = self.splitter.split(input_rows)
        context = self.summarize(group_plan.grouped_members, group_plan.valid_members)
        return context, group_plan

    def summarize(self, X: Tensor, valid_members: Tensor | None = None) -> Tensor:
        grouped_tokens = X
        self.VALIDATOR.validate_input(grouped_tokens, self.feature_dim)
        self.VALIDATOR.validate_valid_members(grouped_tokens, valid_members)
        summary = self._reduce(grouped_tokens, valid_members)
        context = self.__normalize_summary(summary)
        self.VALIDATOR.validate_output(context, grouped_tokens)
        return context.to(dtype=grouped_tokens.dtype)

    def _reduce(self, X: Tensor, valid_members: Tensor | None) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__} must implement _reduce().")

    def __normalize_summary(self, summary: Tensor) -> Tensor:
        if self.cfg.summary_normalization is SummaryNormalizationOptions.DISABLED:
            return summary
        accumulator = self._accumulation_input(summary)
        return F.rms_norm(
            accumulator,
            self.normalizer.normalized_shape,
            self.normalizer.weight.to(dtype=accumulator.dtype),
            self.normalizer.eps,
        )

    def _accumulation_input(self, grouped_tokens: Tensor) -> Tensor:
        if grouped_tokens.dtype in (torch.float16, torch.bfloat16):
            return grouped_tokens.to(dtype=torch.float32)
        return grouped_tokens
