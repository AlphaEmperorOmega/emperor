import math
from dataclasses import fields
from numbers import Real

import torch
from torch import Tensor, nn

from emperor.augmentations.adaptive_parameters._grouping.config import (
    AttentionGroupingConfig,
    GroupingConfig,
    MeanStdGroupingConfig,
)
from emperor.augmentations.adaptive_parameters._grouping.options import (
    SummaryNormalizationOptions,
)
from emperor.config import ConfigBase


class GroupingValidator:
    @classmethod
    def validate_grouping_value(cls, grouping: object) -> None:
        from emperor.augmentations.adaptive_parameters._options import (
            AdaptiveParameterGroupingScopeOptions,
            AdaptiveParameterInputOrderOptions,
        )

        cls.validate_grouping_config(grouping)
        if not isinstance(grouping.scope, AdaptiveParameterGroupingScopeOptions):
            raise TypeError(
                "scope must be an AdaptiveParameterGroupingScopeOptions value."
            )
        if (grouping.group_count is None) == (grouping.chunk_size is None):
            raise ValueError("Exactly one of group_count or chunk_size is required.")
        if grouping.group_count is not None:
            cls._validate_group_count(grouping.group_count)
        elif type(grouping.chunk_size) is not int or grouping.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive non-Boolean integer.")
        if grouping.scope is AdaptiveParameterGroupingScopeOptions.ROWS:
            if grouping.sequence_length is not None or grouping.input_order is not None:
                raise ValueError(
                    "ROWS grouping forbids sequence_length and input_order."
                )
            return
        length = grouping.sequence_length
        if type(length) is not int or length <= 0:
            raise ValueError("sequence_length must be a positive integer for SEQUENCE.")
        if not isinstance(grouping.input_order, AdaptiveParameterInputOrderOptions):
            raise TypeError(
                "input_order must be an AdaptiveParameterInputOrderOptions value for SEQUENCE."
            )
        if grouping.group_count is not None and (
            grouping.group_count > length or length % grouping.group_count
        ):
            raise ValueError(
                "sequence_length must be divisible by group_count and at least group_count."
            )

    @staticmethod
    def _validate_group_count(group_count: int | None) -> None:
        if (
            isinstance(group_count, bool)
            or not isinstance(group_count, int)
            or group_count <= 0
        ):
            raise ValueError(
                "group_count must be a positive integer when provided, "
                f"received {group_count!r}."
            )

    @classmethod
    def validate_grouping_config(cls, config: object) -> None:
        if not isinstance(config, GroupingConfig):
            raise TypeError("grouping configuration must be a GroupingConfig.")
        if type(config) is GroupingConfig:
            raise ValueError("GroupingConfig is abstract; select a concrete variant.")
        cls._validate_config_fields(config, type(config))
        if config.feature_dim is not None:
            cls.validate_feature_dim(config.feature_dim)
        normalization = config.summary_normalization
        if normalization is None:
            normalization = SummaryNormalizationOptions.DISABLED
        if not isinstance(normalization, SummaryNormalizationOptions):
            raise TypeError(
                "summary_normalization must be a SummaryNormalizationOptions value."
            )
        if isinstance(
            config,
            (
                AttentionGroupingConfig,
                MeanStdGroupingConfig,
            ),
        ):
            cls.validate_grouping_model_config(config.model_config)
        epsilon = config.rms_norm_epsilon
        if normalization is SummaryNormalizationOptions.RMS_NORM:
            if epsilon is None:
                epsilon = 1e-6
            if (
                isinstance(epsilon, bool)
                or not isinstance(epsilon, Real)
                or not math.isfinite(epsilon)
                or epsilon <= 0
            ):
                raise ValueError(
                    "rms_norm_epsilon must be a finite positive real number."
                )
        elif epsilon is not None:
            raise ValueError(
                "rms_norm_epsilon requires RMS_NORM summary normalization."
            )

    @classmethod
    def validate_grouping_model_config(cls, model_config: object) -> None:
        if model_config is None:
            raise ValueError("Attention and mean/std grouping require a model_config.")
        if not isinstance(model_config, ConfigBase):
            raise TypeError("grouping model_config must be a ConfigBase instance.")
        hidden_dim = getattr(model_config, "hidden_dim", None)
        if hidden_dim is not None:
            cls.validate_feature_dim(hidden_dim)

    @staticmethod
    def validate_model(model: object) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("grouping model_config must build a torch.nn.Module.")

    @staticmethod
    def validate_model_output(
        state: object, model_input: Tensor, features: int
    ) -> None:
        from emperor.layers import LayerState

        if not isinstance(state, LayerState):
            raise TypeError("Grouping models must return a LayerState.")
        if state.loss is not None:
            raise ValueError(
                "Grouping model_config must produce hidden values without auxiliary loss."
            )
        rows = model_input.size(0)
        hidden = state.hidden
        if not isinstance(hidden, Tensor) or tuple(hidden.shape) != (rows, features):
            raise ValueError(
                f"Grouping model output must have shape {(rows, features)}."
            )
        if not hidden.is_floating_point() or hidden.device != model_input.device:
            raise ValueError(
                "Grouping model output must be floating-point on the input device."
            )

    @staticmethod
    def _validate_config_fields(config: object, expected_type: type) -> None:
        if not isinstance(config, expected_type):
            raise TypeError(
                f"grouping configuration must be a {expected_type.__name__}."
            )
        if (
            vars(config).keys()
            - {field.name for field in fields(expected_type)}
            - {"_passed_args"}
        ):
            raise ValueError(
                f"{expected_type.__name__} contains unsupported fields; rebuild it using the current configuration schema."
            )

    @staticmethod
    def validate_feature_dim(feature_dim: object) -> None:
        if type(feature_dim) is not int or feature_dim <= 0:
            raise ValueError(
                "grouping feature_dim must be a positive non-Boolean integer."
            )

    @staticmethod
    def validate_input(grouped_tokens: object, feature_dim: int) -> None:
        if (
            not isinstance(grouped_tokens, Tensor)
            or not grouped_tokens.is_floating_point()
        ):
            raise TypeError("grouped_tokens must be a floating-point Tensor.")
        if grouped_tokens.dim() != 3:
            raise ValueError(
                "grouped_tokens must have shape [chunks, members, features]."
            )
        if any(dimension == 0 for dimension in grouped_tokens.shape):
            raise ValueError(
                "grouping requires nonempty chunks, members, and features."
            )
        if grouped_tokens.size(-1) != feature_dim:
            raise ValueError(
                f"grouping input feature dimension must equal {feature_dim}."
            )

    @staticmethod
    def validate_valid_members(
        grouped_tokens: Tensor, valid_members: Tensor | None
    ) -> None:
        if valid_members is None:
            return
        if not isinstance(valid_members, Tensor) or valid_members.dtype != torch.bool:
            raise TypeError("valid_members must be a Boolean Tensor.")
        if valid_members.shape != grouped_tokens.shape[:2]:
            raise ValueError("valid_members must have shape [chunks, members].")
        if valid_members.device != grouped_tokens.device:
            raise ValueError("valid_members must be on the input device.")
        if not valid_members.any(dim=1).all():
            raise ValueError("Every chunk must contain at least one valid member.")

    @staticmethod
    def validate_output(context: object, grouped_tokens: Tensor) -> None:
        expected = (grouped_tokens.size(0), grouped_tokens.size(-1))
        if not isinstance(context, Tensor) or tuple(context.shape) != expected:
            raise ValueError(f"summarized context must have shape {expected}.")
        if context.device != grouped_tokens.device or not context.is_floating_point():
            raise ValueError(
                "summarized context must be floating-point on the input device."
            )

    @staticmethod
    def validate_flat_input(input_rows: object, cfg: GroupingConfig) -> None:
        from emperor.augmentations.adaptive_parameters._options import (
            AdaptiveParameterGroupingScopeOptions,
        )

        if not isinstance(input_rows, Tensor) or not input_rows.is_floating_point():
            raise TypeError("input_rows must be a floating-point Tensor.")
        if input_rows.dim() != 2:
            raise ValueError("input_rows must be a two-dimensional matrix.")
        row_count, feature_dim = input_rows.shape
        if row_count == 0 or feature_dim == 0:
            raise ValueError("Grouping requires nonempty input rows and features.")
        if feature_dim != cfg.feature_dim:
            raise ValueError(
                f"grouping input feature dimension must equal {cfg.feature_dim}."
            )
        if cfg.scope is AdaptiveParameterGroupingScopeOptions.ROWS:
            if cfg.group_count is not None and (
                cfg.group_count > row_count or row_count % cfg.group_count
            ):
                raise ValueError(
                    f"row count {row_count} must be divisible by group_count={cfg.group_count} "
                    "and at least group_count."
                )
        elif row_count % cfg.sequence_length:
            raise ValueError(
                f"row count {row_count} must be divisible by sequence_length={cfg.sequence_length}."
            )

    @staticmethod
    def validate_grouped_output(
        grouped_output: object,
        expected_leading_shape: tuple[int, int],
        output_dim: int | None,
    ) -> None:
        if not isinstance(grouped_output, Tensor):
            raise TypeError("grouped_output must be a Tensor.")
        if grouped_output.dim() != 3:
            raise ValueError(
                "grouped_output must have shape (context_count, members_per_group, output_dim)."
            )
        if tuple(grouped_output.shape[:2]) != expected_leading_shape:
            raise ValueError(
                f"grouped_output leading dimensions must equal {expected_leading_shape}."
            )
        if output_dim is not None and grouped_output.size(-1) != output_dim:
            raise ValueError(
                f"grouped_output feature dimension must equal {output_dim}."
            )
