"""Package-owned scalar grouping resolution."""

from dataclasses import dataclass, fields, replace
from typing import Any

from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterAugmentationConfig,
    AdaptiveParameterGroupingScopeOptions,
    AdaptiveParameterInputOrderOptions,
    AttentionGroupingConfig,
    GroupingConfig,
    MeanStdGroupingConfig,
    SumGroupingConfig,
    SummaryNormalizationOptions,
)
from emperor.config import ConfigBase
from models.experts.linear_adaptive import config

GROUPING_FIELDS = (
    "grouping_scope",
    "group_count",
    "chunk_size",
    "grouping_sequence_length",
    "grouping_input_order",
    "grouping_method",
    "grouping_model_config",
    "grouping_summary_normalization",
    "grouping_attention_hidden_dim",
    "grouping_rms_norm_epsilon",
)


def grouping_from_fields(
    values: dict[str, Any], *, prefix: str
) -> GroupingConfig | None:
    if all(values.get(field) is None for field in GROUPING_FIELDS):
        return None
    grouping_values = {
        "summary_normalization": values.get("grouping_summary_normalization"),
        "rms_norm_epsilon": values.get("grouping_rms_norm_epsilon"),
    }
    model_config = values.get("grouping_model_config")
    hidden_dim = values.get("grouping_attention_hidden_dim")
    method = values.get("grouping_method")
    config_type = SumGroupingConfig if method is None else method
    if not isinstance(config_type, type) or not issubclass(config_type, GroupingConfig):
        raise ValueError(
            "grouping_method must be a concrete grouping configuration class."
        )
    if hidden_dim is not None:
        if not issubclass(config_type, AttentionGroupingConfig):
            raise ValueError(
                "grouping_attention_hidden_dim requires attention grouping."
            )
        if not isinstance(model_config, ConfigBase):
            raise ValueError(
                "grouping_attention_hidden_dim requires a supplied grouping_model_config."
            )
        if "hidden_dim" not in {field.name for field in fields(model_config)}:
            raise ValueError(
                "grouping_attention_hidden_dim requires a model config with a hidden_dim field."
            )
        model_config = replace(model_config, hidden_dim=hidden_dim)
    if model_config is not None:
        if not issubclass(
            config_type, (AttentionGroupingConfig, MeanStdGroupingConfig)
        ):
            raise ValueError(
                "grouping_model_config requires attention or mean/std grouping."
            )
        grouping_values["model_config"] = model_config
    grouping = config_type(
        **grouping_values,
        scope=values.get("grouping_scope"),
        group_count=values.get("group_count"),
        chunk_size=values.get("chunk_size"),
        sequence_length=values.get("grouping_sequence_length"),
        input_order=values.get("grouping_input_order"),
    )
    validator = AdaptiveParameterAugmentationConfig().registry_owner().VALIDATOR
    try:
        validator.validate_grouping_value(grouping)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{prefix} grouping requires complete scope-specific settings; "
            "set all grouping fields to None to disable it. " + str(exc)
        ) from exc
    return grouping


@dataclass(frozen=True, kw_only=True)
class GroupingDefaultValues:
    grouping_scope: AdaptiveParameterGroupingScopeOptions | None = config.GROUPING_SCOPE
    group_count: int | None = config.GROUP_COUNT
    chunk_size: int | None = config.CHUNK_SIZE
    grouping_sequence_length: int | None = config.GROUPING_SEQUENCE_LENGTH
    grouping_input_order: AdaptiveParameterInputOrderOptions | None = (
        config.GROUPING_INPUT_ORDER
    )
    grouping_method: type[GroupingConfig] | None = config.GROUPING_METHOD
    grouping_summary_normalization: SummaryNormalizationOptions | None = (
        config.GROUPING_SUMMARY_NORMALIZATION
    )
    grouping_model_config: ConfigBase | None = config.GROUPING_MODEL_CONFIG
    grouping_attention_hidden_dim: int | None = config.GROUPING_ATTENTION_HIDDEN_DIM
    grouping_rms_norm_epsilon: float | None = config.GROUPING_RMS_NORM_EPSILON
    router_grouping_scope: AdaptiveParameterGroupingScopeOptions | None = (
        config.ROUTER_GROUPING_SCOPE
    )
    router_group_count: int | None = config.ROUTER_GROUP_COUNT
    router_chunk_size: int | None = config.ROUTER_CHUNK_SIZE
    router_grouping_sequence_length: int | None = config.ROUTER_GROUPING_SEQUENCE_LENGTH
    router_grouping_input_order: AdaptiveParameterInputOrderOptions | None = (
        config.ROUTER_GROUPING_INPUT_ORDER
    )
    router_grouping_method: type[GroupingConfig] | None = config.ROUTER_GROUPING_METHOD
    router_grouping_summary_normalization: SummaryNormalizationOptions | None = (
        config.ROUTER_GROUPING_SUMMARY_NORMALIZATION
    )
    router_grouping_model_config: ConfigBase | None = (
        config.ROUTER_GROUPING_MODEL_CONFIG
    )
    router_grouping_attention_hidden_dim: int | None = (
        config.ROUTER_GROUPING_ATTENTION_HIDDEN_DIM
    )
    router_grouping_rms_norm_epsilon: float | None = (
        config.ROUTER_GROUPING_RMS_NORM_EPSILON
    )
    input_layer_grouping_scope: AdaptiveParameterGroupingScopeOptions | None = (
        config.INPUT_LAYER_GROUPING_SCOPE
    )
    input_layer_group_count: int | None = config.INPUT_LAYER_GROUP_COUNT
    input_layer_chunk_size: int | None = config.INPUT_LAYER_CHUNK_SIZE
    input_layer_grouping_sequence_length: int | None = (
        config.INPUT_LAYER_GROUPING_SEQUENCE_LENGTH
    )
    input_layer_grouping_input_order: AdaptiveParameterInputOrderOptions | None = (
        config.INPUT_LAYER_GROUPING_INPUT_ORDER
    )
    input_layer_grouping_method: type[GroupingConfig] | None = (
        config.INPUT_LAYER_GROUPING_METHOD
    )
    input_layer_grouping_summary_normalization: SummaryNormalizationOptions | None = (
        config.INPUT_LAYER_GROUPING_SUMMARY_NORMALIZATION
    )
    input_layer_grouping_model_config: ConfigBase | None = (
        config.INPUT_LAYER_GROUPING_MODEL_CONFIG
    )
    input_layer_grouping_attention_hidden_dim: int | None = (
        config.INPUT_LAYER_GROUPING_ATTENTION_HIDDEN_DIM
    )
    input_layer_grouping_rms_norm_epsilon: float | None = (
        config.INPUT_LAYER_GROUPING_RMS_NORM_EPSILON
    )
    output_layer_grouping_scope: AdaptiveParameterGroupingScopeOptions | None = (
        config.OUTPUT_LAYER_GROUPING_SCOPE
    )
    output_layer_group_count: int | None = config.OUTPUT_LAYER_GROUP_COUNT
    output_layer_chunk_size: int | None = config.OUTPUT_LAYER_CHUNK_SIZE
    output_layer_grouping_sequence_length: int | None = (
        config.OUTPUT_LAYER_GROUPING_SEQUENCE_LENGTH
    )
    output_layer_grouping_input_order: AdaptiveParameterInputOrderOptions | None = (
        config.OUTPUT_LAYER_GROUPING_INPUT_ORDER
    )
    output_layer_grouping_method: type[GroupingConfig] | None = (
        config.OUTPUT_LAYER_GROUPING_METHOD
    )
    output_layer_grouping_summary_normalization: SummaryNormalizationOptions | None = (
        config.OUTPUT_LAYER_GROUPING_SUMMARY_NORMALIZATION
    )
    output_layer_grouping_model_config: ConfigBase | None = (
        config.OUTPUT_LAYER_GROUPING_MODEL_CONFIG
    )
    output_layer_grouping_attention_hidden_dim: int | None = (
        config.OUTPUT_LAYER_GROUPING_ATTENTION_HIDDEN_DIM
    )
    output_layer_grouping_rms_norm_epsilon: float | None = (
        config.OUTPUT_LAYER_GROUPING_RMS_NORM_EPSILON
    )
