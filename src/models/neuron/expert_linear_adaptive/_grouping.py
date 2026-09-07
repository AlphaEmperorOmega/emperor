"""Package-owned scalar grouping resolution."""

from dataclasses import fields, replace
from typing import Any

from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterAugmentationConfig,
    AttentionGroupingConfig,
    GroupingConfig,
    MeanStdGroupingConfig,
    SumGroupingConfig,
)
from emperor.config import ConfigBase

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


def grouping_keys(config_module):
    return {
        name.lower()
        for name in vars(config_module)
        if name.isupper()
        and any(name.lower().endswith(field) for field in GROUPING_FIELDS)
    }


def grouping_from_overrides(values, config_module, prefix=""):
    resolved = {}
    for name in GROUPING_FIELDS:
        key = prefix + name
        if key in values:
            resolved[name] = values[key]
        elif prefix in ("attn_", "ff_") and name in values:
            resolved[name] = values[name]
        else:
            resolved[name] = getattr(config_module, key.upper(), None)
    return grouping_from_fields(resolved, prefix=prefix or "main")


def apply_grouping_options(options, values, config_module):
    result = dict(options)
    for role, prefix in (
        ("", ""),
        ("attention_", "attn_"),
        ("feed_forward_", "ff_"),
        ("router_", "router_"),
    ):
        weight_key = role + "hidden_adaptive_weight_options"
        if role == "router_":
            weight_key = "router_adaptive_weight_options"
        if weight_key in result:
            result[role + "grouping_config"] = grouping_from_overrides(
                values, config_module, prefix
            )
    for role, prefix in (
        ("input_boundary_options", "input_layer_"),
        ("output_boundary_options", "output_layer_"),
    ):
        if result.get(role) is not None:
            result[role] = replace(
                result[role],
                grouping_config=grouping_from_overrides(values, config_module, prefix),
            )
    return result
