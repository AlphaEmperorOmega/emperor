"""Executable package contracts for explicit grouping migration and scalar overrides."""

import importlib
from dataclasses import fields

import pytest
import torch

from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterGroupingScopeOptions as Scope,
)
from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterInputOrderOptions as Order,
)
from emperor.augmentations.adaptive_parameters import (
    AdditiveDynamicBiasConfig,
    SumGroupingConfig,
)
from model_runtime.inspection import configuration_schema
from model_runtime.packages.configuration import (
    parse_config_value,
    serialize_config_value,
)
from models.catalog import model_package

PACKAGES = ("transformer/linear_adaptive", "transformer/expert_linear_adaptive")
FIELDS = (
    "grouping_scope",
    "group_count",
    "grouping_sequence_length",
    "grouping_input_order",
)
TINY = dict(
    batch_size=2,
    vocab_size=32,
    model_dim=8,
    source_sequence_length=4,
    target_sequence_length=4,
    encoder_num_layers=1,
    decoder_num_layers=1,
    attn_num_heads=2,
    ff_stack_hidden_dim=8,
    dropout_probability=0.0,
)


@pytest.mark.parametrize("key", PACKAGES)
def test_every_grouping_alias_is_discoverable_optional_and_round_trips(key):
    package = model_package(key)
    config = importlib.import_module("models." + key.replace("/", ".") + ".config")
    schema_keys = {field.key for field in configuration_schema(package).fields}
    prefixes = [
        name.removesuffix("GROUPING_SCOPE")
        for name in vars(config)
        if name.endswith("GROUPING_SCOPE")
    ]
    assert prefixes
    for prefix in prefixes:
        overrides = {}
        for suffix, value in zip(
            FIELDS, (Scope.SEQUENCE, 2, 4, Order.SEQUENCE_FIRST), strict=True
        ):
            name = prefix + suffix.upper()
            assert (
                name in schema_keys
                and name in package.runtime_defaults_spec.supported_keys
            )
            assert getattr(config, name) is None
            encoded = serialize_config_value(value)
            overrides[name.lower()] = parse_config_value(config, name, str(encoded))
            assert overrides[name.lower()] == value
            assert parse_config_value(config, name, "None") is None
        runtime = package.bind_runtime_defaults(overrides)
        assert any(
            getattr(getattr(runtime, field.name), "grouping_config", None) is not None
            for field in fields(runtime)
        )
    with pytest.raises(
        ValueError, match="Use None to leave this optional setting unset"
    ):
        parse_config_value(config, "GROUPING_SCOPE", "INVALID_OPTION")


@pytest.mark.parametrize("key", PACKAGES)
@pytest.mark.parametrize(
    "values",
    [
        {"group_count": 2},
        {"grouping_scope": Scope.SEQUENCE},
        {"grouping_sequence_length": 4},
        {"grouping_input_order": Order.BATCH_FIRST},
        {"grouping_scope": Scope.ROWS, "group_count": 2, "grouping_sequence_length": 4},
    ],
)
def test_orphan_and_incomplete_final_grouping_specs_are_rejected(key, values):
    with pytest.raises(ValueError, match="grouping"):
        model_package(key).bind_runtime_defaults(values)


@pytest.mark.parametrize("key", PACKAGES)
def test_precedence_resolves_before_constructing_nested_grouping_and_clear_is_explicit(
    key,
):
    package = model_package(key)
    category = (
        "projection_adaptive_"
        if key.endswith("/linear_adaptive")
        else "attention_projection_adaptive_"
    )
    values = dict(
        grouping_scope=Scope.SEQUENCE,
        group_count=1,
        grouping_sequence_length=4,
        grouping_input_order=Order.BATCH_FIRST,
    )
    values.update(
        {
            category + "group_count": 2,
            category + "grouping_input_order": Order.SEQUENCE_FIRST,
            "encoder_attn_adaptive_group_count": 4,
        }
    )
    runtime = package.bind_runtime_defaults(values)
    assert (
        runtime.encoder_attention_adaptive_options.grouping_config
        == SumGroupingConfig(
            scope=Scope.SEQUENCE,
            group_count=4,
            sequence_length=4,
            input_order=Order.SEQUENCE_FIRST,
        )
    )
    assert (
        runtime.decoder_self_attention_adaptive_options.grouping_config.group_count == 2
    )
    assert (
        runtime.encoder_feed_forward_adaptive_options.grouping_config.input_order
        is Order.BATCH_FIRST
    )
    values["encoder_attn_adaptive_grouping_scope"] = None
    with pytest.raises(ValueError, match="grouping"):
        package.bind_runtime_defaults(values)
    values.update({"encoder_attn_adaptive_" + field: None for field in FIELDS})
    assert (
        package.bind_runtime_defaults(
            values
        ).encoder_attention_adaptive_options.grouping_config
        is None
    )


@pytest.mark.parametrize("key", PACKAGES)
def test_default_package_executes_with_grouping_absent_and_padding_supported(key):
    package = model_package(key)
    configuration = package.build_configuration(config_overrides=TINY)
    model = package.build_model(configuration)
    ids = torch.tensor([[2, 5, 6, 0], [2, 7, 8, 3]])
    output, loss = model(ids, ids)
    assert output.shape == (2, 4, 32) and torch.isfinite(output).all()
    (output.square().mean() + loss).backward()


def test_encoder_projection_and_feed_forward_grouping_execute_with_distinct_layouts():
    package = model_package("transformer/linear_adaptive")
    overrides = dict(TINY)
    for prefix, order in [
        ("encoder_attn_adaptive_", Order.SEQUENCE_FIRST),
        ("encoder_ff_adaptive_", Order.BATCH_FIRST),
    ]:
        overrides.update(
            {
                prefix + "bias_option": AdditiveDynamicBiasConfig,
                prefix + "bias_option_flag": True,
            }
        )
        overrides.update(
            {
                prefix + field: value
                for field, value in zip(
                    FIELDS, (Scope.SEQUENCE, 2, 4, order), strict=True
                )
            }
        )
    configuration = package.build_configuration(config_overrides=overrides)
    model = package.build_model(configuration)
    ids = torch.tensor([[2, 5, 6, 3], [2, 7, 8, 3]])
    output, loss = model(ids, ids)
    assert output.shape == (2, 4, 32) and torch.isfinite(output).all()
    (output.square().mean() + loss).backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_package_preserves_incompatible_projection_order_for_owner_diagnostic():
    package = model_package("transformer/linear_adaptive")
    overrides = {
        **TINY,
        **{
            "encoder_attn_adaptive_" + field: value
            for field, value in zip(
                FIELDS, (Scope.SEQUENCE, 2, 4, Order.BATCH_FIRST), strict=True
            )
        },
    }
    overrides.update(
        encoder_attn_adaptive_bias_option=AdditiveDynamicBiasConfig,
        encoder_attn_adaptive_bias_option_flag=True,
    )
    configuration = package.build_configuration(config_overrides=overrides)
    with pytest.raises(ValueError, match="SEQUENCE_FIRST"):
        package.build_model(configuration)


def test_expert_package_keeps_grouping_rejected_on_routed_attention():
    package = model_package("transformer/expert_linear_adaptive")
    overrides = {
        **TINY,
        **{
            "encoder_attn_expert_adaptive_" + field: value
            for field, value in zip(
                FIELDS, (Scope.SEQUENCE, 2, 4, Order.SEQUENCE_FIRST), strict=True
            )
        },
    }
    overrides.update(
        encoder_attn_expert_adaptive_bias_option=AdditiveDynamicBiasConfig,
        encoder_attn_expert_adaptive_bias_option_flag=True,
    )
    configuration = package.build_configuration(config_overrides=overrides)
    with pytest.raises(ValueError, match="grouping"):
        package.build_model(configuration)
