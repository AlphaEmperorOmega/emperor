"""Public scalar chunking selection, precedence, and executable package contracts."""

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
    GroupingConfig,
)
from emperor.augmentations.adaptive_parameters import (
    AttentionGroupingConfig as AttentionConfig,
)
from emperor.augmentations.adaptive_parameters import (
    MeanGroupingConfig as MeanConfig,
)
from emperor.augmentations.adaptive_parameters import (
    MeanStdGroupingConfig as MeanStdConfig,
)
from emperor.augmentations.adaptive_parameters import (
    SumGroupingConfig as SumConfig,
)
from emperor.augmentations.adaptive_parameters import (
    SummaryNormalizationOptions as Normalization,
)
from model_runtime.inspection import configuration_schema
from model_runtime.packages import parse_config_value, serialize_config_value
from models.catalog import model_package
from support.adaptive_grouping_variants import GROUPING_CONFIGS, grouping_model_config

PACKAGES = ("transformer/linear_adaptive", "transformer/expert_linear_adaptive")
CHUNKING_FIELDS = (
    "grouping_method",
    "grouping_model_config",
    "grouping_summary_normalization",
    "grouping_attention_hidden_dim",
    "grouping_rms_norm_epsilon",
)
GROUPING_FIELDS = (
    "grouping_scope",
    "group_count",
    "grouping_sequence_length",
    "grouping_input_order",
)
GROUPING = dict(
    zip(GROUPING_FIELDS, (Scope.SEQUENCE, 2, 4, Order.BATCH_FIRST), strict=True)
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
def test_every_chunking_alias_round_trips_through_actual_schema_and_runtime(key):
    package = model_package(key)
    config = importlib.import_module("models." + key.replace("/", ".") + ".config")
    schema = {field.key: field for field in configuration_schema(package).fields}
    prefixes = [
        name.removesuffix("GROUPING_SCOPE")
        for name in vars(config)
        if name.endswith("GROUPING_SCOPE")
    ]
    for prefix in prefixes:
        overrides = {prefix.lower() + name: value for name, value in GROUPING.items()}
        for suffix, value in zip(
            CHUNKING_FIELDS,
            (
                AttentionConfig,
                grouping_model_config(AttentionConfig),
                Normalization.RMS_NORM,
                3,
                0.01,
            ),
            strict=True,
        ):
            name = prefix + suffix.upper()
            assert (
                name in schema and name in package.runtime_defaults_spec.supported_keys
            )
            assert getattr(config, name) is None
            if suffix == "grouping_model_config":
                overrides[name.lower()] = value
            else:
                overrides[name.lower()] = parse_config_value(
                    config, name, str(serialize_config_value(value))
                )
            assert overrides[name.lower()] == value
        choice = prefix + "GROUPING_SUMMARY_NORMALIZATION"
        for member in Normalization:
            encoded = serialize_config_value(member)
            assert encoded in schema[choice].choices
            assert parse_config_value(config, choice, encoded) is member
        assert parse_config_value(config, choice, "None") is None
        runtime = package.bind_runtime_defaults(overrides)
        groupings = [
            getattr(getattr(runtime, field.name), "grouping_config", None)
            for field in fields(runtime)
        ]
        chunking_configs = [grouping for grouping in groupings if grouping is not None]
        assert any(
            type(chunking) is AttentionConfig and chunking.model_config.hidden_dim == 3
            for chunking in chunking_configs
            if chunking is not None
        )


@pytest.mark.parametrize("key", PACKAGES)
@pytest.mark.parametrize(
    "values",
    [
        {**GROUPING, "grouping_method": GroupingConfig},
        {"grouping_method": SumConfig},
        {"grouping_summary_normalization": Normalization.DISABLED},
        {"grouping_attention_hidden_dim": 3},
        {"grouping_rms_norm_epsilon": 0.1},
        {
            **GROUPING,
            "grouping_method": MeanConfig,
            "grouping_attention_hidden_dim": 3,
        },
        {**GROUPING, "grouping_rms_norm_epsilon": 0.1},
        {
            **GROUPING,
            "grouping_method": AttentionConfig,
            "grouping_attention_hidden_dim": True,
        },
        {
            **GROUPING,
            "grouping_summary_normalization": Normalization.RMS_NORM,
            "grouping_rms_norm_epsilon": float("nan"),
        },
    ],
)
def test_orphan_and_irrelevant_chunking_settings_are_rejected(key, values):
    with pytest.raises((TypeError, ValueError)):
        model_package(key).bind_runtime_defaults(values)


@pytest.mark.parametrize("key", PACKAGES)
def test_final_precedence_preserves_unspecified_values_and_requires_explicit_clearing(
    key,
):
    package = model_package(key)
    category = "feed_forward_adaptive_"
    path = "encoder_ff_adaptive_"
    values = {
        **GROUPING,
        "grouping_method": AttentionConfig,
        "grouping_model_config": grouping_model_config(AttentionConfig),
        "grouping_summary_normalization": Normalization.RMS_NORM,
    }
    # Width/epsilon defaults remain unspecified while the method changes by path.
    values[path + "grouping_method"] = MeanConfig
    with pytest.raises(ValueError, match="grouping_model_config"):
        package.bind_runtime_defaults(values)
    values[path + "grouping_model_config"] = None
    runtime = package.bind_runtime_defaults(values)
    chunking = runtime.encoder_feed_forward_adaptive_options.grouping_config
    assert type(chunking) is MeanConfig
    assert chunking.rms_norm_epsilon is None
    # Explicit inherited width must instead be cleared when changing method.
    values[category + "grouping_attention_hidden_dim"] = 3
    with pytest.raises(ValueError, match="grouping_attention_hidden_dim"):
        package.bind_runtime_defaults(values)
    values[path + "grouping_attention_hidden_dim"] = None
    values[category + "grouping_rms_norm_epsilon"] = 0.01
    values[path + "grouping_summary_normalization"] = Normalization.DISABLED
    with pytest.raises(ValueError, match="rms_norm_epsilon"):
        package.bind_runtime_defaults(values)
    values[path + "grouping_rms_norm_epsilon"] = None
    runtime = package.bind_runtime_defaults(values)
    chunking = runtime.encoder_feed_forward_adaptive_options.grouping_config
    assert chunking.summary_normalization is Normalization.DISABLED
    assert (
        runtime.decoder_feed_forward_adaptive_options.grouping_config.model_config.hidden_dim
        == 3
    )
    values.update({path + name: None for name in GROUPING_FIELDS})
    with pytest.raises(ValueError, match="grouping"):
        package.bind_runtime_defaults(values)
    values.update({path + name: None for name in CHUNKING_FIELDS})
    assert (
        package.bind_runtime_defaults(
            values
        ).encoder_feed_forward_adaptive_options.grouping_config
        is None
    )


@pytest.mark.parametrize("method", GROUPING_CONFIGS)
def test_encoder_and_feed_forward_use_selected_groupers_in_a_real_package(method):
    package = model_package("transformer/linear_adaptive")
    overrides = dict(TINY)
    for prefix, order in [
        ("encoder_attn_adaptive_", Order.SEQUENCE_FIRST),
        ("encoder_ff_adaptive_", Order.BATCH_FIRST),
    ]:
        overrides.update({prefix + name: value for name, value in GROUPING.items()})
        overrides.update(
            {
                prefix + "grouping_input_order": order,
                prefix + "grouping_method": method,
                prefix + "grouping_model_config": grouping_model_config(method),
                prefix + "grouping_summary_normalization": Normalization.RMS_NORM,
                prefix + "bias_option": AdditiveDynamicBiasConfig,
                prefix + "bias_option_flag": True,
            }
        )
        if method is AttentionConfig:
            overrides[prefix + "grouping_attention_hidden_dim"] = 3
    model = package.build_model(package.build_configuration(config_overrides=overrides))
    augmentations = [
        module
        for module in model.modules()
        if getattr(module, "grouper", None) is not None
    ]
    assert augmentations and all(
        type(module.grouping_config) is method for module in augmentations
    )
    ids = torch.tensor([[2, 5, 6, 3], [2, 7, 8, 3]])
    output, loss = model(ids, ids)
    assert output.shape == (2, 4, 32) and torch.isfinite(output).all()
    (output.square().mean() + loss).backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    calls = []
    hooks = [
        module.grouper.register_forward_pre_hook(lambda *_: calls.append(True))
        for module in augmentations
    ]
    try:
        with pytest.raises(ValueError, match="all-valid"):
            model(torch.tensor([[2, 5, 6, 0], [2, 7, 8, 3]]), ids)
    finally:
        for hook in hooks:
            hook.remove()
    assert not calls


@pytest.mark.parametrize("method", GROUPING_CONFIGS)
def test_expert_package_retains_routed_grouping_rejections(method):
    package = model_package("transformer/expert_linear_adaptive")
    prefix = "encoder_attn_expert_adaptive_"
    overrides = {
        **TINY,
        **{prefix + name: value for name, value in GROUPING.items()},
        prefix + "grouping_input_order": Order.SEQUENCE_FIRST,
        prefix + "grouping_method": method,
        prefix + "grouping_model_config": grouping_model_config(method),
        prefix + "bias_option": AdditiveDynamicBiasConfig,
        prefix + "bias_option_flag": True,
    }
    config = package.build_configuration(config_overrides=overrides)
    with pytest.raises(ValueError, match="grouping"):
        package.build_model(config)


@pytest.mark.parametrize(
    "key", (*PACKAGES, "linears/linear_adaptive", "experts/linear_adaptive")
)
@pytest.mark.parametrize("method", [AttentionConfig, MeanStdConfig])
def test_learned_chunking_requires_model_config_at_package_boundary(key, method):
    package = model_package(key)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="model_config"):
        package.bind_runtime_defaults(
            {
                "grouping_scope": Scope.ROWS,
                "chunk_size": 5,
                "grouping_method": method,
            }
        )
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


@pytest.mark.parametrize("key", ["linears/linear_adaptive", "experts/linear_adaptive"])
@pytest.mark.parametrize("method", [AttentionConfig, MeanStdConfig])
def test_standalone_packages_build_the_supplied_chunking_model(key, method):
    import copy

    package = model_package(key)
    model_config = grouping_model_config(method, width=3)
    original = copy.deepcopy(model_config)
    config = package.build_configuration(
        config_overrides={
            "input_dim": 2,
            "hidden_dim": 2,
            "output_dim": 2,
            "grouping_scope": Scope.ROWS,
            "chunk_size": 5,
            "grouping_method": method,
            "grouping_model_config": model_config,
            "bias_option_flag": True,
            "bias_option": AdditiveDynamicBiasConfig,
        }
    )
    model = package.build_model(config)
    groupers = [
        module.grouper
        for module in model.modules()
        if getattr(module, "grouper", None) is not None
    ]
    assert groupers
    for grouper in groupers:
        assert grouper.cfg.model_config == model_config
    inputs = torch.randn(7, 2, requires_grad=True)
    output = model(inputs)
    hidden = output[0] if isinstance(output, tuple) else output
    assert hidden.shape == (7, 2) and torch.isfinite(hidden).all()
    hidden.square().sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert model_config == original
