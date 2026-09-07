"""Variable chunks are selectable through ordinary Model Package configuration."""

import pytest
import torch

from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterGroupingScopeOptions as Scope,
)
from emperor.augmentations.adaptive_parameters import (
    AdditiveDynamicBiasConfig,
)
from models.catalog import model_package


@pytest.mark.parametrize(
    "prefix",
    [
        "encoder_attn_expert_adaptive_",
        "decoder_self_attn_expert_adaptive_",
        "decoder_cross_attn_expert_adaptive_",
    ],
)
def test_optional_attention_expert_component_defaults_inherit_shared_settings(prefix):
    package = model_package("transformer/expert_linear_adaptive")
    common = dict(
        attention_expert_adaptive_bias_option_flag=True,
        attention_expert_adaptive_bias_option=AdditiveDynamicBiasConfig,
    )
    values = {
        prefix + suffix: None
        for suffix in (
            "generator_stack_residual_model_flag",
            "generator_depth",
            "bias_option_flag",
        )
    }
    runtime = package.bind_runtime_defaults({**common, **values})
    for field in (
        "encoder_attention_expert_adaptive_options",
        "decoder_self_attention_expert_adaptive_options",
        "decoder_cross_attention_expert_adaptive_options",
    ):
        options = getattr(runtime, field)
        assert (
            options.bias_option_flag
            and options.bias_option is AdditiveDynamicBiasConfig
        )
        assert options.generator_depth is not None
        assert options.generator_stack_options.residual_model_flag is False
    package.build_configuration(config_overrides={**common, **values})


def test_attention_expert_path_source_defaults_inherit_shared_then_override(
    monkeypatch,
):
    from models.transformer.expert_linear_adaptive import config, runtime_defaults

    assert config.ENCODER_ATTN_EXPERT_ADAPTIVE_BIAS_OPTION_FLAG is None
    monkeypatch.setattr(config, "ATTENTION_EXPERT_ADAPTIVE_BIAS_OPTION_FLAG", True)
    monkeypatch.setattr(
        config, "ATTENTION_EXPERT_ADAPTIVE_BIAS_OPTION", AdditiveDynamicBiasConfig
    )
    monkeypatch.setattr(
        config, "ENCODER_ATTN_EXPERT_ADAPTIVE_GROUPING_SCOPE", Scope.ROWS
    )
    monkeypatch.setattr(config, "ENCODER_ATTN_EXPERT_ADAPTIVE_CHUNK_SIZE", 5)
    runtime = runtime_defaults.runtime_from_config()
    encoder = runtime.encoder_attention_expert_adaptive_options
    assert encoder.bias_option_flag is True
    assert encoder.bias_option is AdditiveDynamicBiasConfig
    assert encoder.grouping_config.chunk_size == 5
    assert runtime.decoder_self_attention_expert_adaptive_options is None
    resolved = runtime_defaults.runtime_from_flat({}, runtime)
    assert resolved.decoder_self_attention_expert_adaptive_options.bias_option_flag
    assert (
        resolved.decoder_self_attention_expert_adaptive_options.grouping_config is None
    )


@pytest.mark.parametrize(
    "key",
    [
        "linears/linear_adaptive",
        "experts/linear_adaptive",
        "transformer/linear_adaptive",
        "transformer/expert_linear_adaptive",
    ],
)
def test_chunk_size_scalar_round_trip_and_resolved_sizing_conflict(key):
    import importlib

    from model_runtime.inspection import configuration_schema
    from model_runtime.packages import parse_config_value, serialize_config_value

    package = model_package(key)
    config = importlib.import_module("models." + key.replace("/", ".") + ".config")
    schema = {field.key: field for field in configuration_schema(package).fields}
    for name in vars(config):
        if not name.endswith("CHUNK_SIZE"):
            continue
        assert name in schema and getattr(config, name) is None
        assert parse_config_value(config, name, str(serialize_config_value(5))) == 5
        assert parse_config_value(config, name, "None") is None
    with pytest.raises(ValueError, match="Exactly one"):
        package.bind_runtime_defaults(
            dict(grouping_scope=Scope.ROWS, chunk_size=5, group_count=2)
        )


@pytest.mark.parametrize(
    "key", ["transformer/linear_adaptive", "transformer/expert_linear_adaptive"]
)
def test_path_sizing_switch_and_all_grouping_field_clearing(key):
    package = model_package(key)
    common = dict(grouping_scope=Scope.ROWS, group_count=2)
    with pytest.raises(ValueError, match="Exactly one"):
        package.bind_runtime_defaults({**common, "encoder_ff_adaptive_chunk_size": 5})
    switched = package.bind_runtime_defaults(
        {
            **common,
            "encoder_ff_adaptive_chunk_size": 5,
            "encoder_ff_adaptive_group_count": None,
        }
    )
    assert (
        switched.encoder_feed_forward_adaptive_options.grouping_config.chunk_size == 5
    )
    fields = (
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
    cleared = package.bind_runtime_defaults(
        {
            "grouping_scope": Scope.ROWS,
            "chunk_size": 5,
            **dict.fromkeys("encoder_ff_adaptive_" + name for name in fields),
        }
    )
    assert cleared.encoder_feed_forward_adaptive_options.grouping_config is None


def test_expert_package_main_grouping_leaves_router_and_boundaries_disabled():
    runtime = model_package("experts/linear_adaptive").bind_runtime_defaults(
        dict(grouping_scope=Scope.ROWS, chunk_size=5)
    )
    assert runtime.grouping_config.chunk_size == 5
    assert runtime.router_grouping_config is None
    assert runtime.input_boundary_options.grouping_config is None
    assert runtime.output_boundary_options.grouping_config is None


@pytest.mark.parametrize("key", ["linears/linear_adaptive", "experts/linear_adaptive"])
def test_standalone_packages_expose_variable_chunk_size(key):
    package = model_package(key)
    configuration = package.build_configuration(
        config_overrides={
            "input_dim": 2,
            "hidden_dim": 2,
            "output_dim": 2,
            "grouping_scope": Scope.ROWS,
            "chunk_size": 5,
            "bias_option_flag": True,
            "bias_option": AdditiveDynamicBiasConfig,
        }
    )
    model = package.build_model(configuration)
    for count in (7, 12, 15):
        inputs = torch.randn(count, 2, requires_grad=True)
        output = model(inputs)
        hidden = output[0] if isinstance(output, tuple) else output
        assert hidden.shape == (count, 2)
        hidden.square().sum().backward()
        assert torch.isfinite(inputs.grad).all()


def test_expert_transformer_enables_encoder_attention_and_feed_forward_only():
    package = model_package("transformer/expert_linear_adaptive")
    overrides = dict(
        batch_size=2,
        vocab_size=32,
        model_dim=8,
        source_sequence_length=7,
        target_sequence_length=7,
        encoder_num_layers=1,
        decoder_num_layers=1,
        attn_num_heads=2,
        ff_stack_hidden_dim=8,
        dropout_probability=0.0,
    )
    for prefix in ("encoder_attn_expert_adaptive_", "encoder_ff_adaptive_"):
        overrides.update(
            {
                prefix + "grouping_scope": Scope.ROWS,
                prefix + "chunk_size": 5,
                prefix + "bias_option": AdditiveDynamicBiasConfig,
                prefix + "bias_option_flag": True,
            }
        )
    configuration = package.build_configuration(config_overrides=overrides)
    model = package.build_model(configuration)
    ids = torch.tensor([[2, 5, 6, 7, 8, 9, 3], [2, 8, 7, 6, 5, 4, 3]])
    output, loss = model(ids, ids)
    assert output.shape == (2, 7, 32)
    (output.square().mean() + loss).backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
