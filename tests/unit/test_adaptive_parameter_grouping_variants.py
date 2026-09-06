"""Configuration-owned chunking variants and model overrides through real models."""

import copy
from dataclasses import fields, replace

import pytest
import torch

from emperor._validation import _adaptive_grouping_configs
from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterAugmentationConfig,
    GroupingConfig,
)
from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterGroupingScopeOptions as Scope,
)
from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterInputOrderOptions as Order,
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
    SumGroupingConfig as SumGroupingSettings,
)
from emperor.augmentations.adaptive_parameters import (
    SummaryNormalizationOptions as Normalization,
)
from emperor.config import ConfigBase
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerStackConfig,
    MirroredLayerStackConfig,
)
from emperor.nn import Module
from support.adaptive_grouping import bias_linear, linear_stack_config
from support.adaptive_grouping_variants import (
    GROUPING_CONFIGS,
    grouping_config,
    grouping_model_config,
    with_reduction,
)


def test_optional_configuration_and_default_grouping_do_not_mutate_caller_values():
    for config in (GroupingConfig(), *(option() for option in GROUPING_CONFIGS)):
        for field in fields(config):
            assert getattr(config, field.name) is None
            assert field.default is None and field.metadata["help"]
    grouping = SumGroupingSettings(scope=Scope.ROWS, group_count=2)
    empty = SumConfig()
    torch.manual_seed(174)
    default = bias_linear(grouping)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(174)
    explicit = bias_linear(with_reduction(grouping, empty))
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert empty == SumConfig() and grouping.feature_dim is None
    assert default.adaptive_behaviour.grouper is not None
    assert bias_linear().adaptive_behaviour.grouper is None
    torch.testing.assert_close(
        default.state_dict(), explicit.state_dict(), rtol=0, atol=0
    )
    assert list(default.adaptive_behaviour.grouper.parameters()) == []
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [-1.0, 3.0], [2.0, -2.0]])
    torch.testing.assert_close(default(inputs), explicit(inputs), rtol=0, atol=0)


def test_abstract_config_has_no_owner_and_concrete_configs_use_standard_build():
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="abstract"):
        GroupingConfig().build()
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    owners = set()
    for option in GROUPING_CONFIGS:
        assert option.build is ConfigBase.build
        config = grouping_config(option)
        model = config.build(option(feature_dim=2))
        assert type(model) is config.registry_owner()
        assert isinstance(model, Module)
        assert not hasattr(model, "reducer")
        assert model.cfg.feature_dim == 2 and config.feature_dim is None
        owners.add(type(model))
    assert len(owners) == len(GROUPING_CONFIGS)


@pytest.mark.parametrize("scope", list(Scope))
@pytest.mark.parametrize(
    "chunking",
    [
        {},
        2,
        "SUM",
        GroupingConfig(),
        AttentionConfig(),
        MeanStdConfig(),
        *[
            SumConfig(summary_normalization=value)
            for value in (0, False, "NONE", SumConfig)
        ],
        *[AttentionConfig(model_config=value) for value in ({}, "linear", 1)],
        *[
            AttentionConfig(model_config=LayerStackConfig(hidden_dim=value))
            for value in (True, 0, -1, 1.5, "4")
        ],
        SumConfig(rms_norm_epsilon=1e-6),
        *[
            SumConfig(
                summary_normalization=Normalization.RMS_NORM, rms_norm_epsilon=value
            )
            for value in (True, False, 0, -1, float("nan"), float("inf"), "1e-6")
        ],
    ],
)
def test_invalid_chunking_fails_before_rng_and_during_discovery(scope, chunking):
    grouping = chunking
    if isinstance(grouping, GroupingConfig):
        grouping = replace(grouping, scope=scope, group_count=2)
        if scope is Scope.SEQUENCE:
            grouping = replace(
                grouping, sequence_length=4, input_order=Order.BATCH_FIRST
            )
    config = copy.deepcopy(bias_linear().cfg)
    config.adaptive_augmentation_config.grouping_config = grouping
    before = torch.get_rng_state().clone()
    with pytest.raises((TypeError, ValueError)):
        config.build()
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    with pytest.raises((TypeError, ValueError)):
        tuple(_adaptive_grouping_configs(config, root="owner"))
    if isinstance(chunking, GroupingConfig):
        with pytest.raises((TypeError, ValueError)):
            chunking.build(type(chunking)(feature_dim=2))
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


@pytest.mark.parametrize("option", GROUPING_CONFIGS)
@pytest.mark.parametrize("normalization", list(Normalization))
def test_config_build_preserves_module_construction_and_caller_value(
    option, normalization
):
    config = grouping_config(option, summary_normalization=normalization)
    original = copy.deepcopy(config)
    owner_config = bias_linear(
        replace(config, scope=Scope.ROWS, group_count=2)
    ).cfg.adaptive_augmentation_config
    torch.manual_seed(273)
    direct = config.build(option(feature_dim=2)).double()
    after_build_rng = torch.get_rng_state().clone()
    inputs = torch.tensor(
        [[[1.0, 2.0], [-2.0, 4.0]], [[3.0, 1.0], [4.0, -1.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    expected = direct.summarize(inputs)
    expected.square().sum().backward()
    torch.manual_seed(273)
    owner = owner_config.build(
        AdaptiveParameterAugmentationConfig(input_dim=2, output_dim=2)
    )
    owned = owner.grouper.double()
    torch.testing.assert_close(direct.state_dict(), owned.state_dict(), rtol=0, atol=0)
    other_inputs = inputs.detach().clone().requires_grad_()
    actual = owned.summarize(other_inputs)
    actual.square().sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(other_inputs.grad, inputs.grad, rtol=0, atol=0)
    assert config == original
    torch.manual_seed(273)
    rebuilt = config.build(option(feature_dim=2))
    torch.testing.assert_close(torch.get_rng_state(), after_build_rng, rtol=0, atol=0)
    assert not (
        {id(p) for p in direct.parameters()} & {id(p) for p in rebuilt.parameters()}
    )


@pytest.mark.parametrize("feature_dim", [None, True, False, 0, -1, 2.5, "2"])
def test_config_build_rejects_invalid_dimensions_before_initialization(feature_dim):
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="feature_dim"):
        grouping_config(AttentionConfig).build(AttentionConfig(feature_dim=feature_dim))
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


@pytest.mark.parametrize("option", GROUPING_CONFIGS)
@pytest.mark.parametrize("normalization", list(Normalization))
def test_only_selected_parameters_are_registered_and_defaults_stay_unspecified(
    option, normalization
):
    config = grouping_config(option, summary_normalization=normalization)
    grouping = replace(config, scope=Scope.ROWS, group_count=2)
    first = bias_linear(grouping).adaptive_behaviour.grouper
    second = bias_linear(grouping).adaptive_behaviour.grouper
    expected_count = (
        256 if option is AttentionConfig else 10 if option is MeanStdConfig else 0
    )
    if normalization is Normalization.RMS_NORM:
        expected_count += 2
        assert first.normalizer.eps == 1e-6
        torch.testing.assert_close(first.normalizer.weight, torch.ones(2))
        assert tuple(dict(first.named_parameters()))[-1] == "normalizer.weight"
    assert sum(p.numel() for p in first.parameters()) == expected_count
    assert not (
        {id(p) for p in first.parameters()} & {id(p) for p in second.parameters()}
    )
    assert config.feature_dim is None and config.rms_norm_epsilon is None
    with pytest.raises(TypeError):
        option(None)


@pytest.mark.parametrize("option", [AttentionConfig, MeanStdConfig])
@pytest.mark.parametrize("stack_type", [LayerStackConfig, MirroredLayerStackConfig])
def test_supplied_stack_owns_architecture_and_bias_with_only_dimension_overrides(
    option,
    stack_type,
):
    template = linear_stack_config(7, 9)
    stack = stack_type(
        **{field.name: getattr(template, field.name) for field in fields(template)}
    )
    stack.hidden_dim = 5
    stack.num_layers = 3
    stack.layer_config.activation = ActivationOptions.RELU
    stack.last_layer_bias_option = (
        LastLayerBiasOptions.ENABLED
        if option is AttentionConfig
        else LastLayerBiasOptions.DISABLED
    )
    original = copy.deepcopy(stack)
    config = option(scope=Scope.ROWS, group_count=2, model_config=stack)
    module = config.build(option(feature_dim=2)).double()
    network = module.scorer if option is AttentionConfig else module.projection
    assert network.cfg.input_dim == (2 if option is AttentionConfig else 4)
    assert network.cfg.output_dim == (1 if option is AttentionConfig else 2)
    assert type(network) is stack.registry_owner()
    model_attribute = "scorer" if option is AttentionConfig else "projection"
    assert module.get_submodule(model_attribute) is network
    assert not hasattr(module, "reducer")
    state_keys = tuple(module.state_dict())
    assert any(key.startswith(f"{model_attribute}.") for key in state_keys)
    assert not any(key.startswith("reducer.") for key in state_keys)
    assert network.cfg.hidden_dim == 5
    assert len(network) == (6 if stack_type is MirroredLayerStackConfig else 3)
    assert network[0].cfg.activation is ActivationOptions.RELU
    assert network[-1].model.bias_flag is (option is AttentionConfig)
    assert stack == original and config.feature_dim is None
    inputs = torch.randn(2, 3, 2, dtype=torch.float64, requires_grad=True)
    output = module.summarize(inputs)
    assert output.shape == (2, 2)
    output.square().sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in network.parameters()
    )
    restored = config.build(option(feature_dim=2)).double()
    restored.load_state_dict(module.state_dict(), strict=True)
    torch.testing.assert_close(restored.summarize(inputs.detach()), output)


@pytest.mark.parametrize("option", GROUPING_CONFIGS)
@pytest.mark.parametrize("field", ["method", "attention_config", "unexpected_field"])
def test_removed_or_unsupported_fields_fail_before_initialization(option, field):
    config = option()
    setattr(config, field, None)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="unsupported fields"):
        config.build(option(feature_dim=2))
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


@pytest.mark.parametrize("scope", list(Scope))
def test_unknown_grouping_fields_cannot_silently_select_default_chunking(scope):
    grouping = SumGroupingSettings(scope=scope, group_count=2)
    if scope is Scope.SEQUENCE:
        grouping = replace(grouping, sequence_length=4, input_order=Order.BATCH_FIRST)
    object.__setattr__(grouping, "unexpected_field", None)
    config = copy.deepcopy(bias_linear().cfg)
    config.adaptive_augmentation_config.grouping_config = grouping
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="unsupported fields"):
        config.build()
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    with pytest.raises(ValueError, match="unsupported fields"):
        tuple(_adaptive_grouping_configs(config, root="owner"))


def test_chunking_inherits_or_replaces_as_one_grouping_value():
    original = AttentionConfig(
        model_config=grouping_model_config(AttentionConfig, width=3),
        summary_normalization=Normalization.RMS_NORM,
        scope=Scope.ROWS,
        group_count=2,
    )
    owner = copy.deepcopy(bias_linear(original).cfg.adaptive_augmentation_config)
    inherited = owner.build(
        AdaptiveParameterAugmentationConfig(input_dim=2, output_dim=2)
    )
    assert inherited.grouping_config == original
    replacement = with_reduction(original, MeanConfig())
    updated = owner.build(
        AdaptiveParameterAugmentationConfig(
            input_dim=2,
            output_dim=2,
            grouping_config=replacement,
        )
    )
    assert updated.grouping_config == replacement
    assert not list(updated.grouper.parameters())
    assert owner.grouping_config == original
    assert (
        replace(owner, grouping_config=None)
        .build(AdaptiveParameterAugmentationConfig(input_dim=2, output_dim=2))
        .grouper
        is None
    )


@pytest.mark.parametrize("option", [AttentionConfig, MeanStdConfig])
@pytest.mark.parametrize("single_layer", [False, True])
def test_supplied_model_owns_initialization_and_can_be_a_single_layer(
    option, single_layer
):
    from emperor.layers import LayerConfig, LayerState

    model_config = grouping_model_config(option)
    if single_layer:
        model_config = model_config.layer_config
    original = copy.deepcopy(model_config)
    input_dim = 2 if option is AttentionConfig else 4
    output_dim = 1 if option is AttentionConfig else 2
    overrides = type(model_config)(input_dim=input_dim, output_dim=output_dim)
    torch.manual_seed(92)
    expected = model_config.build(overrides)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(92)
    grouper = option(
        scope=Scope.ROWS, group_count=2, feature_dim=2, model_config=model_config
    ).build()
    actual = grouper.scorer if option is AttentionConfig else grouper.projection
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.state_dict(), expected.state_dict(), rtol=0, atol=0
    )
    assert model_config == original
    if single_layer:
        assert type(actual) is LayerConfig().registry_owner()
    inputs = torch.randn(3, 4, 2, requires_grad=True)
    if option is AttentionConfig:
        scores = expected(LayerState(hidden=inputs.reshape(-1, 2))).hidden.reshape(
            3, 4, 1
        )
        reference = (scores.softmax(1) * inputs).sum(1)
    else:
        deviation, mean = torch.std_mean(inputs, dim=1, correction=0)
        statistics = torch.cat((mean, deviation), dim=-1)
        reference = expected(LayerState(hidden=statistics)).hidden
    output = grouper.summarize(inputs)
    torch.testing.assert_close(output, reference)
    output.square().sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


@pytest.mark.parametrize("option", [AttentionConfig, MeanStdConfig])
@pytest.mark.parametrize("invalid_output", ["tensor", "shape", "integer", "loss"])
def test_supplied_model_output_contract_is_validated(option, invalid_output):
    from emperor.layers import LayerState

    grouper = grouping_config(option, feature_dim=2).build()
    network = grouper.scorer if option is AttentionConfig else grouper.projection

    def replace_output(_module, _inputs, output):
        if invalid_output == "tensor":
            return output.hidden
        if invalid_output == "shape":
            return LayerState(hidden=output.hidden[:, :0])
        if invalid_output == "integer":
            return LayerState(hidden=output.hidden.long())
        return LayerState(hidden=output.hidden, loss=output.hidden.sum())

    messages = {
        "tensor": "LayerState",
        "shape": "shape",
        "integer": "floating-point",
        "loss": "auxiliary loss",
    }
    handle = network.register_forward_hook(replace_output)
    try:
        with pytest.raises((TypeError, ValueError), match=messages[invalid_output]):
            grouper.summarize(torch.randn(2, 3, 2))
    finally:
        handle.remove()
