"""Every adaptive package exposes usable generation and grouping settings."""

from copy import deepcopy
from dataclasses import fields, is_dataclass
from importlib import import_module

import pytest
import torch

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AdaptiveParameterGroupingScopeOptions,
    AdaptiveParameterInputOrderOptions,
    AttentionGroupingConfig,
    DiagonallyModulatedLowRankDynamicWeightConfig,
    LowRankFactorSourceOptions,
    MatrixBiasMixtureConfig,
    MatrixWeightsMixtureConfig,
    MeanGroupingConfig,
    MeanStdGroupingConfig,
    RMSGroupingConfig,
    SumGroupingConfig,
)
from model_runtime.inspection import configuration_schema
from model_runtime.packages import parse_config_value, serialize_config_value
from models.catalog import model_package
from support.adaptive_grouping_variants import grouping_model_config

PACKAGES = (
    "linears/linear_adaptive",
    "experts/linear_adaptive",
    "transformer/linear_adaptive",
    "transformer/expert_linear_adaptive",
    *(
        f"{family}/{backend}"
        for family in ("bert", "gpt", "vit", "neuron", "mlp_mixer")
        for backend in ("linear_adaptive", "expert_linear_adaptive")
    ),
)


def _augmentations(value):
    if isinstance(value, AdaptiveParameterAugmentationConfig):
        yield value
    elif is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            yield from _augmentations(getattr(value, item.name))
    elif isinstance(value, dict):
        for item in value.values():
            yield from _augmentations(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _augmentations(item)


@pytest.mark.parametrize("package_name", PACKAGES)
@pytest.mark.parametrize(
    "weight_option",
    (MatrixWeightsMixtureConfig, DiagonallyModulatedLowRankDynamicWeightConfig),
)
def test_generation_settings_reach_executable_adaptive_layers(
    package_name, weight_option
):
    package = model_package(package_name)
    configuration = package.build_configuration(
        config_overrides=dict(
            weight_option_flag=True,
            weight_option=weight_option,
            bias_option_flag=True,
            bias_option=MatrixBiasMixtureConfig,
            weight_mixture_num_experts=3,
            weight_mixture_top_k=2,
            bias_mixture_num_experts=5,
            bias_mixture_top_k=3,
            weight_mixture_router_generator_stack_independent_flag=True,
            weight_mixture_router_generator_stack_hidden_dim=7,
            bias_mixture_router_generator_stack_independent_flag=True,
            bias_mixture_router_generator_stack_hidden_dim=9,
            weight_input_factor_source=LowRankFactorSourceOptions.SHARED_PARAMETER,
            weight_output_factor_source=LowRankFactorSourceOptions.GENERATED,
            weight_output_factor_generator_stack_independent_flag=True,
            weight_output_factor_generator_stack_hidden_dim=7,
            weight_coefficient_generator_stack_independent_flag=True,
            weight_coefficient_generator_stack_hidden_dim=5,
            adaptive_generator_stack_hidden_dim=8,
            adaptive_generator_stack_num_layers=1,
        )
    )
    augmentations = [
        augmentation
        for augmentation in _augmentations(configuration)
        if isinstance(augmentation.weight_config, weight_option)
    ]
    assert augmentations
    for augmentation in augmentations:
        weights = augmentation.weight_config
        biases = augmentation.bias_config
        assert isinstance(biases, MatrixBiasMixtureConfig)
        assert (biases.num_experts, biases.top_k) == (5, 3)
        assert biases.sampler_config.router_config.model_config.hidden_dim == 9
        if weight_option is MatrixWeightsMixtureConfig:
            assert (weights.num_experts, weights.top_k) == (3, 2)
            assert weights.sampler_config.router_config.model_config.hidden_dim == 7
            assert weights.sampler_config is not biases.sampler_config
        else:
            assert (
                weights.input_factor_source
                is LowRankFactorSourceOptions.SHARED_PARAMETER
            )
            assert weights.input_factor_model_config is None
            assert weights.output_factor_model_config.hidden_dim == 7
            assert weights.coefficient_model_config.hidden_dim == 5

    layer_config = AdaptiveLinearLayerConfig(
        input_dim=6,
        output_dim=4,
        bias_flag=True,
        adaptive_augmentation_config=deepcopy(augmentations[0]),
    )
    layer = layer_config.registry_owner()(layer_config)
    inputs = torch.randn(6, 6, requires_grad=True)
    outputs = layer(inputs)
    assert outputs.shape == (6, 4)
    assert layer.weight_params is not None and layer.bias_params is not None
    outputs.square().sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert any(
        parameter.grad is not None
        for parameter in layer.adaptive_behaviour.parameters()
    )


@pytest.mark.parametrize("package_name", PACKAGES)
@pytest.mark.parametrize(
    "method",
    (
        SumGroupingConfig,
        MeanGroupingConfig,
        RMSGroupingConfig,
        MeanStdGroupingConfig,
        AttentionGroupingConfig,
    ),
)
def test_grouping_summaries_can_generate_mixture_parameters(package_name, method):
    overrides = dict(
        weight_option_flag=True,
        weight_option=MatrixWeightsMixtureConfig,
        weight_mixture_num_experts=3,
        weight_mixture_top_k=2,
        grouping_scope=AdaptiveParameterGroupingScopeOptions.ROWS,
        chunk_size=4,
        grouping_method=method,
        adaptive_generator_stack_hidden_dim=8,
        adaptive_generator_stack_num_layers=1,
    )
    if method in (MeanStdGroupingConfig, AttentionGroupingConfig):
        overrides["grouping_model_config"] = grouping_model_config(method)
    configuration = model_package(package_name).build_configuration(
        config_overrides=overrides
    )
    augmentations = [
        augmentation
        for augmentation in _augmentations(configuration)
        if isinstance(augmentation.weight_config, MatrixWeightsMixtureConfig)
    ]
    assert augmentations
    assert all(
        isinstance(augmentation.grouping_config, method)
        for augmentation in augmentations
    )
    layer_config = AdaptiveLinearLayerConfig(
        input_dim=6,
        output_dim=4,
        bias_flag=True,
        adaptive_augmentation_config=deepcopy(augmentations[0]),
    )
    layer = layer_config.registry_owner()(layer_config)
    inputs = torch.randn(7, 6, requires_grad=True)
    outputs = layer(inputs)
    assert outputs.shape == (7, 4)
    outputs.square().sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


@pytest.mark.parametrize("package_name", PACKAGES)
def test_new_options_are_visible_in_configuration_schema(package_name):
    schema = {
        field.key.lower(): field
        for field in configuration_schema(model_package(package_name)).fields
    }
    for key in (
        "weight_mixture_num_experts",
        "bias_mixture_num_experts",
        "weight_mixture_top_k",
        "bias_mixture_top_k",
        "weight_input_factor_source",
        "weight_output_factor_source",
        "weight_coefficient_generator_stack_independent_flag",
        "grouping_scope",
        "chunk_size",
        "grouping_method",
        "grouping_model_config",
    ):
        assert key in schema
    config_module = import_module(
        "models." + package_name.replace("/", ".") + ".config"
    )
    for key, value in (
        ("weight_option", MatrixWeightsMixtureConfig),
        ("weight_option", DiagonallyModulatedLowRankDynamicWeightConfig),
        ("bias_option", MatrixBiasMixtureConfig),
        ("weight_input_factor_source", LowRankFactorSourceOptions.SHARED_PARAMETER),
        ("grouping_method", AttentionGroupingConfig),
    ):
        serialized = serialize_config_value(value)
        assert serialized in schema[key].choices
        assert parse_config_value(config_module, key.upper(), serialized) is value


@pytest.mark.parametrize("package_name", PACKAGES)
@pytest.mark.parametrize("method", (AttentionGroupingConfig, MeanStdGroupingConfig))
def test_learned_grouping_requires_its_model_configuration(package_name, method):
    with pytest.raises(ValueError, match="model_config"):
        model_package(package_name).build_configuration(
            config_overrides=dict(
                grouping_scope=AdaptiveParameterGroupingScopeOptions.ROWS,
                chunk_size=3,
                grouping_method=method,
            )
        )


@pytest.mark.parametrize("package_name", PACKAGES)
def test_independent_coefficients_inherit_the_weight_generator_defaults(package_name):
    configuration = model_package(package_name).build_configuration(
        config_overrides=dict(
            weight_option_flag=True,
            weight_option=DiagonallyModulatedLowRankDynamicWeightConfig,
            adaptive_generator_stack_hidden_dim=8,
            weight_generator_stack_independent_flag=True,
            weight_generator_stack_hidden_dim=11,
            weight_coefficient_generator_stack_independent_flag=True,
        )
    )
    weights = [
        augmentation.weight_config
        for augmentation in _augmentations(configuration)
        if isinstance(
            augmentation.weight_config, DiagonallyModulatedLowRankDynamicWeightConfig
        )
    ]
    assert weights
    assert all(weight.coefficient_model_config.hidden_dim == 11 for weight in weights)


@pytest.mark.parametrize(
    "package_name,prefix",
    [
        *(
            (f"{family}/linear_adaptive", prefix)
            for family in ("bert", "gpt", "vit")
            for prefix in ("attn_", "ff_")
        ),
        *(
            (f"{family}/expert_linear_adaptive", "router_")
            for family in ("bert", "gpt", "vit", "neuron")
        ),
        *(
            ("experts/linear_adaptive", prefix)
            for prefix in ("router_", "input_layer_", "output_layer_")
        ),
        *(
            ("neuron/linear_adaptive", prefix)
            for prefix in ("input_layer_", "output_layer_")
        ),
        *(
            ("neuron/expert_linear_adaptive", prefix)
            for prefix in ("input_layer_", "output_layer_")
        ),
    ],
)
def test_role_mixture_and_grouping_overrides_remain_local(package_name, prefix):
    overrides = dict(
        weight_option_flag=True,
        weight_option=MatrixWeightsMixtureConfig,
        weight_mixture_num_experts=3,
    )
    overrides.update(
        {
            prefix + key: value
            for key, value in dict(
                weight_option=MatrixWeightsMixtureConfig,
                weight_mixture_num_experts=5,
                weight_mixture_top_k=2,
                grouping_scope=AdaptiveParameterGroupingScopeOptions.ROWS,
                grouping_method=MeanGroupingConfig,
                chunk_size=3,
            ).items()
        }
    )
    if prefix not in ("input_layer_", "output_layer_"):
        overrides[prefix + "weight_option_flag"] = True
    configuration = model_package(package_name).build_configuration(
        config_overrides=overrides
    )
    mixtures = [
        augmentation
        for augmentation in _augmentations(configuration)
        if isinstance(augmentation.weight_config, MatrixWeightsMixtureConfig)
    ]
    selected = [
        augmentation
        for augmentation in mixtures
        if augmentation.weight_config.num_experts == 5
    ]
    assert selected
    assert all(
        isinstance(augmentation.grouping_config, MeanGroupingConfig)
        for augmentation in selected
    )
    inherited = [
        augmentation
        for augmentation in mixtures
        if augmentation.weight_config.num_experts == 3
    ]
    assert inherited
    assert all(augmentation.grouping_config is None for augmentation in inherited)


@pytest.mark.parametrize(
    "package_name",
    [name for name in PACKAGES if not name.startswith(("linears/", "transformer/"))],
)
@pytest.mark.parametrize(
    "weight_option",
    (MatrixWeightsMixtureConfig, DiagonallyModulatedLowRankDynamicWeightConfig),
)
@pytest.mark.parametrize("grouped", (False, True))
def test_complete_models_run_with_new_parameter_generators(
    package_name, weight_option, grouped
):
    family = package_name.split("/")[0]
    config_module = import_module(
        "models." + package_name.replace("/", ".") + ".config"
    )
    dimensions = dict(
        batch_size=2,
        input_dim=6,
        output_dim=4,
        hidden_dim=8,
        sequence_length=4,
        stack_num_layers=1,
        stack_dropout_probability=0.0,
        attn_num_heads=2,
        submodule_stack_hidden_dim=8,
        submodule_stack_num_layers=1,
        expert_stack_hidden_dim=8,
        expert_stack_num_layers=1,
        router_stack_hidden_dim=8,
        router_stack_num_layers=1,
        num_experts=3,
        top_k=2,
        cluster_initial_x_axis_total_neurons=1,
        cluster_initial_y_axis_total_neurons=1,
        cluster_initial_z_axis_total_neurons=1,
        cluster_max_steps=1,
        token_mixer_stack_hidden_dim=6,
        token_mixer_num_layers=2,
        channel_mixer_stack_hidden_dim=8,
        channel_mixer_num_layers=2,
    )
    if family in ("bert", "gpt"):
        dimensions.update(input_dim=16, output_dim=16)
        inputs = torch.randint(1, 16, (2, 4))
    elif family in ("vit", "mlp_mixer"):
        dimensions.update(
            input_dim=64, image_height=8, image_patch_size=4, input_channels=1
        )
        inputs = torch.randn(2, 1, 8, 8)
    else:
        inputs = torch.randn(2, 6)
    overrides = {
        name: value
        for name, value in dimensions.items()
        if hasattr(config_module, name.upper())
    }
    overrides.update(
        weight_option_flag=True,
        weight_option=weight_option,
        bias_option_flag=True,
        bias_option=MatrixBiasMixtureConfig,
        weight_mixture_num_experts=3,
        weight_mixture_top_k=2,
        bias_mixture_num_experts=3,
        bias_mixture_top_k=2,
        weight_input_factor_source=LowRankFactorSourceOptions.SHARED_PARAMETER,
        adaptive_generator_stack_hidden_dim=8,
        adaptive_generator_stack_num_layers=1,
        weight_coefficient_generator_stack_independent_flag=True,
        weight_coefficient_generator_stack_hidden_dim=5,
    )
    if grouped:
        overrides.update(
            grouping_scope=AdaptiveParameterGroupingScopeOptions.ROWS,
            chunk_size=3,
            grouping_method=MeanGroupingConfig,
        )
        if family in ("bert", "gpt", "vit") and package_name.endswith(
            "/linear_adaptive"
        ):
            overrides.update(
                grouping_scope=AdaptiveParameterGroupingScopeOptions.SEQUENCE,
                grouping_sequence_length=5 if family == "vit" else 4,
                grouping_input_order=AdaptiveParameterInputOrderOptions.BATCH_FIRST,
                attn_grouping_input_order=AdaptiveParameterInputOrderOptions.SEQUENCE_FIRST,
            )
        elif family in ("bert", "gpt", "vit"):
            overrides.update(
                attn_grouping_scope=AdaptiveParameterGroupingScopeOptions.SEQUENCE,
                attn_grouping_sequence_length=5 if family == "vit" else 4,
                attn_grouping_input_order=AdaptiveParameterInputOrderOptions.SEQUENCE_FIRST,
            )
            if family == "vit":
                overrides.update(
                    ff_grouping_scope=AdaptiveParameterGroupingScopeOptions.SEQUENCE,
                    ff_grouping_sequence_length=5,
                    ff_grouping_input_order=AdaptiveParameterInputOrderOptions.BATCH_FIRST,
                )
        elif family == "mlp_mixer":
            for name in ("grouping_scope", "chunk_size", "grouping_method"):
                overrides["channel_mixer_" + name] = overrides.pop(name)
            if package_name.endswith("/linear_adaptive"):
                overrides.update(
                    channel_mixer_grouping_scope=AdaptiveParameterGroupingScopeOptions.SEQUENCE,
                    channel_mixer_grouping_sequence_length=4,
                    channel_mixer_grouping_input_order=AdaptiveParameterInputOrderOptions.BATCH_FIRST,
                )
    package = model_package(package_name)
    if grouped and family == "gpt":
        with pytest.raises(ValueError, match="causal"):
            model = package.build_model(
                package.build_configuration(config_overrides=overrides)
            )
            model(inputs)
        return
    model = package.build_model(package.build_configuration(config_overrides=overrides))
    outputs = model(inputs)
    tensors = outputs if isinstance(outputs, tuple) else (outputs,)
    assert all(torch.isfinite(output).all() for output in tensors)
    loss = sum(output.square().mean() for output in tensors if output.requires_grad)
    loss.backward()
    adaptive_layers = [
        module
        for module in model.modules()
        if isinstance(module, AdaptiveLinearLayerConfig().registry_owner())
    ]
    assert adaptive_layers
    assert any(
        parameter.grad is not None
        for layer in adaptive_layers
        for parameter in layer.parameters()
    )
