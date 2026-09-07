"""Numerical contracts for contextual rank modulation and complete parameter sets."""

import pytest
import torch

from emperor.augmentations.adaptive_parameters import (
    DynamicDepthOptions,
    LowRankFactorSourceOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
)
from emperor.layers import Layer, LayerStack
from support.adaptive_generation import modulated_config
from support.adaptive_grouping import linear_stack_config


def test_shared_factors_with_identity_coefficients_add_to_base():
    shared = LowRankFactorSourceOptions.SHARED_PARAMETER
    model = (
        modulated_config(input_factor_source=shared, output_factor_source=shared)
        .build()
        .double()
    )
    with torch.no_grad():
        coefficient_output = model.coefficient_model[-1].model
        coefficient_output.weight_params.zero_()
        coefficient_output.bias_params.fill_(1.0)
    inputs = torch.tensor([[1.0, 2.0, 3.0], [2.0, -1.0, 0.0]], dtype=torch.float64)
    base = torch.arange(6, dtype=torch.float64).reshape(3, 2).requires_grad_()
    expected_update = model.input_factor @ model.output_factor
    actual = model(base, inputs)
    torch.testing.assert_close(actual, (base + expected_update).expand(2, -1, -1))
    actual.square().sum().backward()
    assert model.input_factor.grad.abs().sum() > 0
    assert model.output_factor.grad.abs().sum() > 0
    assert base.grad.abs().sum() > 0


@pytest.mark.parametrize("input_source", list(LowRankFactorSourceOptions))
@pytest.mark.parametrize("output_source", list(LowRankFactorSourceOptions))
def test_all_factor_sources_match_dense_diagonal_and_propagate_gradients(
    input_source, output_source
):
    model = (
        modulated_config(
            input_factor_source=input_source, output_factor_source=output_source
        )
        .build()
        .double()
    )
    with torch.no_grad():
        coefficient_output = model.coefficient_model[-1].model
        coefficient_output.weight_params.fill_(0.17)
        coefficient_output.bias_params.copy_(torch.tensor([-0.7, 1.3]))
    context = torch.tensor(
        [[1.0, -2.0, 0.5], [3.0, 0.25, -1.0]], dtype=torch.float64, requires_grad=True
    )
    base = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    u = (
        model.input_model(context).transpose(1, 2)
        if model.input_model is not None
        else model.input_factor.expand(2, -1, -1)
    )
    v = (
        model.output_model(context)
        if model.output_model is not None
        else model.output_factor.expand(2, -1, -1)
    )
    d = Layer.run_model_from_hidden(model.coefficient_model, context).hidden
    expected = torch.stack([base + u[c] @ torch.diag(d[c]) @ v[c] for c in range(2)])
    actual = model(base, context)
    torch.testing.assert_close(actual, expected)
    differentiated = (
        context,
        base,
        *(
            p
            for name, p in model.named_parameters()
            if name
            not in ("_normalization_policy.scale", "_normalization_policy.clamp_limit")
        ),
    )
    actual_gradients = torch.autograd.grad(
        actual.square().sum(), differentiated, retain_graph=True
    )
    reference_gradients = torch.autograd.grad(
        expected.square().sum(), differentiated, retain_graph=True
    )
    for observed, reference in zip(actual_gradients, reference_gradients, strict=True):
        torch.testing.assert_close(observed, reference)
    actual.square().sum().backward()
    assert context.grad.abs().sum() > 0
    for name, parameter in model.named_parameters():
        if name in ("_normalization_policy.scale", "_normalization_policy.clamp_limit"):
            continue
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert (model.input_factor is None) == (
        input_source is LowRankFactorSourceOptions.GENERATED
    )
    assert (model.output_factor is None) == (
        output_source is LowRankFactorSourceOptions.GENERATED
    )


def normalize_reference(values, option, scale, limit):
    if option is WeightNormalizationOptions.DISABLED:
        return values
    if option is WeightNormalizationOptions.CLAMP:
        return values.clamp(-limit.abs(), limit.abs())
    if option is WeightNormalizationOptions.L2_SCALE:
        return values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12) * scale
    if option is WeightNormalizationOptions.RMS:
        return (
            values / (values.square().mean(dim=-1, keepdim=True).sqrt() + 1e-8) * scale
        )
    if option is WeightNormalizationOptions.SOFT_CLAMP:
        return limit.abs() * (values / limit.abs()).tanh()
    return (2 * values.sigmoid() - 1) * scale


@pytest.mark.parametrize(
    "sources",
    [(a, b) for a in LowRankFactorSourceOptions for b in LowRankFactorSourceOptions],
)
@pytest.mark.parametrize("normalization", list(WeightNormalizationOptions))
@pytest.mark.parametrize("rank,contexts", [(1, 1), (2, 3)])
def test_normalization_precedes_rank_coefficients_on_feature_axes(
    sources, normalization, rank, contexts
):
    model = (
        modulated_config(
            input_factor_source=sources[0],
            output_factor_source=sources[1],
            normalization_option=normalization,
            generator_depth=DynamicDepthOptions(rank),
        )
        .build()
        .double()
    )
    with torch.no_grad():
        coefficient_output = model.coefficient_model[-1].model
        coefficient_output.weight_params.zero_()
        coefficient_output.bias_params.copy_(
            torch.tensor([0.0] if rank == 1 else [-0.5, 1.7])
        )
    x = torch.randn(contexts, 3, dtype=torch.float64)
    base = torch.randn(3, 2, dtype=torch.float64)
    u = (
        model.input_model(x)
        if model.input_model is not None
        else model.input_factor.T.expand(contexts, -1, -1)
    )
    v = (
        model.output_model(x)
        if model.output_model is not None
        else model.output_factor.expand(contexts, -1, -1)
    )
    u = normalize_reference(
        u,
        normalization,
        model._normalization_policy.scale,
        model._normalization_policy.clamp_limit,
    )
    v = normalize_reference(
        v,
        normalization,
        model._normalization_policy.scale,
        model._normalization_policy.clamp_limit,
    )
    d = coefficient_output.bias_params
    expected = torch.stack(
        [base + u[c].T @ torch.diag(d) @ v[c] for c in range(contexts)]
    )
    torch.testing.assert_close(model(base, x), expected)


@pytest.mark.parametrize("schedule", list(WeightDecayScheduleOptions))
@pytest.mark.parametrize("normalization", list(WeightNormalizationOptions))
def test_identity_coefficients_match_original_low_rank_outputs_and_gradients(
    schedule, normalization
):
    from dataclasses import fields

    from emperor.augmentations.adaptive_parameters import LowRankDynamicWeightConfig

    config = modulated_config(
        normalization_option=normalization, decay_schedule=schedule, decay_rate=0.1
    )
    current = config.build().double()
    with torch.no_grad():
        coefficient_output = current.coefficient_model[-1].model
        coefficient_output.weight_params.zero_()
        coefficient_output.bias_params.fill_(1.0)
    old = (
        LowRankDynamicWeightConfig(
            **{
                f.name: getattr(config, f.name)
                for f in fields(LowRankDynamicWeightConfig)
            }
        )
        .build()
        .double()
    )
    old.input_model.load_state_dict(current.input_model.state_dict())
    old.output_model.load_state_dict(current.output_model.state_dict())
    for training in (True, False):
        current.train(training)
        old.train(training)
        x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        base = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
        actual = current(base, x)
        expected = old(base, x)
        torch.testing.assert_close(actual, expected)
        actual_grads = torch.autograd.grad(actual.square().sum(), (x, base))
        expected_grads = torch.autograd.grad(expected.square().sum(), (x, base))
        for actual_grad, expected_grad in zip(
            actual_grads, expected_grads, strict=True
        ):
            torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize(
    "sources",
    [(a, b) for a in LowRankFactorSourceOptions for b in LowRankFactorSourceOptions],
)
def test_state_roundtrip_and_context_isolation(sources):
    cfg = modulated_config(
        input_factor_source=sources[0], output_factor_source=sources[1]
    )
    model = cfg.build().double().eval()
    with torch.no_grad():
        model.coefficient_model[-1].model.weight_params.fill_(0.3)
    clone = cfg.build().double().eval()
    clone.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(3, 3, dtype=torch.float64)
    base = torch.randn(3, 2, dtype=torch.float64)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    expected = model(base, x)
    torch.testing.assert_close(clone(base, x), expected)
    perturbed = x.clone()
    perturbed[0] += 1
    actual = model(base, perturbed)
    torch.testing.assert_close(actual[1:], expected[1:], rtol=0, atol=0)
    assert not torch.equal(actual[0], expected[0])
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"input_factor_source": True},
        {
            "input_factor_source": LowRankFactorSourceOptions.SHARED_PARAMETER,
            "input_factor_model_config": linear_stack_config(3, 2),
        },
        {"generator_depth": DynamicDepthOptions.DISABLED},
        {"coefficient_model_config": object()},
    ],
)
def test_invalid_modulated_sources_and_trunks_fail_before_parameter_initialization(
    overrides,
):
    rng = torch.get_rng_state().clone()
    with pytest.raises((TypeError, ValueError)):
        modulated_config(**overrides).build()
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


def test_coefficient_hidden_width_fallback_and_configuration_ownership():
    from copy import deepcopy
    from dataclasses import fields, replace

    from emperor.augmentations.adaptive_parameters import (
        DiagonallyModulatedLowRankDynamicWeightConfig,
        MatrixWeightsMixtureConfig,
    )

    for config_type in (
        MatrixWeightsMixtureConfig,
        DiagonallyModulatedLowRankDynamicWeightConfig,
    ):
        assert all(
            field.default is None and field.metadata["help"]
            for field in fields(config_type)
        )
    coefficient_config = replace(linear_stack_config(3, 2), hidden_dim=None)
    config = modulated_config(
        model_config=None,
        coefficient_model_config=coefficient_config,
        input_factor_source=LowRankFactorSourceOptions.SHARED_PARAMETER,
        output_factor_source=LowRankFactorSourceOptions.SHARED_PARAMETER,
    )
    before = deepcopy(config)
    model = config.build().double()
    assert config == before and coefficient_config.hidden_dim is None
    assert model.coefficient_model.hidden_dim == 3
    assert model.coefficient_model.output_dim == 2
    assert model.input_model is None and model.output_model is None
    coefficient_output = model.coefficient_model[-1].model
    context = torch.randn(4, 3, dtype=torch.float64)
    coefficients = Layer.run_model_from_hidden(model.coefficient_model, context).hidden
    torch.testing.assert_close(
        coefficients,
        context @ coefficient_output.weight_params + coefficient_output.bias_params,
    )
    with torch.no_grad():
        coefficient_output.weight_params.zero_()
        coefficient_output.bias_params.copy_(torch.tensor([-2.0, 7.0]))
    torch.testing.assert_close(
        Layer.run_model_from_hidden(
            model.coefficient_model, torch.randn(4, 3, dtype=torch.float64)
        ).hidden,
        torch.tensor([[-2.0, 7.0]], dtype=torch.float64).expand(4, -1),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("bias_flag", [False, True])
def test_coefficient_stack_directly_outputs_rank_using_configured_layers(
    num_layers, bias_flag
):
    from copy import deepcopy
    from dataclasses import replace

    coefficient_config = replace(
        linear_stack_config(3, 7), hidden_dim=5, num_layers=num_layers
    )
    coefficient_config.layer_config.layer_model_config.bias_flag = bias_flag
    original = deepcopy(coefficient_config)
    model = (
        modulated_config(
            coefficient_model_config=coefficient_config,
            input_factor_source=LowRankFactorSourceOptions.SHARED_PARAMETER,
            output_factor_source=LowRankFactorSourceOptions.SHARED_PARAMETER,
        )
        .build()
        .double()
    )
    generator = model.coefficient_model
    assert isinstance(generator, LayerStack)
    assert coefficient_config == original
    assert len(generator) == num_layers
    assert generator.output_dim == model.rank
    assert not hasattr(generator, "readout")

    context = torch.tensor(
        [[1.0, -2.0, 0.5], [3.0, 0.25, -1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    expected = context
    for layer in generator:
        linear = layer.model
        assert (linear.bias_params is not None) is bias_flag
        expected = expected @ linear.weight_params
        if bias_flag:
            expected = expected + linear.bias_params
    actual = Layer.run_model_from_hidden(generator, context).hidden
    assert actual.shape == (2, model.rank)
    torch.testing.assert_close(actual, expected)
    parameters = (context, *generator.parameters())
    actual_gradients = torch.autograd.grad(actual.square().sum(), parameters)
    expected_gradients = torch.autograd.grad(expected.square().sum(), parameters)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)
        assert actual_gradient.abs().sum() > 0
