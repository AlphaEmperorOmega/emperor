"""Independent selection/reduction oracles for complete parameter banks."""

import pytest
import torch

from emperor.sampler import SamplerConfig
from support.adaptive_generation import bias_mixture_config, weight_mixture_config


@pytest.mark.parametrize("with_indices", [True, False])
@pytest.mark.parametrize(
    "kind", ["MatrixWeightsMixtureConfig", "MatrixBiasMixtureConfig"]
)
def test_singleton_bank_preserves_independent_contexts(kind, with_indices, monkeypatch):
    from emperor.augmentations import adaptive_parameters
    from support.adaptive_grouping import linear_stack_config

    config_type = getattr(adaptive_parameters, kind)
    model = (
        config_type(
            input_dim=2,
            output_dim=3,
            num_experts=1,
            top_k=1,
            sampler_config=SamplerConfig(),
            model_config=linear_stack_config(2, 1),
        )
        .build()
        .double()
    )
    probabilities = torch.tensor(
        [[0.2], [0.8]], dtype=torch.float64, requires_grad=True
    )
    indices = torch.zeros(2, 1, dtype=torch.long) if with_indices else None
    expected = torch.stack([p * model.parameter_bank[0] for p in probabilities[:, 0]])
    monkeypatch.setattr(
        model.sampler,
        "sample_probabilities_and_indices",
        lambda context: (probabilities, indices, None, context.new_zeros(())),
    )
    output = model(
        torch.zeros_like(model.parameter_bank[0]), torch.ones(2, 2, dtype=torch.float64)
    )
    torch.testing.assert_close(output, expected)
    output.square().sum().backward()
    assert probabilities.grad.abs().sum() > 0
    assert model.parameter_bank.grad.abs().sum() > 0


def test_independent_banks_add_to_base_with_one_route_per_selected_variant(
    monkeypatch,
):
    from emperor.augmentations.adaptive_parameters import (
        AdaptiveLinearLayerConfig,
        AdaptiveParameterAugmentationConfig,
    )

    model = (
        AdaptiveLinearLayerConfig(
            input_dim=2,
            output_dim=3,
            bias_flag=True,
            adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                weight_config=weight_mixture_config(),
                bias_config=bias_mixture_config(top_k=1),
            ),
        )
        .build()
        .double()
    )
    weight = model.adaptive_behaviour.weight_model
    bias = model.adaptive_behaviour.bias_model
    with torch.no_grad():
        for variant, distribution in (
            (weight, [0.25, 0.05, 0.7]),
            (bias, [0.6, 0.3, 0.1]),
        ):
            router = variant.sampler.router.model[0].model
            router.weight_params.zero_()
            router.bias_params.copy_(torch.tensor(distribution).log())
    calls = []
    for label, variant in (("weight", weight), ("bias", bias)):
        original = variant.sampler.sample_probabilities_and_indices

        def observe(*args, _label=label, _original=original, **kwargs):
            calls.append(_label)
            return _original(*args, **kwargs)

        monkeypatch.setattr(
            variant.sampler, "sample_probabilities_and_indices", observe
        )
    x = torch.tensor([[2.0, 1.0], [-1.0, 3.0]], dtype=torch.float64)
    expected_weight = (
        0.7 * weight.parameter_bank[2] + 0.25 * weight.parameter_bank[0]
    ) / (0.95 + 1e-6)
    expected_bias = 0.6 * bias.parameter_bank[0]
    torch.testing.assert_close(
        model(x),
        x @ (model.weight_params + expected_weight) + model.bias_params + expected_bias,
    )
    assert calls == ["weight", "bias"]
    assert model.weight_params is not None and model.bias_params is not None
    assert "weight_params" in model.state_dict() and "bias_params" in model.state_dict()


@pytest.mark.parametrize("config_factory", [weight_mixture_config, bias_mixture_config])
@pytest.mark.parametrize("experts,k", [(1, 1), (3, 1), (3, 2), (3, 3)])
@pytest.mark.parametrize("normalize", [False, True])
def test_routes_follow_existing_sampler_probabilities_and_gradients(
    experts, k, normalize, config_factory
):
    cfg = config_factory(
        num_experts=experts,
        top_k=k,
        sampler_config=SamplerConfig(normalize_probabilities_flag=normalize),
    )
    if k == 1 and normalize:
        with pytest.raises(
            ValueError, match="normalize_probabilities_flag must be False"
        ):
            cfg.build()
        return
    model = cfg.build().double()
    x = torch.tensor([[0.1, 1.7], [-2.0, 0.6]], dtype=torch.float64, requires_grad=True)
    logits = model.sampler.router.compute_logit_scores(x)
    full = logits.softmax(-1)
    if k == experts:
        indices = torch.arange(experts).expand(2, -1)
        alpha = full
    else:
        alpha, indices = full.topk(k, dim=-1)
    if normalize:
        alpha = alpha / (alpha.sum(-1, keepdim=True) + 1e-6).detach()
    expected_w = torch.stack(
        [
            sum(alpha[c, j] * model.parameter_bank[indices[c, j]] for j in range(k))
            for c in range(2)
        ]
    )
    actual = model(torch.zeros_like(model.parameter_bank[0]), x)
    torch.testing.assert_close(actual, expected_w)
    inputs = (x, model.parameter_bank, *model.sampler.router.parameters())
    actual_grads = torch.autograd.grad(actual.square().sum(), inputs)
    expected_grads = torch.autograd.grad(expected_w.square().sum(), inputs)
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected)
    if experts > 1:
        assert any(gradient.abs().sum() > 0 for gradient in actual_grads[2:])
    if k < experts:
        selected = set(indices.flatten().tolist())
        for index in set(range(experts)) - selected:
            assert actual_grads[1][index].count_nonzero() == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 0.1},
        {"num_topk_samples": 1},
        {"noisy_topk_flag": True},
        {"switch_loss_weight": 0.2},
        {"coefficient_of_variation_loss_weight": 0.1},
        {"zero_centred_loss_weight": 0.1},
        {"mutual_information_loss_weight": 0.1},
        {"num_experts": 5},
        {"top_k": 3},
    ],
)
def test_unsupported_or_disagreeing_sampler_options_fail_before_initialization(kwargs):
    rng = torch.get_rng_state().clone()
    with pytest.raises((ValueError, TypeError)):
        weight_mixture_config(sampler_config=SamplerConfig(**kwargs)).build()
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_experts": True},
        {"num_experts": 0},
        {"top_k": 0},
        {"top_k": 4},
    ],
)
def test_invalid_weight_mixture_configuration_fails_early(kwargs):
    with pytest.raises((TypeError, ValueError)):
        weight_mixture_config(**kwargs).build()


@pytest.mark.parametrize(
    "bad_probs,bad_indices",
    [
        (torch.ones(2, 2), torch.tensor([[0, 3], [0, 1]])),
        (torch.ones(2, 2), torch.tensor([[0.0, 1.0], [0.0, 1.0]])),
        (torch.ones(2, 2), torch.tensor([[0], [1]])),
        (torch.ones(3, 2), torch.tensor([[0, 1], [1, 2]])),
        (torch.tensor([[float("nan"), 1.0]]), torch.tensor([[0, 1]])),
        (None, torch.tensor([[0, 1]])),
        (torch.ones(2, 2), None),
    ],
)
@pytest.mark.parametrize("config_factory", [weight_mixture_config, bias_mixture_config])
def test_malformed_sampler_routes_cannot_broadcast_silently(
    bad_probs, bad_indices, config_factory, monkeypatch
):
    model = config_factory().build()
    monkeypatch.setattr(
        model.sampler,
        "sample_probabilities_and_indices",
        lambda context: (bad_probs, bad_indices, None, context.new_zeros(())),
    )
    with pytest.raises((TypeError, ValueError)):
        model(torch.zeros_like(model.parameter_bank[0]), torch.ones(2, 2))


@pytest.mark.parametrize("bias", [False, True])
def test_linear_owns_bias_policy_and_constructs_base_parameters(bias, monkeypatch):
    from emperor.augmentations.adaptive_parameters import (
        AdaptiveLinearLayerConfig,
        AdaptiveParameterAugmentationConfig,
    )
    from emperor.linears import LinearAbstract

    calls = []
    original = LinearAbstract._create_weight_parameters

    def observe(model):
        calls.append(model)
        return original(model)

    monkeypatch.setattr(LinearAbstract, "_create_weight_parameters", observe)
    layer = AdaptiveLinearLayerConfig(
        input_dim=2,
        output_dim=3,
        bias_flag=bias,
        adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
            weight_config=weight_mixture_config(),
            bias_config=bias_mixture_config() if bias else None,
        ),
    ).build()
    assert sum(owner is layer for owner in calls) == 1
    assert layer.weight_params is not None
    assert (layer.bias_params is not None) is bias
    assert (layer.adaptive_behaviour.bias_model is not None) is bias
    assert layer(torch.randn(2, 2)).shape == (2, 3)
