"""Independent additive variants through the ordinary augmentation slots."""

import pytest
import torch

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    DynamicBiasConfig,
    DynamicWeightConfig,
    MatrixBiasMixtureConfig,
    MatrixWeightsMixtureConfig,
)
from emperor.sampler import SamplerConfig
from support.adaptive_grouping import linear_stack_config


@pytest.mark.parametrize(
    "config_type, base, owner",
    [
        (MatrixWeightsMixtureConfig, DynamicWeightConfig, "_weights"),
        (MatrixBiasMixtureConfig, DynamicBiasConfig, "_biases"),
    ],
)
def test_matrix_mixture_configuration_belongs_to_its_parameter_family(
    config_type, base, owner
):
    from typing import get_type_hints

    assert issubclass(config_type, base)
    assert config_type.__module__.endswith(f"{owner}.config")
    assert {
        "num_experts",
        "top_k",
        "sampler_config",
    } <= config_type.__annotations__.keys()
    assert (
        config_type()
        .registry_owner()
        .__module__.endswith(f"{owner}.variants.matrix_mixture")
    )
    annotations = get_type_hints(config_type().registry_owner().__init__)
    assert annotations["cfg"] is config_type
    assert annotations["overrides"] == config_type | None


@pytest.mark.parametrize(
    "config_type, owner",
    [(MatrixWeightsMixtureConfig, "_weights"), (MatrixBiasMixtureConfig, "_biases")],
)
def test_matrix_mixture_runtime_and_validation_are_owned_by_each_family(
    config_type, owner
):
    from emperor.nn import Module

    variant = config_type().registry_owner()
    assert variant.__bases__ == (Module,)
    assert "forward" in vars(variant)
    assert not hasattr(variant, "compute_mixture")
    assert variant.VALIDATOR.__module__.endswith(f"{owner}.validation")
    assert not hasattr(variant.VALIDATOR, "resolve_config")
    assert f"_{variant.__name__}__resolve_config" in vars(variant)
    assert "validate_route" in vars(variant.VALIDATOR)


@pytest.mark.parametrize(
    "config_type", [MatrixWeightsMixtureConfig, MatrixBiasMixtureConfig]
)
def test_matrix_mixture_validators_only_check_configuration(config_type):
    import ast
    import inspect
    import textwrap

    validator = config_type().registry_owner().VALIDATOR
    validator_source = textwrap.dedent(inspect.getsource(validator))
    validator_tree = ast.parse(validator_source)
    assert not any(
        isinstance(node, ast.Return) and node.value is not None
        for node in ast.walk(validator_tree)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"deepcopy", "replace", "setattr"}
        for node in ast.walk(validator_tree)
    )


@pytest.mark.parametrize(
    "slot, config_type",
    [
        ("weight_config", MatrixWeightsMixtureConfig),
        ("bias_config", MatrixBiasMixtureConfig),
    ],
)
@pytest.mark.parametrize("top_k", [1, 2, 3])
@pytest.mark.parametrize("model_source", ["parent", "variant", "router"])
def test_owner_preflight_and_build_preserve_configuration_and_router_defaults(
    slot, config_type, top_k, model_source
):
    from copy import deepcopy

    from emperor.sampler import RouterConfig

    parent_model = linear_stack_config(2, 3)
    variant_model = linear_stack_config(2, 3)
    variant_model.hidden_dim = 5
    router_model = linear_stack_config(2, 3)
    router_model.hidden_dim = 7
    router_config = None
    selected_model = None
    expected_model = parent_model
    if model_source != "parent":
        selected_model = variant_model
        expected_model = variant_model
    if model_source == "router":
        router_config = RouterConfig(model_config=router_model)
        expected_model = router_model
    source = config_type(
        num_experts=3,
        top_k=top_k,
        model_config=selected_model,
        sampler_config=SamplerConfig(router_config=router_config),
    )
    original = deepcopy(source)
    original_parent_model = deepcopy(parent_model)
    rng = torch.get_rng_state().clone()
    assert (
        source.registry_owner().validate_owner_config(
            source, input_dim=2, output_dim=3, model_config=parent_model
        )
        is None
    )
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    layer = AdaptiveLinearLayerConfig(
        input_dim=2,
        output_dim=3,
        bias_flag=True,
        adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
            model_config=parent_model, **{slot: source}
        ),
    ).build()
    variant = getattr(layer.adaptive_behaviour, slot.replace("_config", "_model"))
    sampler = variant.cfg.sampler_config
    assert sampler.num_experts == 3 and sampler.top_k == top_k
    assert sampler.normalize_probabilities_flag is (top_k > 1)
    assert sampler.threshold == 0.0 and sampler.num_topk_samples == 0
    assert sampler.noisy_topk_flag is False
    assert sampler.router_config.input_dim == 2
    assert sampler.router_config.num_experts == 3
    assert sampler.router_config.noisy_topk_flag is False
    assert sampler.router_config.model_config == expected_model
    assert sampler.router_config.model_config is not expected_model
    assert source == original and parent_model == original_parent_model
    assert source.sampler_config is not sampler


@pytest.mark.parametrize(
    "slot, config_type",
    [
        ("weight_config", MatrixWeightsMixtureConfig),
        ("bias_config", MatrixBiasMixtureConfig),
    ],
)
@pytest.mark.parametrize(
    "invalid_options, error",
    [
        ({"num_experts": 4}, "num_experts.*match"),
        ({"top_k": 3}, "top_k.*match"),
        ({"switch_loss_weight": 0.5}, "switch_loss_weight"),
        ({"normalize_probabilities_flag": "yes"}, "normalize_probabilities_flag"),
        ({"router_config": object()}, "router_config"),
    ],
)
def test_parent_preflight_rejects_invalid_sampler_without_mutation_or_initialization(
    slot, config_type, invalid_options, error
):
    source = config_type(
        num_experts=3,
        top_k=2,
        sampler_config=SamplerConfig(**invalid_options),
    )
    config = AdaptiveLinearLayerConfig(
        input_dim=2,
        output_dim=3,
        bias_flag=True,
        adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
            model_config=linear_stack_config(2, 3), **{slot: source}
        ),
    )
    rng = torch.get_rng_state().clone()
    with pytest.raises((TypeError, ValueError), match=error):
        config.build()
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for name, value in invalid_options.items():
        assert getattr(source.sampler_config, name) is value
    assert source.input_dim is None and source.output_dim is None


@pytest.mark.parametrize(
    "config_type", [MatrixWeightsMixtureConfig, MatrixBiasMixtureConfig]
)
def test_forward_samples_once_and_uses_the_selected_probabilities_and_indices(
    config_type, monkeypatch
):
    model = config_type(
        input_dim=2,
        output_dim=3,
        num_experts=3,
        top_k=2,
        sampler_config=SamplerConfig(),
        model_config=linear_stack_config(2, 3),
    ).build()
    context = torch.ones(2, 2)
    samples = []
    forward_outputs = []
    original_sample = model.sampler.sample_probabilities_and_indices

    def observe_sample(input_matrix):
        assert input_matrix is context
        sample = original_sample(input_matrix)
        samples.append(sample)
        return sample

    monkeypatch.setattr(
        model.sampler, "sample_probabilities_and_indices", observe_sample
    )
    output_hook = model.register_forward_hook(
        lambda _module, _inputs, output: forward_outputs.append(output)
    )
    try:
        output = model(torch.zeros_like(model.parameter_bank[0]), context)
        assert len(samples) == 1
        probabilities, indices, _, _ = samples[0]
        expected = torch.stack(
            [
                sum(
                    probability * model.parameter_bank[index]
                    for probability, index in zip(
                        row_probabilities, row_indices, strict=True
                    )
                )
                for row_probabilities, row_indices in zip(
                    probabilities, indices, strict=True
                )
            ]
        )
        torch.testing.assert_close(output, expected)
        assert len(forward_outputs) == 1 and forward_outputs[0] is output
    finally:
        output_hook.remove()


@pytest.mark.parametrize(
    "config_type", [MatrixWeightsMixtureConfig, MatrixBiasMixtureConfig]
)
def test_forward_has_no_external_route_argument(config_type):
    import inspect

    model = config_type(
        input_dim=2,
        output_dim=3,
        num_experts=2,
        top_k=2,
        sampler_config=SamplerConfig(),
        model_config=linear_stack_config(2, 2),
    ).build()
    assert "route" not in inspect.signature(model.forward).parameters
    with pytest.raises(TypeError, match="unexpected keyword argument 'route'"):
        model(None, torch.ones(2, 2), route=(torch.ones(2, 2), None))


@pytest.mark.parametrize(
    "config_type", [MatrixWeightsMixtureConfig, MatrixBiasMixtureConfig]
)
@pytest.mark.parametrize("sampler_options", [{}, {"sampler_config": None}])
def test_matrix_mixture_requires_sampler_configuration_before_initialization(
    config_type,
    sampler_options,
):
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="sampler_config.*required"):
        config_type(
            input_dim=2,
            output_dim=3,
            num_experts=2,
            top_k=2,
            model_config=linear_stack_config(2, 2),
            **sampler_options,
        ).build()
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize(
    "slot, config_type",
    [
        ("weight_config", MatrixWeightsMixtureConfig),
        ("bias_config", MatrixBiasMixtureConfig),
    ],
)
def test_owner_rejects_missing_mixture_sampler_before_parameter_initialization(
    slot,
    config_type,
):
    cfg = AdaptiveLinearLayerConfig(
        input_dim=2,
        output_dim=3,
        bias_flag=True,
        adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
            model_config=linear_stack_config(2, 3),
            **{slot: config_type(num_experts=3, top_k=2)},
        ),
    )
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="sampler_config.*required"):
        cfg.build()
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize(
    "config_type", [MatrixWeightsMixtureConfig, MatrixBiasMixtureConfig]
)
def test_matrix_mixture_requires_a_router_model_as_well_as_sampler_config(config_type):
    rng = torch.get_rng_state().clone()
    with pytest.raises((TypeError, ValueError), match="model_config|LayerStackConfig"):
        config_type(
            input_dim=2,
            output_dim=3,
            num_experts=2,
            top_k=2,
            sampler_config=SamplerConfig(),
        ).build()
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize(
    "config_type", [MatrixWeightsMixtureConfig, MatrixBiasMixtureConfig]
)
@pytest.mark.parametrize("flag", [False, True, None])
def test_matrix_mixture_rejects_removed_weighting_flag(config_type, flag):
    with pytest.raises(TypeError, match="weighted_parameters_flag"):
        config_type(weighted_parameters_flag=flag)


@pytest.mark.parametrize(
    "weight_enabled,bias_enabled", [(True, False), (False, True), (True, True)]
)
def test_independent_mixture_selection_preserves_base_parameters(
    weight_enabled, bias_enabled
):
    config = AdaptiveLinearLayerConfig(
        input_dim=2,
        output_dim=3,
        bias_flag=True,
        adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
            model_config=linear_stack_config(2, 3),
            weight_config=MatrixWeightsMixtureConfig(
                num_experts=3, top_k=2, sampler_config=SamplerConfig()
            )
            if weight_enabled
            else None,
            bias_config=MatrixBiasMixtureConfig(
                num_experts=2, top_k=1, sampler_config=SamplerConfig()
            )
            if bias_enabled
            else None,
        ),
    )
    model = config.build().double()
    assert model.weight_params is not None and model.bias_params is not None
    x = torch.tensor([[1.0, 2.0], [3.0, -1.0]], dtype=torch.float64, requires_grad=True)
    augmentation = model.adaptive_behaviour
    weights = (
        augmentation.weight_model(model.weight_params, x)
        if weight_enabled
        else model.weight_params
    )
    bias = (
        augmentation.bias_model(model.bias_params, x)
        if bias_enabled
        else model.bias_params
    )
    expected = torch.einsum("bi,bio->bo", x, weights) if weight_enabled else x @ weights
    expected = expected + bias
    torch.testing.assert_close(model(x), expected)
    model(x).square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert model.weight_params.grad is not None and model.bias_params.grad is not None
    for variant in (augmentation.weight_model, augmentation.bias_model):
        if variant is not None:
            assert variant.parameter_bank.grad is not None
            assert torch.isfinite(variant.parameter_bank.grad).all()
    if weight_enabled and bias_enabled:
        assert augmentation.weight_model.sampler is not augmentation.bias_model.sampler
        assert augmentation.weight_model.num_experts == 3
        assert augmentation.bias_model.num_experts == 2


@pytest.mark.parametrize(
    "config_type,base_shape",
    [
        (MatrixWeightsMixtureConfig, (2, 3)),
        (MatrixBiasMixtureConfig, (3,)),
    ],
)
def test_mixture_adds_supplied_base_without_mutating_configuration(
    config_type, base_shape
):
    from copy import deepcopy

    source = config_type(
        num_experts=3,
        top_k=2,
        model_config=linear_stack_config(2, 3),
        sampler_config=SamplerConfig(normalize_probabilities_flag=False),
    )
    original = deepcopy(source)
    model = source.build(config_type(input_dim=2, output_dim=3)).double()
    assert source == original and source.sampler_config is not model.cfg.sampler_config
    context = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    base = torch.full(base_shape, 1000.0, dtype=torch.float64, requires_grad=True)
    output = model(base, context)
    torch.testing.assert_close(output, base + model(torch.zeros_like(base), context))
    base_gradient, context_gradient = torch.autograd.grad(
        output.sum(), (base, context), allow_unused=True
    )
    torch.testing.assert_close(base_gradient, torch.full_like(base, 4.0))
    assert context_gradient is not None and torch.isfinite(context_gradient).all()


def test_bias_mixture_cannot_enable_a_bias_disabled_linear():
    with pytest.raises(ValueError, match="bias_flag is False"):
        AdaptiveLinearLayerConfig(
            input_dim=2,
            output_dim=3,
            bias_flag=False,
            adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                model_config=linear_stack_config(2, 3),
                bias_config=MatrixBiasMixtureConfig(
                    num_experts=2, top_k=1, sampler_config=SamplerConfig()
                ),
            ),
        ).build()


def test_linear_matrix_mixtures_retain_base_parameters_and_decay_independently():
    from emperor.augmentations.adaptive_parameters import (
        AdaptiveParameterGroupingScopeOptions,
        SumGroupingConfig,
        WeightDecayScheduleOptions,
    )
    from support.adaptive_generation import bias_mixture_config, weight_mixture_config

    config = AdaptiveLinearLayerConfig(
        input_dim=2,
        output_dim=3,
        bias_flag=True,
        adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
            weight_config=weight_mixture_config(
                sampler_config=SamplerConfig(normalize_probabilities_flag=False),
                decay_schedule=WeightDecayScheduleOptions.LINEAR,
                decay_rate=0.25,
                decay_warmup_batches=1,
            ),
            bias_config=bias_mixture_config(
                sampler_config=SamplerConfig(normalize_probabilities_flag=False),
                decay_schedule=WeightDecayScheduleOptions.MULTIPLICATIVE,
                decay_rate=0.5,
                decay_warmup_batches=0,
            ),
            grouping_config=SumGroupingConfig(
                scope=AdaptiveParameterGroupingScopeOptions.ROWS, chunk_size=2
            ),
        ),
    )
    layer = config.build().double()
    assert layer.weight_params is not None and layer.bias_params is not None
    weight = layer.adaptive_behaviour.weight_model
    bias = layer.adaptive_behaviour.bias_model
    with torch.no_grad():
        layer.weight_params.fill_(2.0)
        layer.bias_params.fill_(3.0)
        weight.parameter_bank.fill_(0.5)
        bias.parameter_bank.fill_(0.25)
        for variant in (weight, bias):
            router = variant.sampler.router.model[0].model
            router.weight_params.zero_()
            router.bias_params.copy_(
                torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64).log()
            )
    base_weight = layer.weight_params.detach().clone()
    base_bias = layer.bias_params.detach().clone()
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
    for weight_factor, bias_factor in ((1.0, 1.0), (1.0, 0.5), (0.75, 0.25)):
        expected = (
            inputs @ (base_weight * weight_factor + 0.4) + base_bias * bias_factor + 0.2
        )
        output = layer(inputs)
        torch.testing.assert_close(output, expected)
    assert (
        weight._decay_policy.warmup_step.item() == 1
        and weight._decay_policy.decay_step.item() == 2
    )
    assert (
        bias._decay_policy.warmup_step.item() == 0
        and bias._decay_policy.decay_step.item() == 3
    )
    output.sum().backward()
    torch.testing.assert_close(
        layer.weight_params.grad, inputs.sum(0)[:, None].expand(2, 3) * 0.75
    )
    torch.testing.assert_close(layer.bias_params.grad, torch.full_like(base_bias, 0.75))
    assert weight.parameter_bank.grad.abs().sum() > 0
    assert bias.parameter_bank.grad.abs().sum() > 0
    torch.testing.assert_close(layer.weight_params, base_weight)
    torch.testing.assert_close(layer.bias_params, base_bias)

    restored = config.build().double()
    restored.load_state_dict(layer.state_dict(), strict=True)
    layer.eval()
    restored.eval()
    expected = inputs @ (base_weight * 0.5 + 0.4) + base_bias * 0.125 + 0.2
    torch.testing.assert_close(layer(inputs), expected)
    torch.testing.assert_close(restored(inputs), expected)
    torch.testing.assert_close(layer(inputs), expected)
    assert (
        weight._decay_policy.decay_step.item() == 2
        and bias._decay_policy.decay_step.item() == 3
    )


@pytest.mark.parametrize(
    "config_type", [MatrixWeightsMixtureConfig, MatrixBiasMixtureConfig]
)
@pytest.mark.parametrize(
    "schedule_name", ["DISABLED", "EXPONENTIAL", "LINEAR", "MULTIPLICATIVE"]
)
def test_matrix_mixture_decay_schedules_scale_only_the_base(config_type, schedule_name):
    import math

    from emperor.augmentations.adaptive_parameters import WeightDecayScheduleOptions

    schedule = WeightDecayScheduleOptions[schedule_name]
    model = (
        config_type(
            input_dim=2,
            output_dim=3,
            num_experts=3,
            top_k=3,
            sampler_config=SamplerConfig(normalize_probabilities_flag=False),
            model_config=linear_stack_config(2, 3),
            decay_schedule=schedule,
            decay_rate=0.25,
            decay_warmup_batches=2,
        )
        .build()
        .double()
    )
    with torch.no_grad():
        model.parameter_bank.fill_(1.25)
    base = torch.full_like(model.parameter_bank[0], 2.0, requires_grad=True)
    context = torch.ones(2, 2, dtype=torch.float64)
    first_factor = math.exp(-0.25) if schedule_name == "EXPONENTIAL" else 0.75
    if schedule_name == "DISABLED":
        first_factor = 1.0
    for factor in (1.0, 1.0, 1.0, first_factor):
        output = model(base, context)
        torch.testing.assert_close(output, (base * factor + 1.25).expand_as(output))
    output.sum().backward()
    torch.testing.assert_close(base.grad, torch.full_like(base, 2.0 * first_factor))
    torch.testing.assert_close(base, torch.full_like(base, 2.0))
    torch.testing.assert_close(
        model.parameter_bank, torch.full_like(model.parameter_bank, 1.25)
    )
    expected_steps = 0 if schedule_name == "DISABLED" else 2
    assert model._decay_policy.decay_step.item() == expected_steps
    assert model._decay_policy.warmup_step.item() == expected_steps


@pytest.mark.parametrize(
    "config_type", [MatrixWeightsMixtureConfig, MatrixBiasMixtureConfig]
)
@pytest.mark.parametrize(
    "invalid",
    [
        {"decay_rate": None},
        {"decay_rate": 0.0},
        {"decay_rate": -0.1},
        {"decay_rate": 1.0},
        {"decay_rate": float("inf")},
        {"decay_rate": float("nan")},
        {"decay_warmup_batches": -1},
    ],
)
def test_invalid_matrix_decay_config_fails_before_parameter_initialization(
    config_type, invalid
):
    from emperor.augmentations.adaptive_parameters import WeightDecayScheduleOptions

    config = config_type(
        **{
            "input_dim": 2,
            "output_dim": 3,
            "num_experts": 3,
            "top_k": 2,
            "sampler_config": SamplerConfig(),
            "model_config": linear_stack_config(2, 3),
            "decay_schedule": WeightDecayScheduleOptions.LINEAR,
            "decay_rate": 0.25,
            "decay_warmup_batches": 0,
            **invalid,
        }
    )
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="decay_rate|decay_warmup_batches"):
        config.build()
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize(
    "config_type", [MatrixWeightsMixtureConfig, MatrixBiasMixtureConfig]
)
def test_matrix_mixture_requires_base_parameters_before_sampling(
    config_type, monkeypatch
):
    model = config_type(
        input_dim=2,
        output_dim=3,
        num_experts=3,
        top_k=2,
        sampler_config=SamplerConfig(),
        model_config=linear_stack_config(2, 3),
    ).build()
    sampler_calls = []
    monkeypatch.setattr(
        model.sampler,
        "sample_probabilities_and_indices",
        lambda _: sampler_calls.append(True),
    )
    with pytest.raises((TypeError, ValueError), match="weight_params|bias_params"):
        model(None, torch.ones(2, 2))
    assert not sampler_calls
    assert (
        model._decay_policy.decay_step.item() == 0
        and model._decay_policy.warmup_step.item() == 0
    )
