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
