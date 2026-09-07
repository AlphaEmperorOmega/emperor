"""Adaptive chunks respect the real assignments supplied by expert routing."""

import pytest
import torch

from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    MixtureOfExpertsConfig,
    RoutingInitializationMode,
)
from support.adaptive_grouping_variants import reference_mixture, variable_chunk_stack


def mixture_config(top_k=1, num_experts=3):
    return MixtureOfExpertsConfig(
        input_dim=2,
        output_dim=2,
        top_k=top_k,
        num_experts=num_experts,
        capacity_factor=0.0,
        dropped_token_behavior=DroppedTokenOptions.ZEROS,
        compute_expert_mixture_flag=True,
        weighted_parameters_flag=True,
        weighting_position_option=ExpertWeightingPositionOptions.AFTER_EXPERTS,
        routing_initialization_mode=RoutingInitializationMode.DISABLED,
        sampler_config=None,
        expert_model_config=variable_chunk_stack(),
    )


def initialize_experts(model):
    with torch.no_grad():
        for index, stack in enumerate(model.expert_modules):
            linear = stack[0].model
            linear.weight_params.copy_(torch.eye(2))
            linear.bias_params.zero_()
            generator = linear.adaptive_behaviour.bias_model.model[0].model
            generator.weight_params.copy_(torch.eye(2) * (index + 1))
            generator.bias_params.zero_()


def routed_model_config(mode):
    from dataclasses import fields

    from emperor.experts import MixtureOfExpertsLayerConfig, MixtureOfExpertsModelConfig
    from emperor.sampler import RouterConfig, SamplerConfig
    from support.adaptive_grouping import linear_stack_config

    sampler = SamplerConfig(
        top_k=1,
        num_experts=3,
        threshold=0.0,
        filter_above_threshold=False,
        num_topk_samples=0,
        normalize_probabilities_flag=False,
        noisy_topk_flag=False,
        coefficient_of_variation_loss_weight=0.0,
        switch_loss_weight=0.0,
        zero_centred_loss_weight=0.0,
        mutual_information_loss_weight=0.0,
        router_config=RouterConfig(
            input_dim=2,
            num_experts=3,
            noisy_topk_flag=False,
            model_config=linear_stack_config(2, 3),
        ),
    )
    mixture = mixture_config()
    mixture.routing_initialization_mode = mode
    mixture.sampler_config = sampler
    stack = linear_stack_config(2, 2)
    values = {
        field.name: getattr(stack.layer_config, field.name)
        for field in fields(stack.layer_config)
    }
    values["layer_model_config"] = mixture
    stack.layer_config = MixtureOfExpertsLayerConfig(**values)
    return MixtureOfExpertsModelConfig(
        input_dim=2,
        output_dim=2,
        top_k=1,
        routing_initialization_mode=mode,
        sampler_config=sampler,
        stack_config=stack,
    )


@pytest.mark.parametrize(
    "mode", [RoutingInitializationMode.LAYER, RoutingInitializationMode.SHARED]
)
def test_real_layer_and_shared_routing_backward_and_generated_masks(mode):
    from emperor.experts import MixtureOfExpertsLayerState

    config = routed_model_config(mode)
    model = config.build().double().eval()
    inputs = torch.randn(12, 2, dtype=torch.float64, requires_grad=True)
    state = model(
        MixtureOfExpertsLayerState(hidden=inputs, skip_mask=torch.ones(12, 1))
    )
    state.hidden.square().sum().backward()
    assert torch.isfinite(inputs.grad).all()
    config.sampler_config.threshold = 1.0
    config.stack_config.layer_config.layer_model_config.sampler_config.threshold = 1.0
    masked = config.build().double().eval()
    calls = []
    hooks = [
        expert.register_forward_pre_hook(lambda *_: calls.append(True))
        for expert in masked.expert_stack[0].model.expert_modules
    ]
    try:
        with pytest.raises(ValueError, match="all-active skip_mask"):
            masked(
                MixtureOfExpertsLayerState(hidden=inputs, skip_mask=torch.ones(12, 1))
            )
    finally:
        for hook in hooks:
            hook.remove()
    assert not calls


def test_shared_routing_rejects_supplied_inactive_mask_before_grouped_router():
    from dataclasses import replace

    from emperor.experts import MixtureOfExpertsLayerState

    config = routed_model_config(RoutingInitializationMode.SHARED)
    config.sampler_config.router_config.model_config = replace(
        variable_chunk_stack(), output_dim=3
    )
    model = config.build().double().eval()
    calls = []
    hook = model.shared_sampler.router.model.register_forward_pre_hook(
        lambda *_: calls.append(True)
    )
    try:
        with pytest.raises(ValueError, match="all-active skip_mask"):
            model(
                MixtureOfExpertsLayerState(
                    hidden=torch.randn(12, 2, dtype=torch.float64),
                    skip_mask=torch.zeros(12, 1),
                )
            )
    finally:
        hook.remove()
    assert not calls


def test_real_experts_restore_unequal_assignments_and_gradients():
    model = mixture_config().build().double().eval()
    initialize_experts(model)
    inputs = torch.randn(19, 2, dtype=torch.float64, requires_grad=True)
    indices = torch.tensor([0] * 12 + [1] * 7)
    probabilities = torch.ones(19, dtype=torch.float64)
    expected = torch.cat(
        [
            chunk + (index + 1) * chunk.mean(0)
            for index, part in enumerate((inputs[:12], inputs[12:]))
            for chunk in part.split(5)
        ]
    )
    output, skip, loss = model(inputs, probabilities, indices)
    assert output.shape == inputs.shape and skip is None
    torch.testing.assert_close(output, expected)
    actual_grad = torch.autograd.grad(output.square().sum() + loss, inputs)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), inputs)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert all(p.grad is None for p in model.expert_modules[2].parameters())


@pytest.mark.parametrize("top_k", [1, 2, 3])
@pytest.mark.parametrize("before", [False, True])
@pytest.mark.parametrize(
    "capacity,fallback",
    [
        (0.0, DroppedTokenOptions.ZEROS),
        (0.5, DroppedTokenOptions.ZEROS),
        (0.5, DroppedTokenOptions.IDENTITY),
    ],
)
def test_weighting_capacity_and_assignment_gradients_match_real_chunk_reference(
    top_k, before, capacity, fallback
):
    if top_k == 3 and capacity:
        pytest.skip("Dense routing does not support capacity limiting")
    config = mixture_config(top_k)
    config.capacity_factor = capacity
    config.dropped_token_behavior = fallback
    if before:
        config.weighting_position_option = ExpertWeightingPositionOptions.BEFORE_EXPERTS
    model = config.build().double().eval()
    inputs = torch.randn(19, 2, dtype=torch.float64, requires_grad=True)
    probabilities = torch.full(
        (19, top_k), 0.3, dtype=torch.float64, requires_grad=True
    )
    indices = (torch.arange(19)[:, None] % 2 + torch.arange(top_k)) % 3
    if top_k == 3:
        indices = None
    actual, _, loss = model(inputs, probabilities, indices)
    expected = reference_mixture(model, inputs, probabilities, indices)
    torch.testing.assert_close(actual, expected)
    targets = (inputs, probabilities, *model.parameters())
    actual_grads = torch.autograd.grad(
        actual.square().sum() + loss, targets, retain_graph=True, allow_unused=True
    )
    expected_grads = torch.autograd.grad(
        expected.square().sum(), targets, allow_unused=True
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_map_reduce_and_skip_mask_contracts():
    from emperor.experts._layers.map import MixtureOfExpertsMap
    from emperor.experts._layers.reduce import MixtureOfExpertsReduce

    mapping = MixtureOfExpertsMap(mixture_config(2)).double().eval()
    reduction = MixtureOfExpertsReduce(mixture_config(2)).double().eval()
    inputs = torch.randn(13, 2, dtype=torch.float64, requires_grad=True)
    indices = (torch.arange(13)[:, None] + torch.arange(2)) % 3
    probabilities = torch.full((13, 2), 0.4, dtype=torch.float64)
    mapped, _, _ = mapping(inputs, probabilities, indices)
    torch.testing.assert_close(
        mapped, reference_mixture(mapping, inputs, probabilities, indices)
    )
    actual, _, _ = reduction(mapped, probabilities, indices)
    expected = reference_mixture(
        reduction, mapped, probabilities, indices, reduce_input=True
    )
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(inputs.grad).all()
    inactive = torch.ones(13, 1)
    inactive[1] = 0
    for model, rows in ((mapping, inputs), (reduction, mapped)):
        with pytest.raises(ValueError, match="all-active skip_mask"):
            model(rows, probabilities, indices, inactive)
