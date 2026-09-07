"""Routed attention keeps padding within its adaptive Map/Reduce invocations."""

import pytest
import torch

from emperor.attention import MixtureOfAttentionHeadsConfig
from support.adaptive_grouping import linear_stack_config
from support.adaptive_grouping_variants import reference_mixture, variable_chunk_stack
from support.attention import build_attention_config


def attention_config(top_k=1, expert_kv=True):
    config = build_attention_config(
        config_class=MixtureOfAttentionHeadsConfig,
        batch_size=2,
        num_heads=1,
        embedding_dim=2,
        target_sequence_length=7,
        source_sequence_length=7,
        dropout_probability=0.0,
        experts_top_k=top_k,
        experts_num_experts=3,
    )
    config.experts_config.expert_model_config = variable_chunk_stack()
    config.use_kv_expert_models_flag = expert_kv
    config.target_dtype = torch.float64
    config.projection_model_config = linear_stack_config(2, 2)
    return config


@pytest.mark.parametrize("expert_kv", [False, True])
@pytest.mark.parametrize("top_k", [1, 2])
def test_routed_attention_chunking_runs_forward_and_backward(top_k, expert_kv):
    model = attention_config(top_k, expert_kv).build().eval()
    query = torch.randn(7, 2, 2, dtype=torch.float64, requires_grad=True)
    output, weights, loss = model(query, query, query)
    assert output.shape == query.shape and weights is None
    (output.square().sum() + loss).backward()
    assert torch.isfinite(query.grad).all()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert model.projector.indices is None


@pytest.mark.parametrize("expert_kv", [False, True])
@pytest.mark.parametrize("top_k", [1, 2])
def test_routed_attention_matches_independent_map_attention_reduce_equations(
    top_k, expert_kv
):
    from dataclasses import replace

    from emperor.augmentations.adaptive_parameters import (
        AdaptiveParameterGroupingScopeOptions as Scope,
    )
    from emperor.augmentations.adaptive_parameters import (
        AdaptiveParameterInputOrderOptions as Order,
    )

    config = attention_config(top_k, expert_kv)
    if not expert_kv:
        config.projection_model_config = variable_chunk_stack()
        augmentation = config.projection_model_config.layer_config.layer_model_config.adaptive_augmentation_config
        augmentation.grouping_config = replace(
            augmentation.grouping_config,
            scope=Scope.SEQUENCE,
            sequence_length=7,
            input_order=Order.SEQUENCE_FIRST,
        )
    model = config.build().eval()
    query = torch.randn(7, 2, 2, dtype=torch.float64, requires_grad=True)
    routes = []
    hook = model.projector.query_model.register_forward_pre_hook(
        lambda module, args: routes.append((args[1], args[2]))
    )
    try:
        actual, _, loss = model(query, query, query)
    finally:
        hook.remove()
    probabilities, indices = routes[0]
    rows = query.reshape(-1, 2)
    q = reference_mixture(model.projector.query_model, rows, probabilities, indices)
    q = q.reshape(7, 2, top_k, 2).permute(1, 2, 0, 3)
    projections = []
    for projection in (model.projector.key_model, model.projector.value_model):
        if expert_kv:
            values = reference_mixture(projection, rows, probabilities, indices)
            values = values.reshape(7, 2, top_k, 2).permute(1, 2, 0, 3)
        else:
            linear = projection[0].model
            contexts = (
                torch.stack(
                    [
                        torch.cat(
                            [
                                chunk.mean(0).expand_as(chunk)
                                for chunk in sample.split(5)
                            ]
                        )
                        for sample in query.transpose(0, 1)
                    ]
                )
                .transpose(0, 1)
                .reshape(-1, 2)
            )
            generator = linear.adaptive_behaviour.bias_model.model[0].model
            bias = (
                linear.bias_params
                + contexts @ generator.weight_params
                + generator.bias_params
            )
            values = (
                (rows @ linear.weight_params + bias)
                .reshape(7, 2, 2)
                .permute(1, 0, 2)
                .unsqueeze(1)
            )
        projections.append(values)
    key, value = projections
    weights = (q @ key.transpose(-1, -2) / 2**0.5).softmax(-1)
    attended = (weights @ value).permute(2, 0, 1, 3).reshape(-1, 2)
    expected = reference_mixture(
        model.projector.output_model,
        attended,
        probabilities,
        indices,
        reduce_input=True,
    ).reshape_as(query)
    torch.testing.assert_close(actual, expected)
    targets = (query, *model.parameters())
    actual_grads = torch.autograd.grad(
        actual.square().sum() + loss, targets, retain_graph=True, allow_unused=True
    )
    expected_grads = torch.autograd.grad(
        expected.square().sum(), targets, allow_unused=True
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize(
    "restriction", ["causal", "padding", "mask", "cross", "static"]
)
def test_restrictive_attention_inputs_reject_before_grouped_generation(restriction):
    config = attention_config(expert_kv=False)
    if restriction == "causal":
        config.causal_attention_mask_flag = True
        with pytest.raises(ValueError, match="causal"):
            config.build()
        return
    model = config.build().eval()
    query = torch.randn(7, 2, 2, dtype=torch.float64)
    kwargs = {}
    if restriction == "padding":
        kwargs["k_padding_mask"] = torch.tensor([[False] * 6 + [True]] * 2)
    elif restriction == "mask":
        kwargs["attention_mask"] = torch.zeros(7, 7, dtype=torch.bool)
    elif restriction == "static":
        kwargs["static_k"] = torch.randn(2, 7, 2, dtype=torch.float64)
    calls = []
    hook = model.projector.query_model.register_forward_pre_hook(
        lambda *_: calls.append(True)
    )
    try:
        with pytest.raises(ValueError, match="grouping"):
            model(
                query,
                query.clone() if restriction == "cross" else query,
                query,
                **kwargs,
            )
    finally:
        hook.remove()
    assert not calls and model.projector.indices is None
