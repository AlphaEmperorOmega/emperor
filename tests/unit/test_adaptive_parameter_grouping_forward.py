"""Grouping owns splitting and summarizing flat token input in one invocation."""

import weakref
from dataclasses import replace

import pytest
import torch

from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterGroupingScopeOptions as Scope,
)
from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterInputOrderOptions as Order,
)
from emperor.augmentations.adaptive_parameters import (
    MeanGroupingConfig,
    RMSGroupingConfig,
    SumGroupingConfig,
)
from support.adaptive_grouping import bias_linear
from support.adaptive_grouping_variants import (
    GROUPING_CONFIGS,
    grouping_config,
    initialize_mean_summary,
)


@pytest.mark.parametrize("variant", GROUPING_CONFIGS)
@pytest.mark.parametrize("layout", ["rows", "batch_first", "sequence_first"])
@pytest.mark.parametrize("sizing", ["count", "size"])
def test_forward_splits_flat_tokens_and_summarizes_each_group(variant, layout, sizing):
    sequence_length = 6 if sizing == "count" else 7
    logical = (
        torch.arange(3 * sequence_length * 2, dtype=torch.float64).reshape(
            3, sequence_length, 2
        )
        / 7
        - 5
    )
    logical.requires_grad_()
    settings = {"group_count": 3} if sizing == "count" else {"chunk_size": 3}
    if layout == "rows":
        settings["scope"] = Scope.ROWS
        ordered = logical
        sequences = [logical.reshape(-1, 2)]
    else:
        order = Order.BATCH_FIRST if layout == "batch_first" else Order.SEQUENCE_FIRST
        settings.update(
            scope=Scope.SEQUENCE, sequence_length=sequence_length, input_order=order
        )
        ordered = logical if order is Order.BATCH_FIRST else logical.transpose(0, 1)
        sequences = list(logical)
    flat_input = ordered.reshape(-1, 2)
    cfg = grouping_config(variant, feature_dim=2, **settings)
    grouper = initialize_mean_summary(cfg.build().double())
    expected_groups = []
    for sequence in sequences:
        width = sequence.size(0) // 3 if sizing == "count" else 3
        expected_groups.extend(sequence.split(width))
    summaries, plan = grouper(flat_input)
    expected_summaries = []
    for group in expected_groups:
        if variant is SumGroupingConfig:
            summary = group.sum(0)
        elif variant is RMSGroupingConfig:
            summary = group.square().mean(0).sqrt()
        else:
            # Learned test fixtures use uniform attention / mean-only projection.
            summary = group.mean(0)
        expected_summaries.append(summary)
    expected = torch.stack(expected_summaries)
    torch.testing.assert_close(summaries, expected)
    for index, group in enumerate(expected_groups):
        torch.testing.assert_close(plan.grouped_members[index, : group.size(0)], group)
        if plan.valid_members is not None:
            assert plan.valid_members[index].sum().item() == group.size(0)
    torch.testing.assert_close(plan.restore(plan.grouped_members), flat_input)
    actual_gradient = torch.autograd.grad(
        summaries.square().sum(), logical, retain_graph=True
    )[0]
    expected_gradient = torch.autograd.grad(expected.square().sum(), logical)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("variant", GROUPING_CONFIGS)
def test_forward_keeps_layout_and_summary_local_and_reloads_strictly(variant):
    cfg = grouping_config(variant, scope=Scope.ROWS, chunk_size=3, feature_dim=2)
    grouper = cfg.build().double()
    restored = cfg.build().double()
    restored.load_state_dict(grouper.state_dict(), strict=True)
    first_input = torch.randn(7, 2, dtype=torch.float64, requires_grad=True)
    summary, plan = grouper(first_input)
    expected, _ = restored(first_input)
    torch.testing.assert_close(summary, expected, rtol=0, atol=0)
    summary_ref, plan_ref = weakref.ref(summary), weakref.ref(plan)
    del summary, plan
    assert summary_ref() is None and plan_ref() is None
    second_input = torch.randn(4, 2, dtype=torch.float64)
    second_summary, second_plan = grouper(second_input)
    assert second_summary.shape == (2, 2)
    torch.testing.assert_close(
        second_plan.restore(second_plan.grouped_members), second_input
    )


def test_augmentation_sends_flat_tokens_to_grouper_and_applies_summary_to_members():
    cfg = MeanGroupingConfig(scope=Scope.ROWS, chunk_size=3)
    model = bias_linear(cfg).double()
    inputs = torch.arange(14, dtype=torch.float64).reshape(7, 2)
    grouped_calls = []
    generated_contexts = []
    grouping_hook = model.adaptive_behaviour.grouper.register_forward_hook(
        lambda module, args, result: grouped_calls.append((args[0], result))
    )
    generation_hook = model.adaptive_behaviour.bias_model.register_forward_pre_hook(
        lambda module, args: generated_contexts.append(args[-1])
    )
    try:
        actual = model(inputs)
    finally:
        grouping_hook.remove()
        generation_hook.remove()
    assert len(grouped_calls) == len(generated_contexts) == 1
    flat_input, (summary, plan) = grouped_calls[0]
    assert flat_input is inputs
    assert generated_contexts[0] is summary
    assert plan.grouped_members.shape == (3, 3, 2)
    expected = torch.cat([group.mean(0).expand_as(group) for group in inputs.split(3)])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "inputs",
    [
        None,
        torch.ones(2),
        torch.ones(2, 3, 2),
        torch.ones(2, 2, dtype=torch.int64),
        torch.empty(0, 2),
        torch.ones(2, 3),
    ],
)
def test_forward_rejects_invalid_flat_input_before_reduction(inputs):
    grouper = SumGroupingConfig(scope=Scope.ROWS, chunk_size=3, feature_dim=2).build()
    with pytest.raises((TypeError, ValueError)):
        grouper(inputs)


def test_variant_config_rejects_the_removed_nested_reduction_field():
    cfg = SumGroupingConfig(scope=Scope.ROWS, group_count=2)
    cfg.chunking_config = MeanGroupingConfig()
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="unsupported fields"):
        cfg.build(replace(cfg, feature_dim=2))
    torch.testing.assert_close(torch.get_rng_state(), before)
