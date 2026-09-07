"""Public contracts for configuration-owned adaptive grouping."""

import copy
from dataclasses import fields, replace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from emperor._validation import _adaptive_grouping_configs, _adaptive_grouping_paths
from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterAugmentationConfig,
    AdaptiveParameterGroupingScopeOptions,
    SumGroupingConfig,
)
from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterInputOrderOptions as Order,
)
from support.adaptive_grouping import bias_linear


def test_nested_grouping_applies_each_chunk_context_to_original_members():
    grouping = SumGroupingConfig()
    for config_field in fields(grouping):
        assert getattr(grouping, config_field.name) is None
        assert config_field.default is None
        assert config_field.metadata["help"]
    model = bias_linear(
        replace(
            grouping,
            scope=AdaptiveParameterGroupingScopeOptions.ROWS,
            group_count=2,
        )
    )
    inputs = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    torch.testing.assert_close(
        model(inputs),
        torch.tensor(
            [
                [3.0, 30.0],
                [3.0, 30.0],
                [7.0, 70.0],
                [7.0, 70.0],
            ]
        ),
    )


def test_absent_grouping_matches_per_row_tensor_reference():
    import json
    from pathlib import Path

    snapshot = json.loads(
        (
            Path(__file__).parents[1] / "support/adaptive_grouping_reference_state.json"
        ).read_text()
    )
    model = bias_linear()
    current = model.state_dict()
    state = {
        key: torch.tensor(value, dtype=current[key].dtype)
        for key, value in snapshot["state_dict"].items()
    }
    model.load_state_dict(state, strict=True)
    torch.testing.assert_close(
        model(torch.tensor(snapshot["input"])), torch.tensor(snapshot["output"])
    )
    assert not model.adaptive_behaviour.adaptive_parameter_grouping_enabled


def test_sequence_orders_restore_samples_and_match_member_gradients():
    from emperor.augmentations.adaptive_parameters import (
        AdaptiveParameterInputOrderOptions as Order,
    )

    for order in Order:
        model = bias_linear(
            SumGroupingConfig(
                scope=AdaptiveParameterGroupingScopeOptions.SEQUENCE,
                group_count=2,
                sequence_length=6,
                input_order=order,
            )
        ).double()
        for batch_size in (3, 1):
            logical = (
                torch.arange(batch_size * 6 * 2, dtype=torch.float64)
                .reshape(batch_size, 6, 2)
                .requires_grad_()
            )
            ordered = logical if order is Order.BATCH_FIRST else logical.transpose(0, 1)
            actual = model(ordered.reshape(-1, 2)).reshape(ordered.shape)
            expected = (
                logical.reshape(batch_size, 2, 3, 2)
                .sum(2, keepdim=True)
                .expand(-1, -1, 3, -1)
                .reshape(logical.shape)
            )
            if order is Order.SEQUENCE_FIRST:
                expected = expected.transpose(0, 1)
            torch.testing.assert_close(actual, expected)
            actual.sum().backward()
            torch.testing.assert_close(logical.grad, torch.full_like(logical, 3.0))


def test_attention_rejects_padding_before_generation_and_recovers():
    import pytest

    from emperor.attention import SelfAttentionConfig
    from emperor.augmentations.adaptive_parameters import (
        AdaptiveParameterInputOrderOptions as Order,
    )
    from support.attention import build_attention_config, make_projection_model_config

    projection = make_projection_model_config()
    projection.layer_config.layer_model_config = bias_linear(
        SumGroupingConfig(
            scope=AdaptiveParameterGroupingScopeOptions.SEQUENCE,
            group_count=2,
            sequence_length=4,
            input_order=Order.SEQUENCE_FIRST,
        )
    ).cfg
    config = build_attention_config(
        config_class=SelfAttentionConfig,
        embedding_dim=2,
        num_heads=1,
        batch_size=2,
        target_sequence_length=4,
        source_sequence_length=4,
    )
    config.batch_first_flag = True
    config.projection_model_config = projection
    model = config.build()
    calls = []
    handles = [
        module.register_forward_pre_hook(lambda *_: calls.append(True))
        for module in model.modules()
        if hasattr(module, "adaptive_parameter_grouping_enabled")
    ]
    inputs = torch.randn(2, 4, 2)
    with pytest.raises(ValueError, match="all-valid"):
        model(
            inputs,
            inputs,
            inputs,
            k_padding_mask=torch.tensor([[False, False, False, True], [False] * 4]),
        )
    assert not calls
    output, _, _ = model(
        inputs, inputs, inputs, k_padding_mask=torch.zeros(2, 4, dtype=torch.bool)
    )
    assert output.shape == inputs.shape and calls
    for handle in handles:
        handle.remove()


# Construction must fail before any generator or base parameter consumes randomness.


Scope = AdaptiveParameterGroupingScopeOptions


@pytest.mark.parametrize(
    "grouping",
    [
        {},
        2,
        "ROWS",
        SumGroupingConfig(),
        SumGroupingConfig(scope=Scope.ROWS),
        SumGroupingConfig(group_count=1),
        SumGroupingConfig(scope=Scope.SEQUENCE, group_count=2),
        SumGroupingConfig(scope=Scope.SEQUENCE, group_count=2, sequence_length=4),
        *[
            SumGroupingConfig(scope=value, group_count=1)
            for value in (None, 1, True, "ROWS", Order.BATCH_FIRST)
        ],
        *[
            SumGroupingConfig(scope=Scope.ROWS, group_count=value)
            for value in (None, True, False, 0, -1, 1.5, "2")
        ],
        SumGroupingConfig(scope=Scope.ROWS, group_count=1, sequence_length=4),
        SumGroupingConfig(
            scope=Scope.ROWS, group_count=1, input_order=Order.BATCH_FIRST
        ),
        *[
            SumGroupingConfig(
                scope=Scope.SEQUENCE,
                group_count=2,
                sequence_length=value,
                input_order=Order.BATCH_FIRST,
            )
            for value in (None, True, 0, -2, 4.0, "4", 1, 3)
        ],
        *[
            SumGroupingConfig(
                scope=Scope.SEQUENCE,
                group_count=2,
                sequence_length=4,
                input_order=value,
            )
            for value in (None, 0, False, "BATCH_FIRST", Scope.ROWS)
        ],
    ],
)
def test_invalid_grouping_fails_before_parameter_initialization(grouping):
    config = copy.deepcopy(bias_linear().cfg)
    config.adaptive_augmentation_config.grouping_config = grouping
    before = torch.get_rng_state().clone()
    with pytest.raises((TypeError, ValueError)):
        config.build()
    torch.testing.assert_close(torch.get_rng_state(), before)


def test_grouping_is_keyword_only_and_replaced_as_a_whole():
    rows = SumGroupingConfig(scope=Scope.ROWS, group_count=2)
    with pytest.raises(TypeError):
        SumGroupingConfig(Scope.ROWS, 2)
    assert copy.deepcopy(rows) == rows
    assert Scope.ROWS.value == 1 and Scope.SEQUENCE.value == 2
    assert Order.BATCH_FIRST.value == 0 and Order.SEQUENCE_FIRST.value == 1
    assert set(Scope.__members__) == {"ROWS", "SEQUENCE"}
    assert set(Order.__members__) == {"BATCH_FIRST", "SEQUENCE_FIRST"}
    assert {field.name for field in fields(rows)} == {
        "scope",
        "group_count",
        "chunk_size",
        "sequence_length",
        "input_order",
        "feature_dim",
        "summary_normalization",
        "rms_norm_epsilon",
    }
    assert {field.name for field in fields(AdaptiveParameterAugmentationConfig)} == {
        "input_dim",
        "output_dim",
        "diagonal_config",
        "weight_config",
        "bias_config",
        "mask_config",
        "model_config",
        "grouping_config",
    }
    config = copy.deepcopy(bias_linear(rows).cfg)
    replacement = SumGroupingConfig(
        scope=Scope.SEQUENCE,
        group_count=1,
        sequence_length=4,
        input_order=Order.SEQUENCE_FIRST,
    )
    inherited = config.adaptive_augmentation_config.build(
        AdaptiveParameterAugmentationConfig(input_dim=2, output_dim=2)
    )
    assert inherited.grouping_config == rows
    replaced = config.adaptive_augmentation_config.build(
        AdaptiveParameterAugmentationConfig(
            input_dim=2, output_dim=2, grouping_config=replacement
        )
    )
    assert replaced.grouping_config == replacement
    config.adaptive_augmentation_config = replace(
        config.adaptive_augmentation_config, grouping_config=None
    )
    assert config.build().adaptive_behaviour.grouping_config is None


@pytest.mark.parametrize("value", [None, 0, 4])
@pytest.mark.parametrize("grouped", [False, True])
def test_deserialized_unknown_fields_fail_before_initialization(value, grouped):
    grouping = SumGroupingConfig(scope=Scope.ROWS, group_count=1) if grouped else None
    config = copy.deepcopy(bias_linear(grouping).cfg)
    config.adaptive_augmentation_config.unexpected_field = value
    before = torch.get_rng_state().clone()
    with pytest.raises(
        ValueError, match="unsupported fields.*current configuration schema"
    ):
        config.build()
    with pytest.raises(ValueError, match="unsupported fields"):
        tuple(_adaptive_grouping_configs(config, root="root"))
    torch.testing.assert_close(torch.get_rng_state(), before)


def test_discovery_preserves_owner_paths_and_survives_shared_cycles():
    config = bias_linear(SumGroupingConfig(scope=Scope.ROWS, group_count=1)).cfg
    graph = {"first": config, "shared": config}
    graph["cycle"] = graph
    owners = tuple(_adaptive_grouping_configs(graph, root="root"))
    paths = _adaptive_grouping_paths(graph, root="root")
    assert len(owners) == 2
    assert owners[0][1] is config.adaptive_augmentation_config
    assert tuple(path for path, _ in owners) == tuple(paths)
    assert "first" in owners[0][0] and "adaptive_augmentation_config" in owners[0][0]


@pytest.mark.parametrize(
    "inputs,grouping",
    [
        (None, SumGroupingConfig(scope=Scope.ROWS, group_count=1)),
        (
            torch.ones(2),
            SumGroupingConfig(scope=Scope.ROWS, group_count=1),
        ),
        (
            torch.ones(2, 2, 2),
            SumGroupingConfig(scope=Scope.ROWS, group_count=1),
        ),
        (
            torch.empty(0, 2),
            SumGroupingConfig(scope=Scope.ROWS, group_count=1),
        ),
        (
            torch.ones(3, 2),
            SumGroupingConfig(scope=Scope.ROWS, group_count=2),
        ),
        (
            torch.ones(2, 2),
            SumGroupingConfig(scope=Scope.ROWS, group_count=4),
        ),
        (
            torch.ones(6, 2),
            SumGroupingConfig(
                scope=Scope.SEQUENCE,
                group_count=2,
                sequence_length=4,
                input_order=Order.BATCH_FIRST,
            ),
        ),
    ],
)
def test_group_plan_rejects_invalid_member_geometry(inputs, grouping):
    with pytest.raises((TypeError, ValueError)):
        replace(grouping, feature_dim=2).build()(inputs)[1]


@pytest.mark.parametrize(
    "output", [None, torch.ones(4, 2), torch.ones(4, 1, 2), torch.ones(1, 4, 2)]
)
def test_restore_rejects_invalid_generated_member_geometry(output):
    plan = replace(
        SumGroupingConfig(scope=Scope.ROWS, group_count=2), feature_dim=2
    ).build()(torch.ones(4, 2))[1]
    with pytest.raises((TypeError, ValueError)):
        plan.restore(output)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("order", list(Order))
def test_noncontiguous_inputs_restore_changed_output_width(dtype, order):
    logical = torch.arange(3 * 6 * 4, dtype=dtype).reshape(3, 6, 4)[..., ::2]
    ordered = logical if order is Order.BATCH_FIRST else logical.transpose(0, 1)
    flat = ordered.reshape(-1, 2)
    grouping = SumGroupingConfig(
        scope=Scope.SEQUENCE, group_count=3, sequence_length=6, input_order=order
    )
    plan = replace(grouping, feature_dim=2).build()(flat)[1]
    torch.testing.assert_close(plan.restore(plan.grouped_members), flat)
    actual = plan.restore(
        torch.cat((plan.grouped_members, plan.grouped_members[..., :1]), dim=-1)
    )
    torch.testing.assert_close(actual, torch.cat((flat, flat[:, :1]), dim=-1))
    assert (
        plan.context_count == 9 and plan.members_per_group == 2 and plan.row_count == 18
    )


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_checkpoint_recomputes_contexts_without_retaining_previous_calls(use_reentrant):
    model = bias_linear(
        SumGroupingConfig(
            scope=Scope.SEQUENCE,
            group_count=2,
            sequence_length=4,
            input_order=Order.BATCH_FIRST,
        )
    ).double()
    reference = copy.deepcopy(model)
    contexts = []
    handle = model.adaptive_behaviour.bias_model.register_forward_pre_hook(
        lambda _, args: contexts.append(args[1].detach().clone())
    )
    for batch_size in (3, 1, 2):
        inputs = torch.randn(batch_size * 4, 2, dtype=torch.float64, requires_grad=True)
        expected_input = inputs.detach().clone().requires_grad_()
        model.zero_grad()
        reference.zero_grad()
        actual = checkpoint(model, inputs, use_reentrant=use_reentrant)
        expected = reference(expected_input)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(inputs.grad, expected_input.grad)
        for parameter, expected_parameter in zip(
            model.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(parameter.grad, expected_parameter.grad)
        assert contexts[-1].shape == (batch_size * 2, 2)
        torch.testing.assert_close(
            contexts[-1], inputs.detach().reshape(batch_size * 2, 2, 2).sum(1)
        )
    assert len(contexts) == 6
    handle.remove()


def test_invalid_then_valid_direct_call_never_reuses_contexts():
    model = bias_linear(SumGroupingConfig(scope=Scope.ROWS, group_count=2))
    contexts = []
    handle = model.adaptive_behaviour.bias_model.register_forward_pre_hook(
        lambda _, args: contexts.append(args[1].clone())
    )
    for inputs in (torch.empty(0, 2), torch.ones(3, 2), torch.ones(4, 3)):
        with pytest.raises(ValueError):
            model(inputs)
        assert not contexts
    inputs = torch.ones(4, 2)
    torch.testing.assert_close(model(inputs), inputs * 2)
    inputs.fill_(3)
    torch.testing.assert_close(model(inputs), inputs * 2)
    torch.testing.assert_close(contexts[0], torch.full((2, 2), 2.0))
    torch.testing.assert_close(contexts[1], torch.full((2, 2), 6.0))
    handle.remove()


def test_strict_loading_of_combined_component_tensor_reference():
    import json
    from pathlib import Path

    from support.adaptive_grouping import combined_linear

    snapshot = json.loads(
        (
            Path(__file__).parents[1]
            / "support/adaptive_grouping_reference_combined_state.json"
        ).read_text()
    )
    model = combined_linear()
    current = model.state_dict()
    state = {
        key: torch.tensor(value, dtype=current[key].dtype)
        for key, value in snapshot["state_dict"].items()
    }
    assert tuple(current) == tuple(state)
    for key in current:
        assert current[key].shape == state[key].shape
    model.load_state_dict(state, strict=True)
    torch.testing.assert_close(
        model(torch.tensor(snapshot["input"])), torch.tensor(snapshot["output"])
    )
