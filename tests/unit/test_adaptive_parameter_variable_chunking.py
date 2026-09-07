"""Variable chunks retain real members and exclude locally inserted padding."""

from copy import deepcopy
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
    MeanStdGroupingConfig,
    SumGroupingConfig,
)
from emperor.augmentations.adaptive_parameters import (
    SummaryNormalizationOptions as Normalization,
)
from support.adaptive_grouping import bias_linear
from support.adaptive_grouping_variants import (
    GROUPING_CONFIGS,
    grouping_config,
    initialize_mean_summary,
)


def test_twelve_real_rows_use_three_contexts_and_return_twelve_outputs():
    model = bias_linear(MeanGroupingConfig(scope=Scope.ROWS, chunk_size=5)).double()
    inputs = torch.arange(24, dtype=torch.float64).reshape(12, 2).requires_grad_()
    expected = torch.cat([chunk.mean(0).expand_as(chunk) for chunk in inputs.split(5)])
    contexts = []
    hook = model.adaptive_behaviour.bias_model.register_forward_pre_hook(
        lambda module, arguments: contexts.append(arguments[-1].detach().clone())
    )
    try:
        output = model(inputs)
    finally:
        hook.remove()
    assert output.shape == (12, 2)
    assert contexts[0].shape == (3, 2)
    torch.testing.assert_close(output, expected)
    actual_gradient = torch.autograd.grad(output.square().sum(), inputs)[0]
    expected_gradient = torch.autograd.grad(expected.square().sum(), inputs)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_padding_positions_are_removed_even_when_padded_bias_outputs_are_nonzero():
    grouping = SumGroupingConfig(scope=Scope.ROWS, chunk_size=5)
    plan = replace(grouping, feature_dim=2).build()(torch.zeros(12, 2))[1]
    assert plan.row_count == 12 and plan.physical_row_count == 15
    assert plan.valid_members.sum(1).tolist() == [5, 5, 2]
    assert plan.restore(torch.ones(3, 5, 3), output_dim=3).shape == (12, 3)
    for output in (torch.ones(3, 5, 2), torch.ones(3, 4, 3), torch.ones(4, 5, 3)):
        with pytest.raises(ValueError):
            plan.restore(output, output_dim=3)


def test_shared_configuration_is_validated_independently_per_execution_owner():
    from emperor._validation import (
        _adaptive_grouping_configs,
        _validate_adaptive_sequence_input,
        _validate_grouped_row_preservation,
    )
    from emperor.experts import MixtureOfExpertsConfig
    from support.adaptive_grouping_variants import variable_chunk_stack

    shared = variable_chunk_stack()
    expert = MixtureOfExpertsConfig(
        top_k=2, compute_expert_mixture_flag=False, expert_model_config=shared
    )
    graph = {"expert": expert, "direct": shared}
    graph["cycle"] = graph
    all_paths = tuple(_adaptive_grouping_configs(graph, root="root"))
    direct_paths = tuple(
        _adaptive_grouping_configs(graph, root="root", direct_only=True)
    )
    assert len(all_paths) == 2 and len(direct_paths) == 1
    assert "['direct']" in direct_paths[0][0]
    _validate_grouped_row_preservation(expert, root="expert")
    with pytest.raises(ValueError, match="unreduced"):
        _validate_grouped_row_preservation(graph, root="root")
    # Only the direct occurrence consumes the physical sequence layout.
    with pytest.raises(ValueError, match="ROWS"):
        _validate_adaptive_sequence_input(
            graph, root="root", sequence_length=7, input_order="BATCH_FIRST"
        )


@pytest.mark.parametrize("config_type", GROUPING_CONFIGS)
@pytest.mark.parametrize("normalization", list(Normalization))
def test_masked_reducers_match_real_member_outputs_and_gradients(
    config_type, normalization
):
    model = (
        grouping_config(config_type, feature_dim=2, summary_normalization=normalization)
        .build()
        .double()
    )
    if config_type is MeanStdGroupingConfig:
        with torch.no_grad():
            model.projection[-1].model.weight_params[2:].copy_(torch.eye(2))
    reference = deepcopy(model)
    values = torch.tensor(
        [
            [
                [2.0, 1.0],
                [4.0, 3.0],
                [float("nan"), float("inf")],
                [float("inf"), float("nan")],
            ],
            [
                [0.0, 0.0],
                [float("nan"), float("nan")],
                [float("inf"), float("inf")],
                [1e30, -1e30],
            ],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    valid = torch.tensor([[True, True, False, False], [True, False, False, False]])
    expected_inputs = [
        values.detach()[i, mask].clone().requires_grad_()
        for i, mask in enumerate(valid)
    ]
    expected = torch.cat(
        [reference.summarize(part.unsqueeze(0)) for part in expected_inputs]
    )
    actual = model.summarize(values, valid)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for i, mask in enumerate(valid):
        torch.testing.assert_close(values.grad[i, mask], expected_inputs[i].grad)
    assert torch.count_nonzero(values.grad[~valid]) == 0
    for actual_parameter, expected_parameter in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)


@pytest.mark.parametrize("config_type", GROUPING_CONFIGS)
def test_one_instance_accepts_variable_counts_and_empty_backward(config_type):
    model = bias_linear(grouping_config(config_type, scope=Scope.ROWS, chunk_size=5))
    for count in (0, 1, 2, 4, 5, 6, 7, 10, 12, 15, 16):
        inputs = torch.randn(2, count).t().requires_grad_()
        calls = []
        hook = model.adaptive_behaviour.grouper.register_forward_hook(
            lambda module, args, output, calls=calls: calls.append(output[0].shape[0])
        )
        try:
            output = model(inputs)
        finally:
            hook.remove()
        assert output.shape == (count, 2)
        assert calls == ([] if count == 0 else [(count + 4) // 5])
        output.sum().backward()
        assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


@pytest.mark.parametrize("order", list(Order))
def test_sequence_tail_padding_is_removed_separately_for_each_sample(order):
    model = bias_linear(
        MeanGroupingConfig(
            scope=Scope.SEQUENCE, sequence_length=7, input_order=order, chunk_size=5
        )
    )
    canonical = torch.arange(28, dtype=torch.float32).reshape(2, 7, 2)
    expected = torch.stack(
        [
            torch.cat([chunk.mean(0).expand_as(chunk) for chunk in sample.split(5)])
            for sample in canonical
        ]
    )
    if order is Order.SEQUENCE_FIRST:
        canonical, expected = canonical.transpose(0, 1), expected.transpose(0, 1)
    torch.testing.assert_close(model(canonical.reshape(-1, 2)), expected.reshape(-1, 2))


@pytest.mark.parametrize(
    "sizing",
    [
        {},
        {"chunk_size": True},
        {"chunk_size": 0},
        {"chunk_size": -1},
        {"chunk_size": 1.5},
        {"chunk_size": 5, "group_count": 2},
    ],
)
def test_invalid_chunk_sizing_fails_at_construction(sizing):
    with pytest.raises((TypeError, ValueError), match="chunk_size|group_count"):
        bias_linear(SumGroupingConfig(scope=Scope.ROWS, **sizing))


@pytest.mark.parametrize("config_type", GROUPING_CONFIGS)
def test_grouper_rejects_a_chunk_with_no_real_members(config_type):
    grouper = grouping_config(config_type, feature_dim=2).build()
    with pytest.raises(ValueError, match="at least one valid member"):
        grouper.summarize(torch.ones(1, 5, 2), torch.zeros(1, 5, dtype=torch.bool))


def test_empty_rectangular_output_keeps_input_dtype_and_skips_all_generation():
    from dataclasses import replace

    model = bias_linear(SumGroupingConfig(scope=Scope.ROWS, chunk_size=5)).cfg
    model = replace(model, output_dim=3).build()
    inputs = torch.empty(0, 2, requires_grad=True)
    calls = []
    handles = [
        module.register_forward_pre_hook(lambda *_: calls.append(True))
        for module in (
            model.adaptive_behaviour.grouper,
            model.adaptive_behaviour.bias_model,
        )
    ]
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = model(inputs)
        assert result.shape == (0, 3) and result.dtype == inputs.dtype
        result.sum().backward()
    finally:
        for handle in handles:
            handle.remove()
    assert not calls and inputs.grad.shape == inputs.shape
    output = model(torch.randn(12, 2))
    assert output.shape == (12, 3)


def test_empty_calls_preserve_decay_counters_and_validate_base_parameters():
    from emperor.augmentations.adaptive_parameters import WeightDecayScheduleOptions

    config = bias_linear(SumGroupingConfig(scope=Scope.ROWS, chunk_size=5)).cfg
    config.adaptive_augmentation_config.bias_config.decay_schedule = (
        WeightDecayScheduleOptions.EXPONENTIAL
    )
    config.adaptive_augmentation_config.bias_config.decay_rate = 0.1
    model = config.build().train()
    model(torch.ones(12, 2))
    generator = model.adaptive_behaviour.bias_model
    before = (
        generator._decay_policy.decay_step.clone(),
        generator._decay_policy.warmup_step.clone(),
    )
    model(torch.empty(0, 2, requires_grad=True)).sum().backward()
    assert torch.equal(generator._decay_policy.decay_step, before[0])
    assert torch.equal(generator._decay_policy.warmup_step, before[1])
    calls = []

    def affine(*args):
        calls.append(True)

    with pytest.raises(ValueError, match="weight_params"):
        model.adaptive_behaviour(
            affine, torch.empty(2, 3), model.bias_params, torch.empty(0, 2)
        )
    assert not calls


def test_variable_chunks_recompute_on_checkpoint_and_round_trip_state():
    from torch.utils.checkpoint import checkpoint

    config = grouping_config(GROUPING_CONFIGS[3], scope=Scope.ROWS, chunk_size=5)
    model = bias_linear(config).double()
    restored = bias_linear(config).double()
    restored.load_state_dict(model.state_dict(), strict=True)
    for count in (12, 7, 15):
        inputs = torch.randn(count, 2, dtype=torch.float64, requires_grad=True)
        ordinary = model(inputs)
        recomputed = checkpoint(restored, inputs, use_reentrant=False)
        torch.testing.assert_close(ordinary, recomputed)
        actual_grad = torch.autograd.grad(ordinary.square().sum(), inputs)[0]
        expected_grad = torch.autograd.grad(recomputed.square().sum(), inputs)[0]
        torch.testing.assert_close(actual_grad, expected_grad)
    assert not any("valid_members" in key for key in restored.state_dict())


@pytest.mark.parametrize(
    "inputs",
    [[], torch.empty(0, 2, dtype=torch.long), torch.empty(0, 0), torch.empty(0, 3)],
)
def test_empty_shortcut_still_validates_input(inputs):
    model = bias_linear(SumGroupingConfig(scope=Scope.ROWS, chunk_size=5))
    with pytest.raises((TypeError, ValueError)):
        model(inputs)


def test_grouping_monitor_reports_detached_counts_and_cleans_up():
    from lightning import LightningModule

    from emperor.augmentations.adaptive_parameters import (
        AdaptiveParameterMonitorCallback,
    )

    class Observed(LightningModule):
        def __init__(self):
            super().__init__()
            self.model = bias_linear(SumGroupingConfig(scope=Scope.ROWS, chunk_size=5))
            self.metrics = {}

        def log(self, name, value, *args, **kwargs):
            self.metrics[name] = value

    wrapped = Observed()
    callback = AdaptiveParameterMonitorCallback(log_every_n_steps=1)
    callback.on_fit_start(None, wrapped)
    callback.on_fit_start(None, wrapped)
    for count, contexts, padding in ((12, 3, 3), (0, 0, 0), (7, 2, 3)):
        wrapped.metrics.clear()
        wrapped.model(torch.randn(count, 2, requires_grad=True)).sum().backward()
        metrics = {
            name.rsplit("/", 1)[1]: value
            for name, value in wrapped.metrics.items()
            if "/grouping/" in name
        }
        assert {name: value.item() for name, value in metrics.items()} == dict(
            real_row_count=count, context_count=contexts, padding_row_count=padding
        )
        assert all(
            value.grad_fn is None and not value.requires_grad
            for value in metrics.values()
        )
    with pytest.raises(ValueError):
        wrapped.model(torch.empty(7, 3))
    callback.on_exception(None, wrapped, ValueError("invalid features"))
    wrapped.metrics.clear()
    wrapped.model(torch.randn(12, 2))
    assert not wrapped.metrics
    assert all(not module._forward_hooks for module in wrapped.model.modules())


@pytest.mark.parametrize("config_type", GROUPING_CONFIGS)
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_masked_statistics_preserve_precision_and_valid_zeros(config_type, dtype):
    grouper = initialize_mean_summary(
        grouping_config(config_type, feature_dim=1).build()
    ).to(dtype=dtype)
    inputs = torch.tensor(
        [[[2.0], [4.0], [float("nan")], [float("inf")], [0.0]]],
        dtype=dtype,
        requires_grad=True,
    )
    valid = torch.tensor([[True, True, False, False, False]])
    output = grouper.summarize(inputs, valid)
    method = config_type.__name__
    expected = 6.0 if "Sum" in method else 10.0**0.5 if "RMS" in method else 3.0
    torch.testing.assert_close(output, torch.tensor([[expected]], dtype=dtype))
    output.float().sum().backward()
    assert torch.isfinite(inputs.grad).all()
    zero = torch.zeros_like(inputs, requires_grad=True)
    grouper.summarize(zero, valid).float().sum().backward()
    assert torch.isfinite(zero.grad).all()


def test_attention_scores_only_real_members_with_unequal_logits_and_rng():
    from emperor.augmentations.adaptive_parameters import (
        AttentionGroupingConfig,
    )
    from emperor.layers import LayerState
    from support.adaptive_grouping import linear_stack_config

    stack = linear_stack_config(2, 1)
    stack.num_layers = 2
    stack.hidden_dim = 2
    stack.layer_config.dropout_probability = 0.3
    model = (
        AttentionGroupingConfig(
            scope=Scope.ROWS, chunk_size=5, feature_dim=2, model_config=stack
        )
        .build()
        .double()
        .train()
    )
    reference = deepcopy(model)
    inputs = torch.randn(2, 5, 2, dtype=torch.float64, requires_grad=True)
    valid = torch.tensor([[True] * 5, [True, True, False, False, False]])
    real = inputs[valid]
    rng = torch.random.get_rng_state()
    actual = model.summarize(inputs, valid)
    actual_rng = torch.random.get_rng_state()
    torch.random.set_rng_state(rng)
    scores = reference.scorer(LayerState(hidden=real)).hidden
    assert scores.unique().numel() > 1
    expected = torch.cat(
        [
            (values * logits.softmax(0)).sum(0, keepdim=True)
            for values, logits in zip(
                real.split([5, 2]), scores.split([5, 2]), strict=True
            )
        ]
    )
    assert torch.equal(torch.random.get_rng_state(), actual_rng)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    actual_gradient = inputs.grad.clone()
    inputs.grad = None
    expected.sum().backward()
    torch.testing.assert_close(actual_gradient, inputs.grad)


@pytest.mark.parametrize("config_type", GROUPING_CONFIGS)
def test_exact_chunks_preserve_initialization_state_rng_and_gradients(config_type):
    torch.manual_seed(381)
    fixed = bias_linear(
        grouping_config(config_type, scope=Scope.ROWS, group_count=3)
    ).double()
    fixed_rng = torch.random.get_rng_state()
    torch.manual_seed(381)
    variable = bias_linear(
        grouping_config(config_type, scope=Scope.ROWS, chunk_size=5)
    ).double()
    assert torch.equal(torch.random.get_rng_state(), fixed_rng)
    assert tuple(dict(fixed.named_modules())) == tuple(dict(variable.named_modules()))
    for key, value in fixed.state_dict().items():
        torch.testing.assert_close(value, variable.state_dict()[key], rtol=0, atol=0)
    inputs = torch.randn(15, 2, dtype=torch.float64, requires_grad=True)
    expected, actual = fixed(inputs), variable(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected_grad = torch.autograd.grad(expected.sum(), (inputs, *fixed.parameters()))
    actual_grad = torch.autograd.grad(actual.sum(), (inputs, *variable.parameters()))
    for actual_value, expected_value in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
