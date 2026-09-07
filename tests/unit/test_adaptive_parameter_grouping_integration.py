"""Chunking through adaptive affine operations, recurrence, and monitoring."""

import copy
import math
from dataclasses import replace

import pytest
import torch
from lightning import LightningModule
from torch.utils.checkpoint import checkpoint

from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterGroupingScopeOptions as Scope,
)
from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterInputOrderOptions as Order,
)
from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterMonitorCallback,
)
from emperor.augmentations.adaptive_parameters import (
    AttentionGroupingConfig as AttentionConfig,
)
from emperor.augmentations.adaptive_parameters import (
    MeanGroupingConfig as MeanConfig,
)
from emperor.augmentations.adaptive_parameters import (
    MeanStdGroupingConfig as MeanStdConfig,
)
from emperor.augmentations.adaptive_parameters import (
    RMSGroupingConfig as RMSConfig,
)
from emperor.augmentations.adaptive_parameters import (
    SumGroupingConfig as SumConfig,
)
from emperor.augmentations.adaptive_parameters import (
    SummaryNormalizationOptions as Normalization,
)
from emperor.layers import LayerNormPositionOptions, LayerState, RecurrentLayerConfig
from support.adaptive_grouping import bias_linear, combined_linear, linear_stack_config
from support.adaptive_grouping_variants import (
    GROUPING_CONFIGS,
    grouping_config,
    initialize_mean_summary,
    with_reduction,
)


def make_model(
    method, normalization=Normalization.DISABLED, scope=Scope.ROWS, order=None
):
    grouping = grouping_config(
        method, summary_normalization=normalization, scope=scope, group_count=2
    )
    if scope is Scope.SEQUENCE:
        grouping = replace(grouping, sequence_length=6, input_order=order)
    model = bias_linear(grouping).double()
    initialize_mean_summary(model.adaptive_behaviour.grouper)
    with torch.no_grad():
        model.weight_params.copy_(torch.eye(2))
    return model


@pytest.mark.parametrize("method", GROUPING_CONFIGS)
@pytest.mark.parametrize("normalization", list(Normalization))
@pytest.mark.parametrize(
    "scope,order",
    [
        (Scope.ROWS, None),
        (Scope.SEQUENCE, Order.BATCH_FIRST),
        (Scope.SEQUENCE, Order.SEQUENCE_FIRST),
    ],
)
def test_original_member_affine_outputs_restoration_and_dynamic_batches(
    method, normalization, scope, order
):
    model = make_model(method, normalization, scope, order)
    identities = {name: id(parameter) for name, parameter in model.named_parameters()}
    for batch in (2, 1, 3):
        logical = (
            torch.arange(batch * 12, dtype=torch.float64).reshape(batch, 6, 2) / 9 - 2
        )
        ordered = logical.transpose(0, 1) if order is Order.SEQUENCE_FIRST else logical
        inputs = ordered.reshape(-1, 2).detach().requires_grad_()
        chunks = (
            logical.reshape(-1, 3, 2)
            if scope is Scope.SEQUENCE
            else inputs.detach().reshape(2, -1, 2)
        )
        expected_chunks = []
        for chunk in chunks.tolist():
            context = []
            for d in range(2):
                values = [token[d] for token in chunk]
                if method is SumConfig:
                    value = math.fsum(values)
                elif method is RMSConfig:
                    value = math.sqrt(math.fsum(v * v for v in values) / len(values))
                else:
                    value = math.fsum(values) / len(values)
                context.append(value)
            if normalization is Normalization.RMS_NORM:
                scale = math.sqrt(math.fsum(v * v for v in context) / 2 + 1e-6)
                context = [v / scale for v in context]
            expected_chunks.extend(
                [[token[d] + context[d] for d in range(2)] for token in chunk]
            )
        expected = torch.tensor(expected_chunks, dtype=torch.float64)
        if scope is Scope.SEQUENCE and order is Order.SEQUENCE_FIRST:
            expected = expected.reshape(batch, 6, 2).transpose(0, 1).reshape(-1, 2)
        output = model(inputs)
        torch.testing.assert_close(output, expected)
        output.square().sum().backward()
        assert torch.isfinite(inputs.grad).all()
        assert {
            name: id(parameter) for name, parameter in model.named_parameters()
        } == identities
        model.zero_grad()


@pytest.mark.parametrize("method", [MeanStdConfig, AttentionConfig])
@pytest.mark.parametrize("reentrant", [False, True])
def test_learned_parameters_round_trip_deepcopy_and_checkpoint_gradients(
    method, reentrant
):
    model = make_model(method, Normalization.RMS_NORM)
    with torch.no_grad():
        if method is AttentionConfig:
            model.adaptive_behaviour.grouper.scorer[-1].model.weight_params.T.fill_(
                0.15
            )
        else:
            model.adaptive_behaviour.grouper.projection[-1].model.weight_params.T[
                :, 2:
            ].fill_(0.2)
    reference = copy.deepcopy(model)
    restored = make_model(method, Normalization.RMS_NORM)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert all(
        a is not b
        for a, b in zip(model.parameters(), reference.parameters(), strict=True)
    )
    ungrouped_keys = bias_linear().state_dict().keys()
    assert all(
        "grouper" in name for name in model.state_dict() if name not in ungrouped_keys
    )
    initial_keys = tuple(model.state_dict())
    for batch in (2, 3):
        inputs = (
            torch.arange(batch * 8, dtype=torch.float64).reshape(-1, 2) / 11 - 1
        ).requires_grad_()
        other = inputs.detach().clone().requires_grad_()
        actual = checkpoint(model, inputs, use_reentrant=reentrant)
        expected = reference(other)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(restored(inputs.detach()), expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(inputs.grad, other.grad)
        for (name, parameter), (_, other_parameter) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            torch.testing.assert_close(parameter.grad, other_parameter.grad)
            if "grouper" in name:
                assert (
                    parameter.grad is not None and torch.isfinite(parameter.grad).all()
                )
                assert parameter.grad.abs().sum() > 0
        generator = model.adaptive_behaviour.bias_model.model[0].model
        assert generator.weight_params.grad.abs().sum() > 0
        model.zero_grad()
        reference.zero_grad()
    assert tuple(model.state_dict()) == initial_keys
    for training in (False, True):
        model.train(training)
        assert all(
            child.training is training
            for child in model.adaptive_behaviour.grouper.modules()
        )
    assert not any(
        isinstance(value, torch.Tensor)
        for child in model.adaptive_behaviour.grouper.modules()
        for value in vars(child).values()
    )


@pytest.mark.parametrize("method", GROUPING_CONFIGS)
@pytest.mark.parametrize("normalization", list(Normalization))
def test_one_context_is_shared_in_generator_order_and_weights_stay_compact(
    method, normalization
):
    config = copy.deepcopy(combined_linear().cfg)
    config.adaptive_augmentation_config.grouping_config = with_reduction(
        config.adaptive_augmentation_config.grouping_config,
        grouping_config(method, summary_normalization=normalization),
    )
    model = config.build().double()
    augmentation = model.adaptive_behaviour
    # Keep the real hard mask open so its random initialization cannot block
    # the generator-gradient contract being tested here.
    with torch.no_grad():
        model.weight_params.fill_(0.2)
        model.bias_params.fill_(0.1)
        for component in ("weight", "diagonal", "bias", "mask"):
            for parameter in getattr(augmentation, component + "_model").parameters():
                parameter.fill_(0.15)
    events = []
    contexts = []
    weights = []
    hooks = []

    def observe_chunking(_module, args, output):
        assert args[0].dim() == 2
        assert output[1].grouped_members.dim() == 3
        events.append("grouper")
        contexts.append(output[0])

    hooks.append(augmentation.grouper.register_forward_hook(observe_chunking))

    def observe(name):
        def hook(_module, args):
            events.append(name)
            contexts.append(args[-1])

        return hook

    for name in ("weight", "diagonal", "bias", "mask"):
        hooks.append(
            getattr(augmentation, name + "_model").register_forward_pre_hook(
                observe(name)
            )
        )
    hooks.append(
        augmentation.weight_model.register_forward_hook(
            lambda _module, _args, output: weights.append(output.shape)
        )
    )
    inputs = torch.tensor(
        [[1.0, 2.0], [2.0, -1.0], [3.0, 4.0], [-2.0, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    try:
        output = model(inputs)
        output.square().sum().backward()
    finally:
        for hook in hooks:
            hook.remove()
    assert events == ["grouper", "weight", "diagonal", "bias", "mask"]
    assert len(contexts) == 5 and all(context is contexts[0] for context in contexts)
    assert contexts[0].shape == (2, 2) and weights == [torch.Size([2, 2, 3])]
    assert output.shape == (4, 3) and torch.isfinite(inputs.grad).all()
    for name in ("weight", "diagonal", "bias", "mask"):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in getattr(augmentation, name + "_model").parameters()
        )


def test_recurrence_pools_each_current_transition_input():
    adaptive = make_model(MeanConfig)
    stack = linear_stack_config(2, 2)
    stack.layer_config.layer_model_config = adaptive.cfg
    recurrent = (
        RecurrentLayerConfig(
            input_dim=2,
            output_dim=2,
            max_steps=3,
            initial_iterations=3,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
            block_config=stack,
            gate_config=None,
            residual_config=None,
            halting_config=None,
            memory_config=None,
        )
        .build()
        .double()
    )
    groupers = [
        module.grouper
        for module in recurrent.modules()
        if hasattr(module, "grouper") and module.grouper is not None
    ]
    assert len(groupers) == 1
    recorded = []
    hook = groupers[0].register_forward_hook(
        lambda _module, args, output: recorded.append(
            (args[0].detach().clone(), output[0].detach().clone())
        )
    )
    inputs = torch.arange(8, dtype=torch.float64).reshape(4, 2).requires_grad_()
    try:
        result = recurrent(LayerState(hidden=inputs))
    finally:
        hook.remove()
    assert len(recorded) == 3
    assert not torch.equal(recorded[0][0], recorded[1][0])
    for flat_input, context in recorded:
        assert flat_input.shape == (4, 2)
        grouped = flat_input.reshape(2, 2, 2)
        torch.testing.assert_close(context, (grouped[:, 0] + grouped[:, 1]) / 2)
    result.hidden.square().sum().backward()
    assert torch.isfinite(inputs.grad).all()


class ObservedAdaptiveModule(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.metrics = []

    def log(self, name, value, *args, **kwargs):
        self.metrics.append((name, value.detach().clone()))


@pytest.mark.parametrize("method", [MeanStdConfig, AttentionConfig])
def test_monitor_hooks_keep_generator_slots_and_cleanup_without_training(method):
    wrapped = ObservedAdaptiveModule(make_model(method, Normalization.RMS_NORM))
    callback = AdaptiveParameterMonitorCallback(log_every_n_steps=1)
    for _ in range(2):
        callback.on_fit_start(None, wrapped)
    grouper = wrapped.model.adaptive_behaviour.grouper
    assert all(not child._forward_hooks for child in grouper.modules())
    assert len(wrapped.model.adaptive_behaviour.bias_model._forward_hooks) == 1
    inputs = torch.arange(8, dtype=torch.float64).reshape(4, 2)
    wrapped.model(inputs)
    assert wrapped.metrics and all("/bias/" in name for name, _ in wrapped.metrics)
    assert all("input_adaptivity" not in name for name, _ in wrapped.metrics)
    count = len(wrapped.metrics)
    callback.on_fit_end(None, wrapped)
    wrapped.model(inputs)
    assert len(wrapped.metrics) == count
    callback.on_fit_start(None, wrapped)
    with pytest.raises(ValueError):
        wrapped.model(torch.ones(3, 2, dtype=torch.float64))
    callback.on_exception(None, wrapped, ValueError("invalid input"))
    assert not wrapped.model.adaptive_behaviour.bias_model._forward_hooks
    wrapped.model(inputs)
    assert len(wrapped.metrics) == count


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("method", GROUPING_CONFIGS)
def test_cuda_half_model_chunking_and_backward(method):
    model = make_model(method, Normalization.RMS_NORM).cuda().half()
    inputs = torch.tensor(
        [[1.0, 2.0], [2.0, 3.0], [-2.0, 1.0], [3.0, -1.0]],
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    model(inputs).float().sum().backward()
    assert torch.isfinite(inputs.grad).all()
