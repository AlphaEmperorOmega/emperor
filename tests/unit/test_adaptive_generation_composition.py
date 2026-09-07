"""Whole-pipeline oracles use explicit grouping, routing, rank and affine math."""

import pytest
import torch

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    MaskDimensionOptions,
    PerAxisScoreMaskConfig,
    StandardDynamicDiagonalConfig,
    SumGroupingConfig,
)
from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterGroupingScopeOptions as Scope,
)
from emperor.augmentations.adaptive_parameters import (
    LowRankFactorSourceOptions as Source,
)
from emperor.augmentations.adaptive_parameters import (
    RMSGroupingConfig as RMSConfig,
)
from emperor.augmentations.adaptive_parameters import (
    SumGroupingConfig as SumConfig,
)
from emperor.augmentations.adaptive_parameters import (
    SummaryNormalizationOptions as SummaryNorm,
)
from support.adaptive_generation import (
    bias_mixture_config,
    modulated_config,
    weight_mixture_config,
)
from support.adaptive_grouping import grouping_value, linear_stack_config
from support.adaptive_grouping_variants import (
    GROUPING_CONFIGS,
    grouping_config,
    initialize_mean_summary,
    with_reduction,
)


def combined_config(grouping=None):
    return AdaptiveLinearLayerConfig(
        input_dim=3,
        output_dim=2,
        bias_flag=True,
        adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
            model_config=linear_stack_config(3, 2),
            grouping_config=grouping,
            weight_config=weight_mixture_config(
                input_dim=3, output_dim=2, model_config=linear_stack_config(3, 2)
            ),
            diagonal_config=StandardDynamicDiagonalConfig(),
            bias_config=bias_mixture_config(
                input_dim=3, output_dim=2, model_config=linear_stack_config(3, 2)
            ),
            mask_config=PerAxisScoreMaskConfig(
                mask_dimension_option=MaskDimensionOptions.COLUMN,
                mask_threshold=0.5,
                mask_surrogate_scale=1.0,
                mask_floor=0.2,
            ),
        ),
    )


def affine(stack, context):
    linear = stack[0].model
    return context @ linear.weight_params + linear.bias_params


def staged_reference(model, members, context):
    augmentation = model.adaptive_behaviour

    def mixture_reference(variant):
        probabilities = affine(variant.sampler.router.model, context).softmax(-1)
        alpha, indices = probabilities.topk(variant.top_k, dim=-1)
        alpha = alpha / (alpha.sum(-1, keepdim=True) + 1e-6).detach()
        return torch.stack(
            [
                sum(
                    alpha[c, k] * variant.parameter_bank[indices[c, k]]
                    for k in range(variant.top_k)
                )
                for c in range(len(context))
            ]
        )

    weight = model.weight_params + mixture_reference(augmentation.weight_model)
    bias = model.bias_params + mixture_reference(augmentation.bias_model)
    diagonal = affine(augmentation.diagonal_model.model, context)
    weight = weight + torch.nn.functional.pad(torch.diag_embed(diagonal), (0, 0, 0, 1))
    scores = affine(augmentation.mask_model.model, context).sigmoid()
    mask = (0.2 + 0.8 * (scores >= 0.5).to(scores.dtype)) * (scores - 0.5).sigmoid()
    weight = weight * mask[:, None, :]
    return torch.stack([members[c] @ weight[c] + bias[c] for c in range(len(context))])


def test_variable_chunks_preserve_bank_low_rank_diagonal_bias_mask_composition():
    model = (
        combined_config(SumGroupingConfig(scope=Scope.ROWS, chunk_size=5))
        .build()
        .double()
    )
    inputs = torch.randn(12, 3, dtype=torch.float64, requires_grad=True)
    expected = torch.cat(
        [
            staged_reference(model, part[None], part.sum(0, keepdim=True))[0]
            for part in inputs.split(5)
        ]
    )
    actual = model(inputs)
    torch.testing.assert_close(actual, expected)
    targets = (inputs, *model.parameters())
    actual_gradients = torch.autograd.grad(
        actual.square().sum(), targets, retain_graph=True, allow_unused=True
    )
    expected_gradients = torch.autograd.grad(
        expected.square().sum(), targets, allow_unused=True
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("method", GROUPING_CONFIGS)
@pytest.mark.parametrize("normalization", list(SummaryNorm))
@pytest.mark.parametrize(
    "scope,order",
    [
        (Scope.ROWS, "BATCH_FIRST"),
        (Scope.SEQUENCE, "BATCH_FIRST"),
        (Scope.SEQUENCE, "SEQUENCE_FIRST"),
    ],
)
def test_context_once_composition_order_original_members_and_chunk_isolation(
    method, normalization, scope, order
):
    grouping = grouping_value(scope, 4 if scope is Scope.ROWS else 2, input_order=order)
    grouping = with_reduction(
        grouping,
        grouping_config(
            method,
            summary_normalization=normalization,
            rms_norm_epsilon=0.01 if normalization is SummaryNorm.RMS_NORM else None,
        ),
    )
    model = combined_config(grouping).build().double()
    initialize_mean_summary(model.adaptive_behaviour.grouper)
    canonical = torch.arange(24, dtype=torch.float64).reshape(2, 4, 3) / 20 - 0.3
    flat = (
        (canonical.transpose(0, 1) if order == "SEQUENCE_FIRST" else canonical)
        .reshape(8, 3)
        .requires_grad_()
    )
    members = canonical.reshape(4, 2, 3)
    if method is SumConfig:
        context = members.sum(1)
    elif method is RMSConfig:
        context = members.square().mean(1).sqrt()
    else:  # The learned test fixtures are explicitly initialized to the mean.
        context = members.mean(1)
    if normalization is SummaryNorm.RMS_NORM:
        context = context / (context.square().mean(-1, keepdim=True) + 0.01).sqrt()
    expected = staged_reference(model, members, context).reshape(2, 4, 2)
    expected = (
        expected.transpose(0, 1) if order == "SEQUENCE_FIRST" else expected
    ).reshape(8, 2)
    contexts = []
    handle = model.adaptive_behaviour.grouper.register_forward_hook(
        lambda _m, _i, output: contexts.append(output[0].detach())
    )
    actual = model(flat)
    handle.remove()
    assert len(contexts) == 1
    torch.testing.assert_close(contexts[0], context)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert torch.isfinite(flat.grad).all() and flat.grad.abs().sum() > 0
    altered = canonical.clone()
    altered[0, :2] += 0.7
    changed = model(
        (altered.transpose(0, 1) if order == "SEQUENCE_FIRST" else altered).reshape(
            8, 3
        )
    )
    changed = (
        changed.reshape(4, 2, 2).transpose(0, 1)
        if order == "SEQUENCE_FIRST"
        else changed.reshape(2, 4, 2)
    )
    original = (
        actual.reshape(4, 2, 2).transpose(0, 1)
        if order == "SEQUENCE_FIRST"
        else actual.reshape(2, 4, 2)
    )
    torch.testing.assert_close(changed[1], original[1])
    torch.testing.assert_close(changed[0, 2:], original[0, 2:])


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_per_row_composition_dtype_movement_and_strict_state_roundtrip(dtype):
    model = combined_config().build().to(dtype=dtype)
    other = combined_config().build().to(dtype=dtype)
    other.load_state_dict(model.state_dict(), strict=True)
    context = torch.tensor(
        [[0.2, 0.4, -0.1], [0.7, -0.2, 0.3]], dtype=dtype, requires_grad=True
    )
    output = model(context)
    torch.testing.assert_close(output, other(context))
    torch.testing.assert_close(
        output,
        staged_reference(model, context[:, None, :], context).squeeze(1),
        rtol=0.02 if dtype in (torch.float16, torch.bfloat16) else None,
        atol=0.005 if dtype in (torch.float16, torch.bfloat16) else None,
    )
    assert output.dtype == dtype
    output.float().square().sum().backward()
    assert torch.isfinite(context.grad).all()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    assert len(list(model.parameters())) == len({id(p) for p in model.parameters()})


def test_inspection_reports_owned_banks_and_factors_and_unique_parameter_count():
    from model_runtime.inspection.model_graph import inspect_model_graph

    model = combined_config().build()
    graph = inspect_model_graph(model)
    root = graph.nodes[0]
    assert root.parameter_count == sum(p.numel() for p in model.parameters())
    assert root.details["weight_shape"] == "3 x 2"
    assert root.details["bias_shape"] == "2"
    bank = next(node for node in graph.nodes if node.path.endswith("weight_model"))
    bias = next(node for node in graph.nodes if node.path.endswith("bias_model"))
    assert bank.details["parameter_bank_shape"] == "3 x 3 x 2"
    assert bias.details["parameter_bank_shape"] == "3 x 2"
    assert bank.parameter_count == 30 and bias.parameter_count == 18
    low_rank = modulated_config(
        input_factor_source=Source.SHARED_PARAMETER,
        output_factor_source=Source.SHARED_PARAMETER,
    ).build()
    factors = inspect_model_graph(low_rank).nodes[0]
    assert factors.details["input_factor_shape"] == "3 x 2"
    assert factors.details["output_factor_shape"] == "2 x 2"
