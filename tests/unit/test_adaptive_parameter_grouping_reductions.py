"""Independent numerical and gradient oracles for chunk chunking."""

import math
from dataclasses import replace

import pytest
import torch

from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterGroupingScopeOptions as Scope,
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
from support.adaptive_grouping import bias_linear
from support.adaptive_grouping_variants import (
    GROUPING_CONFIGS,
    grouping_config,
    grouping_model_config,
    initialize_mean_summary,
)


def grouper(method, normalization=Normalization.DISABLED, *, epsilon=None, width=None):
    options = dict(summary_normalization=normalization, rms_norm_epsilon=epsilon)
    if width is not None:
        options["model_config"] = grouping_model_config(method, width=width)
    chunking = grouping_config(method, **options)
    return initialize_mean_summary(
        bias_linear(
            replace(chunking, scope=Scope.ROWS, group_count=2)
        ).adaptive_behaviour.grouper
    )


def tokens(dtype=torch.float64):
    return torch.tensor(
        [[[1.0, 2.0], [3.0, -2.0], [5.0, 6.0]], [[-4.0, 1.0], [-2.0, 1.0], [0.0, 1.0]]],
        dtype=dtype,
    )


@pytest.mark.parametrize("method", GROUPING_CONFIGS)
@pytest.mark.parametrize("normalization", list(Normalization))
def test_all_methods_and_normalizers_match_hand_computed_contexts(
    method, normalization
):
    module = grouper(
        method,
        normalization,
        epsilon=0.25 if normalization is Normalization.RMS_NORM else None,
    ).double()
    expected = {
        SumConfig: [[9.0, 6.0], [-6.0, 3.0]],
        MeanConfig: [[3.0, 2.0], [-2.0, 1.0]],
        MeanStdConfig: [[3.0, 2.0], [-2.0, 1.0]],
        AttentionConfig: [[3.0, 2.0], [-2.0, 1.0]],
        RMSConfig: [[math.sqrt(35 / 3), math.sqrt(44 / 3)], [math.sqrt(20 / 3), 1.0]],
    }[method]
    if normalization is Normalization.RMS_NORM:
        with torch.no_grad():
            module.normalizer.weight.copy_(torch.tensor([2.0, 0.5]))
        expected = [
            [
                2 * a / math.sqrt((a * a + b * b) / 2 + 0.25),
                0.5 * b / math.sqrt((a * a + b * b) / 2 + 0.25),
            ]
            for a, b in expected
        ]
    torch.testing.assert_close(
        module.summarize(tokens()), torch.tensor(expected, dtype=torch.float64)
    )
    assert module.summarize(tokens()).shape == (2, 2)


def test_population_deviation_projection_bias_and_normalization_after_projection():
    module = grouper(MeanStdConfig, Normalization.RMS_NORM, epsilon=0.25).double()
    with torch.no_grad():
        module.projection[-1].model.weight_params.T.copy_(
            torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 2.0]])
        )
        module.projection[-1].model.bias_params.copy_(torch.tensor([1.0, -0.5]))
    expected = []
    for a, b in [
        (math.sqrt(8 / 3) + 1, 2 * math.sqrt(32 / 3) - 0.5),
        (math.sqrt(8 / 3) + 1, -0.5),
    ]:
        denominator = math.sqrt((a * a + b * b) / 2 + 0.25)
        expected.append([a / denominator, b / denominator])
    torch.testing.assert_close(
        module.summarize(tokens()), torch.tensor(expected, dtype=torch.float64)
    )
    plain = grouper(MeanStdConfig).double()
    with torch.no_grad():
        plain.projection[-1].model.weight_params.T.copy_(
            torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 2.0]])
        )
    same_mean = torch.tensor(
        [[[0.0, 0.0], [2.0, 2.0]], [[-1.0, -2.0], [3.0, 4.0]]], dtype=torch.float64
    )
    torch.testing.assert_close(
        plain.summarize(same_mean),
        torch.tensor([[2.0, 3.0], [3.0, 7.0]], dtype=torch.float64),
    )


def test_content_attention_matches_scalar_reference_and_reaches_both_scoring_layers():
    module = grouper(AttentionConfig, width=3).double()
    weights = [[1.0, 0.0], [0.0, -0.5], [0.25, 0.25]]
    biases = [0.2, -0.1, 0.0]
    final = [0.3, -0.6, 0.9]
    with torch.no_grad():
        module.scorer[0].model.weight_params.T.copy_(torch.tensor(weights))
        module.scorer[0].model.bias_params.copy_(torch.tensor(biases))
        module.scorer[-1].model.weight_params.T.copy_(torch.tensor([final]))
    inputs = tokens().requires_grad_()
    expected = []
    for chunk in inputs.detach().tolist():
        scores = [
            sum(
                v * math.tanh(w[0] * x + w[1] * y + b)
                for w, b, v in zip(weights, biases, final, strict=True)
            )
            for x, y in chunk
        ]
        probabilities = [math.exp(score - max(scores)) for score in scores]
        denominator = sum(probabilities)
        expected.append(
            [
                sum(p * token[d] for p, token in zip(probabilities, chunk, strict=True))
                / denominator
                for d in range(2)
            ]
        )
    actual = module.summarize(inputs)
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64))
    actual.square().sum().backward()
    assert torch.isfinite(inputs.grad).all() and inputs.grad.abs().sum() > 0
    for parameter in module.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_explicit_uniform_attention_fixture_and_final_scorer_gradient():
    module = grouper(AttentionConfig).double()
    assert module.scorer[0].model.weight_params.T.shape == (64, 2)
    assert module.scorer[-1].model.bias_params is None
    inputs = tokens().requires_grad_()
    module.summarize(inputs).square().sum().backward()
    assert module.scorer[-1].model.weight_params.grad.abs().sum() > 0
    assert module.scorer[0].model.weight_params.grad.count_nonzero() == 0
    assert module.scorer[0].model.bias_params.count_nonzero() == 0


@pytest.mark.parametrize("method", GROUPING_CONFIGS)
@pytest.mark.parametrize("normalization", list(Normalization))
@pytest.mark.parametrize(
    "values",
    [
        torch.zeros(2, 3, 2),
        torch.ones(2, 3, 2) * -2,
        torch.tensor([[[-3.0, 0.0]], [[0.0, 2.0]]]),
    ],
)
def test_zero_constant_and_single_member_chunks_have_finite_gradients(
    method, normalization, values
):
    module = grouper(method, normalization).double()
    if method is MeanStdConfig:
        with torch.no_grad():
            module.projection[-1].model.weight_params.T[:, 2:].fill_(0.5)
    inputs = values.double().requires_grad_()
    output = module.summarize(inputs)
    output.sum().backward()
    assert torch.isfinite(output).all() and torch.isfinite(inputs.grad).all()
    for parameter in module.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    if not values.count_nonzero():
        assert not output.count_nonzero()
        if method is RMSConfig:
            assert not inputs.grad.count_nonzero()


def test_rms_obeys_population_identity_and_single_member_absolute_value():
    inputs = tokens()
    rms = grouper(RMSConfig).double().summarize(inputs)
    mean = grouper(MeanConfig).double().summarize(inputs)
    deviation = grouper(MeanStdConfig).double()
    with torch.no_grad():
        deviation.projection[-1].model.weight_params.T.copy_(
            torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        )
    torch.testing.assert_close(
        rms.square(), mean.square() + deviation.summarize(inputs).square()
    )
    torch.testing.assert_close(
        grouper(RMSConfig).double().summarize(inputs[:, :1]), inputs[:, 0].abs()
    )


@pytest.mark.parametrize("method", GROUPING_CONFIGS)
def test_permutation_invariance_noncontiguous_input_and_chunk_isolation(method):
    module = grouper(method).double()
    if method is AttentionConfig:
        with torch.no_grad():
            module.scorer[-1].model.weight_params.T.fill_(0.2)
    inputs = tokens().transpose(0, 1).contiguous().transpose(0, 1)
    assert not inputs.is_contiguous()
    expected = module.summarize(inputs)
    torch.testing.assert_close(module.summarize(inputs[:, [2, 0, 1]]), expected)
    changed = inputs.clone()
    changed[0] *= 3
    torch.testing.assert_close(module.summarize(changed)[1], expected[1])


@pytest.mark.parametrize("method", [MeanStdConfig, AttentionConfig, RMSConfig])
def test_gradcheck_on_nondegenerate_double_inputs(method):
    module = grouper(
        method, Normalization.RMS_NORM, width=3 if method is AttentionConfig else None
    ).double()
    with torch.no_grad():
        if method is MeanStdConfig:
            module.projection[-1].model.weight_params.T[:, 2:].fill_(0.3)
        if method is AttentionConfig:
            module.scorer[-1].model.weight_params.T.fill_(0.2)
    # Population deviation has no second derivative at zero variance.
    # Constant/single-member chunks have separate finite first-backward tests.
    inputs = tokens()
    inputs[1, :, 1] = torch.tensor([1.0, 2.0, 4.0])
    assert torch.autograd.gradcheck(module.summarize, (inputs.requires_grad_(),))
    assert torch.autograd.gradgradcheck(module.summarize, (inputs,))


@pytest.mark.parametrize("method", GROUPING_CONFIGS)
@pytest.mark.parametrize("normalization", list(Normalization))
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_dtype_device_and_backward_contract(method, normalization, dtype):
    module = grouper(method, normalization).to(dtype=dtype)
    inputs = tokens(dtype).requires_grad_()
    output = module.summarize(inputs)
    assert output.dtype == dtype and output.device == inputs.device
    output.float().sum().backward()
    assert torch.isfinite(inputs.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("method", GROUPING_CONFIGS)
def test_cpu_autocast_preserves_input_context_dtype(method, dtype):
    module = grouper(method, Normalization.RMS_NORM)
    inputs = tokens(torch.float32).requires_grad_()
    with torch.autocast("cpu", dtype=dtype):
        output = module.summarize(inputs)
    assert output.dtype == inputs.dtype
    output.sum().backward()
    assert torch.isfinite(inputs.grad).all()


@pytest.mark.parametrize(
    "bad_input",
    [
        None,
        torch.ones(2, 3, 2, dtype=torch.int64),
        torch.ones(2, 3, 2, dtype=torch.bool),
        torch.ones(2, 3, 2, dtype=torch.complex64),
        torch.ones(3, 2),
        torch.ones(2, 3, 3),
        torch.empty(0, 3, 2),
        torch.empty(2, 0, 2),
        torch.empty(2, 3, 0),
    ],
)
def test_grouper_rejects_invalid_tensor_contract(bad_input):
    module = grouper(SumConfig)
    with pytest.raises((TypeError, ValueError)):
        module.summarize(bad_input)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("normalization", list(Normalization))
@pytest.mark.parametrize(
    "method", [MeanConfig, MeanStdConfig, AttentionConfig, RMSConfig]
)
def test_low_precision_completed_context_matches_explicit_accumulation(
    method, normalization, dtype
):
    module = grouper(
        method, normalization, width=3 if method is AttentionConfig else None
    ).to(dtype=dtype)
    inputs = torch.tensor(
        [
            [[125.25, -41.5], [333.75, 10.125], [-55.5, 16.25]],
            [[1.25, 0.0], [0.125, -3.75], [-4.5, 5.125]],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    accumulator = inputs.float()
    if method is RMSConfig:
        reduced = (accumulator.square().sum(1) / 3).sqrt()
    elif method is MeanStdConfig:
        with torch.no_grad():
            module.projection[-1].model.weight_params.T[:, 2:].copy_(
                torch.tensor([[0.25, 0.0], [0.0, -0.5]], dtype=dtype)
            )
        mean = accumulator.sum(1) / 3
        deviation = ((accumulator - mean[:, None]).square().sum(1) / 3).sqrt()
        statistics = torch.cat((mean, deviation), -1).to(dtype)
        reduced = torch.nn.functional.linear(
            statistics,
            module.projection[-1].model.weight_params.T,
            module.projection[-1].model.bias_params,
        )
    elif method is AttentionConfig:
        with torch.no_grad():
            module.scorer[-1].model.weight_params.T.copy_(
                torch.tensor([[0.2, -0.4, 0.1]], dtype=dtype)
            )
        # The learned boundaries use module precision; softmax and weighted sum use float32.
        hidden = torch.nn.functional.linear(
            inputs,
            module.scorer[0].model.weight_params.T,
            module.scorer[0].model.bias_params,
        ).tanh()
        scores = torch.nn.functional.linear(
            hidden, module.scorer[-1].model.weight_params.T
        ).float()
        probabilities = (scores - scores.amax(1, keepdim=True)).exp()
        probabilities = probabilities / probabilities.sum(1, keepdim=True)
        reduced = (probabilities * accumulator).sum(1)
    else:
        reduced = accumulator.sum(1) / 3
    if normalization is Normalization.RMS_NORM:
        with torch.no_grad():
            module.normalizer.weight.copy_(torch.tensor([1.5, 0.5], dtype=dtype))
        reduced = reduced.float()
        reduced = (
            reduced
            / (reduced.square().mean(-1, keepdim=True) + 1e-6).sqrt()
            * module.normalizer.weight.float()
        )
    actual = module.summarize(inputs)
    torch.testing.assert_close(actual, reduced.to(dtype))
    actual.float().sum().backward()
    assert torch.isfinite(inputs.grad).all()


def test_half_rms_does_not_square_in_half_or_change_native_sum():
    inputs = torch.tensor(
        [[[40000.0, -40000.0], [40000.0, -40000.0]]],
        dtype=torch.float16,
        requires_grad=True,
    )
    rms = grouper(RMSConfig).half().summarize(inputs)
    torch.testing.assert_close(
        rms, torch.tensor([[40000.0, 40000.0]], dtype=torch.float16)
    )
    rms.sum().backward()
    torch.testing.assert_close(
        inputs.grad, torch.tensor([[[0.5, -0.5], [0.5, -0.5]]], dtype=torch.float16)
    )
    summed = grouper(SumConfig).half().summarize(inputs)
    torch.testing.assert_close(summed, inputs.sum(1), rtol=0, atol=0)
    assert summed.isinf().all()
