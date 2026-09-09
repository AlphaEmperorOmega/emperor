import unittest
from dataclasses import dataclass, replace

import torch

from emperor.config import ConfigBase, optional_field
from emperor.nn import Module
from emperor.sampler import RouterConfig, TokenSamplerConfig, TokenSamplerModel
from support.layers import _set_affine_parameters, linear_stack_config


def sampler_config(**overrides):
    return replace(
        TokenSamplerConfig(
            input_dim=2,
            selection_ratio=0.5,
            router_config=RouterConfig(
                num_experts=1,
                noisy_topk_flag=False,
                model_config=linear_stack_config(2, output_dim=1),
            ),
        ),
        **overrides,
    )


def identity_score_sampler(**overrides):
    sampler = sampler_config(**overrides).build()
    _set_affine_parameters(sampler.router.model, torch.tensor([[1.0], [0.0]]), None)
    return sampler


@dataclass
class NonlinearScorerConfig(ConfigBase):
    input_dim: int | None = optional_field("Input width.")
    output_dim: int | None = optional_field("Output width.")

    def _registry_owner(self):
        return NonlinearScorer


class NonlinearScorer(Module):
    def __init__(self, cfg, overrides=None):
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.cfg.input_dim, 5),
            torch.nn.Tanh(),
            torch.nn.Linear(5, self.cfg.output_dim),
        )

    def forward(self, state):
        return replace(state, hidden=self.net(state.hidden))


class TestTokenSampler(unittest.TestCase):
    def test_sampler_selects_per_sequence_and_ranks_logits_before_sigmoid_saturation(
        self,
    ):
        sampler = identity_score_sampler()
        hidden = torch.tensor(
            [
                [[100.0, 0], [120.0, 0], [110.0, 0], [90.0, 0]],
                [[130.0, 0], [90.0, 0], [80.0, 0], [120.0, 0]],
            ]
        )
        result = sampler(hidden)
        torch.testing.assert_close(result.indices, torch.tensor([[1, 2], [0, 3]]))
        assert result.valid.all()
        assert result.sequence_length == 4

    def test_one_sampler_handles_variable_lengths_without_rebuilding_parameters(self):
        for token_count in [1, 3, 8]:
            with self.subTest(token_count=token_count):
                sampler = identity_score_sampler(selection_ratio=0.3)
                parameters = tuple(sampler.parameters())
                sampled = sampler(torch.randn(2, token_count, 2))
                assert sampled.indices.shape == (2, max(1, int(0.3 * token_count)))
                assert all(
                    first is second
                    for first, second in zip(
                        parameters, sampler.parameters(), strict=True
                    )
                )

    def test_tied_scores_still_select_exactly_k_distinct_tokens(self):
        sampled = identity_score_sampler()(torch.ones(2, 6, 2))
        assert sampled.indices.shape == (2, 3)
        assert (sampled.indices.diff(dim=-1) > 0).all()

    def test_float_padding_excluded_and_filler_weights_have_no_gradient(self):
        hidden = torch.tensor(
            [[[0.0, 1], [2.0, 2], [3.0, 3], [4.0, 4]]], requires_grad=True
        )
        sampled = identity_score_sampler(selection_ratio=0.75)(
            hidden,
            torch.tensor([[-torch.inf, 0, -torch.inf, -torch.inf]]),
        )
        torch.testing.assert_close(sampled.indices, torch.tensor([[1, 1, 1]]))
        torch.testing.assert_close(sampled.valid, torch.tensor([[True, False, False]]))
        sampled.weights.sum().backward()
        assert hidden.grad[0, 1, 0] > 0
        assert hidden.grad[0, [0, 2, 3]].count_nonzero() == 0

    def test_generic_nonlinear_scoring_network_accepts_dimension_overrides_and_gradients(
        self,
    ):
        cfg = sampler_config(
            input_dim=2,
            router_config=RouterConfig(
                num_experts=1,
                noisy_topk_flag=False,
                model_config=NonlinearScorerConfig(),
            ),
        )
        sampler = cfg.build_with_router_input_dim(3).double()
        assert isinstance(sampler, TokenSamplerModel)
        assert isinstance(sampler.router.model, NonlinearScorer)
        assert cfg.input_dim == 2
        assert cfg.router_config.input_dim is None
        hidden = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
        sampled = sampler(hidden)
        sampled.weights.sum().backward()
        assert hidden.grad.abs().sum() > 0
        assert all(parameter.grad is not None for parameter in sampler.parameters())
        assert sampled.weights.dtype == hidden.dtype

    def test_invalid_sampler_configuration_rejected(self):
        for overrides in [
            {"selection_ratio": 0},
            {"selection_ratio": float("nan")},
            {"selection_ratio": True},
            {"input_dim": True},
            {"router_config": None},
            {"router_config": RouterConfig(num_experts=2, noisy_topk_flag=False)},
            {"router_config": RouterConfig(num_experts=1, noisy_topk_flag=True)},
        ]:
            with self.subTest(overrides=overrides):
                with self.assertRaises((TypeError, ValueError)):
                    sampler_config(**overrides).build()

    def test_invalid_token_inputs_rejected(self):
        for hidden, padding in [
            (torch.ones(2, 3, 2), torch.ones(2, 3, dtype=torch.bool)),
            (torch.ones(2, 3, 2), torch.zeros(2, 2, dtype=torch.bool)),
            (torch.ones(2, 3, 2), torch.zeros(2, 3, dtype=torch.int64)),
            (torch.ones(2), None),
            (torch.ones(2, 3, 4), None),
            (torch.ones(2, 3, 2, dtype=torch.int64), None),
            (torch.ones(2, 0, 2), None),
        ]:
            with self.subTest(hidden=hidden, padding=padding):
                with self.assertRaises((TypeError, ValueError)):
                    identity_score_sampler()(hidden, padding)

    def test_invalid_router_output_rejected(self):
        sampler = identity_score_sampler()
        sampler.router.compute_logit_scores = lambda hidden: hidden
        with self.assertRaisesRegex(ValueError, "one logit per input token"):
            sampler(torch.randn(2, 3, 2))
