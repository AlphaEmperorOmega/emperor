# ruff: noqa: I001

import unittest

import torch

from emperor.sampler import SamplerConfig


def build_sampler(
    top_k: int,
    *,
    normalize_probabilities: bool = False,
):
    return SamplerConfig(
        top_k=top_k,
        threshold=0.0,
        filter_above_threshold=False,
        num_topk_samples=0,
        normalize_probabilities_flag=normalize_probabilities,
        noisy_topk_flag=False,
        num_experts=3,
        coefficient_of_variation_loss_weight=0.0,
        switch_loss_weight=0.0,
        zero_centred_loss_weight=0.0,
        mutual_information_loss_weight=0.0,
        router_config=None,
    ).build()


class SamplerLogScoreTests(unittest.TestCase):
    def test_each_strategy_preserves_underflowed_log_scores(self) -> None:
        logits = torch.tensor([[-30.0, -29.0, 0.0]], dtype=torch.float16)
        full_log_probabilities = torch.log_softmax(logits.float(), dim=1)
        cases = (
            (1, torch.tensor([2]), full_log_probabilities[:, 2]),
            (
                2,
                torch.tensor([[2, 1]]),
                full_log_probabilities.gather(1, torch.tensor([[2, 1]])),
            ),
            (3, None, full_log_probabilities),
        )

        for top_k, expected_indices, expected_log_scores in cases:
            with self.subTest(top_k=top_k):
                sampler = build_sampler(top_k)

                _, log_scores, indices, _, _ = (
                    sampler.sample_probabilities_log_scores_and_indices(logits)
                )

                if expected_indices is None:
                    self.assertIsNone(indices)
                else:
                    torch.testing.assert_close(indices, expected_indices)
                torch.testing.assert_close(log_scores, expected_log_scores)
                self.assertTrue(torch.isfinite(log_scores).all())

    def test_normalized_log_scores_match_selected_probabilities(self) -> None:
        sampler = build_sampler(2, normalize_probabilities=True)
        logits = torch.tensor([[0.2, -0.4, 1.7]], dtype=torch.float64)

        probabilities, log_scores, indices, _, _ = (
            sampler.sample_probabilities_log_scores_and_indices(logits)
        )

        torch.testing.assert_close(indices, torch.tensor([[2, 0]]))
        torch.testing.assert_close(log_scores.exp(), probabilities)


if __name__ == "__main__":
    unittest.main()
