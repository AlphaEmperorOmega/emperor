import unittest
from unittest.mock import patch

import torch

from emperor.sampler import SamplerConfig
from emperor.sampler._usage import SamplerUsageTrackerManager


def build_sampler(
    top_k: int,
    *,
    normalize_probabilities: bool = False,
    noisy_topk: bool = False,
    num_topk_samples: int = 0,
):
    return SamplerConfig(
        top_k=top_k,
        threshold=0.0,
        filter_above_threshold=False,
        num_topk_samples=num_topk_samples,
        normalize_probabilities_flag=normalize_probabilities,
        noisy_topk_flag=noisy_topk,
        num_experts=3,
        coefficient_of_variation_loss_weight=0.0,
        switch_loss_weight=0.0,
        zero_centred_loss_weight=0.0,
        mutual_information_loss_weight=0.0,
        router_config=None,
    ).build()


class SamplerRoutingContractTests(unittest.TestCase):
    def test_probability_routing_records_usage_once(self) -> None:
        sampler = build_sampler(2)
        logits = torch.tensor([[0.1, 0.2, 0.3]])

        with patch.object(
            SamplerUsageTrackerManager,
            "maybe_record_sampler_output",
        ) as record_output:
            result = sampler.sample_probabilities_and_indices(logits)

        self.assertEqual(len(result), 4)
        record_output.assert_called_once_with(sampler, result)

    def test_probability_routing_advances_rng_for_one_selection_call(self) -> None:
        sampler = build_sampler(
            2,
            noisy_topk=True,
            num_topk_samples=1,
        )
        logits = torch.tensor(
            [
                [0.8, 0.2, -0.4, -1.0, 0.5, 1.5],
                [-0.3, 0.4, 0.9, 0.7, -0.2, 1.1],
            ],
            dtype=torch.float64,
        )

        with torch.random.fork_rng():
            torch.manual_seed(314159)
            expected_result = sampler.sampler_model.get_probabilities_and_indices(
                logits
            )
            expected_rng_state = torch.random.get_rng_state()

        with torch.random.fork_rng():
            torch.manual_seed(314159)
            with patch.object(
                sampler.sampler_model,
                "get_probabilities_and_indices",
                wraps=sampler.sampler_model.get_probabilities_and_indices,
            ) as select_routes:
                actual_result = sampler.sample_probabilities_and_indices(logits)
            actual_rng_state = torch.random.get_rng_state()

        select_routes.assert_called_once_with(logits, None)
        for actual_value, expected_value in zip(
            actual_result,
            expected_result,
            strict=True,
        ):
            if expected_value is None:
                self.assertIsNone(actual_value)
            else:
                torch.testing.assert_close(actual_value, expected_value)
        torch.testing.assert_close(actual_rng_state, expected_rng_state)


if __name__ == "__main__":
    unittest.main()
