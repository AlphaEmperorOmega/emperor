import unittest

import torch

from emperor.sampler import SamplerConfig, SamplerModel
from emperor.sampler._usage import (
    SamplerUsageTracker,
    SamplerUsageTrackerManager,
)


class TestSamplerUsageTracker(unittest.TestCase):
    def test_init_creates_zeroed_buffers(self):
        tracker = SamplerUsageTracker(num_experts=4)

        self.assertEqual(tracker.num_experts, 4)
        for name in (
            "last_expert_usage_counts",
            "last_expert_usage_mass",
            "cumulative_expert_usage_counts",
            "cumulative_expert_usage_mass",
        ):
            buffer = getattr(tracker, name)
            self.assertEqual(buffer.shape, (4,))
            self.assertTrue(torch.all(buffer == 0))

    def test_compute_usage_with_indices_scatters_counts_and_mass(self):
        tracker = SamplerUsageTracker(num_experts=4)
        probabilities = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.9, 0.1]])
        indices = torch.tensor([[0, 1], [0, 2], [3, 1]])

        counts, mass = tracker.compute_usage(probabilities, indices)

        torch.testing.assert_close(counts, torch.tensor([2.0, 2.0, 1.0, 1.0]))
        torch.testing.assert_close(
            mass,
            torch.tensor([0.7 + 0.6, 0.3 + 0.1, 0.4, 0.9]),
        )

    def test_compute_usage_without_indices_uses_dense_path(self):
        tracker = SamplerUsageTracker(num_experts=3)
        probabilities = torch.tensor(
            [
                [0.5, 0.0, 0.5],
                [0.2, 0.8, 0.0],
            ]
        )

        counts, mass = tracker.compute_usage(probabilities, indices=None)

        torch.testing.assert_close(counts, torch.tensor([2.0, 1.0, 1.0]))
        torch.testing.assert_close(mass, torch.tensor([0.7, 0.8, 0.5]))

    def test_record_updates_last_and_accumulates_cumulative(self):
        tracker = SamplerUsageTracker(num_experts=3)
        probabilities = torch.tensor([[0.6, 0.4]])
        indices = torch.tensor([[0, 2]])

        tracker.record(probabilities, indices)
        tracker.record(probabilities, indices)

        torch.testing.assert_close(
            tracker.last_expert_usage_counts, torch.tensor([1.0, 0.0, 1.0])
        )
        torch.testing.assert_close(
            tracker.last_expert_usage_mass, torch.tensor([0.6, 0.0, 0.4])
        )
        torch.testing.assert_close(
            tracker.cumulative_expert_usage_counts, torch.tensor([2.0, 0.0, 2.0])
        )
        torch.testing.assert_close(
            tracker.cumulative_expert_usage_mass, torch.tensor([1.2, 0.0, 0.8])
        )

    def test_record_sampler_output_unpacks_and_detaches(self):
        tracker = SamplerUsageTracker(num_experts=3)
        probabilities = torch.tensor([[0.6, 0.4]], requires_grad=True)
        indices = torch.tensor([[0, 1]])
        output = (probabilities, indices, None, torch.tensor(0.0))

        tracker.record_sampler_output(output)

        self.assertFalse(tracker.last_expert_usage_counts.requires_grad)
        torch.testing.assert_close(
            tracker.last_expert_usage_mass, torch.tensor([0.6, 0.4, 0.0])
        )

    def test_record_sampler_output_handles_none_indices(self):
        tracker = SamplerUsageTracker(num_experts=2)
        probabilities = torch.tensor([[0.3, 0.7], [0.4, 0.6]])
        output = (probabilities, None, None, torch.tensor(0.0))

        tracker.record_sampler_output(output)

        torch.testing.assert_close(
            tracker.last_expert_usage_counts, torch.tensor([2.0, 2.0])
        )
        torch.testing.assert_close(
            tracker.last_expert_usage_mass, torch.tensor([0.7, 1.3])
        )

    def test_reset_zeros_all_buffers(self):
        tracker = SamplerUsageTracker(num_experts=2)
        tracker.record(torch.tensor([[0.5, 0.5]]), torch.tensor([[0, 1]]))

        tracker.reset()

        for name in (
            "last_expert_usage_counts",
            "last_expert_usage_mass",
            "cumulative_expert_usage_counts",
            "cumulative_expert_usage_mass",
        ):
            self.assertTrue(torch.all(getattr(tracker, name) == 0))


class TestSamplerUsageTrackerManager(unittest.TestCase):
    def sampler_config(self, **overrides) -> SamplerConfig:
        values = {
            "top_k": 2,
            "threshold": 0.0,
            "filter_above_threshold": False,
            "num_topk_samples": 0,
            "normalize_probabilities_flag": False,
            "noisy_topk_flag": False,
            "num_experts": 4,
            "coefficient_of_variation_loss_weight": 0.0,
            "switch_loss_weight": 0.0,
            "zero_centred_loss_weight": 0.0,
            "mutual_information_loss_weight": 0.0,
            "router_config": None,
        }
        values.update(overrides)
        return SamplerConfig(**values)

    def test_attach_installs_tracker_module(self):
        sampler = SamplerModel(self.sampler_config())
        manager = SamplerUsageTrackerManager()

        tracker = manager.attach(sampler)

        self.assertIsInstance(tracker, SamplerUsageTracker)
        self.assertIs(sampler.usage_tracker, tracker)
        self.assertEqual(tracker.num_experts, sampler.num_experts)

    def test_attach_is_idempotent(self):
        sampler = SamplerModel(self.sampler_config())
        manager = SamplerUsageTrackerManager()

        first = manager.attach(sampler)
        second = manager.attach(sampler)

        self.assertIs(first, second)

    def test_detach_removes_tracker_module(self):
        sampler = SamplerModel(self.sampler_config())
        manager = SamplerUsageTrackerManager()
        manager.attach(sampler)

        manager.detach(sampler)

        self.assertIsNone(sampler.usage_tracker)
        self.assertNotIn(
            SamplerUsageTrackerManager.TRACKER_MODULE_NAME, sampler._modules
        )

    def test_detach_when_no_tracker_is_noop(self):
        sampler = SamplerModel(self.sampler_config())
        manager = SamplerUsageTrackerManager()

        manager.detach(sampler)

        self.assertIsNone(sampler.usage_tracker)

    def test_maybe_record_sampler_output_skips_when_no_tracker(self):
        sampler = SamplerModel(self.sampler_config())
        output = (
            torch.tensor([[0.6, 0.4]]),
            torch.tensor([[0, 1]]),
            None,
            torch.tensor(0.0),
        )

        SamplerUsageTrackerManager.maybe_record_sampler_output(sampler, output)

        self.assertIsNone(sampler.usage_tracker)

    def test_maybe_record_sampler_output_records_when_attached(self):
        sampler = SamplerModel(self.sampler_config(num_experts=3, top_k=2))
        manager = SamplerUsageTrackerManager()
        manager.attach(sampler)
        probabilities = torch.tensor([[0.7, 0.3]])
        indices = torch.tensor([[0, 2]])
        output = (probabilities, indices, None, torch.tensor(0.0))

        SamplerUsageTrackerManager.maybe_record_sampler_output(sampler, output)

        torch.testing.assert_close(
            sampler.usage_tracker.last_expert_usage_counts,
            torch.tensor([1.0, 0.0, 1.0]),
        )

    def test_end_to_end_sample_records_usage_when_tracker_attached(self):
        sampler = SamplerModel(self.sampler_config(top_k=2, num_experts=4))
        SamplerUsageTrackerManager().attach(sampler)

        logits = torch.tensor(
            [
                [5.0, 4.0, 0.0, 0.0],
                [0.0, 5.0, 4.0, 0.0],
                [0.0, 0.0, 5.0, 4.0],
            ]
        )
        sampler.sample_probabilities_and_indices(logits)

        probabilities = torch.softmax(logits, dim=-1)
        expected_indices = torch.tensor(
            [
                [0, 1],
                [1, 2],
                [2, 3],
            ]
        )
        expected_counts = torch.tensor([1.0, 2.0, 2.0, 1.0])
        expected_mass = torch.zeros(4)
        expected_mass.scatter_add_(
            0,
            expected_indices.reshape(-1),
            torch.gather(probabilities, 1, expected_indices).reshape(-1),
        )

        tracker = sampler.usage_tracker
        torch.testing.assert_close(tracker.last_expert_usage_counts, expected_counts)
        torch.testing.assert_close(
            tracker.cumulative_expert_usage_counts, expected_counts
        )
        torch.testing.assert_close(tracker.last_expert_usage_mass, expected_mass)
        torch.testing.assert_close(tracker.cumulative_expert_usage_mass, expected_mass)


if __name__ == "__main__":
    unittest.main()
