import unittest

from emperor.sampler import SamplerModel


class SamplerPublicInterfaceTests(unittest.TestCase):
    def test_routing_interface_returns_only_sampler_probabilities(self) -> None:
        self.assertTrue(hasattr(SamplerModel, "sample_probabilities_and_indices"))
        self.assertFalse(
            hasattr(SamplerModel, "sample_probabilities_log_scores_and_indices")
        )


if __name__ == "__main__":
    unittest.main()
