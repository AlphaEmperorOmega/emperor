import unittest

from emperor.sampler.core._validator import SamplerSparseValidator
from emperor.sampler.core.config import SamplerConfig
from emperor.sampler.core.variants import SamplerSparse


def make_config(**overrides) -> SamplerConfig:
    values = {
        "top_k": 1,
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


class TestSamplerSparseValidatorAdapter(unittest.TestCase):
    def test_module_exposes_specialized_validator_adapter(self):
        self.assertIs(SamplerSparse.VALIDATOR, SamplerSparseValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(SamplerSparseValidator):
            @staticmethod
            def _validate_sparse_configuration(model):
                raise RuntimeError("substituted construction validator was called")

        class TrackingSamplerSparse(SamplerSparse):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingSamplerSparse(make_config())

    def test_sparse_top_k_normalization_behavior_is_preserved(self):
        model = SamplerSparse(make_config(top_k=2))

        self.assertEqual(model.cfg.top_k, 2)
        self.assertEqual(model.top_k, 1)

    def test_sparse_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            ValueError,
            "normalize_probabilities_flag must be False when using "
            "SamplerSparse, received True",
        ):
            SamplerSparse(make_config(normalize_probabilities_flag=True))


if __name__ == "__main__":
    unittest.main()
