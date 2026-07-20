import unittest

from emperor.sampler import SamplerConfig
from emperor.sampler._selection.full import SamplerFull
from emperor.sampler._selection.validation import SamplerFullValidator


def make_config(**overrides) -> SamplerConfig:
    values = {
        "top_k": 4,
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


class TestSamplerFullValidatorAdapter(unittest.TestCase):
    def test_module_exposes_specialized_validator_adapter(self):
        self.assertIs(SamplerFull.VALIDATOR, SamplerFullValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(SamplerFullValidator):
            @staticmethod
            def _validate_full_configuration(model):
                raise RuntimeError("substituted construction validator was called")

        class TrackingSamplerFull(SamplerFull):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingSamplerFull(make_config())

    def test_full_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            ValueError,
            "switch_loss_weight must be 0.0 when using SamplerFull, "
            "received 0.5",
        ):
            SamplerFull(make_config(switch_loss_weight=0.5))


if __name__ == "__main__":
    unittest.main()
