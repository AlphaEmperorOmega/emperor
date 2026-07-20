import unittest

from emperor.sampler import SamplerConfig
from emperor.sampler._selection.top_k import SamplerTopk
from emperor.sampler._selection.validation import SamplerTopkValidator


def make_config(**overrides) -> SamplerConfig:
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


class TestSamplerTopkValidatorAdapter(unittest.TestCase):
    def test_module_exposes_specialized_validator_adapter(self):
        self.assertIs(SamplerTopk.VALIDATOR, SamplerTopkValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(SamplerTopkValidator):
            @staticmethod
            def _validate_topk_configuration(model):
                raise RuntimeError("substituted construction validator was called")

        class TrackingSamplerTopk(SamplerTopk):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingSamplerTopk(make_config())

    def test_topk_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            ValueError,
            "top_k must be greater than 0 and less than num_experts when using "
            "SamplerTopk, received top_k=4, num_experts=4",
        ):
            SamplerTopk(make_config(top_k=4))


if __name__ == "__main__":
    unittest.main()
