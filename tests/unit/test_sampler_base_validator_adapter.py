import unittest

from emperor.sampler.core._validator import SamplerBaseValidator
from emperor.sampler.core.base import SamplerBase
from emperor.sampler.core.config import SamplerConfig


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


class TestSamplerBaseValidatorAdapter(unittest.TestCase):
    def test_module_exposes_validator_adapter(self):
        self.assertIs(SamplerBase.VALIDATOR, SamplerBaseValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(SamplerBaseValidator):
            @staticmethod
            def _validate_num_topk_samples(num_topk_samples, top_k):
                raise RuntimeError("substituted construction validator was called")

        class TrackingSampler(SamplerBase):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingSampler(make_config())

    def test_runtime_dispatches_through_substituted_validator(self):
        class RejectingValidator(SamplerBaseValidator):
            @staticmethod
            def _validate_router_logit_scores(model, router_logit_scores):
                raise RuntimeError("substituted runtime validator was called")

        class RejectingSampler(SamplerBase):
            VALIDATOR = RejectingValidator

        model = RejectingSampler(make_config())

        with self.assertRaisesRegex(
            RuntimeError, "substituted runtime validator was called"
        ):
            model.get_probabilities_and_indices([])

    def test_positive_integer_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            ValueError,
            "top_k must be a positive integer, received 0",
        ):
            SamplerBase(make_config(top_k=0))


if __name__ == "__main__":
    unittest.main()
