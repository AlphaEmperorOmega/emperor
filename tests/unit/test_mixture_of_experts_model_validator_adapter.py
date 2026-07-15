import unittest

from emperor.experts.config import MixtureOfExpertsModelConfig
from emperor.experts.core._validator import MixtureOfExpertsModelValidator
from emperor.experts.model import MixtureOfExpertsModel


class TestMixtureOfExpertsModelValidatorAdapter(unittest.TestCase):
    def test_model_declares_its_validator_adapter(self):
        self.assertIs(
            MixtureOfExpertsModel.VALIDATOR,
            MixtureOfExpertsModelValidator,
        )

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(MixtureOfExpertsModelValidator):
            @staticmethod
            def validate_cfg_type(model):
                raise RuntimeError("substituted construction validator was called")

        class TrackingModel(MixtureOfExpertsModel):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingModel(MixtureOfExpertsModelConfig())


if __name__ == "__main__":
    unittest.main()
