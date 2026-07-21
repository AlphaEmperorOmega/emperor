import unittest

from emperor.experts import MixtureOfExpertsModelConfig
from emperor.experts._model import MixtureOfExpertsModel
from emperor.experts._validation.model import MixtureOfExpertsModelValidator


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
