import unittest

import torch
from emperor.parametric.core.mixtures._validator import AdaptiveMixtureValidator
from emperor.parametric.core.mixtures.base import AdaptiveMixtureBase
from emperor.parametric.core.mixtures.config import AdaptiveMixtureConfig
from emperor.parametric.core.mixtures.options import ClipParameterOptions
from emperor.parametric.core.mixtures.types.generator import GeneratorWeightsMixture
from emperor.parametric.core.mixtures.types.matrix import (
    MatrixBiasMixture,
    MatrixWeightsMixture,
)
from emperor.parametric.core.mixtures.types.vector import VectorWeightsMixture


def make_config(**overrides) -> AdaptiveMixtureConfig:
    values = {
        "input_dim": 3,
        "output_dim": 4,
        "top_k": 2,
        "num_experts": 4,
        "weighted_parameters_flag": False,
        "clip_parameter_option": ClipParameterOptions.DISABLED,
        "clip_range": 1.0,
    }
    values.update(overrides)
    return AdaptiveMixtureConfig(**values)


class TestAdaptiveMixtureValidatorAdapter(unittest.TestCase):
    def test_mixture_modules_share_the_base_owner_adapter(self):
        module_types = (
            AdaptiveMixtureBase,
            VectorWeightsMixture,
            MatrixWeightsMixture,
            MatrixBiasMixture,
            GeneratorWeightsMixture,
        )

        for module_type in module_types:
            with self.subTest(module_type=module_type.__name__):
                self.assertIs(module_type.VALIDATOR, AdaptiveMixtureValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(AdaptiveMixtureValidator):
            @staticmethod
            def _validate_clip_range(cfg):
                raise RuntimeError("substituted construction validator was called")

        class TrackingMixture(AdaptiveMixtureBase):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingMixture(make_config())

    def test_runtime_dispatches_through_substituted_validator(self):
        class RejectingValidator(AdaptiveMixtureValidator):
            @staticmethod
            def validate_input_batch_2d(input_batch):
                raise RuntimeError("substituted runtime validator was called")

        class RejectingGeneratorMixture(GeneratorWeightsMixture):
            VALIDATOR = RejectingValidator

        model = RejectingGeneratorMixture.__new__(RejectingGeneratorMixture)
        torch.nn.Module.__init__(model)
        model.cfg = make_config()

        with self.assertRaisesRegex(
            RuntimeError, "substituted runtime validator was called"
        ):
            model.compute_mixture(None, None, torch.ones(1, 3))

    def test_top_k_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            ValueError,
            "top_k cannot exceed num_experts for AdaptiveMixtureConfig, "
            "received top_k=5, num_experts=4",
        ):
            AdaptiveMixtureBase(make_config(top_k=5))


if __name__ == "__main__":
    unittest.main()
