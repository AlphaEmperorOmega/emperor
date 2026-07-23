import unittest
from types import SimpleNamespace

import torch

from emperor.augmentations.adaptive_parameters import (
    AdditiveDynamicBiasConfig,
    BankExpansionFactorOptions,
    DynamicBiasConfig,
    WeightDecayScheduleOptions,
    WeightedBankDynamicBiasConfig,
)
from emperor.augmentations.adaptive_parameters._biases.base import DynamicBiasAbstract
from emperor.augmentations.adaptive_parameters._biases.validation import (
    DynamicBiasValidator,
)
from emperor.augmentations.adaptive_parameters._biases.variants.additive import (
    AdditiveDynamicBias,
)
from emperor.augmentations.adaptive_parameters._biases.variants.affine import (
    AffineTransformDynamicBias,
)
from emperor.augmentations.adaptive_parameters._biases.variants.gated import (
    SigmoidGatedDynamicBias,
    TanhGatedDynamicBias,
)
from emperor.augmentations.adaptive_parameters._biases.variants.generator import (
    GeneratorDynamicBias,
)
from emperor.augmentations.adaptive_parameters._biases.variants.multiplicative import (
    MultiplicativeDynamicBias,
)
from emperor.augmentations.adaptive_parameters._biases.variants.weighted_bank import (
    WeightedBankDynamicBias,
)
from emperor.layers import LayerStackConfig


class TestDynamicBiasValidatorAdapter(unittest.TestCase):
    @staticmethod
    def valid_config() -> AdditiveDynamicBiasConfig:
        return AdditiveDynamicBiasConfig(
            input_dim=2,
            output_dim=3,
            decay_schedule=WeightDecayScheduleOptions.DISABLED,
            decay_rate=0.0,
            decay_warmup_batches=0,
            model_config=LayerStackConfig(),
        )

    def test_bias_modules_share_the_base_owner_adapter(self):
        module_types = (
            DynamicBiasAbstract,
            AdditiveDynamicBias,
            AffineTransformDynamicBias,
            MultiplicativeDynamicBias,
            SigmoidGatedDynamicBias,
            TanhGatedDynamicBias,
            GeneratorDynamicBias,
            WeightedBankDynamicBias,
        )

        for module_type in module_types:
            with self.subTest(module_type=module_type.__name__):
                self.assertIs(module_type.VALIDATOR, DynamicBiasValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(DynamicBiasValidator):
            @staticmethod
            def validate_required_fields(cfg):
                raise RuntimeError("substituted construction validator was called")

        class TrackingBias(DynamicBiasAbstract):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingBias(DynamicBiasConfig())

    def test_runtime_dispatches_through_substituted_validator(self):
        class RejectingValidator(DynamicBiasValidator):
            @staticmethod
            def ensure_parameters_exist(bias_params):
                raise RuntimeError("substituted runtime validator was called")

        class RejectingBias(AdditiveDynamicBias):
            VALIDATOR = RejectingValidator

        model = RejectingBias.__new__(RejectingBias)
        torch.nn.Module.__init__(model)

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted runtime validator was called",
        ):
            model(None, torch.ones(1, 3))

    def test_validator_methods_are_check_only_for_valid_values(self):
        model = SimpleNamespace(cfg=self.valid_config())
        weighted_model = SimpleNamespace(
            cfg=WeightedBankDynamicBiasConfig(
                bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_TWO
            )
        )

        self.assertIsNone(DynamicBiasValidator.validate(model))
        self.assertIsNone(DynamicBiasValidator.ensure_parameters_exist(torch.ones(3)))
        self.assertIsNone(
            DynamicBiasValidator.validate_bank_expansion_factor(weighted_model)
        )

    def test_validate_rejects_invalid_field_types_through_the_bias_adapter(self):
        cfg = self.valid_config()
        cfg.decay_schedule = "disabled"

        with self.assertRaisesRegex(
            TypeError,
            "^decay_schedule must be WeightDecayScheduleOptions for "
            "AdditiveDynamicBiasConfig, got str$",
        ):
            DynamicBiasValidator.validate(SimpleNamespace(cfg=cfg))


if __name__ == "__main__":
    unittest.main()
