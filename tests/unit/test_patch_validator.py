import unittest

from emperor.patch.core._validator import PatchValidator
from emperor.patch.core.config import PatchConfig
from emperor.patch.core.layers import (
    PatchBase,
    PatchEmbeddingConv,
    PatchEmbeddingLinear,
)


class TestPatchValidatorAdapter(unittest.TestCase):
    def test_patch_modules_share_the_base_owner_adapter(self):
        for module_type in (PatchBase, PatchEmbeddingLinear, PatchEmbeddingConv):
            with self.subTest(module_type=module_type.__name__):
                self.assertIs(module_type.VALIDATOR, PatchValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        validated_probabilities = []

        class TrackingPatchValidator(PatchValidator):
            @staticmethod
            def _validate_dropout_probability(value):
                validated_probabilities.append(value)

        class TrackingPatch(PatchBase):
            VALIDATOR = TrackingPatchValidator

        model = TrackingPatch(
            PatchConfig(
                embedding_dim=8,
                num_input_channels=3,
                patch_size=2,
                dropout_probability=0.25,
            )
        )

        self.assertEqual(validated_probabilities, [0.25])
        self.assertIs(model.VALIDATOR, TrackingPatchValidator)

    def test_dropout_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            ValueError,
            r"dropout_probability must be in \[0\.0, 1\.0\], received 1\.1",
        ):
            PatchBase(
                PatchConfig(
                    embedding_dim=8,
                    num_input_channels=3,
                    patch_size=2,
                    dropout_probability=1.1,
                )
            )


if __name__ == "__main__":
    unittest.main()
