import unittest

from emperor.attention.core._validator import MultiHeadAttentionValidator
from emperor.attention.core.layers import MultiHeadAttentionAbstract

from support.attention import build_attention_config


class TestMultiHeadAttentionValidatorAdapter(unittest.TestCase):
    def test_base_module_declares_its_validator_adapter(self):
        self.assertIs(
            MultiHeadAttentionAbstract.VALIDATOR,
            MultiHeadAttentionValidator,
        )

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(MultiHeadAttentionValidator):
            @classmethod
            def validate_required_fields(cls, cfg):
                raise RuntimeError("substituted construction validator was called")

        class TrackingAttention(MultiHeadAttentionAbstract):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingAttention(build_attention_config())


if __name__ == "__main__":
    unittest.main()
