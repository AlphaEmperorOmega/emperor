import unittest

from model_runtime.inspection import configuration_schema
from model_runtime.packages.configuration import iter_supported_config_keys
from models.catalog import model_package

_TRANSFORMER_PACKAGES = (
    "transformer/linear",
    "transformer/linear_adaptive",
    "transformer/expert_linear",
    "transformer/expert_linear_adaptive",
)


class TestTransformerConfigurationFlexibility(unittest.TestCase):
    def test_every_package_exposes_independent_transformer_path_defaults(self):
        expected_fields = {
            "ENCODER_ATTN_STACK_HIDDEN_DIM",
            "ENCODER_ATTN_GATE_STACK_INDEPENDENT_FLAG",
            "DECODER_SELF_ATTN_STACK_HIDDEN_DIM",
            "DECODER_SELF_ATTN_MEMORY_FLAG",
            "DECODER_CROSS_ATTN_STACK_HIDDEN_DIM",
            "DECODER_CROSS_ATTN_RECURRENT_STACK_GATE_FLAG",
            "ENCODER_FF_STACK_HIDDEN_DIM",
            "ENCODER_FF_GATE_STACK_INDEPENDENT_FLAG",
            "DECODER_FF_STACK_HIDDEN_DIM",
            "DECODER_FF_MEMORY_FLAG",
        }
        overrides = {
            "encoder_attn_stack_hidden_dim": 11,
            "decoder_self_attn_stack_hidden_dim": 12,
            "decoder_cross_attn_stack_hidden_dim": 13,
            "encoder_ff_stack_hidden_dim": 14,
            "decoder_ff_stack_hidden_dim": 15,
        }

        for package_key in _TRANSFORMER_PACKAGES:
            with self.subTest(package=package_key):
                package = model_package(package_key)
                self.assertTrue(
                    expected_fields.issubset(
                        iter_supported_config_keys(package.runtime_defaults)
                    )
                )
                self.assertTrue(
                    expected_fields.issubset(
                        {field.key for field in configuration_schema(package).fields}
                    )
                )

                runtime = package.bind_runtime_defaults(overrides)
                self.assertEqual(
                    runtime.encoder_attention_options.stack_options.hidden_dim,
                    11,
                )
                self.assertEqual(
                    runtime.decoder_self_attention_options.stack_options.hidden_dim,
                    12,
                )
                self.assertEqual(
                    runtime.decoder_cross_attention_options.stack_options.hidden_dim,
                    13,
                )
                self.assertEqual(
                    runtime.encoder_feed_forward_options.stack_options.hidden_dim,
                    14,
                )
                self.assertEqual(
                    runtime.decoder_feed_forward_options.stack_options.hidden_dim,
                    15,
                )


if __name__ == "__main__":
    unittest.main()
