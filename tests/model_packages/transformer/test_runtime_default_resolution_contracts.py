from __future__ import annotations

import re
import unittest

from models.catalog import model_package

_TRANSFORMER_VARIANTS = (
    "linear",
    "linear_adaptive",
    "expert_linear",
    "expert_linear_adaptive",
)


class TransformerRuntimeDefaultResolutionContracts(unittest.TestCase):
    def test_scoped_nested_paths_override_broadcasts_through_package_interface(
        self,
    ) -> None:
        for variant in _TRANSFORMER_VARIANTS:
            with self.subTest(variant=variant):
                values = {
                    "attn_gate_stack_hidden_dim": 11,
                    "encoder_attn_gate_stack_hidden_dim": 13,
                }
                original = dict(values)
                package = model_package(f"transformer/{variant}")
                self.assertIsNotNone(package)
                assert package is not None

                resolved = package.bind_runtime_defaults(values)

                self.assertEqual(
                    resolved.encoder_attention_options.layer_controller_options.gate_stack_options.hidden_dim,
                    13,
                )
                self.assertEqual(
                    resolved.decoder_self_attention_options.layer_controller_options.gate_stack_options.hidden_dim,
                    11,
                )
                self.assertEqual(
                    resolved.decoder_cross_attention_options.layer_controller_options.gate_stack_options.hidden_dim,
                    11,
                )
                self.assertEqual(values, original)

    def test_invalid_component_and_leaf_names_keep_exact_package_errors(
        self,
    ) -> None:
        for variant in _TRANSFORMER_VARIANTS:
            package_name = f"models.transformer.{variant}"
            package = model_package(f"transformer/{variant}")
            self.assertIsNotNone(package)
            assert package is not None
            for invalid_key in (
                "encoder_attention_stack_hidden_dim",
                "encoder_attn_gate_stack_hidden_dimensions",
            ):
                message = (
                    f"{package_name}: unknown Runtime Defaults field(s): "
                    f"{invalid_key!r}"
                )
                with (
                    self.subTest(variant=variant, invalid_key=invalid_key),
                    self.assertRaisesRegex(ValueError, f"^{re.escape(message)}$"),
                ):
                    package.bind_runtime_defaults(
                        {
                            "attn_gate_stack_hidden_dim": 11,
                            invalid_key: 99,
                        }
                    )


if __name__ == "__main__":
    unittest.main()
