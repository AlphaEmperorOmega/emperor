import unittest

import torch

from emperor.attention._ops.projection_layout import ProjectionRowLayoutManager
from emperor.attention._runtime import (
    AttentionRuntimeLayout,
    MultiHeadAttentionInputs,
)
from emperor.attention._validation import AttentionValidatorBase


class ProjectionRowLayoutManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = ProjectionRowLayoutManager(AttentionValidatorBase)
        self.runtime_layout = AttentionRuntimeLayout(
            batch_size=2,
            target_sequence_length=3,
            source_sequence_length=3,
        )
        self.shared_input = torch.randn(3, 2, 4)

    def test_self_attention_attaches_sequence_major_valid_rows(self):
        key_padding_mask = torch.tensor(
            [[0.0, -torch.inf, -0.25], [-torch.inf, 0.0, 0.0]]
        )
        attention_inputs = MultiHeadAttentionInputs(
            query=self.shared_input,
            key=self.shared_input,
            value=self.shared_input,
            key_padding_mask=key_padding_mask,
            runtime_layout=self.runtime_layout,
        )

        result = self.manager.attach_projection_row_layout(attention_inputs)

        self.assertIsNot(result, attention_inputs)
        self.assertIs(result.query, attention_inputs.query)
        self.assertIs(result.key_padding_mask, key_padding_mask)
        self.assertIsNot(result.runtime_layout, self.runtime_layout)
        row_layout = result.runtime_layout.row_layout
        self.assertEqual(row_layout.leading_shape, (3, 2))
        self.assertEqual(row_layout.batch_axis, 1)
        self.assertEqual(row_layout.sequence_axis, 0)
        self.assertFalse(row_layout.context_sharing_restricted)
        torch.testing.assert_close(
            row_layout.valid_rows,
            torch.tensor([True, False, False, True, True, True]),
        )
        self.assertEqual(row_layout.valid_rows.device, key_padding_mask.device)

    def test_non_shared_or_constrained_sources_restrict_context_sharing(self):
        distinct_source = torch.randn_like(self.shared_input)
        static_source = torch.randn(3, 2, 4)
        cases = (
            {
                "key": distinct_source,
                "value": distinct_source,
            },
            {
                "key": self.shared_input,
                "value": self.shared_input,
                "attention_mask": torch.zeros(3, 3),
            },
            {
                "key": self.shared_input,
                "value": self.shared_input,
                "static_key": static_source,
            },
            {
                "key": self.shared_input,
                "value": self.shared_input,
                "static_value": static_source,
            },
        )

        for case in cases:
            with self.subTest(case=tuple(case)):
                result = self.manager.attach_projection_row_layout(
                    MultiHeadAttentionInputs(
                        query=self.shared_input,
                        runtime_layout=self.runtime_layout,
                        **case,
                    )
                )

                row_layout = result.runtime_layout.row_layout
                self.assertTrue(row_layout.context_sharing_restricted)
                self.assertIsNone(row_layout.valid_rows)

    def test_missing_runtime_layout_is_rejected(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "Projection row layout requires resolved attention runtime layout.",
        ):
            self.manager.attach_projection_row_layout(
                MultiHeadAttentionInputs(
                    query=self.shared_input,
                    key=self.shared_input,
                    value=self.shared_input,
                )
            )

    def test_runtime_layout_validation_uses_the_configured_validator(self):
        class RejectingValidator(AttentionValidatorBase):
            @staticmethod
            def validate_projection_row_layout_runtime_layout(_runtime_layout):
                raise RuntimeError("configured projection-layout validator was called")

        manager = ProjectionRowLayoutManager(RejectingValidator)

        with self.assertRaisesRegex(
            RuntimeError,
            "configured projection-layout validator was called",
        ):
            manager.attach_projection_row_layout(
                MultiHeadAttentionInputs(
                    query=self.shared_input,
                    key=self.shared_input,
                    value=self.shared_input,
                )
            )

    def test_misaligned_prepared_padding_mask_is_rejected(self):
        attention_inputs = MultiHeadAttentionInputs(
            query=self.shared_input,
            key=self.shared_input,
            value=self.shared_input,
            key_padding_mask=torch.zeros(2, 2),
            runtime_layout=self.runtime_layout,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Prepared key padding mask must align with attention source rows, "
            r"expected \(2, 3\), received \(2, 2\)\.",
        ):
            self.manager.attach_projection_row_layout(attention_inputs)


if __name__ == "__main__":
    unittest.main()
