import unittest
from dataclasses import FrozenInstanceError

import torch

from emperor.layers import RowLayout


class RowLayoutTests(unittest.TestCase):
    def test_rows_layout_is_explicit_immutable_and_counts_rows(self):
        layout = RowLayout.rows(
            4,
            context_sharing_restricted=False,
        )

        self.assertEqual(layout.leading_shape, (4,))
        self.assertEqual(layout.row_count, 4)
        self.assertIsNone(layout.batch_axis)
        self.assertIsNone(layout.sequence_axis)
        self.assertIsNone(layout.valid_rows)

        with self.assertRaises(FrozenInstanceError):
            layout.leading_shape = (2, 2)

    def test_sequence_layout_records_exact_batch_major_flattening(self):
        valid_rows = torch.tensor([True, True, False, False, True, True, True, False])

        layout = RowLayout.sequence(
            leading_shape=(2, 4),
            batch_axis=0,
            sequence_axis=1,
            valid_rows=valid_rows,
            context_sharing_restricted=False,
        )

        self.assertEqual(layout.leading_shape, (2, 4))
        self.assertEqual(layout.row_count, 8)
        self.assertEqual(layout.batch_axis, 0)
        self.assertEqual(layout.sequence_axis, 1)
        self.assertIs(layout.valid_rows, valid_rows)

    def test_sequence_layout_records_exact_sequence_major_flattening(self):
        layout = RowLayout.sequence(
            leading_shape=(4, 2),
            batch_axis=1,
            sequence_axis=0,
            context_sharing_restricted=False,
        )

        self.assertEqual(layout.row_count, 8)
        self.assertEqual(layout.batch_axis, 1)
        self.assertEqual(layout.sequence_axis, 0)

    def test_rejects_invalid_structural_metadata(self):
        invalid_cases = (
            (
                {"leading_shape": ()},
                "leading_shape must be a non-empty tuple",
            ),
            (
                {"leading_shape": (0,)},
                "leading_shape dimensions must be positive integers",
            ),
            (
                {"leading_shape": (True,)},
                "leading_shape dimensions must be positive integers",
            ),
            (
                {
                    "leading_shape": (2, 4),
                    "batch_axis": 0,
                    "sequence_axis": 0,
                },
                "batch_axis and sequence_axis must be distinct",
            ),
            (
                {
                    "leading_shape": (2, 4),
                    "batch_axis": 0,
                    "sequence_axis": 2,
                },
                "sequence_axis must index leading_shape",
            ),
            (
                {
                    "leading_shape": (2, 2, 2),
                    "batch_axis": 0,
                    "sequence_axis": 1,
                },
                "sequence layouts require exactly two leading axes",
            ),
            (
                {
                    "leading_shape": (2, 4),
                    "batch_axis": None,
                    "sequence_axis": None,
                },
                "row layouts require exactly one leading axis",
            ),
            (
                {
                    "leading_shape": (2, 4),
                    "batch_axis": 0,
                    "sequence_axis": None,
                },
                "sequence layouts require both batch_axis and sequence_axis",
            ),
            (
                {"batch_axis": True},
                "batch_axis must be an integer",
            ),
            (
                {"context_sharing_restricted": 0},
                "context_sharing_restricted must be a bool",
            ),
        )

        for overrides, message in invalid_cases:
            with self.subTest(overrides=overrides):
                kwargs = {
                    "leading_shape": (2, 4),
                    "batch_axis": 0,
                    "sequence_axis": 1,
                    "context_sharing_restricted": False,
                }
                kwargs.update(overrides)
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    RowLayout(**kwargs)

    def test_rejects_invalid_valid_row_masks(self):
        invalid_masks = (
            ("not-a-tensor", "valid_rows must be a Tensor"),
            (torch.ones(8), "valid_rows must be a Boolean tensor"),
            (torch.ones(2, 4, dtype=torch.bool), "valid_rows must be one-dimensional"),
            (
                torch.ones(7, dtype=torch.bool),
                "valid_rows length must equal row_count=8",
            ),
        )

        for valid_rows, message in invalid_masks:
            with self.subTest(
                shape=getattr(valid_rows, "shape", None),
                dtype=getattr(valid_rows, "dtype", None),
            ):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    RowLayout.sequence(
                        leading_shape=(2, 4),
                        batch_axis=0,
                        sequence_axis=1,
                        valid_rows=valid_rows,
                        context_sharing_restricted=False,
                    )


if __name__ == "__main__":
    unittest.main()
