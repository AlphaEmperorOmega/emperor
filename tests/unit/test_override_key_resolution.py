from __future__ import annotations

import unittest

from model_runtime.inspection import (
    InspectionError,
    canonicalize_overrides,
    parse_overrides,
)
from models.catalog import model_package


class OverrideKeyResolutionTests(unittest.TestCase):
    def test_retired_residual_flag_override_is_rejected(self) -> None:
        package = model_package("linears/linear")
        assert package is not None

        for apply_overrides in (parse_overrides, canonicalize_overrides):
            with self.subTest(apply_overrides=apply_overrides.__name__):
                with self.assertRaisesRegex(
                    InspectionError,
                    r"^Unknown override 'stack_residual_flag'\.$",
                ):
                    apply_overrides(package, {"stack_residual_flag": "false"})


if __name__ == "__main__":
    unittest.main()
