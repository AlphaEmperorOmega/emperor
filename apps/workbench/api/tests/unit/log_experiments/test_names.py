from __future__ import annotations

import unittest

from emperor_workbench.log_experiments import (
    LOG_EXPERIMENT_NAME_RE,
    LogExperimentFailure,
    is_valid_log_experiment_name,
    validate_log_experiment_name,
)


class LogExperimentNameTests(unittest.TestCase):
    def test_log_experiment_name_regex_pattern_is_stable(self) -> None:
        self.assertEqual(
            LOG_EXPERIMENT_NAME_RE.pattern,
            r"^[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*$",
        )

    def test_is_valid_log_experiment_name_accepts_allowed_names(self) -> None:
        for name in ("abc", "abc_123", "A1_B2"):
            with self.subTest(name=name):
                self.assertTrue(is_valid_log_experiment_name(name))

    def test_is_valid_log_experiment_name_rejects_disallowed_names(self) -> None:
        for name in (
            "",
            ".",
            "..",
            "_abc",
            "abc_",
            "abc__def",
            "abc-def",
            "abc.def",
            "abc/def",
            "abcé",
        ):
            with self.subTest(name=name):
                self.assertFalse(is_valid_log_experiment_name(name))

    def test_validate_log_experiment_name_returns_allowed_names(self) -> None:
        for name in ("abc", "abc_123", "A1_B2"):
            with self.subTest(name=name):
                self.assertEqual(validate_log_experiment_name(name), name)

    def test_validate_log_experiment_name_rejects_disallowed_names(self) -> None:
        for name in (
            "",
            ".",
            "..",
            "_abc",
            "abc_",
            "abc__def",
            "abc-def",
            "abc.def",
            "abc/def",
            "abcé",
        ):
            with self.subTest(name=name):
                with self.assertRaises(LogExperimentFailure):
                    validate_log_experiment_name(name)


if __name__ == "__main__":
    unittest.main()
