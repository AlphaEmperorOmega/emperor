from __future__ import annotations

import unittest
from enum import Enum
from types import ModuleType

from model_runtime.packages import (
    ConfigValueError,
    parse_config_value,
    serialize_config_value,
)


class _Color(Enum):
    RED = "red"
    BLUE = "blue"


class _RenamedOption(Enum):
    CURRENT = "current"
    HISTORICAL = "current"
    ALTERNATIVE = "alternative"


class _FirstChoice:
    pass


class _SecondChoice:
    pass


def _configuration_module() -> ModuleType:
    module = ModuleType("configuration_value_fixture")
    module._FirstChoice = _FirstChoice
    module._SecondChoice = _SecondChoice
    module.CurrentChoice = _FirstChoice
    module.HistoricalChoice = _FirstChoice
    module.CURRENT_BOOL = True
    module.CURRENT_INT = 1
    module.CURRENT_FLOAT = 1.0
    module.CURRENT_STRING = "old"
    module.CURRENT_ENUM = _Color.RED
    module.RENAMED_ENUM = _RenamedOption.CURRENT
    module.CURRENT_CLASS = _FirstChoice
    module.RENAMED_CLASS = _FirstChoice
    module.CURRENT_LIST = [None, 1, 2]
    module.CURRENT_ENUM_LIST = [None, _Color.RED]
    module.EMPTY_LIST = []
    module.NONE_LIST = [None, None]
    module.FALLBACK = object()
    module.ANNOTATED_NONE = None
    module.__annotations__ = {
        "CURRENT_BOOL": int,
        "CURRENT_INT": bool,
        "CURRENT_FLOAT": str,
        "CURRENT_STRING": int,
        "CURRENT_ENUM": str,
        "CURRENT_CLASS": str,
        "RENAMED_CLASS": type[_FirstChoice],
        "CURRENT_LIST": str,
        "CURRENT_ENUM_LIST": str,
        "EMPTY_LIST": int,
        "NONE_LIST": int,
        "FALLBACK": int,
        "ANNOTATED_NONE": _Color | int | bool,
        "ANNOTATED_BOOL": bool | int,
        "ANNOTATED_INT": int | float,
        "ANNOTATED_FLOAT": float | str,
        "ANNOTATED_STRING": str | _FirstChoice,
        "ANNOTATED_CLASS": type[_FirstChoice] | None,
    }
    return module


class ConfigurationValueParsingTests(unittest.TestCase):
    def test_current_values_take_precedence_and_lists_use_first_non_none(self) -> None:
        module = _configuration_module()
        cases = (
            ("CURRENT_BOOL", "off", False),
            ("CURRENT_INT", "12", 12),
            ("CURRENT_FLOAT", "1.5", 1.5),
            ("CURRENT_STRING", "12", "12"),
            ("CURRENT_ENUM", "package.BLUE", _Color.BLUE),
            ("CURRENT_CLASS", "package._SecondChoice", _SecondChoice),
            ("CURRENT_LIST", "9", 9),
            ("CURRENT_ENUM_LIST", "BLUE", _Color.BLUE),
            ("EMPTY_LIST", "9", "9"),
            ("NONE_LIST", "9", "9"),
            ("FALLBACK", "9", "9"),
        )

        for key, raw_value, expected in cases:
            with self.subTest(key=key):
                parsed = parse_config_value(module, key, raw_value)
                if isinstance(expected, type) or isinstance(expected, Enum):
                    self.assertIs(parsed, expected)
                else:
                    self.assertEqual(parsed, expected)

    def test_none_current_values_follow_annotation_precedence(self) -> None:
        module = _configuration_module()
        cases = (
            ("ANNOTATED_NONE", "BLUE", _Color.BLUE),
            ("ANNOTATED_BOOL", "1", True),
            ("ANNOTATED_INT", "3", 3),
            ("ANNOTATED_FLOAT", "3", 3.0),
            ("ANNOTATED_STRING", "package._SecondChoice", "package._SecondChoice"),
            ("ANNOTATED_CLASS", "package._SecondChoice", _SecondChoice),
            ("UNANNOTATED", "value", "value"),
        )

        for key, raw_value, expected in cases:
            with self.subTest(key=key):
                parsed = parse_config_value(module, key, raw_value)
                if isinstance(expected, type) or isinstance(expected, Enum):
                    self.assertIs(parsed, expected)
                else:
                    self.assertEqual(parsed, expected)

    def test_null_literals_win_before_current_and_annotation_dispatch(self) -> None:
        module = _configuration_module()

        for key in (
            "CURRENT_BOOL",
            "CURRENT_LIST",
            "CURRENT_CLASS",
            "ANNOTATED_NONE",
            "UNANNOTATED",
        ):
            for raw_value in ("none", "NULL"):
                with self.subTest(key=key, raw_value=raw_value):
                    self.assertIsNone(parse_config_value(module, key, raw_value))

    def test_public_parse_failures_retain_exact_types_and_messages(self) -> None:
        module = _configuration_module()

        with self.assertRaisesRegex(
            ConfigValueError,
            "^expected a boolean value, got 'perhaps'$",
        ):
            parse_config_value(module, "CURRENT_BOOL", "perhaps")
        with self.assertRaisesRegex(
            ConfigValueError,
            "^unknown _Color value 'GREEN'. Choices: RED, BLUE$",
        ):
            parse_config_value(module, "CURRENT_ENUM", "GREEN")
        with self.assertRaisesRegex(
            ConfigValueError,
            "^unknown config class 'package.MissingChoice'$",
        ):
            parse_config_value(module, "CURRENT_CLASS", "package.MissingChoice")
        with self.assertRaisesRegex(ValueError, "invalid literal for int"):
            parse_config_value(module, "CURRENT_INT", "not-an-integer")

    def test_package_local_symbol_aliases_parse_but_serialize_canonically(self) -> None:
        module = _configuration_module()

        parsed_enum = parse_config_value(
            module,
            "RENAMED_ENUM",
            "legacy.HISTORICAL",
        )
        parsed_class = parse_config_value(
            module,
            "RENAMED_CLASS",
            "legacy.HistoricalChoice",
        )

        self.assertIs(parsed_enum, _RenamedOption.CURRENT)
        self.assertIs(parsed_class, _FirstChoice)
        self.assertEqual(serialize_config_value(parsed_enum), "CURRENT")
        self.assertEqual(serialize_config_value(parsed_class), "_FirstChoice")

    def test_class_lookup_never_imports_a_supplied_module_path(self) -> None:
        module = _configuration_module()

        with self.assertRaisesRegex(
            ConfigValueError,
            r"^unknown config class 'pathlib\.Path'$",
        ):
            parse_config_value(module, "RENAMED_CLASS", "pathlib.Path")


if __name__ == "__main__":
    unittest.main()
