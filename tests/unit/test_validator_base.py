import unittest
from dataclasses import dataclass
from typing import Literal

from emperor._validation import ValidatorBase
from emperor.config import ConfigBase


@dataclass
class ValidatorBaseTestConfig(ConfigBase):
    required_value: int | None = None
    optional_value: int | None = None


@dataclass
class ValidatorTypeMatrixConfig(ConfigBase):
    direct_integer: int = 1
    optional_integer: int | None = None
    first_union_type: int | str | None = 2
    generic_values: list[int] | None = None


@dataclass
class ValidatorOptionalFirstConfig(ConfigBase):
    optional_first: int | None = None
    required_after_optional: int | None = 1


class ValidatorWithOptionalField(ValidatorBase):
    OPTIONAL_FIELDS = {"optional_value"}


class ValidatorWithOptionalFirst(ValidatorBase):
    OPTIONAL_FIELDS = {"optional_first"}


class TestValidatorBase(unittest.TestCase):
    def test_subclass_optional_fields_drive_required_field_validation(self):
        cfg = ValidatorBaseTestConfig(required_value=1, optional_value=None)

        self.assertIsNone(ValidatorWithOptionalField.validate_required_fields(cfg))

    def test_required_field_validation_rejects_none(self):
        cfg = ValidatorBaseTestConfig(required_value=None, optional_value=1)

        with self.assertRaises(ValueError) as error:
            ValidatorWithOptionalField.validate_required_fields(cfg)
        self.assertEqual(
            str(error.exception),
            "required_value is required for ValidatorBaseTestConfig, received None",
        )

    def test_optional_field_does_not_stop_later_required_field_validation(self):
        cfg = ValidatorOptionalFirstConfig(
            optional_first=None,
            required_after_optional=None,
        )

        with self.assertRaises(ValueError) as error:
            ValidatorWithOptionalFirst.validate_required_fields(cfg)
        self.assertEqual(
            str(error.exception),
            "required_after_optional is required for "
            "ValidatorOptionalFirstConfig, received None",
        )

    def test_field_type_validation_dispatches_through_subclass(self):
        cfg = ValidatorBaseTestConfig(required_value="invalid", optional_value=None)

        with self.assertRaisesRegex(
            TypeError,
            "required_value must be int for ValidatorBaseTestConfig, got str",
        ):
            ValidatorWithOptionalField.validate_field_types(cfg)

    def test_field_type_validation_accepts_direct_optional_and_generic_annotations(
        self,
    ):
        cfg = ValidatorTypeMatrixConfig(
            direct_integer=3,
            optional_integer=4,
            first_union_type=5,
            generic_values=[6, 7],
        )

        self.assertIsNone(ValidatorBase.validate_field_types(cfg))

    def test_field_type_validation_skips_declared_optional_field_values(self):
        cfg = ValidatorBaseTestConfig(
            required_value=3,
            optional_value="ignored by subclass contract",
        )

        self.assertIsNone(ValidatorWithOptionalField.validate_field_types(cfg))

    def test_optional_field_does_not_stop_later_type_validation(self):
        cfg = ValidatorOptionalFirstConfig(
            optional_first=None,
            required_after_optional="invalid",
        )

        with self.assertRaises(TypeError) as error:
            ValidatorWithOptionalFirst.validate_field_types(cfg)
        self.assertEqual(
            str(error.exception),
            "required_after_optional must be int for "
            "ValidatorOptionalFirstConfig, got str",
        )

    def test_union_validation_uses_first_concrete_runtime_type(self):
        cfg = ValidatorTypeMatrixConfig(first_union_type="invalid")

        with self.assertRaisesRegex(
            TypeError,
            "first_union_type must be int for ValidatorTypeMatrixConfig, got str",
        ):
            ValidatorBase.validate_field_types(cfg)

    def test_extract_type_handles_direct_union_and_unsupported_annotations(self):
        self.assertIs(ValidatorBase._extract_type(int), int)
        self.assertIs(ValidatorBase._extract_type(int | None), int)
        self.assertIs(ValidatorBase._extract_type(type(None) | int), int)
        self.assertIs(ValidatorBase._extract_type(int | str | None), int)
        self.assertIs(ValidatorBase._extract_type(list[int]), list)
        self.assertIsNone(ValidatorBase._extract_type("int"))
        self.assertIsNone(
            ValidatorBase._extract_type(type(None) | Literal["unsupported"])
        )

    def test_integer_fields_reject_booleans_before_runtime_use(self):
        cfg = ValidatorBaseTestConfig(required_value=True, optional_value=None)

        with self.assertRaisesRegex(
            TypeError,
            "required_value must be int for ValidatorBaseTestConfig, got bool",
        ):
            ValidatorWithOptionalField.validate_field_types(cfg)

    def test_dimension_validation_rejects_non_positive_values(self):
        with self.assertRaisesRegex(
            ValueError,
            "width must be greater than 0, received 0",
        ):
            ValidatorBase.validate_dimensions(width=0)

    def test_dimension_validation_accepts_multiple_positive_values(self):
        self.assertIsNone(ValidatorBase.validate_dimensions(width=1, height=3))
