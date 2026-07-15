import unittest
from dataclasses import dataclass

from emperor.base.config import ConfigBase
from emperor.base.validator import ValidatorBase


@dataclass
class ValidatorBaseTestConfig(ConfigBase):
    required_value: int | None = None
    optional_value: int | None = None


class ValidatorWithOptionalField(ValidatorBase):
    OPTIONAL_FIELDS = {"optional_value"}


class TestValidatorBase(unittest.TestCase):
    def test_subclass_optional_fields_drive_required_field_validation(self):
        cfg = ValidatorBaseTestConfig(required_value=1, optional_value=None)

        self.assertIsNone(ValidatorWithOptionalField.validate_required_fields(cfg))

    def test_required_field_validation_rejects_none(self):
        cfg = ValidatorBaseTestConfig(required_value=None, optional_value=1)

        with self.assertRaisesRegex(
            ValueError,
            "required_value is required for ValidatorBaseTestConfig, received None",
        ):
            ValidatorWithOptionalField.validate_required_fields(cfg)

    def test_field_type_validation_dispatches_through_subclass(self):
        cfg = ValidatorBaseTestConfig(required_value="invalid", optional_value=None)

        with self.assertRaisesRegex(
            TypeError,
            "required_value must be int for ValidatorBaseTestConfig, got str",
        ):
            ValidatorWithOptionalField.validate_field_types(cfg)

    def test_dimension_validation_rejects_non_positive_values(self):
        with self.assertRaisesRegex(
            ValueError,
            "width must be greater than 0, received 0",
        ):
            ValidatorBase.validate_dimensions(width=0)
