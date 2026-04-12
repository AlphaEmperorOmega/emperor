import types

from typing import get_args, get_origin

from emperor.base.utils import ConfigBase


class ValidatorBase:
    OPTIONAL_FIELDS: set[str] = set()

    @classmethod
    def validate_required_fields(cls, cfg: ConfigBase) -> None:
        for field_name in cfg.__dataclass_fields__:
            if field_name in cls.OPTIONAL_FIELDS:
                continue
            if getattr(cfg, field_name) is None:
                raise ValueError(
                    f"{field_name} is required for {cfg.__class__.__name__}, received None"
                )

    @classmethod
    def validate_field_types(cls, cfg: ConfigBase) -> None:
        for field_name, field_info in cfg.__dataclass_fields__.items():
            if field_name in cls.OPTIONAL_FIELDS:
                continue
            value = getattr(cfg, field_name)
            if value is None:
                continue
            expected = cls.__extract_type(field_info.type)
            if expected and not isinstance(value, expected):
                raise TypeError(
                    f"{field_name} must be {expected.__name__} for "
                    f"{cfg.__class__.__name__}, got {type(value).__name__}"
                )

    @staticmethod
    def __extract_type(annotation) -> type | None:
        if isinstance(annotation, type):
            return annotation
        if get_origin(annotation) is types.UnionType:
            args = [a for a in get_args(annotation) if a is not type(None)]
            return args[0] if args else None
        return None

    @staticmethod
    def validate_dimensions(**dims: int) -> None:
        for name, value in dims.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than 0, received {value}")
