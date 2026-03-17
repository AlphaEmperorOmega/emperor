import types
from dataclasses import dataclass, fields
from typing import get_args

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.sampler.model import SamplerModel
    from emperor.sampler.utils.samplers import (
        SamplerBase,
        SamplerSparse,
        SamplerTopk,
        SamplerFull,
    )


_SKIP_FIELDS = {"override_config"}


def _extract_field_type(field_type) -> tuple[type | str, bool]:
    if isinstance(field_type, types.UnionType):
        args = get_args(field_type)
        base = next(a for a in args if a is not type(None))
        optional = type(None) in args
        return base, optional
    if isinstance(field_type, str):
        parts = field_type.split(" | ")
        optional = "None" in parts
        return parts[0].strip(), optional
    return field_type, False


def _build_fields_from_dataclass(config_cls: type) -> dict[str, dict]:
    result = {}
    for f in fields(config_cls):
        if f.name in _SKIP_FIELDS:
            continue
        base_type, optional = _extract_field_type(f.type)
        result[f.name] = {"type": base_type, "optional": optional}
    return result


class _BaseValidator:
    _FIELDS: dict = {}
    _STRING_TYPES: dict = {}

    def __init__(self, model):
        self.model = model
        self._resolve_string_types()
        self.validate()

    def _resolve_string_types(self) -> None:
        pass

    def validate(self) -> None:
        for name, rules in self._FIELDS.items():
            val = getattr(self.model, name, None)

            if val is None:
                if rules.get("optional"):
                    continue
                raise ValueError(
                    f"Configuration Error: '{name}' is required for "
                    f"{self.model.__class__.__name__}."
                )

            expected = rules["type"]
            expected_type = (
                self._STRING_TYPES.get(expected, expected)
                if isinstance(expected, str)
                else expected
            )
            if not isinstance(val, expected_type):
                raise TypeError(
                    f"Type Error: '{name}' on {self.model.__class__.__name__} "
                    f"expected {expected_type.__name__}, got "
                    f"{type(val).__name__} (value={val!r})."
                )

            validator = rules.get("validate")
            if validator is not None:
                result = validator(val)
                if result is not True and result is not None:
                    raise ValueError(
                        f"Configuration Error: '{name}' {result} (value={val!r})."
                    )


class SamplerModelValidator(_BaseValidator):
    def __init__(self, model: "SamplerModel"):
        from emperor.sampler.utils.samplers import SamplerConfig

        self._FIELDS = _build_fields_from_dataclass(SamplerConfig)
        self._FIELDS["num_experts"] = {
            "type": int,
            "validate": lambda v: v > 0 or "must be > 0",
        }
        super().__init__(model)

    def _resolve_string_types(self) -> None:
        from emperor.sampler.utils.samplers import SamplerConfig
        from emperor.sampler.utils.routers import RouterConfig

        self._STRING_TYPES = {
            "SamplerConfig": SamplerConfig,
            "RouterConfig": RouterConfig,
        }


class SamplerBaseValidator(_BaseValidator):
    _VALIDATORS = {
        "top_k": lambda v: v >= 0 or "must be non-negative",
        "threshold": lambda v: 0.0 <= v <= 1.0
        or "must be between 0.0 and 1.0 (inclusive)",
    }

    def __init__(self, model: "SamplerBase"):
        from emperor.sampler.utils.samplers import SamplerConfig

        self._FIELDS = _build_fields_from_dataclass(SamplerConfig)
        for name, validator in self._VALIDATORS.items():
            if name in self._FIELDS:
                self._FIELDS[name]["validate"] = validator
        super().__init__(model)
        self.__validate_cross_field_constraints()

    def __validate_cross_field_constraints(self) -> None:
        if self.model.num_topk_samples > self.model.top_k:
            raise ValueError(
                f"Configuration Error: 'num_topk_samples' cannot exceed 'top_k'. "
                f"Got num_topk_samples={self.model.num_topk_samples}, top_k={self.model.top_k}."
            )


class SamplerSparseValidator(_BaseValidator):
    _FIELDS = {
        "top_k": {
            "type": int,
            "validate": lambda v: v == 1 or "must be 1 when using SamplerSparse",
        },
        "normalize_probabilities_flag": {
            "type": bool,
            "validate": lambda v: v is False
            or "must be False when using SamplerSparse",
        },
        "num_topk_samples": {
            "type": int,
            "validate": lambda v: v == 0 or "must be 0 when using SamplerSparse",
        },
        "mutual_information_loss_weight": {
            "type": float,
            "validate": lambda v: v == 0.0 or "must be 0.0 when using SamplerSparse",
        },
    }


class SamplerTopkValidator(_BaseValidator):
    def __init__(self, model: "SamplerTopk"):
        super().__init__(model)
        self.__validate_cross_field_constraints()

    def __validate_cross_field_constraints(self) -> None:
        if not (0 < self.model.top_k < self.model.num_experts):
            raise ValueError(
                f"Configuration Error: 'top_k' must be greater than 0 and less than 'num_experts'. "
                f"Got top_k={self.model.top_k}, num_experts={self.model.num_experts}."
            )


class SamplerFullValidator(_BaseValidator):
    _FIELDS = {
        "num_topk_samples": {
            "type": int,
            "validate": lambda v: v == 0 or "must be 0 when using SamplerFull",
        },
        "coefficient_of_variation_loss_weight": {
            "type": float,
            "validate": lambda v: v == 0.0 or "must be 0.0 when using SamplerFull",
        },
        "switch_loss_weight": {
            "type": float,
            "validate": lambda v: v == 0.0 or "must be 0.0 when using SamplerFull",
        },
        "zero_centred_loss_weight": {
            "type": float,
            "validate": lambda v: v == 0.0 or "must be 0.0 when using SamplerFull",
        },
        "mutual_information_loss_weight": {
            "type": float,
            "validate": lambda v: v == 0.0 or "must be 0.0 when using SamplerFull",
        },
    }

    def __init__(self, model: "SamplerFull"):
        super().__init__(model)
        self.__validate_cross_field_constraints()

    def __validate_cross_field_constraints(self) -> None:
        if self.model.top_k != self.model.num_experts:
            raise ValueError(
                f"Configuration Error: 'top_k' must be equal to 'num_experts' when using SamplerFull. "
                f"Got top_k={self.model.top_k}, num_experts={self.model.num_experts}."
            )
