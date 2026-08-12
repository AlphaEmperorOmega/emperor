from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import pytest

from model_runtime.inspection import InspectionError
from model_runtime.inspection.preflight import preflight_inspection_configuration
from model_runtime.packages import (
    InspectionConstructionLimits,
    InspectionFieldProductLimit,
)
from models.catalog import model_package


def test_field_maximums_are_validated_normalized_and_immutable() -> None:
    limits = InspectionConstructionLimits(field_maximums={" hidden_dim ": 12})

    assert limits.maximum_for("HIDDEN_DIM") == 12
    with pytest.raises(TypeError):
        mutable_maximums = cast(dict[str, int | float], limits.field_maximums)
        mutable_maximums["HIDDEN_DIM"] = 13

    for invalid in (True, 0, -1, float("inf"), float("nan"), "12"):
        with pytest.raises((TypeError, ValueError)):
            InspectionConstructionLimits(
                field_maximums=cast(
                    Mapping[str, int | float],
                    {"HIDDEN_DIM": invalid},
                ),
            )


def test_field_maximum_suffix_policy_and_precedence_are_stable() -> None:
    limits = InspectionConstructionLimits(
        maximum_hidden_dimension=101,
        maximum_io_dimension=102,
        maximum_sequence_length=103,
        maximum_layer_count=104,
        maximum_expert_count=105,
        maximum_attention_head_count=106,
        maximum_recurrent_steps=107,
        field_maximums={
            "override_hidden_dim": 201,
            "trainer_max_steps": 202,
            "custom_float": 2.5,
        },
    )
    cases = {
        "HIDDEN_DIM": 101,
        "STACK_HIDDEN_DIM": 101,
        "model_dim": 101,
        "TOKEN_EMBEDDING_DIM": 101,
        "INPUT_DIM": 102,
        "HEAD_OUTPUT_DIM": 102,
        "VOCAB_SIZE": 102,
        "SOURCE_SEQUENCE_LENGTH": 103,
        "NUM_LAYERS": 104,
        "ENCODER_NUM_EXPERTS": 105,
        "ATTN_NUM_HEADS": 106,
        "MAX_STEPS": 107,
        "RECURRENT_MAX_STEPS": 107,
        "NUM_INNER_STEPS": 107,
        "TRAINER_NUM_INNER_STEPS": 107,
        "OVERRIDE_HIDDEN_DIM": 201,
        "trainer_max_steps": 202,
        "CUSTOM_FLOAT": 2.5,
        "TRAINER_OTHER_MAX_STEPS": None,
        "NOT_TRAINER_MAX_STEPS": None,
        "MAX_STEPS_TRAILING": None,
        "HIDDEN_DIM_TRAILING": None,
        " override_hidden_dim ": None,
        "UNKNOWN": None,
    }

    for key, expected in cases.items():
        assert limits.maximum_for(key) == expected


def test_product_limit_values_are_normalized_and_validated() -> None:
    limit = InspectionFieldProductLimit(
        label="  capacity  ",
        factors=((" hidden_dim ",),),
        maximum=12,
    )

    assert limit.label == "capacity"
    assert limit.factors == (("HIDDEN_DIM",),)

    with pytest.raises(ValueError, match="labels"):
        InspectionFieldProductLimit(label=" ", factors=(("A",),), maximum=1)
    with pytest.raises(ValueError, match="factors"):
        InspectionFieldProductLimit(label="x", factors=(), maximum=1)
    with pytest.raises(ValueError, match="maximums"):
        InspectionFieldProductLimit(label="x", factors=(("A",),), maximum=0)


@pytest.mark.parametrize("declaration", ("maximum", "product"))
def test_unknown_limit_fields_fail_closed(declaration: str) -> None:
    package = model_package("linears/linear")
    assert package is not None
    if declaration == "maximum":
        limits = InspectionConstructionLimits(field_maximums={"UNKNOWN": 1})
    else:
        limits = InspectionConstructionLimits(
            field_product_limits=(
                InspectionFieldProductLimit(
                    label="unknown",
                    factors=(("UNKNOWN",),),
                    maximum=1,
                ),
            )
        )
    invalid_package = replace(package, inspection_construction_limits=limits)

    with pytest.raises(InspectionError, match="unknown Runtime Defaults field"):
        preflight_inspection_configuration(
            invalid_package,
            {},
            invalid_package.default_preset,
        )
