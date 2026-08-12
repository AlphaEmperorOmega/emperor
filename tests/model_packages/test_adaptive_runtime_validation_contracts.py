from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest

from models.linears.linear_adaptive.runtime_defaults import (
    runtime_from_flat as linears_runtime_from_flat,
)
from models.neuron.linear_adaptive._hidden.runtime_defaults import (
    runtime_from_flat as neuron_runtime_from_flat,
)

_RUNTIME_CASES = (
    (linears_runtime_from_flat, "models.linears.linear_adaptive"),
    (neuron_runtime_from_flat, "models.neuron.linear_adaptive._hidden"),
)


@pytest.mark.parametrize(
    ("runtime_from_flat", "package_name"),
    _RUNTIME_CASES,
    ids=("linears", "neuron-hidden"),
)
@pytest.mark.parametrize(
    ("overrides", "first_error"),
    (
        (
            {"batch_size": 0, "learning_rate": 0.0},
            "runtime key 'batch_size' must be positive",
        ),
        (
            {"gate_stack_hidden_dim": 0, "halting_dropout": 2.0},
            "runtime key 'gate_stack_hidden_dim' must be positive",
        ),
        (
            {"halting_dropout": 2.0, "weight_decay_rate": -1.0},
            "runtime key 'halting_dropout' must be between 0 and 1",
        ),
        (
            {
                "weight_option_flag": True,
                "weight_option": None,
                "bias_option_flag": True,
                "bias_option": None,
            },
            "runtime key 'weight_option' must be set when 'weight_option_flag' is True",
        ),
    ),
    ids=(
        "positive-order",
        "patterned-before-probability",
        "probability-before-decay",
        "enabled-order",
    ),
)
def test_validation_preserves_exact_first_error_order(
    runtime_from_flat: Callable[[Mapping[str, object]], object],
    package_name: str,
    overrides: Mapping[str, object],
    first_error: str,
) -> None:
    with pytest.raises(ValueError) as error:
        runtime_from_flat(overrides)

    assert str(error.value) == f"{package_name}: {first_error}"
