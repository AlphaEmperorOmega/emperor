from __future__ import annotations

import pytest

from model_runtime.inspection import InspectionError
from model_runtime.inspection.preflight import preflight_inspection_configuration
from models.catalog import model_package

_NEURON_PACKAGES = (
    "neuron/linear",
    "neuron/linear_adaptive",
    "neuron/expert_linear",
    "neuron/expert_linear_adaptive",
)


def _selected_package(catalog_key: str):
    package = model_package(catalog_key)
    assert package is not None
    return package


@pytest.mark.parametrize("catalog_key", _NEURON_PACKAGES)
def test_neuron_packages_bound_all_multiplicative_construction_axes(
    catalog_key: str,
) -> None:
    package = _selected_package(catalog_key)
    limits = package.inspection_construction_limits

    assert limits.maximum_for("CLUSTER_BEAM_WIDTH") == 64
    assert {limit.label for limit in limits.field_product_limits} == {
        "initial neuron count",
        "neuron capacity",
    }


@pytest.mark.parametrize("catalog_key", _NEURON_PACKAGES)
@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        (
            {
                "cluster_initial_x_axis_total_neurons": 17,
                "cluster_initial_y_axis_total_neurons": 17,
                "cluster_initial_z_axis_total_neurons": 17,
            },
            "initial neuron count",
        ),
        (
            {
                "cluster_x_axis_total_neurons": 17,
                "cluster_y_axis_total_neurons": 17,
                "cluster_z_axis_total_neurons": 17,
            },
            "neuron capacity",
        ),
        ({"cluster_beam_width": 65}, "CLUSTER_BEAM_WIDTH"),
    ),
)
def test_neuron_product_and_beam_limits_reject_before_construction(
    catalog_key: str,
    overrides: dict[str, int],
    message: str,
) -> None:
    package = _selected_package(catalog_key)
    preset = package.resolve_preset("baseline")

    with pytest.raises(InspectionError, match=message):
        preflight_inspection_configuration(package, overrides, preset)


@pytest.mark.parametrize("catalog_key", _NEURON_PACKAGES)
def test_initial_neuron_count_scales_the_dense_parameter_estimate(
    catalog_key: str,
) -> None:
    package = _selected_package(catalog_key)
    preset = package.resolve_preset("baseline")

    one_neuron_estimate = preflight_inspection_configuration(
        package,
        {
            "cluster_initial_x_axis_total_neurons": 1,
            "cluster_initial_y_axis_total_neurons": 1,
            "cluster_initial_z_axis_total_neurons": 1,
        },
        preset,
    )
    two_neuron_estimate = preflight_inspection_configuration(
        package,
        {
            "cluster_initial_x_axis_total_neurons": 2,
            "cluster_initial_y_axis_total_neurons": 1,
            "cluster_initial_z_axis_total_neurons": 1,
        },
        preset,
    )
    default_estimate = preflight_inspection_configuration(package, {}, preset)

    one_neuron_dense_increment = two_neuron_estimate - one_neuron_estimate
    assert one_neuron_dense_increment > 0
    assert default_estimate - one_neuron_estimate == one_neuron_dense_increment * 8
