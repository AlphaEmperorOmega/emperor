from __future__ import annotations

from inspect import Parameter, signature
from unittest.mock import patch

import pytest

from emperor.config import ModelConfig
from model_runtime.inspection import InspectionError, InspectionRequest, inspect_model
from model_runtime.inspection.preflight import preflight_inspection_configuration
from model_runtime.packages import ModelPackage
from models.catalog import discover_model_packages, model_package

_MAXIMUM_DEFAULT_ESTIMATE_TO_ACTUAL_RATIO = 200


@pytest.mark.parametrize(
    "package",
    discover_model_packages(),
    ids=lambda package: package.catalog_key,
)
def test_default_parameter_estimate_is_safe_and_bounded(package) -> None:
    preset = package.default_preset
    estimate = preflight_inspection_configuration(package, {}, preset)
    configuration = package.build_configuration(preset)
    model = package.build_model(configuration)
    actual_parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert actual_parameter_count <= estimate
    assert (
        estimate <= actual_parameter_count * _MAXIMUM_DEFAULT_ESTIMATE_TO_ACTUAL_RATIO
    )
    assert estimate <= package.inspection_construction_limits.maximum_parameter_estimate


def _dataset(package: ModelPackage, name: str) -> type:
    return next(
        dataset
        for options in package.dataset_metadata.values()
        for dataset in options
        if dataset.__name__ == name
    )


def test_effective_dataset_dimensions_tighten_linear_inspection_preflight() -> None:
    package = model_package("linears/linear")
    assert package is not None
    dataset = _dataset(package, "Cifar100")
    configuration = package.build_configuration(package.default_preset, dataset)

    assert preflight_inspection_configuration(
        package,
        {},
        package.default_preset,
    ) == 68_416
    assert preflight_inspection_configuration(
        package,
        {},
        package.default_preset,
        effective_configuration=configuration,
    ) == 144_512


def test_effective_configuration_is_an_additive_keyword_only_preflight_input() -> None:
    parameter = signature(preflight_inspection_configuration).parameters[
        "effective_configuration"
    ]

    assert parameter.kind is Parameter.KEYWORD_ONLY
    assert parameter.default is None


def test_effective_root_dimensions_do_not_depend_on_runtime_default_names() -> None:
    package = model_package("transformer/linear")
    assert package is not None
    configuration = ModelConfig(
        input_dim=2_000_000,
        hidden_dim=20_000,
        output_dim=2_000_000,
        sequence_length=64,
    )

    with pytest.raises(
        InspectionError,
        match="field 'INPUT_DIM' value 2000000 exceeds",
    ):
        preflight_inspection_configuration(
            package,
            {},
            package.default_preset,
            effective_configuration=configuration,
        )


@pytest.mark.parametrize(
    ("package_key", "dataset_name", "memory_limit_bytes", "expected_estimate"),
    (
        ("linears/linear", "Cifar100", 1_120_000, 144_512),
        ("gpt/linear", "WikiText103", 160_000_000, 69_429_376),
    ),
)
def test_dataset_effective_preflight_rejects_before_model_allocation(
    package_key: str,
    dataset_name: str,
    memory_limit_bytes: int,
    expected_estimate: int,
) -> None:
    package = model_package(package_key)
    assert package is not None

    with patch.object(
        ModelPackage,
        "build_model",
        side_effect=AssertionError("model constructor was observed"),
    ) as build_model:
        with pytest.raises(
            InspectionError,
            match=(
                rf"estimated parameter count {expected_estimate} exceeds the "
                r"memory-derived maximum"
            ),
        ):
            inspect_model(
                package,
                InspectionRequest(
                    preset="baseline",
                    dataset=dataset_name,
                    memory_limit_bytes=memory_limit_bytes,
                ),
            )

    build_model.assert_not_called()


@pytest.mark.parametrize(
    "package_key",
    ("vit/expert_linear", "vit/expert_linear_adaptive"),
)
def test_effective_configuration_cannot_lower_existing_admission_estimate(
    package_key: str,
) -> None:
    package = model_package(package_key)
    assert package is not None
    raw_estimate = preflight_inspection_configuration(
        package,
        {},
        package.default_preset,
    )
    configuration = package.build_configuration(package.default_preset)
    effective_estimate = preflight_inspection_configuration(
        package,
        {},
        package.default_preset,
        effective_configuration=configuration,
    )
    model = package.build_model(configuration)
    actual_parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert effective_estimate == raw_estimate
    assert actual_parameter_count <= effective_estimate


def test_every_catalog_dataset_configuration_passes_production_preflight() -> None:
    for package in discover_model_packages():
        for datasets in package.dataset_metadata.values():
            for dataset in datasets:
                configuration = package.build_configuration(
                    package.default_preset,
                    dataset,
                )
                estimate = preflight_inspection_configuration(
                    package,
                    {},
                    package.default_preset,
                    effective_configuration=configuration,
                    memory_limit_bytes=4 * 1024**3,
                )

                assert estimate > 0, (package.catalog_key, dataset.__name__)
