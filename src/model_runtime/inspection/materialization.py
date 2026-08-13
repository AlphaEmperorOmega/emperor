from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from emperor.experiments import ExperimentTask
from model_runtime.inspection.errors import InspectionError, model_package_failure
from model_runtime.inspection.overrides import validated_overrides_for_materialization
from model_runtime.inspection.preflight import preflight_inspection_configuration
from model_runtime.inspection.records import (
    InspectionRequest,
    InspectionResult,
    ModelGraph,
    ParsedOverrides,
)
from model_runtime.packages import ModelPackage


@dataclass(frozen=True, slots=True)
class MaterializedConfiguration:
    package: ModelPackage
    request: InspectionRequest
    preset: Any
    experiment_task: ExperimentTask
    dataset: type
    overrides: ParsedOverrides
    configuration: Any


@dataclass(frozen=True, slots=True)
class MaterializedInspection:
    prepared: MaterializedConfiguration
    model: Any

    def result(self, graph: ModelGraph) -> InspectionResult:
        root = graph.nodes[0] if graph.nodes else None
        return InspectionResult(
            identity=self.prepared.package.identity,
            preset=self.prepared.package.preset_name(self.prepared.preset),
            parameter_count=root.parameter_count if root is not None else 0,
            parameter_size_bytes=(root.parameter_size_bytes if root is not None else 0),
            nodes=graph.nodes,
            edges=graph.edges,
        )


def _require_model_package(value: object) -> ModelPackage:
    if not isinstance(value, ModelPackage):
        raise TypeError("Inspection requires a selected ModelPackage.")
    return value


def materialize_configuration(
    package: ModelPackage,
    request: InspectionRequest,
) -> MaterializedConfiguration:
    package = _require_model_package(package)
    try:
        preset = package.resolve_preset(request.preset)
    except ValueError as exc:
        raise InspectionError(str(exc)) from exc
    except Exception as exc:
        raise model_package_failure(package.catalog_key, exc) from exc

    try:
        parsed_overrides = validated_overrides_for_materialization(
            package,
            request.overrides,
            request.preset,
        )
        preflight_inspection_configuration(
            package,
            parsed_overrides.values,
            preset,
            memory_limit_bytes=request.memory_limit_bytes,
        )
        experiment_task = package.resolve_experiment_task(request.experiment_task)
        dataset = package.resolve_dataset(request.dataset, experiment_task)
        configuration = package.build_configuration(
            preset,
            dataset,
            config_overrides=dict(parsed_overrides.values),
        )
        preflight_inspection_configuration(
            package,
            parsed_overrides.values,
            preset,
            memory_limit_bytes=request.memory_limit_bytes,
            effective_configuration=configuration,
        )
    except InspectionError:
        raise
    except (ImportError, ModuleNotFoundError) as exc:
        raise model_package_failure(package.catalog_key, exc) from exc
    except Exception as exc:
        raise InspectionError(
            f"Failed to build preset '{request.preset}' for model "
            f"'{package.catalog_key}': {exc}"
        ) from exc

    return MaterializedConfiguration(
        package=package,
        request=request,
        preset=preset,
        experiment_task=experiment_task,
        dataset=dataset,
        overrides=parsed_overrides,
        configuration=configuration,
    )


def materialize_inspection(
    package: ModelPackage,
    request: InspectionRequest,
) -> MaterializedInspection:
    prepared = materialize_configuration(package, request)
    try:
        model = package.build_model(prepared.configuration)
    except Exception as exc:
        raise InspectionError(
            f"Failed to instantiate model '{package.catalog_key}' preset "
            f"'{request.preset}': {exc}"
        ) from exc
    return MaterializedInspection(prepared=prepared, model=model)


__all__ = [
    "MaterializedConfiguration",
    "MaterializedInspection",
    "materialize_configuration",
    "materialize_inspection",
]
