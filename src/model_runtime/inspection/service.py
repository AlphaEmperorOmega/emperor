from __future__ import annotations

from model_runtime.inspection.errors import InspectionError, _model_package_failure
from model_runtime.inspection.overrides import parse_overrides, reject_locked_overrides
from model_runtime.inspection.preflight import preflight_inspection_configuration
from model_runtime.inspection.records import (
    InspectionRequest,
    InspectionResult,
    ParsedOverrides,
)
from model_runtime.packages import ModelPackage


def inspect_model(
    package: ModelPackage,
    request: InspectionRequest,
) -> InspectionResult:
    from model_runtime.inspection.model_graph import inspect_model_graph

    if not isinstance(package, ModelPackage):
        raise TypeError("Inspection requires a selected ModelPackage.")
    preset, _configuration, model = _instantiate_inspection_model(package, request)

    graph = inspect_model_graph(model)
    return _inspection_result(package, preset, graph)


def _instantiate_inspection_model(
    package: ModelPackage,
    request: InspectionRequest,
):
    preset, configuration = _build_configuration(package, request)
    try:
        model = package.build_model(configuration)
    except Exception as exc:
        raise InspectionError(
            f"Failed to instantiate model '{package.catalog_key}' preset "
            f"'{request.preset}': {exc}"
        ) from exc
    return preset, configuration, model


def _inspection_result(package: ModelPackage, preset, graph) -> InspectionResult:
    parameter_count = graph.nodes[0].parameter_count if graph.nodes else 0
    parameter_size_bytes = graph.nodes[0].parameter_size_bytes if graph.nodes else 0
    return InspectionResult(
        identity=package.identity,
        preset=package.preset_name(preset),
        parameter_count=parameter_count,
        parameter_size_bytes=parameter_size_bytes,
        nodes=graph.nodes,
        edges=graph.edges,
    )


def _build_configuration(
    package: ModelPackage,
    request: InspectionRequest,
):
    try:
        preset = package.resolve_preset(request.preset)
    except ValueError as exc:
        raise InspectionError(str(exc)) from exc
    except Exception as exc:
        raise _model_package_failure(package.catalog_key, exc) from exc
    try:
        if isinstance(request.overrides, ParsedOverrides):
            parsed_overrides = request.overrides
            reject_locked_overrides(
                package,
                request.preset,
                parsed_overrides.values,
            )
        else:
            parsed_overrides = parse_overrides(
                package,
                request.overrides,
                preset=request.preset,
            )
        preflight_inspection_configuration(
            package,
            parsed_overrides.values,
            preset,
        )
        try:
            dataset = package.resolve_dataset(
                request.dataset,
                request.experiment_task,
            )
        except ValueError:
            raise
        except Exception as exc:
            raise _model_package_failure(package.catalog_key, exc) from exc
        configuration = package.build_configuration(
            preset,
            dataset,
            config_overrides=dict(parsed_overrides.values),
        )
    except InspectionError:
        raise
    except (ImportError, ModuleNotFoundError) as exc:
        raise _model_package_failure(package.catalog_key, exc) from exc
    except Exception as exc:
        raise InspectionError(
            f"Failed to build preset '{request.preset}' for model "
            f"'{package.catalog_key}': {exc}"
        ) from exc
    return preset, configuration


def validate_configuration(
    package: ModelPackage,
    request: InspectionRequest,
) -> None:
    if not isinstance(package, ModelPackage):
        raise TypeError("Inspection requires a selected ModelPackage.")
    _build_configuration(package, request)


__all__ = ["inspect_model", "validate_configuration"]
