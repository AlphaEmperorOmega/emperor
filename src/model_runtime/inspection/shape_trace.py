from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.metadata import packages_distributions
from pathlib import Path
from types import FrameType
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from model_runtime.inspection._shape_runtime import shape_trace_runtime
from model_runtime.inspection._tensor_walk import TensorWalkStart, tensor_observations
from model_runtime.inspection.capture_limits import InspectionCapture
from model_runtime.inspection.errors import InspectionError
from model_runtime.inspection.materialization import (
    MaterializedConfiguration,
    materialize_inspection,
)
from model_runtime.inspection.model_graph import inspect_model_graph
from model_runtime.inspection.records import (
    InspectionRequest,
    InspectionResult,
    ModelGraph,
)
from model_runtime.packages import ModelPackage
from model_runtime.task_behavior import SyntheticInputError, experiment_task_behavior

ShapeTraceDetail = Literal["outputs", "variables"]

_FIRST_PARTY_PACKAGE_NAMES = frozenset({"emperor", "model_runtime", "models"})


@dataclass(frozen=True, slots=True)
class TensorShape:
    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str


@dataclass(frozen=True, slots=True)
class ModuleShapeCall:
    inputs: tuple[TensorShape, ...]
    outputs: tuple[TensorShape, ...]


@dataclass(frozen=True, slots=True)
class ModuleShapeTrace:
    node_id: str
    calls: tuple[ModuleShapeCall, ...]


@dataclass(frozen=True, slots=True)
class TensorVariableTrace:
    order: int
    line: int | None
    tensors: tuple[TensorShape, ...]


@dataclass(frozen=True, slots=True)
class MethodShapeTrace:
    id: int
    parent_id: int | None
    order: int
    qualified_name: str
    module_path: str | None
    source_path: str
    first_line: int
    inputs: tuple[TensorShape, ...]
    variables: tuple[TensorVariableTrace, ...]
    outputs: tuple[TensorShape, ...]


@dataclass(frozen=True, slots=True)
class ModelShapeTrace:
    dataset: str
    experiment_task: str
    batch_size: int
    sample_inputs: tuple[TensorShape, ...]
    modules: tuple[ModuleShapeTrace, ...]
    methods: tuple[MethodShapeTrace, ...]


@dataclass(frozen=True, slots=True)
class _ObservedTensor:
    shape: TensorShape
    object_id: int


@dataclass(slots=True)
class _MutableModuleCall:
    inputs: tuple[TensorShape, ...]
    outputs: tuple[TensorShape, ...] = ()


@dataclass(slots=True)
class _MutableMethodTrace:
    id: int
    parent_id: int | None
    order: int
    qualified_name: str
    module_path: str | None
    source_path: str
    first_line: int
    inputs: tuple[TensorShape, ...]
    variables: list[TensorVariableTrace] = field(
        default_factory=list[TensorVariableTrace]
    )
    outputs: tuple[TensorShape, ...] = ()


@dataclass(slots=True)
class _FrameTrace:
    method: _MutableMethodTrace
    tensors: dict[str, _ObservedTensor]
    last_line: int | None = None


def _tensor_shape(name: str, tensor: Tensor) -> TensorShape:
    return TensorShape(
        name=name,
        shape=tuple(int(dimension) for dimension in tensor.shape),
        dtype=str(tensor.dtype).removeprefix("torch."),
        device=str(tensor.device),
    )


def _tensor_observations(
    value: object,
    name: str,
    *,
    capture: InspectionCapture | None = None,
    _seen: set[int] | None = None,
    _depth: int = 0,
) -> tuple[_ObservedTensor, ...]:
    return tensor_observations(
        value,
        name,
        capture,
        _observed_tensor,
        TensorWalkStart(seen=_seen, depth=_depth),
    )


def _observed_tensor(name: str, tensor: Tensor, object_id: int) -> _ObservedTensor:
    return _ObservedTensor(
        shape=_tensor_shape(name, tensor),
        object_id=object_id,
    )


def _tensor_shapes(
    value: object,
    name: str,
    *,
    capture: InspectionCapture | None = None,
) -> tuple[TensorShape, ...]:
    return tuple(
        observation.shape
        for observation in _tensor_observations(
            value,
            name,
            capture=capture,
        )
    )


def _local_tensor_observations(
    local_values: Mapping[str, object],
    *,
    capture: InspectionCapture | None = None,
) -> dict[str, _ObservedTensor]:
    tensors: dict[str, _ObservedTensor] = {}
    for name, value in local_values.items():
        if name in {"self", "cls"}:
            continue
        for tensor in _tensor_observations(value, name, capture=capture):
            tensors[tensor.shape.name] = tensor
    return tensors


def _local_tensor_shapes(
    local_values: Mapping[str, object],
    *,
    capture: InspectionCapture | None = None,
) -> dict[str, TensorShape]:
    return {
        name: observation.shape
        for name, observation in _local_tensor_observations(
            local_values,
            capture=capture,
        ).items()
    }


def _bound_input_shapes(
    module: nn.Module,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    *,
    capture: InspectionCapture,
) -> tuple[TensorShape, ...]:
    try:
        bound = inspect.signature(module.forward).bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        values = {f"input[{index}]": value for index, value in enumerate(args)}
        values.update({str(key): value for key, value in kwargs.items()})
    else:
        values = dict(bound.arguments)
    return tuple(_local_tensor_shapes(values, capture=capture).values())


@lru_cache(maxsize=1)
def _package_distributions() -> Mapping[str, tuple[str, ...]]:
    return {
        package_name: tuple(distributions)
        for package_name, distributions in packages_distributions().items()
    }


def _trace_module_names(model: nn.Module) -> frozenset[str]:
    registered_module_names = {type(module).__module__ for module in model.modules()}
    model_package_name = (type(model).__module__ or "").partition(".")[0]
    runtime_package_name = (__package__ or "").partition(".")[0]
    distributions = _package_distributions()
    owned_distributions = {
        *distributions.get(model_package_name, ()),
        *distributions.get(runtime_package_name, ()),
    }
    return frozenset(
        module_name
        for module_name in registered_module_names
        if module_name.partition(".")[0] in _FIRST_PARTY_PACKAGE_NAMES
        or module_name.partition(".")[0] == model_package_name
        or bool(
            owned_distributions
            & set(distributions.get(module_name.partition(".")[0], ()))
        )
    )


def _source_path(module_name: str, filename: str) -> str:
    source_path = Path(filename)
    module_path = Path(*module_name.split("."))
    if source_path.name == "__init__.py":
        return (module_path / source_path.name).as_posix()
    return module_path.with_suffix(source_path.suffix or ".py").as_posix()


class _TensorVariableTracer:
    def __init__(
        self,
        module_paths: Mapping[int, str],
        trace_module_names: frozenset[str],
        capture: InspectionCapture,
    ) -> None:
        self._module_paths = module_paths
        self._trace_module_names = trace_module_names
        self._capture = capture
        self._frames: dict[int, _FrameTrace] = {}
        self._methods: list[_MutableMethodTrace] = []
        self._next_method_id = 1
        self._next_order = 1

    def _order(self) -> int:
        order = self._next_order
        self._next_order += 1
        return order

    def _is_relevant(self, frame: FrameType) -> bool:
        module_name = str(frame.f_globals.get("__name__", ""))
        filename = frame.f_code.co_filename
        return not filename.startswith("<") and module_name in self._trace_module_names

    def _parent_id(self, frame: FrameType) -> int | None:
        parent = frame.f_back
        while parent is not None:
            parent_trace = self._frames.get(id(parent))
            if parent_trace is not None:
                return parent_trace.method.id
            parent = parent.f_back
        return None

    def _module_path(self, local_values: Mapping[str, object]) -> str | None:
        preferred_names = ("self", "model", "module")
        candidates = [local_values.get(name) for name in preferred_names]
        candidates.extend(local_values.values())
        for candidate in candidates:
            if isinstance(candidate, nn.Module):
                module_path = self._module_paths.get(id(candidate))
                if module_path is not None:
                    return module_path
        return None

    def _start(self, frame: FrameType) -> None:
        self._capture.increment(
            "methods",
            label="method",
            maximum=self._capture.limits.maximum_methods,
        )
        observations = _local_tensor_observations(
            frame.f_locals,
            capture=self._capture,
        )
        inputs = tuple(observation.shape for observation in observations.values())
        method = _MutableMethodTrace(
            id=self._next_method_id,
            parent_id=self._parent_id(frame),
            order=self._order(),
            qualified_name=frame.f_code.co_qualname,
            module_path=self._module_path(frame.f_locals),
            source_path=_source_path(
                str(frame.f_globals.get("__name__", "")),
                frame.f_code.co_filename,
            ),
            first_line=frame.f_code.co_firstlineno,
            inputs=inputs,
        )
        self._next_method_id += 1
        self._capture.reserve_output(method)
        self._methods.append(method)
        self._frames[id(frame)] = _FrameTrace(method=method, tensors=observations)

    def _capture_changes(self, frame: FrameType, trace: _FrameTrace) -> None:
        current = _local_tensor_observations(
            frame.f_locals,
            capture=self._capture,
        )
        changed = tuple(
            observation.shape
            for name, observation in current.items()
            if trace.tensors.get(name) != observation
        )
        if changed:
            self._capture.increment(
                "variable_events",
                label="variable event",
                maximum=self._capture.limits.maximum_variable_events,
            )
            variable = TensorVariableTrace(
                order=self._order(),
                line=trace.last_line,
                tensors=changed,
            )
            self._capture.reserve_output(variable)
            trace.method.variables.append(variable)
        trace.tensors = current

    def __call__(self, frame: FrameType, event: str, argument: object):
        if event == "call":
            if not self._is_relevant(frame):
                return None
            self._capture.increment(
                "trace_events",
                label="trace event",
                maximum=self._capture.limits.maximum_trace_events,
            )
            self._start(frame)
            return self

        trace = self._frames.get(id(frame))
        if trace is None:
            return None
        self._capture.increment(
            "trace_events",
            label="trace event",
            maximum=self._capture.limits.maximum_trace_events,
        )
        if event == "line":
            self._capture_changes(frame, trace)
            trace.last_line = frame.f_lineno
        elif event == "return":
            self._capture_changes(frame, trace)
            outputs = _tensor_shapes(
                argument,
                "return",
                capture=self._capture,
            )
            self._capture.reserve_output(outputs)
            trace.method.outputs = outputs
            self._frames.pop(id(frame), None)
        return self

    def results(self) -> tuple[MethodShapeTrace, ...]:
        return tuple(
            MethodShapeTrace(
                id=method.id,
                parent_id=method.parent_id,
                order=method.order,
                qualified_name=method.qualified_name,
                module_path=method.module_path,
                source_path=method.source_path,
                first_line=method.first_line,
                inputs=method.inputs,
                variables=tuple(method.variables),
                outputs=method.outputs,
            )
            for method in self._methods
        )


def _module_for_node(model: nn.Module, node_id: str, path: str) -> nn.Module | None:
    if node_id == "__root__":
        return model
    try:
        return model.get_submodule(path)
    except (AttributeError, KeyError):
        return None


class _ModuleCallRecorder:
    def __init__(
        self,
        model: nn.Module,
        graph: ModelGraph,
        capture: InspectionCapture,
    ) -> None:
        self._capture = capture
        self.calls: dict[str, list[_MutableModuleCall]] = {
            node.id: [] for node in graph.nodes
        }
        self._node_by_module_id: dict[int, str] = {}
        self.module_paths: dict[int, str] = {}
        self._modules: dict[int, nn.Module] = {}
        for node in graph.nodes:
            module = _module_for_node(model, node.id, node.path)
            if module is None or id(module) in self._modules:
                continue
            module_id = id(module)
            self._modules[module_id] = module
            self._node_by_module_id[module_id] = node.id
            self.module_paths[module_id] = node.path
        self._pending: dict[int, list[_MutableModuleCall]] = {
            module_id: [] for module_id in self._modules
        }

    def _before_forward(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        self._capture.increment(
            "module_calls",
            label="module call",
            maximum=self._capture.limits.maximum_module_calls,
        )
        call = _MutableModuleCall(
            inputs=_bound_input_shapes(
                module,
                args,
                kwargs,
                capture=self._capture,
            ),
        )
        self._capture.reserve_output(call.inputs)
        self._pending[id(module)].append(call)
        self.calls[self._node_by_module_id[id(module)]].append(call)

    def _after_forward(
        self,
        module: nn.Module,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        module_pending = self._pending[id(module)]
        if module_pending:
            outputs = _tensor_shapes(output, "output", capture=self._capture)
            self._capture.reserve_output(outputs)
            module_pending.pop().outputs = outputs

    def install(self, handles: list[RemovableHandle]) -> None:
        for module in self._modules.values():
            handles.append(
                module.register_forward_pre_hook(
                    self._before_forward,
                    with_kwargs=True,
                )
            )
            handles.append(
                module.register_forward_hook(
                    self._after_forward,
                    with_kwargs=True,
                )
            )


def _trace_module_calls(
    model: nn.Module,
    graph: ModelGraph,
    capture: InspectionCapture,
) -> tuple[
    dict[str, list[_MutableModuleCall]],
    dict[int, str],
    list[RemovableHandle],
]:
    recorder = _ModuleCallRecorder(model, graph, capture)
    handles: list[RemovableHandle] = []
    try:
        recorder.install(handles)
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    return recorder.calls, recorder.module_paths, handles


def _sample_inputs(
    materialized: MaterializedConfiguration,
) -> tuple[str, str, tuple[Tensor, ...]]:
    package = materialized.package
    task = materialized.experiment_task
    dataset = materialized.dataset
    configuration = materialized.configuration

    try:
        inputs = experiment_task_behavior(task).synthetic_inputs(
            dataset,
            configuration,
        )
    except SyntheticInputError as exc:
        raise InspectionError(str(exc)) from exc

    return dataset.__name__, package.task_name(task), inputs


def _require_model_package(value: object) -> ModelPackage:
    if not isinstance(value, ModelPackage):
        raise TypeError("Inspection requires a selected ModelPackage.")
    return value


@dataclass(frozen=True, slots=True)
class _PreparedShapeTrace:
    package: ModelPackage
    request: InspectionRequest
    model: nn.Module
    result: InspectionResult
    graph: ModelGraph
    capture: InspectionCapture
    dataset_name: str
    task_name: str
    inputs: tuple[Tensor, ...]


@dataclass(frozen=True, slots=True)
class _ExecutedShapeTrace:
    prepared: _PreparedShapeTrace
    calls: dict[str, list[_MutableModuleCall]]
    variable_tracer: _TensorVariableTracer | None


def _prepare_shape_trace(
    package: ModelPackage,
    request: InspectionRequest,
) -> _PreparedShapeTrace:
    materialized = materialize_inspection(package, request)
    capture = InspectionCapture(request.capture_limits)
    graph = inspect_model_graph(
        materialized.model,
        limits=request.capture_limits,
        _capture=capture,
    )
    result = materialized.result(graph)
    dataset_name, task_name, inputs = _sample_inputs(materialized.prepared)
    return _PreparedShapeTrace(
        package=package,
        request=request,
        model=materialized.model,
        result=result,
        graph=graph,
        capture=capture,
        dataset_name=dataset_name,
        task_name=task_name,
        inputs=inputs,
    )


def _variable_tracer(
    detail: ShapeTraceDetail,
    prepared: _PreparedShapeTrace,
    module_paths: Mapping[int, str],
) -> _TensorVariableTracer | None:
    if detail != "variables":
        return None
    trace_module_names = _trace_module_names(prepared.model)
    return _TensorVariableTracer(
        module_paths,
        trace_module_names,
        prepared.capture,
    )


def _execute_shape_trace(
    prepared: _PreparedShapeTrace,
    detail: ShapeTraceDetail,
) -> _ExecutedShapeTrace:
    with shape_trace_runtime(prepared.model) as runtime:
        try:
            calls, module_paths, handles = _trace_module_calls(
                prepared.model,
                prepared.graph,
                prepared.capture,
            )
            runtime.handles.extend(handles)
            variable_tracer = _variable_tracer(detail, prepared, module_paths)
            runtime.execute(prepared.inputs, variable_tracer)
        except InspectionError:
            raise
        except Exception as exc:
            raise InspectionError(
                "Failed to execute shape trace for model "
                f"'{prepared.package.catalog_key}' preset "
                f"'{prepared.request.preset}': {exc}"
            ) from exc
    return _ExecutedShapeTrace(
        prepared=prepared,
        calls=calls,
        variable_tracer=variable_tracer,
    )


def _module_shape_traces(
    execution: _ExecutedShapeTrace,
) -> tuple[ModuleShapeTrace, ...]:
    module_traces: list[ModuleShapeTrace] = []
    for node in execution.prepared.graph.nodes:
        module_trace = ModuleShapeTrace(
            node_id=node.id,
            calls=tuple(
                ModuleShapeCall(inputs=call.inputs, outputs=call.outputs)
                for call in execution.calls[node.id]
            ),
        )
        execution.prepared.capture.reserve_output({"node_id": module_trace.node_id})
        module_traces.append(module_trace)
    return tuple(module_traces)


def _sample_input_shapes(execution: _ExecutedShapeTrace) -> tuple[TensorShape, ...]:
    sample_inputs = tuple(
        tensor
        for index, value in enumerate(execution.prepared.inputs)
        for tensor in _tensor_shapes(
            value,
            f"input[{index}]",
            capture=execution.prepared.capture,
        )
    )
    execution.prepared.capture.reserve_output(sample_inputs)
    return sample_inputs


def _assemble_shape_trace(
    execution: _ExecutedShapeTrace,
) -> tuple[InspectionResult, ModelShapeTrace]:
    prepared = execution.prepared
    module_traces = _module_shape_traces(execution)
    sample_inputs = _sample_input_shapes(execution)
    trace = ModelShapeTrace(
        dataset=prepared.dataset_name,
        experiment_task=prepared.task_name,
        batch_size=1,
        sample_inputs=sample_inputs,
        modules=module_traces,
        methods=(
            execution.variable_tracer.results()
            if execution.variable_tracer is not None
            else ()
        ),
    )
    prepared.capture.reserve_output(
        {
            "dataset": trace.dataset,
            "experiment_task": trace.experiment_task,
            "batch_size": trace.batch_size,
        }
    )
    prepared.capture.ensure_total_output(
        {"result": prepared.result, "shape_trace": trace}
    )
    return prepared.result, trace


def inspect_model_shapes(
    package: ModelPackage,
    request: InspectionRequest,
    *,
    detail: ShapeTraceDetail = "outputs",
) -> tuple[InspectionResult, ModelShapeTrace]:
    """Execute a shape trace for trusted local Model Packages.

    Execution runs synchronously in the caller process and does not enforce a
    deadline or memory limit. It temporarily changes and restores the
    process-global CPU RNG and the caller thread's Python trace; CUDA RNG state,
    Python/NumPy state, and arbitrary package side effects are not isolated.
    Calls must not overlap another shape trace or other global Torch RNG work.
    Capture limits bound collected data, not execution resources. Use
    caller-owned process containment when stronger isolation is required.
    """
    if detail not in {"outputs", "variables"}:
        raise ValueError(f"Unknown shape-trace detail: {detail!r}")
    package = _require_model_package(package)

    torch_random = cast(Any, torch.random)
    with torch_random.fork_rng(devices=[]):
        cast(Any, torch).manual_seed(0)
        execution = _execute_shape_trace(
            _prepare_shape_trace(package, request),
            detail,
        )
    return _assemble_shape_trace(execution)


__all__ = [
    "MethodShapeTrace",
    "ModelShapeTrace",
    "ModuleShapeCall",
    "ModuleShapeTrace",
    "ShapeTraceDetail",
    "TensorShape",
    "TensorVariableTrace",
    "inspect_model_shapes",
]
