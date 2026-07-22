from __future__ import annotations

import inspect
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from functools import lru_cache
from importlib.metadata import packages_distributions
from pathlib import Path
from types import FrameType
from typing import Literal

import torch
from torch import Tensor, nn

from model_runtime.inspection.errors import InspectionError
from model_runtime.inspection.materialization import (
    MaterializedConfiguration,
    materialize_inspection,
)
from model_runtime.inspection.model_graph import inspect_model_graph
from model_runtime.inspection.records import InspectionRequest, InspectionResult
from model_runtime.packages import ModelPackage
from model_runtime.task_behavior import SyntheticInputError, experiment_task_behavior

ShapeTraceDetail = Literal["outputs", "variables"]


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
    variables: list[TensorVariableTrace] = field(default_factory=list)
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


def _mapping_path(name: str, key: object) -> str:
    if isinstance(key, str) and key.isidentifier():
        return f"{name}.{key}"
    return f"{name}[{key!r}]"


def _tensor_observations(
    value: object,
    name: str,
    *,
    _seen: set[int] | None = None,
) -> tuple[_ObservedTensor, ...]:
    if isinstance(value, Tensor):
        return (
            _ObservedTensor(
                shape=_tensor_shape(name, value),
                object_id=id(value),
            ),
        )
    if value is None or isinstance(value, (str, bytes, int, float, bool, type)):
        return ()
    if isinstance(value, nn.Module):
        return ()

    seen = set() if _seen is None else _seen
    value_id = id(value)
    if value_id in seen:
        return ()

    if is_dataclass(value) and not isinstance(value, type):
        seen.add(value_id)
        tensors: list[_ObservedTensor] = []
        for data_field in fields(value):
            tensors.extend(
                _tensor_observations(
                    getattr(value, data_field.name),
                    f"{name}.{data_field.name}",
                    _seen=seen,
                )
            )
        return tuple(tensors)

    if isinstance(value, Mapping):
        seen.add(value_id)
        tensors = []
        for key, item in value.items():
            tensors.extend(
                _tensor_observations(
                    item,
                    _mapping_path(name, key),
                    _seen=seen,
                )
            )
        return tuple(tensors)

    if isinstance(value, Sequence):
        seen.add(value_id)
        tensors = []
        for index, item in enumerate(value):
            tensors.extend(_tensor_observations(item, f"{name}[{index}]", _seen=seen))
        return tuple(tensors)

    return ()


def _tensor_shapes(value: object, name: str) -> tuple[TensorShape, ...]:
    return tuple(observation.shape for observation in _tensor_observations(value, name))


def _local_tensor_observations(
    local_values: Mapping[str, object],
) -> dict[str, _ObservedTensor]:
    tensors: dict[str, _ObservedTensor] = {}
    for name, value in local_values.items():
        if name in {"self", "cls"}:
            continue
        for tensor in _tensor_observations(value, name):
            tensors[tensor.shape.name] = tensor
    return tensors


def _local_tensor_shapes(local_values: Mapping[str, object]) -> dict[str, TensorShape]:
    return {
        name: observation.shape
        for name, observation in _local_tensor_observations(local_values).items()
    }


def _bound_input_shapes(
    module: nn.Module,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> tuple[TensorShape, ...]:
    try:
        bound = inspect.signature(module.forward).bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        values = {f"input[{index}]": value for index, value in enumerate(args)}
        values.update({str(key): value for key, value in kwargs.items()})
    else:
        values = dict(bound.arguments)
    return tuple(_local_tensor_shapes(values).values())


@lru_cache(maxsize=1)
def _package_distributions() -> Mapping[str, tuple[str, ...]]:
    return {
        package_name: tuple(distributions)
        for package_name, distributions in packages_distributions().items()
    }


def _trace_module_names(model: nn.Module) -> frozenset[str]:
    registered_module_names = {
        type(module).__module__
        for module in model.modules()
        if isinstance(type(module).__module__, str)
    }
    model_package_name = type(model).__module__.partition(".")[0]
    runtime_package_name = __package__.partition(".")[0]
    distributions = _package_distributions()
    owned_distributions = {
        *distributions.get(model_package_name, ()),
        *distributions.get(runtime_package_name, ()),
    }
    return frozenset(
        module_name
        for module_name in registered_module_names
        if module_name.partition(".")[0] == model_package_name
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
    ) -> None:
        self._module_paths = module_paths
        self._trace_module_names = trace_module_names
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
        observations = _local_tensor_observations(frame.f_locals)
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
        self._methods.append(method)
        self._frames[id(frame)] = _FrameTrace(method=method, tensors=observations)

    def _capture_changes(self, frame: FrameType, trace: _FrameTrace) -> None:
        current = _local_tensor_observations(frame.f_locals)
        changed = tuple(
            observation.shape
            for name, observation in current.items()
            if trace.tensors.get(name) != observation
        )
        if changed:
            trace.method.variables.append(
                TensorVariableTrace(
                    order=self._order(),
                    line=trace.last_line,
                    tensors=changed,
                )
            )
        trace.tensors = current

    def __call__(self, frame: FrameType, event: str, argument: object):
        if event == "call":
            if not self._is_relevant(frame):
                return None
            self._start(frame)
            return self

        trace = self._frames.get(id(frame))
        if trace is None:
            return None
        if event == "line":
            self._capture_changes(frame, trace)
            trace.last_line = frame.f_lineno
        elif event == "return":
            self._capture_changes(frame, trace)
            trace.method.outputs = _tensor_shapes(argument, "return")
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


def _trace_module_calls(model: nn.Module, graph):
    calls: dict[str, list[_MutableModuleCall]] = {node.id: [] for node in graph.nodes}
    node_by_module_id: dict[int, str] = {}
    module_paths: dict[int, str] = {}
    modules: dict[int, nn.Module] = {}
    for node in graph.nodes:
        module = _module_for_node(model, node.id, node.path)
        if module is None or id(module) in modules:
            continue
        modules[id(module)] = module
        node_by_module_id[id(module)] = node.id
        module_paths[id(module)] = node.path

    pending: dict[int, list[_MutableModuleCall]] = {
        module_id: [] for module_id in modules
    }
    handles = []

    def before_forward(module, args, kwargs):
        call = _MutableModuleCall(
            inputs=_bound_input_shapes(module, args, kwargs),
        )
        pending[id(module)].append(call)
        calls[node_by_module_id[id(module)]].append(call)

    def after_forward(module, _args, _kwargs, output):
        module_pending = pending[id(module)]
        if module_pending:
            module_pending.pop().outputs = _tensor_shapes(output, "output")

    for module in modules.values():
        handles.append(
            module.register_forward_pre_hook(before_forward, with_kwargs=True)
        )
        handles.append(module.register_forward_hook(after_forward, with_kwargs=True))

    return calls, module_paths, handles


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


def inspect_model_shapes(
    package: ModelPackage,
    request: InspectionRequest,
    *,
    detail: ShapeTraceDetail = "outputs",
) -> tuple[InspectionResult, ModelShapeTrace]:
    if detail not in {"outputs", "variables"}:
        raise ValueError(f"Unknown shape-trace detail: {detail!r}")
    if not isinstance(package, ModelPackage):
        raise TypeError("Inspection requires a selected ModelPackage.")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        materialized = materialize_inspection(package, request)
        model = materialized.model
        graph = inspect_model_graph(model)
        result = materialized.result(graph)
        dataset_name, task_name, inputs = _sample_inputs(materialized.prepared)
        calls, module_paths, handles = _trace_module_calls(model, graph)
        variable_tracer = (
            _TensorVariableTracer(module_paths, _trace_module_names(model))
            if detail == "variables"
            else None
        )
        previous_trace = sys.gettrace()
        model.eval()
        try:
            if variable_tracer is not None:
                sys.settrace(variable_tracer)
            with torch.no_grad():
                model(*inputs)
        except Exception as exc:
            raise InspectionError(
                f"Failed to execute shape trace for model '{package.catalog_key}' "
                f"preset '{request.preset}': {exc}"
            ) from exc
        finally:
            if variable_tracer is not None:
                sys.settrace(previous_trace)
            for handle in handles:
                handle.remove()

    module_traces = tuple(
        ModuleShapeTrace(
            node_id=node.id,
            calls=tuple(
                ModuleShapeCall(inputs=call.inputs, outputs=call.outputs)
                for call in calls[node.id]
            ),
        )
        for node in graph.nodes
    )
    sample_inputs = tuple(
        tensor
        for index, value in enumerate(inputs)
        for tensor in _tensor_shapes(value, f"input[{index}]")
    )
    return result, ModelShapeTrace(
        dataset=dataset_name,
        experiment_task=task_name,
        batch_size=1,
        sample_inputs=sample_inputs,
        modules=module_traces,
        methods=variable_tracer.results() if variable_tracer is not None else (),
    )


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
