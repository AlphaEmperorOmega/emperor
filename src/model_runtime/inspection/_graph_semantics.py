from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Protocol, cast

from torch import Tensor
from torch.nn.parameter import is_lazy

from model_runtime.inspection._graph_accounting import GraphModule
from model_runtime.inspection._graph_component_semantics import (
    component_description,
    component_graph_role,
    display_graph_value,
    module_configuration,
)
from model_runtime.inspection.capture_limits import InspectionCaptureLimits
from model_runtime.inspection.errors import InspectionError
from model_runtime.inspection.records import GraphConfiguration, GraphRole


@dataclass(frozen=True, slots=True)
class ModuleSemanticFacts:
    type_name: str
    description: str | None
    graph_role: GraphRole
    details: dict[str, Any]
    configuration: GraphConfiguration | None


@dataclass(slots=True)
class _SemanticDetailContext:
    module: GraphModule
    direct_parameters: dict[str, Any] | None
    limits: InspectionCaptureLimits
    details: dict[str, Any] = field(default_factory=dict[str, Any])
    cluster_present: bool = False
    halting_model_present: bool = False
    halting_model: Any | None = None


class _SemanticDetailsAdapter(Protocol):
    def apply(self, context: _SemanticDetailContext) -> None: ...


class _MissingAttribute:
    __slots__ = ()


_MISSING_ATTRIBUTE = _MissingAttribute()


class _NeuronClusterRemainder(Protocol):
    y_axis_total_neurons: Any
    z_axis_total_neurons: Any


@dataclass(frozen=True, slots=True)
class _NeuronClusterObservation:
    owner: _NeuronClusterRemainder
    x_axis_total_neurons: Any
    cluster: Any


class _TerminalReachRemainder(Protocol):
    y_axis_position: Any
    z_axis_position: Any


@dataclass(frozen=True, slots=True)
class _TerminalReachObservation:
    owner: _TerminalReachRemainder
    neuron_connections: Any
    x_axis_position: Any


class _RecurrentIterationSchedule(Protocol):
    def snapshot(self) -> object: ...


class _RecurrentScheduleSnapshot(Protocol):
    maximum_transition_count: Any
    active_transition_count: Any
    gradient_transition_count: Any
    iteration_unit: Any
    initial_iterations: Any
    maximum_iterations: Any
    active_iterations: Any
    iteration_increment: Any
    forward_calls_before_iteration_increment: Any
    forward_call_progress: Any
    complete: Any
    no_gradient_transition_count: Any
    smooth_iteration_growth: Any
    settled_iterations: Any
    transitioning: Any
    transition_source_iterations: Any
    transition_target_iterations: Any
    transition_forward_index: Any
    transition_forward_count: Any
    transition_weight: Any


def _neuron_cluster_capability(
    module: GraphModule,
) -> _NeuronClusterObservation | None:
    x_axis_total_neurons = getattr(
        module,
        "x_axis_total_neurons",
        _MISSING_ATTRIBUTE,
    )
    if x_axis_total_neurons is _MISSING_ATTRIBUTE:
        return None
    cluster = getattr(module, "cluster", _MISSING_ATTRIBUTE)
    if cluster is _MISSING_ATTRIBUTE:
        return None
    return _NeuronClusterObservation(
        owner=cast(_NeuronClusterRemainder, module),
        x_axis_total_neurons=x_axis_total_neurons,
        cluster=cluster,
    )


def _shape_value(value: Any) -> str | None:
    if is_lazy(value):
        return None
    dimensions = tuple(value.shape)
    if not dimensions:
        return "scalar"
    return " x ".join(str(dimension) for dimension in dimensions)


@dataclass(frozen=True, slots=True)
class _ParameterShapeDetailsAdapter:
    def apply(self, context: _SemanticDetailContext) -> None:
        direct_parameters = context.direct_parameters
        if direct_parameters is None:
            direct_parameters = dict(context.module.named_parameters(recurse=False))
        uninitialized_parameter_count = sum(
            is_lazy(parameter) for parameter in direct_parameters.values()
        )
        if uninitialized_parameter_count:
            context.details["uninitializedParameterCount"] = (
                uninitialized_parameter_count
            )
        for detail_key, parameter_names in (
            ("weightShape", ("weight", "weight_params", "weights")),
            ("biasShape", ("bias", "bias_params", "biases")),
            ("parameterBankShape", ("parameter_bank",)),
            ("inputFactorShape", ("input_factor",)),
            ("outputFactorShape", ("output_factor",)),
        ):
            for parameter_name in parameter_names:
                parameter = direct_parameters.get(parameter_name)
                if parameter is None:
                    continue
                shape = _shape_value(parameter)
                if shape is not None:
                    context.details[detail_key] = shape
                    break


@dataclass(frozen=True, slots=True)
class _DimensionDetailsAdapter:
    def apply(self, context: _SemanticDetailContext) -> None:
        module = context.module
        input_dim = getattr(module, "input_dim", None)
        output_dim = getattr(module, "output_dim", None)
        hidden_dim = getattr(module, "hidden_dim", None)
        if input_dim is not None:
            context.details["inputDim"] = input_dim
        if hidden_dim is not None:
            context.details["hiddenDim"] = hidden_dim
        if output_dim is not None:
            context.details["outputDim"] = output_dim
        if input_dim is not None and output_dim is not None:
            context.details["dims"] = f"{input_dim} -> {output_dim}"


@dataclass(frozen=True, slots=True)
class _AttributeDetailsAdapter:
    fields: tuple[tuple[str, str], ...]

    def apply(self, context: _SemanticDetailContext) -> None:
        for source_attr, detail_key in self.fields:
            value = getattr(context.module, source_attr, None)
            if value is not None:
                context.details[detail_key] = display_graph_value(value)


def _first_detail_value(module: GraphModule, attr_paths: tuple[str, ...]) -> Any:
    for attr_path in attr_paths:
        value: Any = module
        for attr_name in attr_path.split("."):
            observed = getattr(value, attr_name, _MISSING_ATTRIBUTE)
            if observed is _MISSING_ATTRIBUTE:
                value = None
                break
            value = observed
        if value is not None:
            return value
    return None


@dataclass(frozen=True, slots=True)
class _ExpertDetailsAdapter:
    def apply(self, context: _SemanticDetailContext) -> None:
        for detail_key, attr_paths in (
            ("topK", ("top_k", "sampler_config.top_k", "cfg.top_k")),
            (
                "numExperts",
                ("num_experts", "sampler_config.num_experts", "cfg.num_experts"),
            ),
            (
                "routingMode",
                ("routing_initialization_mode", "cfg.routing_initialization_mode"),
            ),
        ):
            value = _first_detail_value(context.module, attr_paths)
            if value is not None:
                context.details[detail_key] = display_graph_value(value)


def _optional_model(
    module: object,
    attr_name: str,
) -> tuple[bool, Any | None]:
    model = getattr(module, attr_name, _MISSING_ATTRIBUTE)
    if model is _MISSING_ATTRIBUTE:
        return False, None
    return True, model


@dataclass(frozen=True, slots=True)
class _LayerBehaviorDetailsAdapter:
    def apply(self, context: _SemanticDetailContext) -> None:
        module = context.module
        postprocessing = getattr(module, "postprocessing", None)
        if postprocessing is None:
            self.__apply_direct_controller_owner(context)
            return

        normalization = getattr(module, "normalization", None)
        halting_delegate = getattr(module, "halting", None)

        dropout = getattr(postprocessing, "dropout_probability", None)
        if dropout is not None:
            context.details["dropout"] = dropout

        self.__append_gate_details(
            context,
            *(_optional_model(postprocessing, "gate")),
            config_owner=postprocessing,
        )
        halting_present, halting = _optional_model(halting_delegate, "model")
        context.halting_model_present = halting_present
        context.halting_model = halting
        if context.halting_model_present:
            context.details["halting"] = context.halting_model is not None

        activation = getattr(postprocessing, "activation_function", None)
        if activation is not None:
            context.details["activation"] = display_graph_value(activation)

        layer_norm = getattr(normalization, "position", None)
        if layer_norm is not None:
            context.details["layerNorm"] = display_graph_value(layer_norm)

    def __apply_direct_controller_owner(
        self,
        context: _SemanticDetailContext,
    ) -> None:
        module = context.module
        dropout = getattr(module, "dropout_probability", None)
        if dropout is not None:
            context.details["dropout"] = dropout

        self.__append_gate_details(
            context,
            *(_optional_model(module, "gate_model")),
            config_owner=module,
        )
        halting_present, halting = _optional_model(module, "halting_model")
        context.halting_model_present = halting_present
        context.halting_model = halting
        if halting_present:
            context.details["halting"] = halting is not None

        activation = getattr(module, "activation_function", None)
        if activation is not None:
            context.details["activation"] = display_graph_value(activation)

        layer_norm = getattr(module, "layer_norm_position", None)
        if layer_norm is not None:
            context.details["layerNorm"] = display_graph_value(layer_norm)

    @staticmethod
    def __append_gate_details(
        context: _SemanticDetailContext,
        gate_present: bool,
        gate: Any | None,
        *,
        config_owner: GraphModule | Any,
    ) -> None:
        gate_option = getattr(gate, "option", None)
        if gate_option is None:
            gate_config = getattr(config_owner, "gate_config", None)
            gate_option = getattr(gate_config, "option", None)
        if gate_option is not None:
            context.details["gateOption"] = display_graph_value(gate_option)
        if gate_present:
            context.details["gate"] = gate is not None


def _coordinate_from_neuron_name(name: str) -> list[int] | None:
    parts = name.split("_")
    if len(parts) != 4 or parts[0] != "neuron":
        return None
    try:
        return [int(parts[1]), int(parts[2]), int(parts[3])]
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class _NeuronDetailsAdapter:
    @staticmethod
    def _cluster_details(
        module: GraphModule,
        limits: InspectionCaptureLimits,
    ) -> dict[str, Any] | None:
        cluster_observation = _neuron_cluster_capability(module)
        if cluster_observation is None:
            return None
        cluster = cluster_observation.cluster
        total_coordinates = len(cluster)
        cluster_names = tuple(
            islice(
                cast(Iterable[str], cluster.keys()),
                limits.maximum_neuron_coordinates,
            )
        )
        coordinates = sorted(
            coordinate
            for coordinate in (
                _coordinate_from_neuron_name(name) for name in cluster_names
            )
            if coordinate is not None
        )
        coordinates_truncated = total_coordinates > len(cluster_names)
        return {
            "capacity": [
                cluster_observation.x_axis_total_neurons,
                cluster_observation.owner.y_axis_total_neurons,
                cluster_observation.owner.z_axis_total_neurons,
            ],
            "initial": [
                getattr(module, "initial_x_axis_total_neurons", None),
                getattr(module, "initial_y_axis_total_neurons", None),
                getattr(module, "initial_z_axis_total_neurons", None),
            ],
            "initialStart": [
                getattr(module, "initial_x_axis_start", 1),
                getattr(module, "initial_y_axis_start", 1),
                getattr(module, "initial_z_axis_start", 1),
            ],
            "instantiated": total_coordinates,
            "coordinates": coordinates,
            "coordinatesTotal": total_coordinates,
            "coordinatesInvalid": len(cluster_names) - len(coordinates),
            "coordinatesTruncated": coordinates_truncated,
            "coordinatesTruncationReason": (
                "maximum_neuron_coordinates" if coordinates_truncated else None
            ),
            "maxSteps": getattr(module, "max_steps", None),
            "growthThreshold": getattr(module, "growth_threshold", None),
        }

    @staticmethod
    def _terminal_reach_details(
        module: GraphModule,
        limits: InspectionCaptureLimits,
    ) -> dict[str, Any] | None:
        source: object = module
        connections = getattr(module, "neuron_connections", _MISSING_ATTRIBUTE)
        if connections is _MISSING_ATTRIBUTE:
            source = getattr(module, "terminal", module)
            connections = (
                None
                if source is module
                else getattr(source, "neuron_connections", None)
            )
        if connections is None:
            return None
        x_axis_position = getattr(source, "x_axis_position", _MISSING_ATTRIBUTE)
        if x_axis_position is _MISSING_ATTRIBUTE:
            return None
        valid_shape = isinstance(connections, Tensor) and connections.shape[1:] == (3,)
        if not valid_shape:
            raise InspectionError(
                "Terminal neuron_connections must be a Tensor shaped [connections, 3]."
            )
        terminal = _TerminalReachObservation(
            owner=cast(_TerminalReachRemainder, source),
            neuron_connections=connections,
            x_axis_position=x_axis_position,
        )
        observed_connections = terminal.neuron_connections
        total = int(observed_connections.shape[0])
        maximum = limits.maximum_terminal_connections
        selected_connections = observed_connections[:maximum]
        truncated = total > maximum
        return {
            "position": [
                terminal.x_axis_position,
                terminal.owner.y_axis_position,
                terminal.owner.z_axis_position,
            ],
            "connections": selected_connections.detach().cpu().tolist(),
            "total": total,
            "truncated": truncated,
            "truncationReason": ("maximum_terminal_connections" if truncated else None),
        }

    def apply(self, context: _SemanticDetailContext) -> None:
        cluster = self._cluster_details(context.module, context.limits)
        context.cluster_present = cluster is not None
        if cluster is not None:
            context.details["cluster"] = cluster

        terminal_reach = self._terminal_reach_details(
            context.module,
            context.limits,
        )
        if terminal_reach is not None:
            context.details["terminalReach"] = terminal_reach


@dataclass(frozen=True, slots=True)
class _RecurrentDetailsAdapter:
    @staticmethod
    def _schedule(
        module: GraphModule,
    ) -> tuple[_RecurrentScheduleSnapshot | None, Any | None]:
        iteration_schedule = getattr(module, "recurrent_iteration_schedule", None)
        schedule_snapshot: _RecurrentScheduleSnapshot | None = None
        if iteration_schedule is not None:
            schedule = cast(_RecurrentIterationSchedule, iteration_schedule)
            schedule_snapshot = cast(_RecurrentScheduleSnapshot, schedule.snapshot())
        max_steps = (
            schedule_snapshot.maximum_transition_count
            if schedule_snapshot is not None
            else getattr(module, "max_steps", None)
        )
        if max_steps is None:
            max_steps = getattr(module, "recurrent_diagnostic_step_limit", None)
        return schedule_snapshot, max_steps

    @staticmethod
    def _gate_details(module: GraphModule) -> tuple[bool, Any | None]:
        recurrent_gate = getattr(module, "recurrent_gate", None)
        gate_option = getattr(recurrent_gate, "option", None)
        if gate_option is None:
            gate_config = getattr(module, "gate_config", None)
            gate_option = getattr(gate_config, "option", None)
        gate_option_name = (
            display_graph_value(gate_option) if gate_option is not None else None
        )
        gate = (
            recurrent_gate is not None
            and getattr(recurrent_gate, "model", None) is not None
        )
        return gate, gate_option_name

    def _base_details(
        self,
        module: GraphModule,
        max_steps: Any,
        halting_model: Any | None,
    ) -> dict[str, Any]:
        gate, gate_option_name = self._gate_details(module)
        return {
            "maxSteps": max_steps,
            "diagnostics": bool(
                getattr(module, "supports_recurrent_diagnostics", False)
            ),
            "gate": gate,
            "gateOption": gate_option_name,
            "halting": halting_model is not None,
        }

    @staticmethod
    def _append_schedule_details(
        recurrent: dict[str, Any],
        schedule_snapshot: _RecurrentScheduleSnapshot | None,
    ) -> None:
        if schedule_snapshot is None:
            return
        recurrent["activeSteps"] = schedule_snapshot.active_transition_count
        if schedule_snapshot.gradient_transition_count is not None:
            recurrent["gradientTransitionCount"] = (
                schedule_snapshot.gradient_transition_count
            )
        recurrent["iterationSchedule"] = {
            "unit": schedule_snapshot.iteration_unit,
            "initialIterations": schedule_snapshot.initial_iterations,
            "maximumIterations": schedule_snapshot.maximum_iterations,
            "activeIterations": schedule_snapshot.active_iterations,
            "iterationIncrement": schedule_snapshot.iteration_increment,
            "forwardCallsBeforeIterationIncrement": (
                schedule_snapshot.forward_calls_before_iteration_increment
            ),
            "forwardCallProgress": schedule_snapshot.forward_call_progress,
            "complete": schedule_snapshot.complete,
            "smoothIterationGrowth": (schedule_snapshot.smooth_iteration_growth),
            "settledIterations": schedule_snapshot.settled_iterations,
            "transitioning": schedule_snapshot.transitioning,
            "transitionSourceIterations": (
                schedule_snapshot.transition_source_iterations
            ),
            "transitionTargetIterations": (
                schedule_snapshot.transition_target_iterations
            ),
            "transitionForwardIndex": (schedule_snapshot.transition_forward_index),
            "transitionForwardCount": (schedule_snapshot.transition_forward_count),
            "transitionWeight": schedule_snapshot.transition_weight,
        }

    @staticmethod
    def _append_optional_details(
        module: GraphModule,
        recurrent: dict[str, Any],
        schedule_snapshot: _RecurrentScheduleSnapshot | None,
        halting_model: Any | None,
    ) -> None:
        min_steps = getattr(halting_model, "min_steps", None)
        if min_steps is not None:
            recurrent["minSteps"] = min_steps
        if schedule_snapshot is not None:
            recurrent["noGradientTransitionCount"] = (
                schedule_snapshot.no_gradient_transition_count
            )
        recurrent_layer_norm = getattr(
            module,
            "recurrent_layer_norm_position",
            None,
        )
        if recurrent_layer_norm is not None:
            recurrent["layerNorm"] = display_graph_value(recurrent_layer_norm)
        answer_update_count = getattr(module, "answer_update_count", None)
        if answer_update_count is not None:
            recurrent["answerUpdateCount"] = answer_update_count
        latent_updates_per_answer_update = getattr(
            module,
            "latent_updates_per_answer_update",
            None,
        )
        if latent_updates_per_answer_update is not None:
            recurrent["latentUpdatesPerAnswerUpdate"] = latent_updates_per_answer_update
        high_cycles = getattr(module, "high_cycles", None)
        if high_cycles is not None:
            recurrent["highCycles"] = high_cycles
        low_cycles = getattr(module, "low_cycles", None)
        if low_cycles is not None:
            recurrent["lowCycles"] = low_cycles

    def apply(self, context: _SemanticDetailContext) -> None:
        schedule_snapshot, max_steps = self._schedule(context.module)
        if max_steps is None or context.cluster_present:
            return
        recurrent = self._base_details(
            context.module,
            max_steps,
            context.halting_model,
        )
        self._append_schedule_details(recurrent, schedule_snapshot)
        self._append_optional_details(
            context.module,
            recurrent,
            schedule_snapshot,
            context.halting_model,
        )
        context.details["recurrent"] = recurrent


@dataclass(frozen=True, slots=True)
class _CausalAttentionDetailsAdapter:
    def apply(self, context: _SemanticDetailContext) -> None:
        causal = getattr(context.module, "causal_attention_mask_flag", None)
        if causal is not None:
            context.details["causalAttention"] = causal


_DETAIL_ADAPTERS: tuple[_SemanticDetailsAdapter, ...] = (
    _ParameterShapeDetailsAdapter(),
    _DimensionDetailsAdapter(),
    _AttributeDetailsAdapter(
        (
            ("embedding_dim", "embeddingDim"),
            ("num_heads", "numHeads"),
            ("num_layers", "numLayers"),
            ("source_sequence_length", "sourceSequenceLength"),
            ("target_sequence_length", "targetSequenceLength"),
        )
    ),
    _ExpertDetailsAdapter(),
    _LayerBehaviorDetailsAdapter(),
    _NeuronDetailsAdapter(),
    _RecurrentDetailsAdapter(),
    _CausalAttentionDetailsAdapter(),
)


def _snake_case_key(key: str) -> str:
    first_pass = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", key)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first_pass).lower()


def _semantic_detail_value(value: Any) -> Any:
    if isinstance(value, dict):
        mapping_value = cast(dict[object, object], value)
        return {
            _snake_case_key(str(key)): _semantic_detail_value(item)
            for key, item in mapping_value.items()
        }
    if isinstance(value, list):
        list_value = cast(list[object], value)
        return [_semantic_detail_value(item) for item in list_value]
    return value


@dataclass(frozen=True, slots=True)
class ModuleSemanticAdapter:
    module: GraphModule
    direct_parameters: dict[str, Any] | None = None
    limits: InspectionCaptureLimits | None = None

    def graph_role(self) -> GraphRole:
        return component_graph_role(self.module)

    def details(self) -> dict[str, Any]:
        if self.limits is None:
            raise RuntimeError("Semantic detail adaptation requires capture limits.")
        context = _SemanticDetailContext(
            module=self.module,
            direct_parameters=self.direct_parameters,
            limits=self.limits,
        )
        for adapter in _DETAIL_ADAPTERS:
            adapter.apply(context)
        return context.details

    def facts(self) -> ModuleSemanticFacts:
        if self.limits is None:
            raise RuntimeError("Semantic fact adaptation requires capture limits.")
        return ModuleSemanticFacts(
            type_name=type(self.module).__name__,
            description=component_description(self.module),
            graph_role=self.graph_role(),
            details=_semantic_detail_value(self.details()),
            configuration=module_configuration(self.module, self.limits),
        )


__all__ = ["ModuleSemanticAdapter", "ModuleSemanticFacts"]
