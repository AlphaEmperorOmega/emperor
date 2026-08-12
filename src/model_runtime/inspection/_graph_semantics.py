from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Protocol, cast

from torch.nn.parameter import is_lazy

from model_runtime.inspection._graph_accounting import GraphModule
from model_runtime.inspection._graph_component_semantics import (
    component_description,
    component_graph_role,
    display_graph_value,
    module_configuration,
)
from model_runtime.inspection.capture_limits import InspectionCaptureLimits
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


class _SemanticDetailsAdapter(Protocol):
    def apply(self, context: _SemanticDetailContext) -> None: ...


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
            if not hasattr(value, attr_name):
                value = None
                break
            value = getattr(value, attr_name)
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


def _bool_from_optional_model(module: GraphModule, attr_name: str) -> bool | None:
    if not hasattr(module, attr_name):
        return None
    return getattr(module, attr_name) is not None


@dataclass(frozen=True, slots=True)
class _LayerBehaviorDetailsAdapter:
    def apply(self, context: _SemanticDetailContext) -> None:
        module = context.module
        dropout = getattr(module, "dropout_probability", None)
        if dropout is not None:
            context.details["dropout"] = dropout

        gate = getattr(module, "gate_model", None)
        gate_option = getattr(gate, "option", None)
        if gate_option is None:
            gate_config = getattr(module, "gate_config", None)
            gate_option = getattr(gate_config, "option", None)
        gate_option_name = (
            display_graph_value(gate_option) if gate_option is not None else None
        )
        if gate_option_name is not None:
            context.details["gateOption"] = gate_option_name

        gate_model = _bool_from_optional_model(module, "gate_model")
        if gate_model is not None:
            context.details["gate"] = gate_model and gate is not None

        halting = _bool_from_optional_model(module, "halting_model")
        if halting is not None:
            context.details["halting"] = halting

        activation = getattr(module, "activation_function", None)
        if activation is not None:
            context.details["activation"] = display_graph_value(activation)

        layer_norm = getattr(module, "layer_norm_position", None)
        if layer_norm is not None:
            context.details["layerNorm"] = display_graph_value(layer_norm)


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
        if not hasattr(module, "x_axis_total_neurons") or not hasattr(
            module, "cluster"
        ):
            return None
        dynamic_module = cast(Any, module)
        cluster: Any = dynamic_module.cluster
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
                dynamic_module.x_axis_total_neurons,
                dynamic_module.y_axis_total_neurons,
                dynamic_module.z_axis_total_neurons,
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
        source: Any = module
        if not hasattr(source, "neuron_connections") and hasattr(module, "terminal"):
            source = cast(Any, module).terminal
        connections = getattr(source, "neuron_connections", None)
        if connections is None or not hasattr(source, "x_axis_position"):
            return None
        total = int(connections.shape[0])
        selected_connections = connections[: limits.maximum_terminal_connections]
        truncated = total > limits.maximum_terminal_connections
        return {
            "position": [
                source.x_axis_position,
                source.y_axis_position,
                source.z_axis_position,
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
    def _schedule(module: GraphModule) -> tuple[Any | None, Any | None]:
        iteration_schedule = getattr(module, "recurrent_iteration_schedule", None)
        schedule_snapshot = (
            iteration_schedule.snapshot() if iteration_schedule is not None else None
        )
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

    def _base_details(self, module: GraphModule, max_steps: Any) -> dict[str, Any]:
        gate, gate_option_name = self._gate_details(module)
        return {
            "maxSteps": max_steps,
            "diagnostics": bool(
                getattr(module, "supports_recurrent_diagnostics", False)
            ),
            "gate": gate,
            "gateOption": gate_option_name,
            "halting": bool(getattr(module, "halting_model", None) is not None),
        }

    @staticmethod
    def _append_schedule_details(
        recurrent: dict[str, Any],
        schedule_snapshot: Any | None,
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
        }

    @staticmethod
    def _append_optional_details(
        module: GraphModule,
        recurrent: dict[str, Any],
        schedule_snapshot: Any | None,
    ) -> None:
        halting_model = getattr(module, "halting_model", None)
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
        recurrent = self._base_details(context.module, max_steps)
        self._append_schedule_details(recurrent, schedule_snapshot)
        self._append_optional_details(
            context.module,
            recurrent,
            schedule_snapshot,
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
