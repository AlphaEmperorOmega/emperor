from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import fields, is_dataclass
from enum import Enum
from inspect import cleandoc
from typing import Any, Protocol

from torch.nn.parameter import is_lazy

from emperor.config import ConfigBase
from model_runtime.inspection.records import (
    GraphConfiguration,
    GraphConfigurationField,
    GraphEdge,
    GraphNode,
    GraphRole,
    ModelGraph,
)


class _GraphModule(Protocol):
    def named_children(self) -> Iterator[tuple[str, _GraphModule]]: ...

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, Any]]: ...


ROOT_NODE_ID = "__root__"
ROOT_NODE_PATH = "model"

ARCHITECTURE_ROLE: GraphRole = "architecture"
INTERNAL_ROLE: GraphRole = "internal"
RUNTIME_ROLE: GraphRole = "runtime"

INTERNAL_GRAPH_TYPE_NAMES = {
    "Dropout",
    "KeyValueBias",
    "LayerNorm",
    "SamplerAuxiliaryLosses",
    "SelfAttentionProcessor",
    "SelfAttentionProjector",
    "Unfold",
}

RUNTIME_GRAPH_TYPE_NAMES = {
    "ClassifierMetricsLogger",
    "CrossEntropyLoss",
    "LanguageModelMetricsLogger",
    "MulticlassAccuracy",
    "MulticlassF1Score",
    "SequenceClassifierMetricsLogger",
}

COMPONENT_DESCRIPTION_BY_CLASS_NAME = {
    "Model": (
        "Top-level inspected model wrapper that owns the architecture, loss, "
        "metrics, and runtime modules for the selected preset."
    ),
    "ModuleList": (
        "Container that stores an ordered list of child modules; execution is "
        "defined by the parent module."
    ),
    "Sequential": (
        "Container that applies child modules in order, passing each output to "
        "the next child."
    ),
    "Dropout": (
        "Regularization module that randomly zeroes activations during training "
        "and is inactive during evaluation."
    ),
    "LayerNorm": (
        "Normalizes features within each sample to stabilize hidden-state "
        "scale before or after a layer block."
    ),
    "CrossEntropyLoss": ("Runtime loss module for multi-class classification targets."),
    "ClassifierMetricsLogger": (
        "Runtime module that groups classifier metrics for train, validation, "
        "and test reporting."
    ),
    "LanguageModelMetricsLogger": (
        "Runtime module that groups language-model metrics for train, "
        "validation, and test reporting."
    ),
    "SequenceClassifierMetricsLogger": (
        "Runtime module that groups sequence-classifier metrics for train, "
        "validation, and test reporting."
    ),
    "MulticlassAccuracy": (
        "Runtime metric that reports the share of classified examples whose "
        "predicted class matches the target class."
    ),
    "MulticlassF1Score": (
        "Runtime metric that reports the harmonic mean of classifier precision "
        "and recall across classes."
    ),
    "KeyValueBias": (
        "Internal attention helper that adds learned key/value bias terms."
    ),
    "SamplerAuxiliaryLosses": (
        "Internal mixture-of-experts helper that tracks auxiliary routing losses."
    ),
    "SelfAttentionProcessor": (
        "Internal attention helper that prepares attention inputs and masks."
    ),
    "SelfAttentionProjector": (
        "Internal attention helper that projects hidden states into attention "
        "query, key, and value tensors."
    ),
    "Unfold": (
        "Internal tensor reshaping module that extracts sliding local blocks "
        "from an input tensor."
    ),
    "LinearLayer": (
        "Applies a learned linear projection with configured input/output "
        "dimensions and optional bias."
    ),
    "LinearLayerConfig": (
        "Builds a learned linear projection with configured input/output "
        "dimensions and optional bias."
    ),
    "AdaptiveLinearLayer": (
        "Applies a learned linear projection that can optionally augment "
        "parameters from the current input."
    ),
    "AdaptiveLinearLayerConfig": (
        "Builds a linear projection that can optionally augment parameters from "
        "the current input."
    ),
    "Layer": (
        "Applies one configured layer block with optional activation, residuals, "
        "normalization, gating, halting, and memory hooks."
    ),
    "LayerConfig": (
        "Builds a Layer block with optional activation, residuals, normalization, "
        "gating, halting, and memory hooks."
    ),
    "LayerStack": (
        "Runs an ordered stack of Layer blocks, with shared dimensions and "
        "optional shared gate, halting, or memory modules."
    ),
    "LayerStackConfig": (
        "Builds an ordered stack of Layer blocks, with shared dimensions and "
        "optional shared gate, halting, or memory modules."
    ),
    "RecurrentLayer": (
        "Reuses a configured block for multiple recurrent steps, optionally "
        "adding recurrent gating, normalization, halting, or memory."
    ),
    "RecurrentLayerConfig": (
        "Builds a recurrent block that can run for multiple steps with optional "
        "gating, normalization, halting, or memory."
    ),
    "MixtureOfExperts": (
        "Routes inputs across a set of expert modules using sampler probabilities "
        "and combines or maps the selected expert outputs."
    ),
    "MixtureOfExpertsMap": (
        "Routes inputs across experts and returns mapped expert outputs."
    ),
    "MixtureOfExpertsReduce": (
        "Routes inputs across experts and reduces selected expert outputs back "
        "into one representation."
    ),
    "MixtureOfExpertsLayer": (
        "Wraps mixture-of-experts routing in the standard Layer pipeline."
    ),
    "MixtureOfExpertsConfig": (
        "Configures expert count, routing, capacity, weighting, sampler behavior, "
        "and expert model construction."
    ),
    "MixtureOfExpertsLayerConfig": (
        "Builds a mixture-of-experts layer inside the standard Layer pipeline."
    ),
    "MixtureOfExpertsModelConfig": (
        "Builds a model around a mixture-of-experts layer stack."
    ),
    "LayerGate": (
        "Combines a learned gate output with the current layer value by scaling "
        "or addition."
    ),
    "Gate": (
        "Combines a learned gate output with the current layer value by scaling "
        "or addition."
    ),
    "GateConfig": (
        "Configures a layer gate network and how its output is composed with "
        "the current value."
    ),
    "ResidualConnection": (
        "Combines the current and previous hidden values using the configured "
        "residual composition mode."
    ),
    "ResidualConfig": (
        "Configures residual composition and optionally makes weighted mixing "
        "coefficients data-dependent."
    ),
    "Halting": (
        "Controls adaptive computation by deciding when recurrent processing has "
        "accumulated enough probability mass to stop."
    ),
    "HaltingConfig": (
        "Configures adaptive computation halting thresholds, dropout, hidden-state "
        "mode, and gate network."
    ),
    "SoftHalting": (
        "Accumulates weighted recurrent states until the halting threshold is met."
    ),
    "SoftHaltingConfig": (
        "Builds soft halting, which accumulates weighted recurrent states until "
        "the threshold is met."
    ),
    "StickBreaking": (
        "Allocates remaining recurrent probability mass step by step until the "
        "halting threshold is met."
    ),
    "StickBreakingConfig": (
        "Builds stick-breaking halting, which allocates remaining recurrent "
        "probability mass over steps."
    ),
    "NeuronCluster": (
        "Maintains a 3D cluster of routed neurons that can traverse, branch, and "
        "grow during training."
    ),
    "NeuronClusterConfig": (
        "Configures a 3D routed neuron cluster, including capacity, traversal, "
        "sampling, and growth controls."
    ),
}

DEFAULT_RESIDUAL_OPTION_DESCRIPTION = (
    "Residual connection behavior. Enabled options require input_dim == output_dim."
)
DEFAULT_RESIDUAL_MODEL_DESCRIPTION = (
    "Optional model that generates data-dependent coefficients for weighted residual "
    "modes. When omitted, weighted modes use a learned scalar parameter."
)
RESIDUAL_FIELD_DESCRIPTIONS_BY_CONFIG_NAME = {
    "RecurrentLayerConfig": (
        "Residual connection behavior between recurrent steps. Set to null to "
        "disable recurrent residuals.",
        DEFAULT_RESIDUAL_MODEL_DESCRIPTION,
    ),
    "TransformerEncoderLayerConfig": (
        "Residual connection behavior applied to every encoder sub-block join.",
        "Optional data-dependent coefficient model used at each encoder join.",
    ),
    "TransformerDecoderLayerConfig": (
        "Residual connection behavior applied to every decoder sub-block join.",
        "Optional data-dependent coefficient model used at each decoder join.",
    ),
}


def _display_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, type):
        return value.__name__
    return value


def _config_field_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, ConfigBase):
        return type(value).__name__
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return type(value).__name__
    return str(value)


def _module_config_instance(module: _GraphModule) -> Any | None:
    config = getattr(module, "_emperor_config", None)
    if config is None:
        config = getattr(module, "cfg", None)
    if config is None or isinstance(config, type) or not is_dataclass(config):
        return None
    return config


def _metadata_help(metadata: Any) -> str | None:
    help_text = metadata.get("help") if hasattr(metadata, "get") else None
    if not isinstance(help_text, str):
        return None
    help_text = help_text.strip()
    return help_text or None


def _flattened_residual_configuration_fields(
    config: Any,
) -> tuple[GraphConfigurationField, GraphConfigurationField]:
    residual_config = config.residual_config
    option = getattr(residual_config, "option", None)
    model_config = getattr(residual_config, "model_config", None)
    option_description, model_description = (
        RESIDUAL_FIELD_DESCRIPTIONS_BY_CONFIG_NAME.get(
            type(config).__name__,
            (
                DEFAULT_RESIDUAL_OPTION_DESCRIPTION,
                DEFAULT_RESIDUAL_MODEL_DESCRIPTION,
            ),
        )
    )
    return (
        GraphConfigurationField(
            key="residual_connection_option",
            value=_config_field_value(option),
            description=option_description,
        ),
        GraphConfigurationField(
            key="residual_model_config",
            value=_config_field_value(model_config),
            description=model_description,
        ),
    )


def _module_config(module: _GraphModule) -> GraphConfiguration | None:
    config = _module_config_instance(module)
    if config is None:
        return None

    serialized_fields: list[GraphConfigurationField] = []
    flattened_residual_model_field: GraphConfigurationField | None = None
    for field in fields(config):
        if field.name == "residual_config":
            (
                residual_option_field,
                flattened_residual_model_field,
            ) = _flattened_residual_configuration_fields(config)
            serialized_fields.append(residual_option_field)
            continue
        description = _metadata_help(field.metadata)
        serialized_fields.append(
            GraphConfigurationField(
                key=field.name,
                value=_config_field_value(getattr(config, field.name)),
                description=description,
            )
        )
    if flattened_residual_model_field is not None:
        serialized_fields.append(flattened_residual_model_field)

    return GraphConfiguration(
        type_name=type(config).__name__,
        fields=tuple(serialized_fields),
    )


def _explicit_docstring_description(class_type: type[Any]) -> str | None:
    raw_docstring = class_type.__dict__.get("__doc__")
    if not isinstance(raw_docstring, str):
        return None
    docstring = cleandoc(raw_docstring).strip()
    if not docstring or docstring.startswith(f"{class_type.__name__}("):
        return None
    return docstring.split("\n\n", 1)[0].replace("\n", " ")


def _component_description(module: _GraphModule) -> str | None:
    module_type = type(module)
    description = COMPONENT_DESCRIPTION_BY_CLASS_NAME.get(module_type.__name__)
    if description is not None:
        return description

    config = _module_config_instance(module)
    if config is not None:
        description = COMPONENT_DESCRIPTION_BY_CLASS_NAME.get(type(config).__name__)
        if description is not None:
            return description

    if not module_type.__module__.startswith("torch."):
        description = _explicit_docstring_description(module_type)
        if description is not None:
            return description
    if config is not None:
        return _explicit_docstring_description(type(config))
    return None


def _shape_value(value: Any) -> str | None:
    if is_lazy(value):
        return None
    dimensions = tuple(value.shape)
    if not dimensions:
        return "scalar"
    return " x ".join(str(dimension) for dimension in dimensions)


def _bool_from_optional_model(module: _GraphModule, attr_name: str) -> bool | None:
    if not hasattr(module, attr_name):
        return None
    return getattr(module, attr_name) is not None


def _first_detail_value(module: _GraphModule, attr_paths: tuple[str, ...]) -> Any:
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


def _coordinate_from_neuron_name(name: str) -> list[int] | None:
    parts = name.split("_")
    if len(parts) != 4 or parts[0] != "neuron":
        return None
    try:
        return [int(parts[1]), int(parts[2]), int(parts[3])]
    except ValueError:
        return None


def _neuron_cluster_details(module: _GraphModule) -> dict[str, Any] | None:
    if not hasattr(module, "x_axis_total_neurons") or not hasattr(module, "cluster"):
        return None
    coordinates = sorted(
        coordinate
        for coordinate in (
            _coordinate_from_neuron_name(name) for name in module.cluster.keys()
        )
        if coordinate is not None
    )
    return {
        "capacity": [
            module.x_axis_total_neurons,
            module.y_axis_total_neurons,
            module.z_axis_total_neurons,
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
        "instantiated": len(coordinates),
        "coordinates": coordinates,
        "maxSteps": getattr(module, "max_steps", None),
        "growthThreshold": getattr(module, "growth_threshold", None),
    }


def _terminal_reach_details(module: _GraphModule) -> dict[str, Any] | None:
    # The reach lives on a Terminal; surface it on the parent Neuron too so a
    # neuron click shows the area its sampler can route to.
    source = module
    if not hasattr(source, "neuron_connections") and hasattr(module, "terminal"):
        source = module.terminal
    connections = getattr(source, "neuron_connections", None)
    if connections is None or not hasattr(source, "x_axis_position"):
        return None
    return {
        "position": [
            source.x_axis_position,
            source.y_axis_position,
            source.z_axis_position,
        ],
        "connections": connections.detach().cpu().tolist(),
        "total": getattr(source, "total_neuron_connections", connections.shape[0]),
    }


def _parameter_shape_details(module: _GraphModule) -> dict[str, Any]:
    details: dict[str, Any] = {}
    direct_parameters = dict(module.named_parameters(recurse=False))
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
                details[detail_key] = shape
                break
    return details


def _dimension_details(module: _GraphModule) -> dict[str, Any]:
    details: dict[str, Any] = {}
    input_dim = getattr(module, "input_dim", None)
    output_dim = getattr(module, "output_dim", None)
    hidden_dim = getattr(module, "hidden_dim", None)
    if input_dim is not None:
        details["inputDim"] = input_dim
    if hidden_dim is not None:
        details["hiddenDim"] = hidden_dim
    if output_dim is not None:
        details["outputDim"] = output_dim
    if input_dim is not None and output_dim is not None:
        details["dims"] = f"{input_dim} -> {output_dim}"
    return details


def _sequence_or_attention_details(module: _GraphModule) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for source_attr, detail_key in (
        ("embedding_dim", "embeddingDim"),
        ("num_heads", "numHeads"),
        ("num_layers", "numLayers"),
        ("source_sequence_length", "sourceSequenceLength"),
        ("target_sequence_length", "targetSequenceLength"),
    ):
        value = getattr(module, source_attr, None)
        if value is not None:
            details[detail_key] = _display_value(value)
    return details


def _expert_details(module: _GraphModule) -> dict[str, Any]:
    details: dict[str, Any] = {}
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
        value = _first_detail_value(module, attr_paths)
        if value is not None:
            details[detail_key] = _display_value(value)
    return details


def _layer_behavior_details(module: _GraphModule) -> dict[str, Any]:
    details: dict[str, Any] = {}
    dropout = getattr(module, "dropout_probability", None)
    if dropout is not None:
        details["dropout"] = dropout

    gate = getattr(module, "gate_model", None)
    gate_option = getattr(gate, "option", None)
    if gate_option is None:
        gate_config = getattr(module, "gate_config", None)
        gate_option = getattr(gate_config, "option", None)
    gate_option_name = _display_value(gate_option) if gate_option is not None else None
    if gate_option_name is not None:
        details["gateOption"] = gate_option_name

    gate_model = _bool_from_optional_model(module, "gate_model")
    if gate_model is not None:
        details["gate"] = gate_model and gate is not None

    halting = _bool_from_optional_model(module, "halting_model")
    if halting is not None:
        details["halting"] = halting

    activation = getattr(module, "activation_function", None)
    if activation is not None:
        details["activation"] = _display_value(activation)

    layer_norm = getattr(module, "layer_norm_position", None)
    if layer_norm is not None:
        details["layerNorm"] = _display_value(layer_norm)
    return details


def _recurrent_details(
    module: _GraphModule,
    cluster: dict[str, Any] | None,
) -> dict[str, Any]:
    max_steps = getattr(module, "max_steps", None)
    if max_steps is None or cluster is not None:
        return {}
    recurrent_gate = getattr(module, "recurrent_gate", None)
    gate_option = getattr(recurrent_gate, "option", None)
    if gate_option is None:
        gate_config = getattr(module, "gate_config", None)
        gate_option = getattr(gate_config, "option", None)
    gate_option_name = _display_value(gate_option) if gate_option is not None else None
    gate = (
        recurrent_gate is not None
        and getattr(recurrent_gate, "model", None) is not None
    )
    recurrent: dict[str, Any] = {
        "maxSteps": max_steps,
        "gate": gate,
        "gateOption": gate_option_name,
        "halting": bool(getattr(module, "halting_model", None) is not None),
    }
    recurrent_layer_norm = getattr(module, "recurrent_layer_norm_position", None)
    if recurrent_layer_norm is not None:
        recurrent["layerNorm"] = _display_value(recurrent_layer_norm)
    return {
        "recurrent": recurrent,
    }


def _causal_attention_details(module: _GraphModule) -> dict[str, Any]:
    causal = getattr(module, "causal_attention_mask_flag", None)
    if causal is None:
        return {}
    return {"causalAttention": causal}


def module_details(module: _GraphModule) -> dict[str, Any]:
    details: dict[str, Any] = {}

    details.update(_parameter_shape_details(module))
    details.update(_dimension_details(module))
    details.update(_sequence_or_attention_details(module))
    details.update(_expert_details(module))
    details.update(_layer_behavior_details(module))

    cluster = _neuron_cluster_details(module)
    if cluster is not None:
        details["cluster"] = cluster

    terminal_reach = _terminal_reach_details(module)
    if terminal_reach is not None:
        details["terminalReach"] = terminal_reach

    details.update(_recurrent_details(module, cluster))
    details.update(_causal_attention_details(module))

    return details


def graph_role(module: _GraphModule) -> GraphRole:
    type_name = type(module).__name__
    if type_name in INTERNAL_GRAPH_TYPE_NAMES:
        return INTERNAL_ROLE
    if type_name in RUNTIME_GRAPH_TYPE_NAMES:
        return RUNTIME_ROLE
    if type(module).__module__.startswith("torchmetrics."):
        return RUNTIME_ROLE
    return ARCHITECTURE_ROLE


def _unique_registered_parameters(module: _GraphModule):
    seen_parameter_ids: set[int] = set()
    for _name, parameter in module.named_parameters(
        recurse=True,
        remove_duplicate=False,
    ):
        parameter_id = id(parameter)
        if parameter_id in seen_parameter_ids:
            continue
        seen_parameter_ids.add(parameter_id)
        yield parameter


def parameter_count(module: _GraphModule) -> int:
    count = 0
    for parameter in _unique_registered_parameters(module):
        if is_lazy(parameter):
            continue
        count += parameter.numel()
    return count


def parameter_size_bytes(module: _GraphModule) -> int:
    size = 0
    for parameter in _unique_registered_parameters(module):
        if is_lazy(parameter):
            continue
        size += parameter.numel() * parameter.element_size()
    return size


def _snake_case_key(key: str) -> str:
    first_pass = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", key)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first_pass).lower()


def _semantic_detail_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            _snake_case_key(str(key)): _semantic_detail_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_semantic_detail_value(item) for item in value]
    return value


def _node(node_id: str, path: str, module: _GraphModule) -> GraphNode:
    type_name = type(module).__name__
    description = _component_description(module)
    return GraphNode(
        id=node_id,
        type_name=type_name,
        description=description,
        path=path,
        graph_role=graph_role(module),
        parameter_count=parameter_count(module),
        parameter_size_bytes=parameter_size_bytes(module),
        details=_semantic_detail_value(module_details(module)),
        configuration=_module_config(module),
    )


def _child_path(parent_path: str, child_name: str) -> str:
    return child_name if not parent_path else f"{parent_path}.{child_name}"


def _is_transparent_graph_container(
    parent: _GraphModule,
    child_name: str,
    child: _GraphModule,
) -> bool:
    return (
        type(parent).__name__ == "LayerStack"
        and child_name == "layers"
        and type(child).__name__ == "ModuleList"
    )


def inspect_model_graph(module: _GraphModule) -> ModelGraph:
    nodes = [_node(ROOT_NODE_ID, ROOT_NODE_PATH, module)]
    edges: list[GraphEdge] = []

    def visit(parent: _GraphModule, parent_id: str, parent_path: str) -> None:
        for child_name, child in parent.named_children():
            child_path = _child_path(parent_path, child_name)
            if _is_transparent_graph_container(parent, child_name, child):
                visit_transparent_container(child, parent_id, child_path)
                continue
            child_id = child_path
            nodes.append(_node(child_id, child_path, child))
            edges.append(
                GraphEdge(
                    id=f"{parent_id}-{child_id}",
                    source=parent_id,
                    target=child_id,
                )
            )
            visit(child, child_id, child_path)

    def visit_transparent_container(
        container: _GraphModule,
        visible_parent_id: str,
        container_path: str,
    ) -> None:
        for child_name, child in container.named_children():
            child_path = _child_path(container_path, child_name)
            child_id = child_path
            nodes.append(_node(child_id, child_path, child))
            edges.append(
                GraphEdge(
                    id=f"{visible_parent_id}-{child_id}",
                    source=visible_parent_id,
                    target=child_id,
                )
            )
            visit(child, child_id, child_path)

    visit(module, ROOT_NODE_ID, "")
    return ModelGraph(nodes=tuple(nodes), edges=tuple(edges))


__all__ = [
    "ARCHITECTURE_ROLE",
    "INTERNAL_ROLE",
    "ROOT_NODE_ID",
    "ROOT_NODE_PATH",
    "RUNTIME_ROLE",
    "graph_role",
    "inspect_model_graph",
    "module_details",
    "parameter_count",
    "parameter_size_bytes",
]
