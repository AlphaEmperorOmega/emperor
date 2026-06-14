from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Literal

from emperor.base.utils import ConfigBase
from torch.nn import Module
from torch.nn.parameter import is_lazy

GraphRole = Literal["architecture", "internal", "runtime"]

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


def _module_config(module: Module) -> dict[str, Any] | None:
    config = getattr(module, "_emperor_config", None)
    if config is None:
        config = getattr(module, "cfg", None)
    if config is None or isinstance(config, type) or not is_dataclass(config):
        return None

    return {
        "typeName": type(config).__name__,
        "fields": [
            {
                "key": field.name,
                "value": _config_field_value(getattr(config, field.name)),
            }
            for field in fields(config)
        ],
    }


def _shape_value(value: Any) -> str | None:
    if is_lazy(value):
        return None
    dimensions = tuple(value.shape)
    if not dimensions:
        return "scalar"
    return " x ".join(str(dimension) for dimension in dimensions)


def _bool_from_optional_model(module: Module, attr_name: str) -> bool | None:
    if not hasattr(module, attr_name):
        return None
    return getattr(module, attr_name) is not None


def _first_detail_value(module: Module, attr_paths: tuple[str, ...]) -> Any:
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


def _neuron_cluster_details(module: Module) -> dict[str, Any] | None:
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


def _terminal_reach_details(module: Module) -> dict[str, Any] | None:
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


def _parameter_shape_details(module: Module) -> dict[str, Any]:
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


def _dimension_details(module: Module) -> dict[str, Any]:
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


def _sequence_or_attention_details(module: Module) -> dict[str, Any]:
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


def _expert_details(module: Module) -> dict[str, Any]:
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


def _layer_behavior_details(module: Module) -> dict[str, Any]:
    details: dict[str, Any] = {}
    dropout = getattr(module, "dropout_probability", None)
    if dropout is not None:
        details["dropout"] = dropout

    gate = _bool_from_optional_model(module, "gate_model")
    if gate is not None:
        details["gate"] = gate

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
    module: Module,
    cluster: dict[str, Any] | None,
) -> dict[str, Any]:
    max_steps = getattr(module, "max_steps", None)
    if max_steps is None or cluster is not None:
        return {}
    return {
        "recurrent": {
            "maxSteps": max_steps,
            "gate": bool(getattr(module, "gate_model", None) is not None),
            "halting": bool(getattr(module, "halting_model", None) is not None),
        }
    }


def _causal_attention_details(module: Module) -> dict[str, Any]:
    causal = getattr(module, "causal_attention_mask_flag", None)
    if causal is None:
        return {}
    return {"causalAttention": causal}


def module_details(module: Module) -> dict[str, Any]:
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


def graph_role(module: Module) -> GraphRole:
    type_name = type(module).__name__
    if type_name in INTERNAL_GRAPH_TYPE_NAMES:
        return INTERNAL_ROLE
    if type_name in RUNTIME_GRAPH_TYPE_NAMES:
        return RUNTIME_ROLE
    if type(module).__module__.startswith("torchmetrics."):
        return RUNTIME_ROLE
    return ARCHITECTURE_ROLE


def _unique_registered_parameters(module: Module):
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


def parameter_count(module: Module) -> int:
    count = 0
    for parameter in _unique_registered_parameters(module):
        if is_lazy(parameter):
            continue
        count += parameter.numel()
    return count


def parameter_size_bytes(module: Module) -> int:
    size = 0
    for parameter in _unique_registered_parameters(module):
        if is_lazy(parameter):
            continue
        size += parameter.numel() * parameter.element_size()
    return size


def _node(node_id: str, path: str, module: Module) -> dict[str, Any]:
    type_name = type(module).__name__
    return {
        "id": node_id,
        "label": type_name,
        "typeName": type_name,
        "path": path,
        "graphRole": graph_role(module),
        "parameterCount": parameter_count(module),
        "parameterSizeBytes": parameter_size_bytes(module),
        "details": module_details(module),
        "config": _module_config(module),
    }


def serialize_graph(
    module: Module,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    nodes = [_node(ROOT_NODE_ID, ROOT_NODE_PATH, module)]
    edges: list[dict[str, str]] = []

    def visit(parent: Module, parent_id: str, parent_path: str) -> None:
        for child_name, child in parent.named_children():
            child_path = (
                child_name if not parent_path else f"{parent_path}.{child_name}"
            )
            child_id = child_path
            nodes.append(_node(child_id, child_path, child))
            edges.append(
                {
                    "id": f"{parent_id}-{child_id}",
                    "source": parent_id,
                    "target": child_id,
                }
            )
            visit(child, child_id, child_path)

    visit(module, ROOT_NODE_ID, "")
    return nodes, edges
