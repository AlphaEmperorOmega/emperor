from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import torch
from torch import nn

GraphNodePayload: TypeAlias = dict[str, Any]
GraphEdgePayload: TypeAlias = dict[str, str]


class ConfiguredModule(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg


class TinyGraphFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential()
        self.encoder.add_module("linear", nn.Linear(2, 3))
        self.encoder.add_module("relu", nn.ReLU())
        self.head = nn.Linear(3, 1, bias=False)


class SharedParameterBranch(nn.Module):
    def __init__(self, parameter: nn.Parameter) -> None:
        super().__init__()
        self.weight = parameter


class SharedParameterGraph(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared_parameter = nn.Parameter(torch.ones(2, 3))
        self.left = SharedParameterBranch(shared_parameter)
        self.right = SharedParameterBranch(shared_parameter)


@dataclass
class PlainConfig:
    width: int


class DocumentedModule(nn.Module):
    """Documented component description."""


def config_fields(node: GraphNodePayload) -> dict[str, object]:
    config = node["config"]
    if config is None:
        return {}
    return {field["key"]: field["value"] for field in config["fields"]}


def nodes_by_id(nodes: list[GraphNodePayload]) -> dict[str, GraphNodePayload]:
    return {str(node["id"]): node for node in nodes}


__all__ = [
    "ConfiguredModule",
    "DocumentedModule",
    "GraphEdgePayload",
    "GraphNodePayload",
    "PlainConfig",
    "SharedParameterGraph",
    "TinyGraphFixture",
    "config_fields",
    "nodes_by_id",
]
