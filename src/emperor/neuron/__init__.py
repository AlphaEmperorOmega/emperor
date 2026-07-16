"""Public Interface for neuron-cluster modules."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emperor.neuron._cluster.model import NeuronCluster
    from emperor.neuron._config import (
        AxonsConfig,
        NeuronClusterConfig,
        NeuronConfig,
        NucleusConfig,
        TerminalConfig,
    )
    from emperor.neuron._monitoring.callback import NeuronClusterMonitorCallback
    from emperor.neuron._optimizer_sync import NeuronClusterOptimizerSyncCallback
    from emperor.neuron._options import (
        TerminalConnectionShapeOptions,
        TerminalRangeOptions,
        TerminalZAxisOffsetOptions,
    )
    from emperor.neuron._parts import Axons, Neuron, Nucleus, Terminal
    from emperor.neuron._trace import NeuronClusterTrace, NeuronClusterTraceStep

__all__ = (
    "Axons",
    "AxonsConfig",
    "Neuron",
    "NeuronCluster",
    "NeuronClusterConfig",
    "NeuronClusterMonitorCallback",
    "NeuronClusterOptimizerSyncCallback",
    "NeuronClusterTrace",
    "NeuronClusterTraceStep",
    "NeuronConfig",
    "Nucleus",
    "NucleusConfig",
    "Terminal",
    "TerminalConfig",
    "TerminalConnectionShapeOptions",
    "TerminalRangeOptions",
    "TerminalZAxisOffsetOptions",
)

_LAZY_EXPORTS = {
    "Axons": ("emperor.neuron._parts", "Axons"),
    "AxonsConfig": ("emperor.neuron._config", "AxonsConfig"),
    "Neuron": ("emperor.neuron._parts", "Neuron"),
    "NeuronCluster": ("emperor.neuron._cluster.model", "NeuronCluster"),
    "NeuronClusterConfig": ("emperor.neuron._config", "NeuronClusterConfig"),
    "NeuronClusterMonitorCallback": (
        "emperor.neuron._monitoring.callback",
        "NeuronClusterMonitorCallback",
    ),
    "NeuronClusterOptimizerSyncCallback": (
        "emperor.neuron._optimizer_sync",
        "NeuronClusterOptimizerSyncCallback",
    ),
    "NeuronClusterTrace": ("emperor.neuron._trace", "NeuronClusterTrace"),
    "NeuronClusterTraceStep": ("emperor.neuron._trace", "NeuronClusterTraceStep"),
    "NeuronConfig": ("emperor.neuron._config", "NeuronConfig"),
    "Nucleus": ("emperor.neuron._parts", "Nucleus"),
    "NucleusConfig": ("emperor.neuron._config", "NucleusConfig"),
    "Terminal": ("emperor.neuron._parts", "Terminal"),
    "TerminalConfig": ("emperor.neuron._config", "TerminalConfig"),
    "TerminalConnectionShapeOptions": (
        "emperor.neuron._options",
        "TerminalConnectionShapeOptions",
    ),
    "TerminalRangeOptions": (
        "emperor.neuron._options",
        "TerminalRangeOptions",
    ),
    "TerminalZAxisOffsetOptions": (
        "emperor.neuron._options",
        "TerminalZAxisOffsetOptions",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from error

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
