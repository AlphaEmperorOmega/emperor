"""Package Interface for generalized Recurrent Execution."""

from emperor.layers._composition.recurrent.runtime.execution.executor import (
    RecurrentExecution,
)
from emperor.layers._composition.recurrent.runtime.execution.interface import (
    PreparedRecurrentTransition,
    RecurrentExecutionAdapter,
    RecurrentExecutionResult,
    RecurrentExecutionState,
)

__all__ = [
    "PreparedRecurrentTransition",
    "RecurrentExecution",
    "RecurrentExecutionAdapter",
    "RecurrentExecutionResult",
    "RecurrentExecutionState",
]
