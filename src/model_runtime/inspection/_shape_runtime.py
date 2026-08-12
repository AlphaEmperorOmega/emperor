from __future__ import annotations

import sys
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, cast

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


class ShapeTraceRuntime:
    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._previous_trace = sys.gettrace()
        self._training_states = [
            (module, module.training) for module in model.modules()
        ]
        self.handles: list[RemovableHandle] = []

    def execute(
        self,
        inputs: tuple[object, ...],
        variable_tracer: object | None,
    ) -> None:
        self._model.eval()
        if variable_tracer is not None:
            cast(Any, sys).settrace(variable_tracer)
        with torch.no_grad():
            self._model(*inputs)

    def restore(self) -> None:
        sys.settrace(self._previous_trace)
        for handle in self.handles:
            handle.remove()
        for module, was_training in self._training_states:
            module.training = was_training


@contextmanager
def shape_trace_runtime(model: nn.Module) -> Generator[ShapeTraceRuntime]:
    runtime = ShapeTraceRuntime(model)
    try:
        yield runtime
    finally:
        runtime.restore()


__all__ = ["ShapeTraceRuntime", "shape_trace_runtime"]
