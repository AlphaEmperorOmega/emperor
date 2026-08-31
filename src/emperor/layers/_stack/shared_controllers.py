from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from emperor.layers._stack.validation import LayerStackValidator
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.halting import HaltingConfig, HaltingInterface, HaltingStateBase
    from emperor.layers._composition.gate import LayerGate
    from emperor.layers._config import GateConfig, LayerStackConfig
    from emperor.memory import DynamicMemoryConfig, MemoryInterface


class _SharedControllerLayer(Protocol):
    def _bind_shared_gate(self, config: GateConfig, model: LayerGate) -> None: ...

    def _bind_shared_halting(
        self,
        model: HaltingInterface[HaltingStateBase],
    ) -> None: ...

    def _bind_shared_memory(self, model: MemoryInterface) -> None: ...


class LayerStackSharedControllers(Module):
    """Own stack-wide controller construction, validation, and binding."""

    VALIDATOR = LayerStackValidator

    def __init__(
        self,
        cfg: LayerStackConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.__initialize_from_config()

    def __initialize_from_config(self) -> None:
        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.gate_config: GateConfig | None = self.cfg.shared_gate_config
        self.halting_config: HaltingConfig | None = self.cfg.shared_halting_config
        self.memory_config: DynamicMemoryConfig | None = self.cfg.shared_memory_config

    def bind(self, stack_layers: Sequence[_SharedControllerLayer]) -> None:
        self.__maybe_bind_gate(stack_layers)
        self.__maybe_bind_halting(stack_layers)
        self.__maybe_bind_memory(stack_layers)

    def __maybe_bind_gate(
        self,
        stack_layers: Sequence[_SharedControllerLayer],
    ) -> None:
        config = self.gate_config
        if config is None:
            return
        shared_model = self.VALIDATOR.validate_shared_gate_model(
            self._build_from_config(
                config,
                gate_dim=self.output_dim,
            )
        )
        for stack_layer in stack_layers:
            stack_layer._bind_shared_gate(  # pyright: ignore[reportPrivateUsage]
                config,
                shared_model,
            )

    def __maybe_bind_halting(
        self,
        stack_layers: Sequence[_SharedControllerLayer],
    ) -> None:
        config = self.halting_config
        if config is None:
            return
        shared_model = self.VALIDATOR.validate_shared_halting_model(
            self._build_from_config(
                config,
                input_dim=self.output_dim,
            )
        )
        for stack_layer in stack_layers:
            stack_layer._bind_shared_halting(  # pyright: ignore[reportPrivateUsage]
                shared_model
            )

    def __maybe_bind_memory(
        self,
        stack_layers: Sequence[_SharedControllerLayer],
    ) -> None:
        config = self.memory_config
        if config is None:
            return
        shared_model = self.VALIDATOR.validate_shared_memory_model(
            self._build_from_config(
                config,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
            )
        )
        for stack_layer in stack_layers:
            stack_layer._bind_shared_memory(  # pyright: ignore[reportPrivateUsage]
                shared_model
            )
