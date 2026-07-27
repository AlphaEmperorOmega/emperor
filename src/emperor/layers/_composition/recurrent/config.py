from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from emperor.config import ConfigBase, optional_field
from emperor.layers._options import LayerNormPositionOptions

if TYPE_CHECKING:
    from collections.abc import Callable

    from emperor.halting import HaltingConfig
    from emperor.layers._composition.residual.config import ResidualConfig
    from emperor.layers._config import GateConfig
    from emperor.memory import DynamicMemoryConfig


@dataclass
class RecurrentCompositionConfig(ConfigBase):
    _TRANSITION_CONFIG_FIELDS: ClassVar[tuple[str, ...]] = ()

    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    no_gradient_transition_count: int | None = optional_field(
        "Optional number of initial transition invocations executed without "
        "gradient tracking. Every subsequently executed transition uses gradients."
    )
    recurrent_layer_norm_position: LayerNormPositionOptions | None = optional_field(
        "Where layer normalization is applied within each recurrent transition. "
        "Set to None to disable."
    )
    gate_config: GateConfig | None = optional_field(
        "Optional recurrent gate config. Set to None to disable."
    )
    residual_config: ResidualConfig | None = optional_field(
        "Optional residual connection config applied between recurrent transitions. "
        "Set to None to disable recurrent residuals."
    )
    halting_config: HaltingConfig | None = optional_field(
        "Optional recurrent adaptive computation module. Set to None to disable."
    )
    memory_config: DynamicMemoryConfig | None = optional_field(
        "Optional dynamic memory module applied around recurrent transition blocks. "
        "Set to None to disable memory."
    )

    def _registry_owner(self) -> type:
        raise ValueError(
            "RecurrentCompositionConfig is abstract and has no registered recurrent "
            "composition; instantiate a concrete recurrent config instead."
        )

    def _transition_configs(self) -> tuple[ConfigBase, ...]:
        return tuple(
            transition_config
            for _, transition_config in self._transition_config_items()
        )

    def _transition_config_items(self) -> tuple[tuple[str, ConfigBase], ...]:
        return tuple(
            (field_name, transition_config)
            for field_name in self._TRANSITION_CONFIG_FIELDS
            if (transition_config := getattr(self, field_name)) is not None
        )

    def _missing_transition_config_fields(self) -> tuple[str, ...]:
        return tuple(
            field_name
            for field_name in self._TRANSITION_CONFIG_FIELDS
            if getattr(self, field_name) is None
        )

    def _map_transition_configs(
        self,
        transform: Callable[[ConfigBase], ConfigBase],
    ) -> None:
        for field_name in self._TRANSITION_CONFIG_FIELDS:
            transition_config = getattr(self, field_name)
            if transition_config is not None:
                setattr(self, field_name, transform(transition_config))


@dataclass
class RecurrentLayerConfig(RecurrentCompositionConfig):
    _TRANSITION_CONFIG_FIELDS: ClassVar[tuple[str, ...]] = ("block_config",)

    max_steps: int | None = optional_field("Maximum recurrent applications.")
    reinject_original_hidden_flag: bool | None = optional_field(
        "Add the original input hidden tensor to the evolving hidden tensor before "
        "every recurrent block invocation."
    )

    block_config: ConfigBase | None = optional_field(
        "ConfigBase block reused at every recurrent step. The built module must "
        "consume and return LayerState-compatible values and declare input_dim and "
        "output_dim fields."
    )

    def _registry_owner(self) -> type:
        from emperor.layers._composition.recurrent.variants.standard import (
            RecurrentLayer,
        )

        return RecurrentLayer


@dataclass
class TinyRecursiveModelRecurrentConfig(RecurrentCompositionConfig):
    _TRANSITION_CONFIG_FIELDS: ClassVar[tuple[str, ...]] = ("block_config",)

    block_config: ConfigBase | None = optional_field(
        "ConfigBase transition block shared by latent and answer updates."
    )
    latent_updates_per_answer_update: int | None = optional_field(
        "Number of latent-state transition updates performed before each "
        "answer-state update."
    )
    answer_update_count: int | None = optional_field(
        "Number of answer-state updates performed in one recurrent forward."
    )
    initialization_standard_deviation: float | None = optional_field(
        "Standard deviation for persistent answer and latent initialization buffers."
    )

    def _registry_owner(self) -> type:
        from emperor.layers._composition.recurrent.variants.tiny_recursive_model import (
            TinyRecursiveModelRecurrent,
        )

        return TinyRecursiveModelRecurrent


@dataclass
class HierarchicalReasoningModelRecurrentConfig(RecurrentCompositionConfig):
    _TRANSITION_CONFIG_FIELDS: ClassVar[tuple[str, ...]] = (
        "high_block_config",
        "low_block_config",
    )

    high_block_config: ConfigBase | None = optional_field(
        "ConfigBase transition block used for high-level state updates."
    )
    low_block_config: ConfigBase | None = optional_field(
        "ConfigBase transition block used for low-level state updates."
    )
    high_cycles: int | None = optional_field(
        "Number of high-level clock cycles in one recurrent forward."
    )
    low_cycles: int | None = optional_field(
        "Number of low-level updates in each high-level clock cycle."
    )
    initialization_standard_deviation: float | None = optional_field(
        "Standard deviation for persistent high- and low-state initialization buffers."
    )

    def _registry_owner(self) -> type:
        from emperor.layers._composition.recurrent.variants.hierarchical_reasoning_model import (
            HierarchicalReasoningModelRecurrent,
        )

        return HierarchicalReasoningModelRecurrent
