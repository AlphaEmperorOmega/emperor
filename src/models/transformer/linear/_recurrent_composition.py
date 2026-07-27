from __future__ import annotations

from dataclasses import fields

from emperor.config import ConfigBase
from emperor.halting import HaltingConfig
from emperor.layers import (
    GateConfig,
    LayerNormPositionOptions,
    RecurrentCompositionConfig,
    ResidualConfig,
)
from emperor.memory import DynamicMemoryConfig


def build_recurrent_composition(
    *,
    option: type[RecurrentCompositionConfig],
    input_dim: int,
    output_dim: int,
    block_config: ConfigBase,
    max_steps: int,
    recurrent_layer_norm_position: LayerNormPositionOptions,
    gate_config: GateConfig | None,
    residual_config: ResidualConfig | None,
    halting_config: HaltingConfig | None,
    memory_config: DynamicMemoryConfig | None,
    no_gradient_transition_count: int | None,
    reinject_original_hidden_flag: bool,
    latent_updates_per_answer_update: int,
    answer_update_count: int,
    high_cycles: int,
    low_cycles: int,
    initialization_standard_deviation: float,
) -> RecurrentCompositionConfig:
    """Build the selected recurrent config leaf from package-local options."""

    if not isinstance(option, type) or not issubclass(
        option, RecurrentCompositionConfig
    ):
        raise TypeError(
            "recurrent_composition_option must be a concrete "
            "RecurrentCompositionConfig subclass"
        )

    candidate_values = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "block_config": block_config,
        "high_block_config": block_config,
        "low_block_config": block_config,
        "max_steps": max_steps,
        "recurrent_layer_norm_position": recurrent_layer_norm_position,
        "gate_config": gate_config,
        "residual_config": residual_config,
        "halting_config": halting_config,
        "memory_config": memory_config,
        "no_gradient_transition_count": no_gradient_transition_count,
        "reinject_original_hidden_flag": reinject_original_hidden_flag,
        "latent_updates_per_answer_update": latent_updates_per_answer_update,
        "answer_update_count": answer_update_count,
        "high_cycles": high_cycles,
        "low_cycles": low_cycles,
        "initialization_standard_deviation": initialization_standard_deviation,
    }
    option_fields = {field.name for field in fields(option)}
    recurrent_config = option(
        **{
            name: value
            for name, value in candidate_values.items()
            if name in option_fields
        }
    )
    recurrent_config.registry_owner()
    return recurrent_config


__all__ = ["build_recurrent_composition"]
