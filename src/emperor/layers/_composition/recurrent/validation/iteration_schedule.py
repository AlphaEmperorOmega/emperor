from __future__ import annotations

import torch
from torch import Tensor

from emperor._validation import ValidatorBase
from emperor.layers._composition.recurrent.config import (
    HierarchicalReasoningModelRecurrentConfig,
    RecurrentLayerConfig,
    TinyRecursiveModelRecurrentConfig,
)
from emperor.layers._composition.recurrent.validation.common import (
    _validate_smooth_iteration_growth_controls,
)


class RecurrentIterationScheduleValidator(ValidatorBase):
    @staticmethod
    def validate_config(config: object) -> None:
        concrete_config_types = (
            RecurrentLayerConfig,
            TinyRecursiveModelRecurrentConfig,
            HierarchicalReasoningModelRecurrentConfig,
        )
        if not isinstance(config, concrete_config_types):
            raise TypeError(
                "config must be a concrete RecurrentCompositionConfig, got "
                f"{type(config).__name__}."
            )
        if isinstance(config, RecurrentLayerConfig):
            transitions_per_iteration = 1
        elif isinstance(config, TinyRecursiveModelRecurrentConfig):
            transitions_per_iteration = config.latent_updates_per_answer_update + 1
        else:
            transitions_per_iteration = config.low_cycles + 1
        _validate_smooth_iteration_growth_controls(
            config,
            transitions_per_iteration=transitions_per_iteration,
        )

    @staticmethod
    def validate_checkpoint_progress(
        value: object,
        *,
        checkpoint_key: str,
    ) -> int:
        if not isinstance(value, Tensor):
            raise TypeError(
                f"{checkpoint_key} must be a Tensor in a recurrent checkpoint."
            )
        if value.shape != torch.Size([]):
            raise ValueError(f"{checkpoint_key} must be a scalar Tensor.")
        if value.dtype != torch.long:
            raise TypeError(f"{checkpoint_key} must use torch.long dtype.")
        progress = int(value.item())
        if progress < 0:
            raise ValueError(f"{checkpoint_key} must be non-negative.")
        return progress
