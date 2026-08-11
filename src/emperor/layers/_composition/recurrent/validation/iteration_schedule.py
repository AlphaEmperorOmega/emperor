from __future__ import annotations

import torch
from torch import Tensor

from emperor._validation import ValidatorBase
from emperor.layers._composition.recurrent.config import (
    HierarchicalReasoningModelRecurrentConfig,
    RecurrentLayerConfig,
    TinyRecursiveModelRecurrentConfig,
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
