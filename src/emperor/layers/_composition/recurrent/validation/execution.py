"""Validation for shared recurrent execution."""

from torch import nn

from emperor._validation import ValidatorBase


class RecurrentExecutionValidator(ValidatorBase):
    """Validate requirements imposed by shared recurrent execution."""

    @staticmethod
    def validate_adapter_is_module(adapter: object) -> None:
        """Ensure the Adapter exposes PyTorch Module runtime state."""
        if not isinstance(adapter, nn.Module):
            raise TypeError("Recurrent Execution Adapter must be an nn.Module.")
