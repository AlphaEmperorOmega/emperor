import torch.nn as nn


def zero_gate_parameters(module: nn.Module) -> None:
    """Initialize a configurable gate to equal logits when it has parameters."""

    for parameter in module.parameters():
        nn.init.zeros_(parameter)
