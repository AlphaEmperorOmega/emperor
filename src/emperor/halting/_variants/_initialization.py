import torch.nn as nn


def zero_gate_parameters(module: nn.Module) -> None:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
