from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class NeuronClusterTraceStep:
    probabilities: Tensor
    selected_coordinates: Tensor
    valid_mask: Tensor
    escape_mask: Tensor
    chosen_branch_indices: Tensor
    halt_mask: Tensor
    active_mask: Tensor


@dataclass
class NeuronClusterTrace:
    input_shape: tuple[int, ...]
    entry_coordinates: Tensor
    entry_probabilities: Tensor
    entry_selected_coordinates: Tensor
    entry_valid_mask: Tensor
    entry_escape_mask: Tensor
    entry_chosen_branch_indices: Tensor
    entry_halt_mask: Tensor
    entry_active_mask: Tensor
    steps: list[NeuronClusterTraceStep] = field(default_factory=list)
