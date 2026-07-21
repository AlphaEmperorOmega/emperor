from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.hooks import RemovableHandle


@dataclass(frozen=True)
class _MethodReplacement:
    owner: object
    method_name: str
    original_method: Callable[..., object]

    def restore(self) -> None:
        setattr(self.owner, self.method_name, self.original_method)


def _install_method_replacement(
    replacements: list[_MethodReplacement],
    owner: object,
    method_name: str,
    original_method: Callable[..., object],
    replacement_method: Callable[..., object],
) -> None:
    setattr(owner, method_name, replacement_method)
    replacements.append(_MethodReplacement(owner, method_name, original_method))


def _restore_method_replacements(
    replacements: list[_MethodReplacement],
) -> None:
    for replacement in reversed(replacements):
        replacement.restore()
    replacements.clear()


def _remove_hooks(hook_handles: list[RemovableHandle]) -> None:
    for hook_handle in hook_handles:
        hook_handle.remove()
    hook_handles.clear()


def _extract_hidden_tensor(output: object) -> Tensor | None:
    if torch.is_tensor(output):
        return output
    hidden = getattr(output, "hidden", None)
    return hidden if torch.is_tensor(hidden) else None
