from collections.abc import Sequence

import torch
from torch import Tensor

from emperor.datasets.text._bert_vocabulary import BertSpecialTokenIds


def build_mlm_token_windows(
    token_ids: Sequence[int] | Tensor,
    sequence_length: int,
    special_token_ids: BertSpecialTokenIds,
    add_special_tokens: bool = True,
) -> Tensor:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if add_special_tokens and sequence_length < 3:
        raise ValueError(
            "sequence_length must be at least 3 when adding [CLS] and [SEP]."
        )

    if isinstance(token_ids, Tensor):
        token_ids = token_ids.tolist()

    content_length = sequence_length - 2 if add_special_tokens else sequence_length
    windows = []
    for start in range(0, len(token_ids), content_length):
        chunk = list(token_ids[start : start + content_length])
        if not chunk:
            continue
        if add_special_tokens:
            window = [special_token_ids.cls, *chunk, special_token_ids.sep]
        else:
            window = chunk
        window.extend([special_token_ids.pad] * (sequence_length - len(window)))
        windows.append(window)

    if not windows:
        return torch.empty((0, sequence_length), dtype=torch.long)
    return torch.tensor(windows, dtype=torch.long)
