from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class Utf8BitEncoder(nn.Module):
    """Parameter-free, uncached UTF-8 byte-to-bit conversion."""

    def __init__(self, max_token_bytes: int) -> None:
        super().__init__()
        if type(max_token_bytes) is not int:
            raise TypeError(
                "max_token_bytes must be int for Utf8BitEncoder, "
                f"got {type(max_token_bytes).__name__}"
            )
        if max_token_bytes <= 0:
            raise ValueError(
                "max_token_bytes must be greater than 0 for Utf8BitEncoder, "
                f"received {max_token_bytes}"
            )
        self.max_token_bytes = max_token_bytes

    def forward(
        self,
        token_texts: Sequence[Sequence[str]],
        *,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        byte_rows: list[list[list[int]]] = []
        mask_rows: list[list[list[bool]]] = []
        for batch_index, row in enumerate(token_texts):
            encoded_row: list[list[int]] = []
            mask_row: list[list[bool]] = []
            for token_index, token_text in enumerate(row):
                try:
                    retained_bytes = token_text.encode("utf-8")[: self.max_token_bytes]
                except UnicodeEncodeError as error:
                    raise ValueError(
                        f"token_texts[{batch_index}][{token_index}] cannot be "
                        "encoded as UTF-8"
                    ) from error
                retained_count = len(retained_bytes)
                padding_count = self.max_token_bytes - retained_count
                encoded_row.append([*retained_bytes, *([0] * padding_count)])
                mask_row.append(
                    [*([True] * retained_count), *([False] * padding_count)]
                )
            byte_rows.append(encoded_row)
            mask_rows.append(mask_row)

        byte_values = torch.tensor(byte_rows, dtype=torch.uint8, device=device)
        byte_mask = torch.tensor(mask_rows, device=device)
        shifts = torch.arange(7, -1, -1, device=device)
        bits_by_byte = (
            byte_values.unsqueeze(-1).bitwise_right_shift(shifts).bitwise_and(1)
        )
        batch_size, sequence_length, _ = byte_values.shape
        bits = bits_by_byte.bool().reshape(
            batch_size,
            sequence_length,
            self.max_token_bytes * 8,
        )
        return bits, byte_mask
