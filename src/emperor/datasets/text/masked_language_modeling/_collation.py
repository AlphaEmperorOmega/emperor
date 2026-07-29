from collections.abc import Sequence

import torch
from torch import Tensor

from emperor.datasets.text._bert_vocabulary import BertSpecialTokenIds
from emperor.datasets.text._masked_token_policy import _MaskedTokenPolicy


class MaskedLanguageModelingCollator:
    def __init__(
        self,
        special_token_ids: BertSpecialTokenIds,
        vocab_size: int,
        mlm_probability: float = 0.15,
        mask_replace_probability: float = 0.8,
        random_replace_probability: float = 0.1,
        ignore_index: int = -100,
        generator: torch.Generator | None = None,
    ) -> None:
        self.masked_token_policy = _MaskedTokenPolicy(
            special_token_ids=special_token_ids,
            vocab_size=vocab_size,
            mlm_probability=mlm_probability,
            mask_replace_probability=mask_replace_probability,
            random_replace_probability=random_replace_probability,
            ignore_index=ignore_index,
            generator=generator,
        )

    def __call__(self, batch) -> tuple[Tensor, Tensor, Tensor]:
        return self.masked_token_policy(self._stack_batch(batch))

    def _stack_batch(self, batch) -> Tensor:
        if isinstance(batch, Tensor):
            token_ids = batch
        else:
            if len(batch) == 0:
                raise ValueError("MaskedLanguageModelingCollator received no samples.")
            samples = []
            for sample in batch:
                if isinstance(sample, Tensor):
                    samples.append(sample)
                elif (
                    isinstance(sample, Sequence)
                    and len(sample) == 1
                    and isinstance(sample[0], Tensor)
                ):
                    samples.append(sample[0])
                else:
                    raise TypeError(
                        "MaskedLanguageModelingCollator expects tensors or "
                        "single-tensor samples."
                    )
            token_ids = torch.stack(samples)

        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.dim() != 2:
            raise ValueError(
                "MaskedLanguageModelingCollator expects a 2D token tensor."
            )
        return token_ids.long()
