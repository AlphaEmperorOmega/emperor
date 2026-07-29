from collections.abc import Sequence

import torch
from torch import Tensor

from emperor.datasets.text._bert_vocabulary import BertSpecialTokenIds
from emperor.datasets.text._masked_token_policy import _MaskedTokenPolicy


class BertPretrainingCollator:
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

    def __call__(
        self,
        batch,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        input_ids, token_type_ids, next_sentence_labels = self._stack_batch(batch)
        input_ids, mlm_labels, attention_mask = self.masked_token_policy(input_ids)
        return (
            input_ids,
            mlm_labels,
            attention_mask,
            token_type_ids,
            next_sentence_labels,
        )

    def _stack_batch(self, batch) -> tuple[Tensor, Tensor, Tensor]:
        if len(batch) == 0:
            raise ValueError("BertPretrainingCollator received no samples.")

        input_ids = []
        token_type_ids = []
        next_sentence_labels = []
        for sample in batch:
            if not isinstance(sample, Sequence) or len(sample) != 3:
                raise TypeError(
                    "BertPretrainingCollator expects samples containing "
                    "(input_ids, token_type_ids, next_sentence_label)."
                )
            sample_input_ids, sample_token_type_ids, sample_next_sentence_label = sample
            input_ids.append(torch.as_tensor(sample_input_ids, dtype=torch.long))
            token_type_ids.append(
                torch.as_tensor(sample_token_type_ids, dtype=torch.long)
            )
            next_sentence_labels.append(
                torch.as_tensor(sample_next_sentence_label, dtype=torch.long)
            )

        return (
            torch.stack(input_ids),
            torch.stack(token_type_ids),
            torch.stack(next_sentence_labels).view(-1),
        )
