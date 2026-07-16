import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from tokenizers import Tokenizer
from torch import Tensor

from emperor.datasets.text._bert_vocabulary import BertSpecialTokenIds
from emperor.datasets.text.bert_pretraining._next_sentence import (
    build_bert_next_sentence_pairs,
)


@dataclass(frozen=True)
class BertPretrainingExample:
    input_ids: Tensor
    token_type_ids: Tensor
    next_sentence_label: Tensor


def build_bert_sentence_pair_inputs(
    tokens_a: Sequence[int],
    tokens_b: Sequence[int],
    sequence_length: int,
    special_token_ids: BertSpecialTokenIds,
) -> tuple[Tensor, Tensor]:
    if sequence_length < 5:
        raise ValueError("sequence_length must be at least 5 for BERT pairs.")

    tokens_a = list(tokens_a)
    tokens_b = list(tokens_b)
    if not tokens_a or not tokens_b:
        raise ValueError("Both sentence-pair segments must contain tokens.")

    _truncate_longest_first(tokens_a, tokens_b, sequence_length - 3)

    input_ids = [
        special_token_ids.cls,
        *tokens_a,
        special_token_ids.sep,
        *tokens_b,
        special_token_ids.sep,
    ]
    token_type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
    padding_length = sequence_length - len(input_ids)
    if padding_length < 0:
        raise ValueError("Truncated sentence-pair input is still too long.")

    input_ids.extend([special_token_ids.pad] * padding_length)
    token_type_ids.extend([0] * padding_length)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(
        token_type_ids,
        dtype=torch.long,
    )


def build_bert_pretraining_examples(
    text_units: Iterable[str],
    tokenizer: Tokenizer,
    sequence_length: int,
    special_token_ids: BertSpecialTokenIds,
    random_next_probability: float = 0.5,
    rng: random.Random | None = None,
) -> list[BertPretrainingExample]:
    examples = []
    pairs = build_bert_next_sentence_pairs(
        text_units,
        random_next_probability=random_next_probability,
        rng=rng,
    )
    for pair in pairs:
        tokens_a = tokenizer.encode(pair.sentence_a).ids
        tokens_b = tokenizer.encode(pair.sentence_b).ids
        if not tokens_a or not tokens_b:
            continue
        try:
            input_ids, token_type_ids = build_bert_sentence_pair_inputs(
                tokens_a=tokens_a,
                tokens_b=tokens_b,
                sequence_length=sequence_length,
                special_token_ids=special_token_ids,
            )
        except ValueError:
            continue
        examples.append(
            BertPretrainingExample(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                next_sentence_label=torch.tensor(
                    pair.next_sentence_label,
                    dtype=torch.long,
                ),
            )
        )
    return examples


def _truncate_longest_first(
    tokens_a: list[int],
    tokens_b: list[int],
    max_content_length: int,
) -> None:
    while len(tokens_a) + len(tokens_b) > max_content_length:
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) > 1:
            tokens_a.pop()
        elif len(tokens_b) > 1:
            tokens_b.pop()
        elif len(tokens_a) > 1:
            tokens_a.pop()
        else:
            break
