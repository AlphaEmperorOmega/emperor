import random
from collections.abc import Iterable
from dataclasses import dataclass

from emperor.datasets.text.bert_pretraining._tokenizer import (
    _normalise_text_units,
)


@dataclass(frozen=True)
class BertNextSentencePair:
    sentence_a: str
    sentence_b: str
    next_sentence_label: int
    sentence_a_index: int
    sentence_b_index: int


def build_bert_next_sentence_pairs(
    text_units: Iterable[str],
    random_next_probability: float = 0.5,
    rng: random.Random | None = None,
) -> list[BertNextSentencePair]:
    if random_next_probability < 0.0 or random_next_probability > 1.0:
        raise ValueError("random_next_probability must be between 0.0 and 1.0.")

    units = list(_normalise_text_units(text_units))
    if len(units) < 2:
        return []

    rng = rng or random.Random()
    pairs = []
    for sentence_a_index in range(len(units) - 1):
        true_next_index = sentence_a_index + 1
        random_next_candidates = [
            index
            for index in range(len(units))
            if index != sentence_a_index and index != true_next_index
        ]
        use_random_next = (
            bool(random_next_candidates) and rng.random() < random_next_probability
        )
        if use_random_next:
            sentence_b_index = rng.choice(random_next_candidates)
            next_sentence_label = 1
        else:
            sentence_b_index = true_next_index
            next_sentence_label = 0

        pairs.append(
            BertNextSentencePair(
                sentence_a=units[sentence_a_index],
                sentence_b=units[sentence_b_index],
                next_sentence_label=next_sentence_label,
                sentence_a_index=sentence_a_index,
                sentence_b_index=sentence_b_index,
            )
        )
    return pairs
