from collections.abc import Iterable

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.trainers import WordPieceTrainer

from emperor.datasets.text._bert_vocabulary import (
    BERT_SPECIAL_TOKENS,
    BERT_UNK_TOKEN,
)

BERT_PRETRAINING_TARGET_VOCAB_SIZE = 30522


def train_local_wordpiece_tokenizer(
    text_units: Iterable[str],
    vocab_size: int = BERT_PRETRAINING_TARGET_VOCAB_SIZE,
) -> Tokenizer:
    training_units = [unit for unit in _normalise_text_units(text_units)]
    if not training_units:
        training_units = [BERT_UNK_TOKEN]

    tokenizer = Tokenizer(WordPiece(unk_token=BERT_UNK_TOKEN))
    tokenizer.normalizer = BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = BertPreTokenizer()
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=BERT_SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(training_units, trainer=trainer)
    return tokenizer


def _normalise_text_units(text_units: Iterable[str]) -> Iterable[str]:
    for unit in text_units:
        unit = str(unit).strip()
        if unit:
            yield unit
