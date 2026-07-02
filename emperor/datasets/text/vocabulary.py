from dataclasses import dataclass
from collections.abc import Mapping


BERT_PAD_TOKEN = "[PAD]"
BERT_UNK_TOKEN = "[UNK]"
BERT_CLS_TOKEN = "[CLS]"
BERT_SEP_TOKEN = "[SEP]"
BERT_MASK_TOKEN = "[MASK]"

BERT_SPECIAL_TOKENS = [
    BERT_PAD_TOKEN,
    BERT_UNK_TOKEN,
    BERT_CLS_TOKEN,
    BERT_SEP_TOKEN,
    BERT_MASK_TOKEN,
]


@dataclass(frozen=True)
class BertSpecialTokenIds:
    pad: int
    unk: int
    cls: int
    sep: int
    mask: int

    def values(self) -> tuple[int, int, int, int, int]:
        return self.pad, self.unk, self.cls, self.sep, self.mask


def get_bert_special_token_ids(vocab) -> BertSpecialTokenIds:
    token_to_index = _token_to_index_mapping(vocab)
    missing = [token for token in BERT_SPECIAL_TOKENS if token not in token_to_index]
    if missing:
        raise KeyError(f"Missing BERT special tokens in vocabulary: {missing}")
    return BertSpecialTokenIds(
        pad=int(token_to_index[BERT_PAD_TOKEN]),
        unk=int(token_to_index[BERT_UNK_TOKEN]),
        cls=int(token_to_index[BERT_CLS_TOKEN]),
        sep=int(token_to_index[BERT_SEP_TOKEN]),
        mask=int(token_to_index[BERT_MASK_TOKEN]),
    )


def set_bert_default_index(vocab) -> BertSpecialTokenIds:
    token_ids = get_bert_special_token_ids(vocab)
    vocab.set_default_index(token_ids.unk)
    return token_ids


def _token_to_index_mapping(vocab) -> Mapping[str, int]:
    if hasattr(vocab, "get_stoi"):
        return vocab.get_stoi()
    if hasattr(vocab, "get_vocab"):
        return vocab.get_vocab()
    return vocab
