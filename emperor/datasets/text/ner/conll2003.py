import torch
import torch.utils.data

from datasets import load_dataset

from emperor.base.utils import DataModule


# CoNLL-2003 NER tag set (BIO scheme)
_NER_TAGS = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC",
]


class _NERDataset(torch.utils.data.Dataset):
    def __init__(self, samples, sequence_length):
        self.samples = samples
        self.sequence_length = sequence_length
        # build word vocab from token lists
        all_tokens = [tok for sample in samples for tok in sample["tokens"]]
        unique_tokens = sorted(set(all_tokens))
        self.token_to_idx = {"<pad>": 0, "<unk>": 1}
        self.token_to_idx.update({tok: idx + 2 for idx, tok in enumerate(unique_tokens)})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = [self.token_to_idx.get(t, 1) for t in sample["tokens"]]
        labels = sample["ner_tags"]
        # truncate or pad to sequence_length
        tokens = tokens[:self.sequence_length] + [0] * max(0, self.sequence_length - len(tokens))
        labels = labels[:self.sequence_length] + [0] * max(0, self.sequence_length - len(labels))
        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


class CoNLL2003(DataModule):
    vocab_size: int = 23624  # approximate CoNLL-2003 token vocab size
    num_classes: int = 9  # O + 4 entity types x B/I
    sequence_length: int = 128

    def __init__(
        self,
        batch_size: int = 32,
        sequence_length: int = 128,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def prepare_data(self) -> None:
        load_dataset("conll2003", split="train")
        load_dataset("conll2003", split="validation")

    def _setup_fit(self) -> None:
        train_data = list(load_dataset("conll2003", split="train"))
        val_data = list(load_dataset("conll2003", split="validation"))
        self.train = _NERDataset(train_data, self.sequence_length)
        self.val = _NERDataset(val_data, self.sequence_length)

    def _setup_validate(self) -> None:
        val_data = list(load_dataset("conll2003", split="validation"))
        self.val = _NERDataset(val_data, self.sequence_length)

    def get_dataloader(self, train: bool):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def _text_labels(self, indices) -> list:
        return [_NER_TAGS[int(i)] for i in indices]
