import random
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchvision.transforms.transforms import Compose

from emperor.base.utils import DataModule


def _yield_tokens(data_iter, tokenizer):
    for _, captions in data_iter:
        for caption in captions:
            yield tokenizer(caption)


def _encode(caption, tokenizer, vocab, sequence_length):
    tokens = vocab(tokenizer(caption))[:sequence_length]
    padding = [vocab["<pad>"]] * (sequence_length - len(tokens))
    return tokens + padding


class _CaptioningDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, vocab, sequence_length, train):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.sequence_length = sequence_length
        self.train = train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, captions = self.dataset[idx]
        caption = random.choice(captions) if self.train else captions[0]
        tokens = _encode(caption, self.tokenizer, self.vocab, self.sequence_length)
        return image, torch.tensor(tokens, dtype=torch.long)


class CocoCaptions(DataModule):
    default_width: int = 224
    default_height: int = 224
    num_channels: int = 3
    flattened_input_dim: int = default_width * default_height * num_channels
    vocab_size: int = 10000
    num_classes: int = vocab_size
    sequence_length: int = 64

    def __init__(
        self,
        batch_size: int = 64,
        sequence_length: int = 64,
        resize: tuple = (224, 224),
        train_ann_file: str = "data/coco/annotations/captions_train2017.json",
        val_ann_file: str = "data/coco/annotations/captions_val2017.json",
        train_root: str = "data/coco/train2017",
        val_root: str = "data/coco/val2017",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.resize = resize
        self.train_ann_file = train_ann_file
        self.val_ann_file = val_ann_file
        self.train_root = train_root
        self.val_root = val_root
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = None

    def prepare_data(self) -> None:
        pass  # COCO requires manual download

    def _setup_fit(self) -> None:
        raw_train = datasets.CocoCaptions(
            root=self.train_root,
            annFile=self.train_ann_file,
            transform=self._get_transforms(),
        )
        self._build_vocab(raw_train)
        raw_val = datasets.CocoCaptions(
            root=self.val_root,
            annFile=self.val_ann_file,
            transform=self._get_transforms(),
        )
        self.train = _CaptioningDataset(raw_train, self.tokenizer, self.vocab, self.sequence_length, train=True)
        self.val = _CaptioningDataset(raw_val, self.tokenizer, self.vocab, self.sequence_length, train=False)

    def _setup_validate(self) -> None:
        raw_val = datasets.CocoCaptions(
            root=self.val_root,
            annFile=self.val_ann_file,
            transform=self._get_transforms(),
        )
        self._build_vocab(raw_val)
        self.val = _CaptioningDataset(raw_val, self.tokenizer, self.vocab, self.sequence_length, train=False)

    def _build_vocab(self, dataset) -> None:
        if self.vocab is not None:
            return
        self.vocab = build_vocab_from_iterator(
            _yield_tokens(dataset, self.tokenizer),
            specials=["<unk>", "<pad>", "<bos>", "<eos>"],
        )
        self.vocab.set_default_index(self.vocab["<unk>"])
        CocoCaptions.vocab_size = len(self.vocab)
        CocoCaptions.num_classes = len(self.vocab)

    def _get_transforms(self) -> Compose:
        return transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

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
        return [self.vocab.lookup_token(int(i)) for i in indices]
