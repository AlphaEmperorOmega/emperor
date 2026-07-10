from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer

from emperor.base.utils import DataModule

SOURCE_COMMIT = "a3d2e0d26b56f3846f66a952536ffed4e401d05a"
SOURCE_BASE_URL = (
    f"https://raw.githubusercontent.com/multi30k/dataset/{SOURCE_COMMIT}/data/task1/raw"
)

VOCAB_SIZE = 8192
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
MAX_SUBWORD_LENGTH = 6


@dataclass(frozen=True)
class Multi30kFile:
    split: str
    language: str
    filename: str
    sha256: str
    line_count: int

    @property
    def url(self) -> str:
        return f"{SOURCE_BASE_URL}/{self.filename}"

    @property
    def text_filename(self) -> str:
        return self.filename.removesuffix(".gz")


FILES: tuple[Multi30kFile, ...] = (
    Multi30kFile(
        "train",
        "de",
        "train.de.gz",
        "726e39b2fa9eb9ffb6dc763fb35a179f80fae06ffc5d28b6ace5faa883de28a6",
        29_000,
    ),
    Multi30kFile(
        "train",
        "en",
        "train.en.gz",
        "d79cfa999dd4c51d2cb42499b6796d5a882c3a8a961923c25a898c90f8bbd56f",
        29_000,
    ),
    Multi30kFile(
        "val",
        "de",
        "val.de.gz",
        "f0cba2f995189cf5770f29a8a9a537a3ad3f51657ad873405082ff6863a5e75a",
        1_014,
    ),
    Multi30kFile(
        "val",
        "en",
        "val.en.gz",
        "14f7d25ddd868909a9213e361768460edcacdd6ab9d1e77b92560dc10c10dc28",
        1_014,
    ),
    Multi30kFile(
        "test",
        "de",
        "test_2016_flickr.de.gz",
        "9204244e408ccb38d2a55cfcd344df15005fc42a07a6e55ca6c52b6ababb8cc8",
        1_000,
    ),
    Multi30kFile(
        "test",
        "en",
        "test_2016_flickr.en.gz",
        "611d361c6334bc7246101d48097c13cf5c4413c5befc793cc629934359d532d9",
        1_000,
    ),
)

DownloadFunction = Callable[[str, Path], object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, destination: Path) -> None:
    """Download *url* to *destination* without importing TorchText."""

    request = urllib.request.Request(url, headers={"User-Agent": "emperor/0.1"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


class _TranslationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pairs: Sequence[tuple[str, str]],
        tokenizer: Tokenizer,
        source_sequence_length: int,
        target_sequence_length: int,
    ) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.source_sequence_length = source_sequence_length
        self.target_sequence_length = target_sequence_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source, target = self.pairs[index]
        return (
            self._encode(source, self.source_sequence_length),
            self._encode(target, self.target_sequence_length),
        )

    def _encode(self, text: str, maximum_length: int) -> torch.Tensor:
        token_ids = self.tokenizer.encode(text).ids[: maximum_length - 2]
        token_ids = [BOS_ID, *token_ids, EOS_ID]
        token_ids.extend([PAD_ID] * (maximum_length - len(token_ids)))
        return torch.tensor(token_ids, dtype=torch.long)


class _Multi30k(DataModule):
    """Pinned, shared-vocabulary Multi30k translation data module."""

    source_language = "de"
    target_language = "en"
    language_pair = (source_language, target_language)

    src_vocab_size = VOCAB_SIZE
    tgt_vocab_size = VOCAB_SIZE
    vocab_size = VOCAB_SIZE
    flattened_input_dim = VOCAB_SIZE
    num_classes = VOCAB_SIZE
    sequence_length = 64
    source_sequence_length = 64
    target_sequence_length = 64

    pad_token_id = PAD_ID
    unk_token_id = UNK_ID
    bos_token_id = BOS_ID
    eos_token_id = EOS_ID

    def __init__(
        self,
        batch_size: int = 64,
        sequence_length: int | None = None,
        source_sequence_length: int | None = None,
        target_sequence_length: int | None = None,
        language_pair: tuple[str, str] | None = None,
        root: str | os.PathLike[str] = "data",
        num_workers: int = 0,
        downloader: DownloadFunction | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(root=str(root), num_workers=num_workers)
        requested_pair = language_pair or type(self).language_pair
        if tuple(requested_pair) != tuple(type(self).language_pair):
            raise ValueError(
                f"{type(self).__name__} only supports the language pair "
                f"{type(self).language_pair!r}; got {tuple(requested_pair)!r}."
            )
        broadcast_length = sequence_length or type(self).sequence_length
        self.batch_size = int(batch_size)
        self.source_sequence_length = int(source_sequence_length or broadcast_length)
        self.target_sequence_length = int(target_sequence_length or broadcast_length)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.source_sequence_length < 2 or self.target_sequence_length < 2:
            raise ValueError("source and target sequence lengths must be at least 2")
        self.sequence_length = max(
            self.source_sequence_length, self.target_sequence_length
        )
        self.language_pair = tuple(requested_pair)
        self.seed = int(seed)
        self._downloader = downloader or _download_file
        self.cache_dir = Path(self.root) / "multi30k" / SOURCE_COMMIT
        self.archive_dir = self.cache_dir / "archives"
        self.corpus_dir = self.cache_dir / "corpus"
        self.tokenizer_path = self.cache_dir / "tokenizer.json"
        self.tokenizer_manifest_path = self.cache_dir / "tokenizer.manifest.json"
        self.tokenizer: Tokenizer | None = None
        self.train = None
        self.val = None
        self.test = None

    def prepare_data(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.cache_dir / ".prepare.lock"
        with lock_path.open("a", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                for file_spec in FILES:
                    self._prepare_file(file_spec)
                self._prepare_tokenizer()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _prepare_file(self, file_spec: Multi30kFile) -> None:
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        archive_path = self.archive_dir / file_spec.filename
        if not self._archive_is_valid(archive_path, file_spec.sha256):
            self._download_verified_archive(file_spec, archive_path)

        text_path = self.corpus_dir / file_spec.text_filename
        if not self._text_file_is_valid(text_path, file_spec.line_count):
            self._decompress_atomically(archive_path, text_path, file_spec.line_count)

    def _archive_is_valid(self, path: Path, expected_sha256: str) -> bool:
        return path.is_file() and _sha256(path) == expected_sha256

    def _download_verified_archive(
        self,
        file_spec: Multi30kFile,
        archive_path: Path,
    ) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{file_spec.filename}.",
            suffix=".tmp",
            dir=self.archive_dir,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            result = self._downloader(file_spec.url, temporary_path)
            if isinstance(result, (bytes, bytearray)):
                temporary_path.write_bytes(result)
            actual_hash = _sha256(temporary_path)
            if actual_hash != file_spec.sha256:
                raise RuntimeError(
                    f"SHA-256 mismatch for {file_spec.filename}: expected "
                    f"{file_spec.sha256}, got {actual_hash}."
                )
            os.replace(temporary_path, archive_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _text_file_is_valid(self, path: Path, expected_lines: int) -> bool:
        if not path.is_file():
            return False
        try:
            with path.open("r", encoding="utf-8") as handle:
                return sum(1 for _ in handle) == expected_lines
        except (OSError, UnicodeDecodeError):
            return False

    def _decompress_atomically(
        self,
        archive_path: Path,
        text_path: Path,
        expected_lines: int,
    ) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{text_path.name}.",
            suffix=".tmp",
            dir=self.corpus_dir,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            with (
                gzip.open(archive_path, "rb") as source,
                temporary_path.open("wb") as destination,
            ):
                shutil.copyfileobj(source, destination)
            if not self._text_file_is_valid(temporary_path, expected_lines):
                raise RuntimeError(
                    f"Prepared {text_path.name} does not contain "
                    f"{expected_lines} UTF-8 lines."
                )
            os.replace(temporary_path, text_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _prepare_tokenizer(self) -> None:
        expected_manifest = self._tokenizer_manifest()
        if self._cached_tokenizer_is_valid(expected_manifest):
            return
        tokenizer = self._train_tokenizer()
        self._write_tokenizer_atomically(tokenizer)
        self._write_json_atomically(
            self.tokenizer_manifest_path,
            expected_manifest,
        )

    def _cached_tokenizer_is_valid(self, expected_manifest: dict) -> bool:
        if (
            not self.tokenizer_path.is_file()
            or not self.tokenizer_manifest_path.is_file()
        ):
            return False
        try:
            manifest = json.loads(
                self.tokenizer_manifest_path.read_text(encoding="utf-8")
            )
            tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return (
            manifest == expected_manifest
            and tokenizer.get_vocab_size() == VOCAB_SIZE
            and all(
                tokenizer.token_to_id(token) == index
                for index, token in enumerate(SPECIAL_TOKENS)
            )
        )

    def _tokenizer_manifest(self) -> dict:
        return {
            "source_commit": SOURCE_COMMIT,
            "source_files": {
                file_spec.filename: file_spec.sha256 for file_spec in FILES
            },
            "training_splits": ["train.de", "train.en"],
            "settings": {
                "algorithm": "WordPiece",
                "vocabulary_selection": (
                    "frequency-descending,length-descending,lexical-ascending"
                ),
                "case_sensitive": True,
                "normalizer": "BertNormalizer(lowercase=False,strip_accents=False)",
                "pre_tokenizer": "BertPreTokenizer",
                "minimum_frequency": 1,
                "maximum_subword_length": MAX_SUBWORD_LENGTH,
                "continuing_subword_prefix": "##",
                "special_tokens": list(SPECIAL_TOKENS),
                "unused_token_pattern": "[unused{index}]",
            },
            "vocabulary_size": VOCAB_SIZE,
        }

    def _training_text(self) -> Iterator[str]:
        # The iterator deliberately contains only the two training files. Keeping the
        # paired order stable makes tokenizer training reproducible.
        paths = (
            self.corpus_dir / "train.de",
            self.corpus_dir / "train.en",
        )
        handles = [path.open("r", encoding="utf-8") for path in paths]
        try:
            for lines in zip(*handles, strict=True):
                yield from (line.rstrip("\n") for line in lines)
        finally:
            for handle in handles:
                handle.close()

    def _train_tokenizer(self) -> Tokenizer:
        normalizer = BertNormalizer(
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=False,
            lowercase=False,
        )
        pre_tokenizer = BertPreTokenizer()
        vocabulary = self._deterministic_wordpiece_vocabulary(
            normalizer,
            pre_tokenizer,
        )
        tokenizer = Tokenizer(
            WordPiece(
                vocab=vocabulary,
                unk_token=UNK_TOKEN,
                continuing_subword_prefix="##",
            )
        )
        tokenizer.normalizer = normalizer
        tokenizer.pre_tokenizer = pre_tokenizer
        tokenizer.decoder = WordPieceDecoder(prefix="##", cleanup=True)
        tokenizer.add_special_tokens(list(SPECIAL_TOKENS))
        if tokenizer.get_vocab_size() != VOCAB_SIZE:
            raise RuntimeError(
                f"Expected an {VOCAB_SIZE}-token vocabulary, got "
                f"{tokenizer.get_vocab_size()}."
            )
        for index, token in enumerate(SPECIAL_TOKENS):
            if tokenizer.token_to_id(token) != index:
                raise RuntimeError(f"Tokenizer assigned an unstable ID to {token}.")
        return tokenizer

    def _deterministic_wordpiece_vocabulary(
        self,
        normalizer: BertNormalizer,
        pre_tokenizer: BertPreTokenizer,
    ) -> dict[str, int]:
        word_counts: Counter[str] = Counter()
        for text in self._training_text():
            normalized = normalizer.normalize_str(text)
            word_counts.update(
                token for token, _ in pre_tokenizer.pre_tokenize_str(normalized)
            )

        candidate_counts: Counter[str] = Counter()
        observed_characters: set[str] = set()
        for word, frequency in sorted(word_counts.items()):
            if not word:
                continue
            observed_characters.update(word)
            candidate_counts[word] += frequency
            maximum_prefix = min(len(word), MAX_SUBWORD_LENGTH)
            for end in range(1, maximum_prefix + 1):
                candidate_counts[word[:end]] += frequency
            for start in range(1, len(word)):
                maximum_end = min(len(word), start + MAX_SUBWORD_LENGTH)
                for end in range(start + 1, maximum_end + 1):
                    candidate_counts[f"##{word[start:end]}"] += frequency

        required_pieces = sorted(
            observed_characters
            | {f"##{character}" for character in observed_characters}
        )
        vocabulary_tokens = list(SPECIAL_TOKENS)
        vocabulary_tokens.extend(
            token for token in required_pieces if token not in SPECIAL_TOKENS
        )
        if len(vocabulary_tokens) > VOCAB_SIZE:
            raise RuntimeError(
                "Multi30k contains more required WordPiece characters than the "
                f"configured {VOCAB_SIZE}-token vocabulary can hold."
            )

        selected = set(vocabulary_tokens)
        ranked_candidates = sorted(
            candidate_counts,
            key=lambda token: (
                -candidate_counts[token],
                -len(token.removeprefix("##")),
                token,
            ),
        )
        for token in ranked_candidates:
            if token in selected:
                continue
            vocabulary_tokens.append(token)
            selected.add(token)
            if len(vocabulary_tokens) == VOCAB_SIZE:
                break

        unused_index = 0
        while len(vocabulary_tokens) < VOCAB_SIZE:
            token = f"[unused{unused_index}]"
            unused_index += 1
            if token in selected:
                continue
            vocabulary_tokens.append(token)
            selected.add(token)

        return {token: index for index, token in enumerate(vocabulary_tokens)}

    def _write_tokenizer_atomically(self, tokenizer: Tokenizer) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".tokenizer.", suffix=".json.tmp", dir=self.cache_dir
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            tokenizer.save(str(temporary_path))
            os.replace(temporary_path, self.tokenizer_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _write_json_atomically(self, path: Path, payload: dict) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.stem}.", suffix=".json.tmp", dir=path.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            temporary_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def setup(self, stage: str | None = None) -> None:
        self.prepare_data()
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        if stage in (None, "fit"):
            self.train = self._dataset_for_split("train")
            self.val = self._dataset_for_split("val")
        elif stage == "validate":
            self.val = self._dataset_for_split("val")
        elif stage in ("test", "predict"):
            self.test = self._dataset_for_split("test")
        else:
            raise ValueError(f"Unsupported Multi30k setup stage: {stage!r}")

    def _dataset_for_split(self, split: str) -> _TranslationDataset:
        if self.tokenizer is None:
            raise RuntimeError("Multi30k tokenizer is not prepared")
        return _TranslationDataset(
            self._read_pairs(split),
            self.tokenizer,
            self.source_sequence_length,
            self.target_sequence_length,
        )

    def _read_pairs(self, split: str) -> list[tuple[str, str]]:
        source_path = (
            self.corpus_dir / f"{self._split_filename(split)}.{self.source_language}"
        )
        target_path = (
            self.corpus_dir / f"{self._split_filename(split)}.{self.target_language}"
        )
        source_lines = source_path.read_text(encoding="utf-8").splitlines()
        target_lines = target_path.read_text(encoding="utf-8").splitlines()
        if len(source_lines) != len(target_lines):
            raise RuntimeError(
                f"Multi30k {split} source/target line counts do not match: "
                f"{len(source_lines)} != {len(target_lines)}."
            )
        return list(zip(source_lines, target_lines, strict=True))

    def _split_filename(self, split: str) -> str:
        return "test_2016_flickr" if split == "test" else split

    def _loader(self, dataset, *, shuffle: bool) -> torch.utils.data.DataLoader:
        if dataset is None:
            raise RuntimeError("Call setup() before requesting a Multi30k data loader")
        generator = torch.Generator().manual_seed(self.seed) if shuffle else None
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=False,
            generator=generator,
        )

    def get_dataloader(self, train: bool):
        return self._loader(self.train if train else self.val, shuffle=train)

    def _get_test_dataloader(self):
        return self._loader(self.test, shuffle=False)

    def decode_ids(self, token_ids: Sequence[int] | torch.Tensor) -> str:
        if self.tokenizer is None:
            self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        ids = [int(token_id) for token_id in token_ids]
        if EOS_ID in ids:
            ids = ids[: ids.index(EOS_ID)]
        ids = [token_id for token_id in ids if token_id not in (PAD_ID, BOS_ID)]
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    def decode_batch(
        self, batch_token_ids: Sequence[Sequence[int]] | torch.Tensor
    ) -> list[str]:
        return [self.decode_ids(token_ids) for token_ids in batch_token_ids]

    def _text_labels(self, indices) -> list[str]:
        if self.tokenizer is None:
            self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        return [
            self.tokenizer.id_to_token(int(index)) or UNK_TOKEN for index in indices
        ]


class Multi30kDeEn(_Multi30k):
    source_language = "de"
    target_language = "en"
    language_pair = (source_language, target_language)


class Multi30kEnDe(_Multi30k):
    source_language = "en"
    target_language = "de"
    language_pair = (source_language, target_language)


# Historical compatibility name: Multi30k has always meant German-to-English.
Multi30k = Multi30kDeEn


__all__ = [
    "BOS_ID",
    "BOS_TOKEN",
    "EOS_ID",
    "EOS_TOKEN",
    "FILES",
    "Multi30k",
    "Multi30kDeEn",
    "Multi30kEnDe",
    "Multi30kFile",
    "PAD_ID",
    "PAD_TOKEN",
    "SOURCE_COMMIT",
    "UNK_ID",
    "UNK_TOKEN",
    "VOCAB_SIZE",
]
