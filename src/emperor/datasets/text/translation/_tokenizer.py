import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer

from emperor.datasets.text.translation._manifest import (
    FILES,
    MAX_SUBWORD_LENGTH,
    SOURCE_COMMIT,
    SPECIAL_TOKENS,
    UNK_TOKEN,
    VOCAB_SIZE,
)


class _TokenizerSupport:
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
                file_spec.filename: file_spec.sha256 for file_spec in self._files
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

    @property
    def _files(self):
        return FILES

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
