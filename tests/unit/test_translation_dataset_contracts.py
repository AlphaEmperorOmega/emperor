import gzip
import hashlib
import io
import json
import re
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

from tokenizers import Tokenizer
from tokenizers.models import WordPiece

from emperor.datasets.text.translation import Multi30kDeEn
from emperor.datasets.text.translation._download import _download_file
from emperor.datasets.text.translation._manifest import (
    BOS_ID,
    EOS_ID,
    FILES,
    PAD_ID,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    UNK_TOKEN,
    VOCAB_SIZE,
    Multi30kFile,
)
from emperor.datasets.text.translation._tokenizer import _TokenizerSupport


def _wordpiece_tokenizer(*, stable_special_ids: bool, size: int) -> Tokenizer:
    special_tokens = list(SPECIAL_TOKENS)
    if not stable_special_ids:
        special_tokens[0], special_tokens[1] = special_tokens[1], special_tokens[0]
    tokens = [*special_tokens, *(f"token_{index}" for index in range(size - 4))]
    vocabulary = {token: index for index, token in enumerate(tokens)}
    return Tokenizer(WordPiece(vocab=vocabulary, unk_token=UNK_TOKEN))


class _PreparedMulti30k(Multi30kDeEn):
    def prepare_data(self) -> None:
        return None


class _RecordingTokenizer:
    def __init__(self) -> None:
        self.decoded_ids: list[int] | None = None

    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        if not skip_special_tokens:
            raise AssertionError("translation decoding must skip special tokens")
        self.decoded_ids = list(token_ids)
        return " decoded tokens "

    def id_to_token(self, token_id: int) -> str | None:
        return "known" if token_id == 5 else None


class _IdentityNormalizer:
    @staticmethod
    def normalize_str(text: str) -> str:
        return text


class _WholeTextPreTokenizer:
    @staticmethod
    def pre_tokenize_str(text: str):
        return [(text, (0, len(text)))]


class _UnstableSpecialIdMulti30k(Multi30kDeEn):
    def _deterministic_wordpiece_vocabulary(self, normalizer, pre_tokenizer):
        del normalizer, pre_tokenizer
        special_tokens = list(SPECIAL_TOKENS)
        special_tokens[0], special_tokens[1] = special_tokens[1], special_tokens[0]
        tokens = [
            *special_tokens,
            *(f"token_{index}" for index in range(VOCAB_SIZE - 4)),
        ]
        return {token: index for index, token in enumerate(tokens)}


class _WrongSizeMulti30k(Multi30kDeEn):
    def _deterministic_wordpiece_vocabulary(self, normalizer, pre_tokenizer):
        del normalizer, pre_tokenizer
        return {token: index for index, token in enumerate(SPECIAL_TOKENS)}


class _TokenizerHarness(_TokenizerSupport):
    pass


class _LiteralPreTokenizer:
    def __init__(self, words: list[str]) -> None:
        self.words = words

    def pre_tokenize_str(self, _text: str):
        return [(word, (0, len(word))) for word in self.words]


class _FailOnceTokenizer:
    def __init__(self) -> None:
        self.calls = 0

    def save(self, path: str) -> None:
        self.calls += 1
        if self.calls == 1:
            raise OSError("sentinel tokenizer write failure")
        Path(path).write_text("complete tokenizer", encoding="utf-8")


class TestDownloadEdges(unittest.TestCase):
    def test_real_download_boundary_sets_user_agent_and_writes_response(self) -> None:
        response_bytes = b"pinned response bytes"
        requests: list[urllib.request.Request] = []

        def open_response(request: urllib.request.Request):
            requests.append(request)
            return io.BytesIO(response_bytes)

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "download.gz"
            with patch(
                "emperor.datasets.text.translation._download.urllib.request.urlopen",
                side_effect=open_response,
            ):
                _download_file("https://example.test/download.gz", destination)

            self.assertEqual(destination.read_bytes(), response_bytes)

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].full_url, "https://example.test/download.gz")
        self.assertEqual(requests[0].get_header("User-agent"), "emperor/0.1")

    def test_byte_returning_downloader_is_verified_and_published(self) -> None:
        archive_bytes = gzip.compress(b"one line\n", mtime=0)
        file_spec = Multi30kFile(
            "train",
            "de",
            "train.de.gz",
            hashlib.sha256(archive_bytes).hexdigest(),
            1,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            data = Multi30kDeEn(
                root=temporary_directory,
                downloader=lambda _url, _destination: archive_bytes,
            )
            data.archive_dir.mkdir(parents=True)
            archive_path = data.archive_dir / file_spec.filename

            data._download_verified_archive(file_spec, archive_path)

            self.assertEqual(archive_path.read_bytes(), archive_bytes)
            self.assertEqual(list(data.archive_dir.glob("*.tmp")), [])

    def test_downloader_failure_removes_its_temporary_file(self) -> None:
        def fail_download(_url: str, _destination: Path):
            raise OSError("sentinel download failure")

        file_spec = Multi30kFile("train", "de", "train.de.gz", "0" * 64, 1)
        with tempfile.TemporaryDirectory() as temporary_directory:
            data = Multi30kDeEn(root=temporary_directory, downloader=fail_download)
            data.archive_dir.mkdir(parents=True)

            with self.assertRaisesRegex(OSError, "sentinel download failure"):
                data._download_verified_archive(
                    file_spec,
                    data.archive_dir / file_spec.filename,
                )

            self.assertEqual(list(data.archive_dir.glob("*.tmp")), [])

    def test_invalid_text_and_wrong_line_decompression_do_not_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data = Multi30kDeEn(root=temporary_directory)
            data.archive_dir.mkdir(parents=True)
            data.corpus_dir.mkdir(parents=True)
            invalid_utf8 = data.corpus_dir / "invalid.de"
            invalid_utf8.write_bytes(b"\xff\n")
            self.assertFalse(data._text_file_is_valid(invalid_utf8, 1))

            readable = data.corpus_dir / "readable.de"
            readable.write_text("one\n", encoding="utf-8")
            with patch.object(Path, "open", side_effect=OSError("sentinel read")):
                self.assertFalse(data._text_file_is_valid(readable, 1))

            archive = data.archive_dir / "train.de.gz"
            archive.write_bytes(gzip.compress(b"only one\n", mtime=0))
            published_text = data.corpus_dir / "train.de"
            published_text.write_text("stable\ncontent\n", encoding="utf-8")

            with self.assertRaisesRegex(
                RuntimeError,
                "Prepared train.de does not contain 2 UTF-8 lines",
            ):
                data._decompress_atomically(archive, published_text, 2)

            self.assertEqual(
                published_text.read_text(encoding="utf-8"),
                "stable\ncontent\n",
            )
            self.assertEqual(list(data.corpus_dir.glob("*.tmp")), [])


class TestAdapterEdges(unittest.TestCase):
    def test_constructor_stage_and_loader_guards_are_exact(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            Multi30kDeEn(batch_size=0)

        with tempfile.TemporaryDirectory() as temporary_directory:
            data = _PreparedMulti30k(root=temporary_directory)
            data.cache_dir.mkdir(parents=True)
            _wordpiece_tokenizer(stable_special_ids=True, size=4).save(
                str(data.tokenizer_path)
            )

            with self.assertRaisesRegex(
                ValueError,
                "Unsupported Multi30k setup stage: 'unsupported'",
            ):
                data.setup("unsupported")
            data.tokenizer = None
            with self.assertRaisesRegex(
                RuntimeError,
                "Multi30k tokenizer is not prepared",
            ):
                data._dataset_for_split("train")
            for loader in (
                lambda: data.get_dataloader(True),
                lambda: data.get_dataloader(False),
                data._get_test_dataloader,
            ):
                with self.subTest(loader=loader):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        r"Call setup\(\) before requesting a Multi30k data loader",
                    ):
                        loader()

        self.assertEqual(Multi30kDeEn()._split_filename("test"), "test_2016_flickr")

    def test_pair_counts_decode_without_eos_and_unknown_labels_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data = Multi30kDeEn(root=temporary_directory)
            data.corpus_dir.mkdir(parents=True)
            source_path = data.corpus_dir / "train.de"
            target_path = data.corpus_dir / "train.en"
            source_path.write_text("eins\nzwei\n", encoding="utf-8")
            target_path.write_text("one\n", encoding="utf-8")

            with self.assertRaisesRegex(
                RuntimeError,
                "Multi30k train source/target line counts do not match: 2 != 1",
            ):
                data._read_pairs("train")
            self.assertEqual(source_path.read_text(encoding="utf-8"), "eins\nzwei\n")
            self.assertEqual(target_path.read_text(encoding="utf-8"), "one\n")

            tokenizer = _RecordingTokenizer()
            data.tokenizer = tokenizer
            self.assertEqual(
                data.decode_ids([BOS_ID, 5, PAD_ID, 6, PAD_ID]),
                "decoded tokens",
            )
            self.assertEqual(tokenizer.decoded_ids, [5, 6])
            self.assertEqual(data.decode_ids([BOS_ID, 5, EOS_ID, 6]), "decoded tokens")
            self.assertEqual(tokenizer.decoded_ids, [5])
            self.assertEqual(data._text_labels([5, 99]), ["known", UNK_TOKEN])

    def test_decode_and_labels_load_the_published_tokenizer_on_demand(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data = Multi30kDeEn(root=temporary_directory)
            data.cache_dir.mkdir(parents=True)
            _wordpiece_tokenizer(stable_special_ids=True, size=5).save(
                str(data.tokenizer_path)
            )

            self.assertEqual(
                data.decode_ids([BOS_ID, 4, EOS_ID]),
                "token_0",
            )
            data.tokenizer = None
            self.assertEqual(data._text_labels([4, 99]), ["token_0", UNK_TOKEN])
            self.assertEqual(
                data.decode_batch(
                    [
                        [BOS_ID, 4, EOS_ID],
                        [BOS_ID, EOS_ID],
                    ]
                ),
                ["token_0", ""],
            )


class TestTokenizerCacheEdges(unittest.TestCase):
    def test_base_tokenizer_support_uses_the_release_manifest(self) -> None:
        self.assertIs(_TokenizerHarness()._files, FILES)

    def test_cache_requires_matching_manifest_size_and_special_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data = Multi30kDeEn(root=temporary_directory)
            data.cache_dir.mkdir(parents=True)
            expected_manifest = data._tokenizer_manifest()

            self.assertFalse(data._cached_tokenizer_is_valid(expected_manifest))
            data.tokenizer_path.write_text("not tokenizer json", encoding="utf-8")
            data.tokenizer_manifest_path.write_text("not json", encoding="utf-8")
            self.assertFalse(data._cached_tokenizer_is_valid(expected_manifest))

            _wordpiece_tokenizer(stable_special_ids=True, size=4).save(
                str(data.tokenizer_path)
            )
            data.tokenizer_manifest_path.write_text(
                json.dumps(expected_manifest),
                encoding="utf-8",
            )
            self.assertFalse(data._cached_tokenizer_is_valid(expected_manifest))

            _wordpiece_tokenizer(stable_special_ids=False, size=VOCAB_SIZE).save(
                str(data.tokenizer_path)
            )
            self.assertFalse(data._cached_tokenizer_is_valid(expected_manifest))

            _wordpiece_tokenizer(stable_special_ids=True, size=VOCAB_SIZE).save(
                str(data.tokenizer_path)
            )
            self.assertTrue(data._cached_tokenizer_is_valid(expected_manifest))
            self.assertFalse(data._cached_tokenizer_is_valid({"changed": True}))

    def test_unequal_training_files_yield_complete_pairs_before_raising(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data = Multi30kDeEn(root=temporary_directory)
            data.corpus_dir.mkdir(parents=True)
            (data.corpus_dir / "train.de").write_text(
                "eins\nzwei\n",
                encoding="utf-8",
            )
            (data.corpus_dir / "train.en").write_text("one\n", encoding="utf-8")

            training_text = data._training_text()
            self.assertEqual(next(training_text), "eins")
            self.assertEqual(next(training_text), "one")
            with self.assertRaises(ValueError):
                next(training_text)

    def test_vocabulary_overflow_and_unstable_special_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data = Multi30kDeEn(root=temporary_directory)
            distinct_characters = "".join(chr(0x1000 + index) for index in range(4095))
            data._training_text = lambda: iter((distinct_characters,))

            with self.assertRaisesRegex(
                RuntimeError,
                "more required WordPiece characters than the configured",
            ):
                data._deterministic_wordpiece_vocabulary(
                    _IdentityNormalizer(),
                    _WholeTextPreTokenizer(),
                )

            unstable = _UnstableSpecialIdMulti30k(root=temporary_directory)
            with self.assertRaisesRegex(
                RuntimeError,
                re.escape(f"Tokenizer assigned an unstable ID to {PAD_TOKEN}."),
            ):
                unstable._train_tokenizer()

            wrong_size = _WrongSizeMulti30k(root=temporary_directory)
            with self.assertRaisesRegex(
                RuntimeError,
                f"Expected an {VOCAB_SIZE}-token vocabulary, got 4",
            ):
                wrong_size._train_tokenizer()

    def test_empty_words_and_reserved_unused_tokens_are_handled_defensively(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data = Multi30kDeEn(root=temporary_directory)
            data._training_text = lambda: iter(("ignored",))

            vocabulary = data._deterministic_wordpiece_vocabulary(
                _IdentityNormalizer(),
                _LiteralPreTokenizer(["", "[unused0]"]),
            )

            self.assertEqual(len(vocabulary), VOCAB_SIZE)
            self.assertIn("[unused0]", vocabulary)
            self.assertIn("[unused1]", vocabulary)
            self.assertEqual(len(set(vocabulary)), VOCAB_SIZE)

    def test_candidate_vocabulary_stops_exactly_at_configured_size(self) -> None:
        alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"

        def base36(value: int) -> str:
            digits = []
            for _ in range(3):
                value, remainder = divmod(value, len(alphabet))
                digits.append(alphabet[remainder])
            return "".join(reversed(digits))

        with tempfile.TemporaryDirectory() as temporary_directory:
            data = Multi30kDeEn(root=temporary_directory)
            data._training_text = lambda: iter(("ignored",))
            candidate_words = [base36(index) for index in range(8_300)]

            vocabulary = data._deterministic_wordpiece_vocabulary(
                _IdentityNormalizer(),
                _LiteralPreTokenizer(candidate_words),
            )

            self.assertEqual(len(vocabulary), VOCAB_SIZE)
            self.assertNotIn("[unused0]", vocabulary)

    def test_atomic_tokenizer_and_manifest_writes_cleanup_and_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data = Multi30kDeEn(root=temporary_directory)
            data.cache_dir.mkdir(parents=True)
            tokenizer = _FailOnceTokenizer()

            with self.assertRaisesRegex(OSError, "sentinel tokenizer write failure"):
                data._write_tokenizer_atomically(tokenizer)
            self.assertFalse(data.tokenizer_path.exists())
            self.assertEqual(list(data.cache_dir.glob("*.tmp")), [])

            data._write_tokenizer_atomically(tokenizer)
            self.assertEqual(
                data.tokenizer_path.read_text(encoding="utf-8"),
                "complete tokenizer",
            )

            with patch(
                "emperor.datasets.text.translation._tokenizer.os.replace",
                side_effect=OSError("sentinel manifest replace failure"),
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "sentinel manifest replace failure",
                ):
                    data._write_json_atomically(
                        data.tokenizer_manifest_path,
                        {"complete": True},
                    )
            self.assertFalse(data.tokenizer_manifest_path.exists())
            self.assertEqual(list(data.cache_dir.glob("*.tmp")), [])

            data._write_json_atomically(
                data.tokenizer_manifest_path,
                {"complete": True},
            )
            self.assertEqual(
                json.loads(data.tokenizer_manifest_path.read_text(encoding="utf-8")),
                {"complete": True},
            )


if __name__ == "__main__":
    unittest.main()
