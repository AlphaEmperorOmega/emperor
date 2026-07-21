import gzip
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import emperor.datasets.text.translation._adapter as multi30k_module
from emperor.datasets.text.translation import Multi30kDeEn, Multi30kEnDe
from emperor.datasets.text.translation._manifest import (
    BOS_ID,
    BOS_TOKEN,
    EOS_ID,
    EOS_TOKEN,
    PAD_ID,
    PAD_TOKEN,
    UNK_ID,
    UNK_TOKEN,
    Multi30kFile,
)


class TestMulti30kTranslation(unittest.TestCase):
    def preset(self):
        text = {
            "train.de.gz": "Hallo Welt\nEin rotes Haus\nKleine Katze\n",
            "train.en.gz": "Hello world\nA red house\nSmall cat\n",
            "val.de.gz": "Ω validation-only\nNoch eins\n",
            "val.en.gz": "Ω validation-only\nOne more\n",
            "test_2016_flickr.de.gz": "Test Satz\n",
            "test_2016_flickr.en.gz": "Test sentence\n",
        }
        archives = {
            name: gzip.compress(value.encode("utf-8"), mtime=0)
            for name, value in text.items()
        }
        split_by_name = {
            "train.de.gz": ("train", "de", 3),
            "train.en.gz": ("train", "en", 3),
            "val.de.gz": ("val", "de", 2),
            "val.en.gz": ("val", "en", 2),
            "test_2016_flickr.de.gz": ("test", "de", 1),
            "test_2016_flickr.en.gz": ("test", "en", 1),
        }
        files = tuple(
            Multi30kFile(
                split=split,
                language=language,
                filename=name,
                sha256=hashlib.sha256(archives[name]).hexdigest(),
                line_count=line_count,
            )
            for name, (split, language, line_count) in split_by_name.items()
        )
        return archives, files

    def data_module(self, root: Path, dataset_type=Multi30kDeEn, batch_size=2):
        archives, files = self.preset()
        calls = []

        def downloader(url: str, destination: Path) -> None:
            name = url.rsplit("/", 1)[-1]
            calls.append(name)
            destination.write_bytes(archives[name])

        data = dataset_type(
            root=root,
            batch_size=batch_size,
            source_sequence_length=7,
            target_sequence_length=6,
            num_workers=0,
            downloader=downloader,
        )
        return data, files, calls

    def test_supported_direction_metadata(self):
        self.assertEqual(Multi30kDeEn.language_pair, ("de", "en"))
        self.assertEqual(Multi30kEnDe.language_pair, ("en", "de"))
        for dataset in (Multi30kDeEn, Multi30kEnDe):
            with self.subTest(dataset=dataset.__name__):
                self.assertEqual(dataset.flattened_input_dim, 8192)
                self.assertEqual(dataset.num_classes, 8192)

    def test_prepare_verifies_hashes_recovers_corruption_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data, files, calls = self.data_module(Path(temporary_directory))
            with patch.object(Multi30kDeEn, "files", files):
                data.prepare_data()
                self.assertEqual(len(calls), 6)
                data.prepare_data()
                self.assertEqual(len(calls), 6)

                archive = data.archive_dir / "train.de.gz"
                archive.write_bytes(b"corrupt")
                data.prepare_data()
                self.assertEqual(calls.count("train.de.gz"), 2)

                corpus = data.corpus_dir / "train.en"
                corpus.write_text("corrupt\n", encoding="utf-8")
                downloads_before = len(calls)
                data.prepare_data()
                self.assertEqual(len(calls), downloads_before)
                self.assertEqual(
                    len(corpus.read_text(encoding="utf-8").splitlines()),
                    3,
                )
                self.assertFalse(
                    any(path.suffix == ".tmp" for path in data.cache_dir.rglob("*"))
                )

    def test_hash_failure_does_not_publish_partial_archive(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data, files, _ = self.data_module(Path(temporary_directory))
            bad_file = Multi30kFile(
                split=files[0].split,
                language=files[0].language,
                filename=files[0].filename,
                sha256="0" * 64,
                line_count=files[0].line_count,
            )
            with patch.object(Multi30kDeEn, "files", (bad_file,)):
                with self.assertRaises(RuntimeError):
                    data.prepare_data()

            self.assertFalse((data.archive_dir / bad_file.filename).exists())
            self.assertFalse(
                any(path.suffix == ".tmp" for path in data.cache_dir.rglob("*"))
            )

    def test_tokenizer_is_deterministic_train_only_and_has_stable_special_ids(self):
        vocabularies = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as temporary_directory:
                data, files, _ = self.data_module(Path(temporary_directory))
                with patch.object(Multi30kDeEn, "files", files):
                    data.prepare_data()
                tokenizer = multi30k_module.Tokenizer.from_file(
                    str(data.tokenizer_path)
                )
                vocabularies.append(tokenizer.get_vocab())
                for expected_id, token in enumerate(
                    (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)
                ):
                    with self.subTest(token=token):
                        self.assertEqual(tokenizer.token_to_id(token), expected_id)
                self.assertEqual(tokenizer.get_vocab_size(), 8192)
                self.assertIsNone(tokenizer.token_to_id("Ω"))

        self.assertEqual(vocabularies[0], vocabularies[1])

    def test_all_splits_fixed_lengths_incomplete_batches_and_direction_reversal(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            de_en, files, _ = self.data_module(root, Multi30kDeEn)
            with (
                patch.object(Multi30kDeEn, "files", files),
                patch.object(Multi30kEnDe, "files", files),
            ):
                de_en.setup("fit")
                train_batches = list(de_en.train_dataloader())
                de_en.setup("validate")
                validation_batch = next(iter(de_en.val_dataloader()))
                de_en.setup("test")
                test_batch = next(iter(de_en.test_dataloader()))

                en_de, _, _ = self.data_module(root, Multi30kEnDe)
                en_de.setup("fit")

            self.assertEqual([batch[0].size(0) for batch in train_batches], [2, 1])
            for source_ids, target_ids in [
                *train_batches,
                validation_batch,
                test_batch,
            ]:
                with self.subTest(batch_size=source_ids.size(0)):
                    self.assertEqual(source_ids.shape[1], 7)
                    self.assertEqual(target_ids.shape[1], 6)
                    self.assertTrue(torch.all(source_ids[:, 0] == BOS_ID))
                    self.assertTrue(torch.all(target_ids[:, 0] == BOS_ID))
                    self.assertTrue(torch.all((source_ids == EOS_ID).any(dim=1)))
                    self.assertTrue(torch.all((target_ids == EOS_ID).any(dim=1)))

            de_source, de_target = de_en.train.pairs[0]
            en_source, en_target = en_de.train.pairs[0]
            self.assertEqual((en_source, en_target), (de_target, de_source))
            decoded = de_en.decode_ids(
                torch.tensor([BOS_ID, UNK_ID, EOS_ID, PAD_ID, PAD_ID])
            )
            self.assertEqual(decoded, "")


if __name__ == "__main__":
    unittest.main()
