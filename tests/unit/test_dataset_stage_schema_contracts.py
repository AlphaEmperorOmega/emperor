import unittest
from dataclasses import dataclass

from emperor.datasets.image.captioning._coco_captions import CocoCaptions
from emperor.datasets.image.captioning._flickr8k import Flickr8k
from emperor.datasets.image.captioning._flickr30k import Flickr30k
from emperor.datasets.multimodal.vqa._gqa import GQA
from emperor.datasets.text.classification._ag_news import AgNews
from emperor.datasets.text.classification._dbpedia import DBpedia
from emperor.datasets.text.classification._imdb import IMDB
from emperor.datasets.text.classification._yelp_review_full import YelpReviewFull
from emperor.datasets.text.nli._multi_nli import MultiNLI
from emperor.datasets.text.nli._snli import SNLI
from emperor.datasets.text.question_answering._squad_v1 import SQuADv1
from emperor.datasets.text.similarity._stsb import STSb
from emperor.datasets.text.summarization._cnn_dailymail import CnnDailyMail
from emperor.datasets.text.summarization._xsum import XSum


@dataclass(frozen=True)
class _SchemaCase:
    adapter_type: type
    vocabulary_attribute: str
    builder_name: str
    builder_arguments: tuple[object, ...]


_SCHEMA_CASES = (
    _SchemaCase(CocoCaptions, "vocab", "_build_vocab", (object(),)),
    _SchemaCase(Flickr30k, "vocab", "_build_vocab", (object(),)),
    _SchemaCase(Flickr8k, "vocab", "_build_vocab", (object(),)),
    _SchemaCase(GQA, "question_vocab", "_build_vocabs", (object(),)),
    _SchemaCase(AgNews, "vocab", "_build_vocab", ()),
    _SchemaCase(DBpedia, "vocab", "_build_vocab", ()),
    _SchemaCase(IMDB, "vocab", "_build_vocab", ()),
    _SchemaCase(YelpReviewFull, "vocab", "_build_vocab", ()),
    _SchemaCase(MultiNLI, "vocab", "_build_vocab", (object(),)),
    _SchemaCase(SNLI, "vocab", "_build_vocab", (object(),)),
    _SchemaCase(SQuADv1, "vocab", "_build_vocab", (object(),)),
    _SchemaCase(STSb, "vocab", "_build_vocab", (object(),)),
    _SchemaCase(CnnDailyMail, "vocab", "_build_vocab", (object(),)),
    _SchemaCase(XSum, "vocab", "_build_vocab", (object(),)),
)


class TestTrainOwnedSchemas(unittest.TestCase):
    def test_existing_vocabulary_guards_preserve_identity_and_metadata(self) -> None:
        for case in _SCHEMA_CASES:
            with self.subTest(adapter=case.adapter_type.__name__):
                adapter = case.adapter_type()
                vocabulary = {"<unk>": 0, "train-only": 1}
                setattr(adapter, case.vocabulary_attribute, vocabulary)
                answer_vocabulary = None
                if isinstance(adapter, GQA):
                    answer_vocabulary = {"yes": 0}
                    adapter.answer_vocab = answer_vocabulary
                metadata = adapter.resolved_metadata

                getattr(adapter, case.builder_name)(*case.builder_arguments)

                self.assertIs(
                    getattr(adapter, case.vocabulary_attribute),
                    vocabulary,
                )
                self.assertEqual(vocabulary, {"<unk>": 0, "train-only": 1})
                self.assertIs(adapter.resolved_metadata, metadata)
                if answer_vocabulary is not None:
                    self.assertIs(adapter.answer_vocab, answer_vocabulary)


if __name__ == "__main__":
    unittest.main()
