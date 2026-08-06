import importlib
import unittest

from emperor.datasets._base import DataModule


class _FitValidateDataModule(DataModule):
    def __init__(self) -> None:
        super().__init__(num_workers=0)
        self.setup_calls: list[str] = []

    def _setup_fit(self) -> None:
        self.setup_calls.append("fit")
        self.train = "train"
        self.val = "validate"

    def _setup_validate(self) -> None:
        self.setup_calls.append("validate")
        self.val = "validate"

    def get_dataloader(self, train: bool):
        return self.train if train else self.val


class _AllStageDataModule(_FitValidateDataModule):
    def _setup_test(self) -> None:
        self.setup_calls.append("test")
        self.test = "test"

    def _get_test_dataloader(self):
        return self.test


FIT_VALIDATE_ADAPTERS = (
    ("emperor.datasets.image.captioning._coco_captions", "CocoCaptions"),
    ("emperor.datasets.image.captioning._flickr30k", "Flickr30k"),
    ("emperor.datasets.image.captioning._flickr8k", "Flickr8k"),
    ("emperor.datasets.image.classification._svhn", "SVHN"),
    ("emperor.datasets.image.detection._coco", "CocoDetection"),
    ("emperor.datasets.image.detection._voc", "VOCDetection"),
    ("emperor.datasets.image.segmentation._cityscapes", "Cityscapes"),
    ("emperor.datasets.image.segmentation._coco", "CocoSegmentation"),
    ("emperor.datasets.image.segmentation._voc", "VOCSegmentation"),
    ("emperor.datasets.multimodal.vqa._gqa", "GQA"),
    ("emperor.datasets.multimodal.vqa._vqa_v2", "VQAv2"),
    ("emperor.datasets.rl._acrobot", "Acrobot"),
    ("emperor.datasets.rl._cart_pole", "CartPole"),
    ("emperor.datasets.rl._frozen_lake", "FrozenLake"),
    ("emperor.datasets.rl._lunar_lander", "LunarLander"),
    ("emperor.datasets.rl._mountain_car", "MountainCar"),
    ("emperor.datasets.rl._pendulum", "Pendulum"),
    ("emperor.datasets.text.classification._ag_news", "AgNews"),
    ("emperor.datasets.text.classification._dbpedia", "DBpedia"),
    ("emperor.datasets.text.classification._imdb", "IMDB"),
    ("emperor.datasets.text.classification._yelp_review_full", "YelpReviewFull"),
    ("emperor.datasets.text.language_modeling._wiki_text_103", "WikiText103"),
    ("emperor.datasets.text.ner._conll2003", "CoNLL2003"),
    ("emperor.datasets.text.nli._multi_nli", "MultiNLI"),
    ("emperor.datasets.text.nli._snli", "SNLI"),
    ("emperor.datasets.text.question_answering._squad_v1", "SQuADv1"),
    ("emperor.datasets.text.question_answering._squad_v2", "SQuADv2"),
    ("emperor.datasets.text.similarity._stsb", "STSb"),
    ("emperor.datasets.text.summarization._cnn_dailymail", "CnnDailyMail"),
    ("emperor.datasets.text.summarization._xsum", "XSum"),
)

ALL_STAGE_ADAPTERS = (
    ("emperor.datasets.image.classification._cifar_10", "Cifar10"),
    ("emperor.datasets.image.classification._cifar_100", "Cifar100"),
    ("emperor.datasets.image.classification._fashion_mnist", "FashionMNIST"),
    ("emperor.datasets.image.classification._mnist", "Mnist"),
    (
        "emperor.datasets.text.bert_pretraining._datasets",
        "PennTreebankBertPretraining",
    ),
    (
        "emperor.datasets.text.bert_pretraining._datasets",
        "WikiText2BertPretraining",
    ),
    ("emperor.datasets.text.language_modeling._penn_treebank", "PennTreebank"),
    ("emperor.datasets.text.language_modeling._wiki_text_2", "WikiText2"),
    (
        "emperor.datasets.text.masked_language_modeling._datasets",
        "PennTreebankMaskedLanguageModeling",
    ),
    (
        "emperor.datasets.text.masked_language_modeling._datasets",
        "WikiText2MaskedLanguageModeling",
    ),
    (
        "emperor.datasets.text.masked_language_modeling._datasets",
        "WikiText103MaskedLanguageModeling",
    ),
    ("emperor.datasets.text.translation._adapter", "Multi30kDeEn"),
    ("emperor.datasets.text.translation._adapter", "Multi30kEnDe"),
)


class TestDatasetStageContracts(unittest.TestCase):
    def test_setup_none_initializes_each_supported_dataset_once(self) -> None:
        data = _AllStageDataModule()

        data.setup(None)

        self.assertEqual(data.setup_calls, ["fit", "test"])
        self.assertEqual(data.train_dataloader(), "train")
        self.assertEqual(data.val_dataloader(), "validate")
        self.assertEqual(data.test_dataloader(), "test")

    def test_setup_none_does_not_invent_a_test_split(self) -> None:
        data = _FitValidateDataModule()

        data.setup(None)

        self.assertEqual(data.setup_calls, ["fit"])
        self.assertEqual(data.train_dataloader(), "train")
        self.assertEqual(data.val_dataloader(), "validate")
        with self.assertRaisesRegex(
            NotImplementedError,
            "_FitValidateDataModule does not support the 'test' stage",
        ):
            data.test_dataloader()

    def test_unknown_and_unsupported_stages_fail_descriptively(self) -> None:
        data = _FitValidateDataModule()

        with self.assertRaisesRegex(ValueError, "Unsupported dataset setup stage"):
            data.setup("predict")
        with self.assertRaisesRegex(
            NotImplementedError,
            "_FitValidateDataModule does not support the 'test' stage",
        ):
            data.setup("test")

    def test_loaders_fail_before_their_stage_is_ready(self) -> None:
        data = _AllStageDataModule()

        for loader_name, setup_hint in (
            ("train_dataloader", "call setup\\('fit'\\)"),
            (
                "val_dataloader",
                "call setup\\('fit'\\) or setup\\('validate'\\)",
            ),
            ("test_dataloader", "call setup\\('test'\\)"),
        ):
            with self.subTest(loader=loader_name):
                with self.assertRaisesRegex(
                    RuntimeError,
                    setup_hint,
                ):
                    getattr(data, loader_name)()

    def test_wrong_stage_does_not_make_unrelated_loader_ready(self) -> None:
        data = _AllStageDataModule()

        data.setup("validate")

        self.assertEqual(data.val_dataloader(), "validate")
        with self.assertRaisesRegex(RuntimeError, r"call setup\('fit'\)"):
            data.train_dataloader()
        with self.assertRaisesRegex(RuntimeError, r"call setup\('test'\)"):
            data.test_dataloader()

    def test_every_hook_based_adapter_reports_its_actual_stage_capability(self) -> None:
        cases = (
            (FIT_VALIDATE_ADAPTERS, frozenset({"fit", "validate"})),
            (ALL_STAGE_ADAPTERS, frozenset({"fit", "validate", "test"})),
        )
        for adapters, expected_stages in cases:
            for module_name, class_name in adapters:
                with self.subTest(adapter=f"{module_name}.{class_name}"):
                    adapter_type = getattr(
                        importlib.import_module(module_name), class_name
                    )
                    adapter = adapter_type.__new__(adapter_type)
                    self.assertEqual(adapter._supported_stages(), expected_stages)


if __name__ == "__main__":
    unittest.main()
