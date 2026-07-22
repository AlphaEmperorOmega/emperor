import importlib
import unittest


class DatasetInterfaceTests(unittest.TestCase):
    NAMESPACE_MODULES = (
        "emperor.datasets.image.classification",
        "emperor.datasets.text.bert_pretraining",
        "emperor.datasets.text.language_modeling",
        "emperor.datasets.text.translation",
    )
    PACKAGE_EXPORTS = {
        "emperor.datasets.image.classification": (
            "Mnist",
            "FashionMNIST",
            "Cifar10",
            "Cifar100",
        ),
        "emperor.datasets.text.bert_pretraining": (
            "BERT_PRETRAINING_TARGET_VOCAB_SIZE",
            "PennTreebankBertPretraining",
            "WikiText2BertPretraining",
        ),
        "emperor.datasets.text.language_modeling": ("PennTreebank", "WikiText2"),
        "emperor.datasets.text.translation": ("Multi30kDeEn", "Multi30kEnDe"),
    }
    OWNER_CASES = (
        ("emperor.datasets.image.classification._mnist", "Mnist"),
        ("emperor.datasets.image.classification._fashion_mnist", "FashionMNIST"),
        ("emperor.datasets.image.classification._cifar_10", "Cifar10"),
        ("emperor.datasets.image.classification._cifar_100", "Cifar100"),
        (
            "emperor.datasets.text.bert_pretraining._tokenizer",
            "BERT_PRETRAINING_TARGET_VOCAB_SIZE",
        ),
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
        ("emperor.datasets.text.translation._adapter", "Multi30kDeEn"),
        ("emperor.datasets.text.translation._adapter", "Multi30kEnDe"),
    )

    def test_dataset_packages_eagerly_export_supported_classes(self) -> None:
        for module_name in self.NAMESPACE_MODULES:
            module = importlib.import_module(module_name)
            with self.subTest(module=module_name):
                self.assertEqual(module.__all__, self.PACKAGE_EXPORTS[module_name])
                for export_name in module.__all__:
                    self.assertTrue(hasattr(module, export_name))

    def test_dataset_packages_have_no_lazy_facade_mechanics(self) -> None:
        for module_name in self.NAMESPACE_MODULES:
            module = importlib.import_module(module_name)
            with self.subTest(module=module_name):
                self.assertFalse(hasattr(module, "__getattr__"))
                self.assertFalse(hasattr(module, "_LAZY_EXPORTS"))

    def test_dataset_roots_export_symbols_from_their_original_owner_modules(self) -> None:
        for owner_name, export_name in self.OWNER_CASES:
            with self.subTest(owner=owner_name, export=export_name):
                owner = importlib.import_module(owner_name)
                value = getattr(owner, export_name)
                self.assertFalse(hasattr(owner, "__all__"))
                if hasattr(value, "__module__"):
                    self.assertEqual(value.__module__, owner_name)
                package_name = owner_name.rsplit(".", 1)[0]
                package = importlib.import_module(package_name)
                self.assertIs(getattr(package, export_name), value)

    def test_mistaken_renamed_owner_modules_are_absent(self) -> None:
        retired_modules = (
            "emperor.datasets.image.classification.mnist",
            "emperor.datasets.image.classification.fashion_mnist",
            "emperor.datasets.image.classification.cifar_10",
            "emperor.datasets.image.classification.cifar_100",
            "emperor.datasets.text.bert_pretraining.constants",
            "emperor.datasets.text.bert_pretraining.datasets",
            "emperor.datasets.text.language_modeling.penn_treebank",
            "emperor.datasets.text.language_modeling.wiki_text_2",
            "emperor.datasets.text.translation.multi30k",
        )
        for module_name in retired_modules:
            with self.subTest(module=module_name):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
