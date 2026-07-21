import json
import subprocess
import sys
import unittest

from emperor.datasets.image import classification
from emperor.datasets.text import bert_pretraining, language_modeling, translation


class DatasetInterfaceTests(unittest.TestCase):
    LAZY_EXPORT_CASES = (
        (
            "emperor.datasets.image.classification",
            "Mnist",
            "emperor.datasets.image.classification._mnist",
        ),
        (
            "emperor.datasets.image.classification",
            "FashionMNIST",
            "emperor.datasets.image.classification._fashion_mnist",
        ),
        (
            "emperor.datasets.image.classification",
            "Cifar10",
            "emperor.datasets.image.classification._cifar_10",
        ),
        (
            "emperor.datasets.image.classification",
            "Cifar100",
            "emperor.datasets.image.classification._cifar_100",
        ),
        (
            "emperor.datasets.text.bert_pretraining",
            "BERT_PRETRAINING_TARGET_VOCAB_SIZE",
            "emperor.datasets.text.bert_pretraining._tokenizer",
        ),
        (
            "emperor.datasets.text.bert_pretraining",
            "PennTreebankBertPretraining",
            "emperor.datasets.text.bert_pretraining._datasets",
        ),
        (
            "emperor.datasets.text.bert_pretraining",
            "WikiText2BertPretraining",
            "emperor.datasets.text.bert_pretraining._datasets",
        ),
        (
            "emperor.datasets.text.language_modeling",
            "PennTreebank",
            "emperor.datasets.text.language_modeling._penn_treebank",
        ),
        (
            "emperor.datasets.text.language_modeling",
            "WikiText2",
            "emperor.datasets.text.language_modeling._wiki_text_2",
        ),
        (
            "emperor.datasets.text.translation",
            "Multi30kDeEn",
            "emperor.datasets.text.translation._adapter",
        ),
        (
            "emperor.datasets.text.translation",
            "Multi30kEnDe",
            "emperor.datasets.text.translation._adapter",
        ),
    )

    def test_supported_dataset_exports_are_exact(self) -> None:
        self.assertEqual(
            classification.__all__,
            ("Mnist", "FashionMNIST", "Cifar10", "Cifar100"),
        )
        self.assertEqual(
            bert_pretraining.__all__,
            (
                "BERT_PRETRAINING_TARGET_VOCAB_SIZE",
                "PennTreebankBertPretraining",
                "WikiText2BertPretraining",
            ),
        )
        self.assertEqual(language_modeling.__all__, ("PennTreebank", "WikiText2"))
        self.assertEqual(translation.__all__, ("Multi30kDeEn", "Multi30kEnDe"))

    def test_unsupported_historical_names_are_not_exported(self) -> None:
        for module, name in (
            (classification, "SVHN"),
            (language_modeling, "WikiText103"),
            (translation, "Multi30k"),
        ):
            with self.subTest(module=module.__name__, name=name):
                with self.assertRaises(AttributeError):
                    getattr(module, name)

    def test_dataset_facades_are_offline_and_lightweight(self) -> None:
        script = "\n".join(
            (
                "import sys",
                "import emperor.datasets.image.classification",
                "import emperor.datasets.text.bert_pretraining",
                "import emperor.datasets.text.language_modeling",
                "import emperor.datasets.text.translation",
                "heavy = ('torch', 'torchvision', 'torchtext', 'tokenizers')",
                "assert not any(name in sys.modules for name in heavy)",
            )
        )
        completed = subprocess.run(
            [sys.executable, "-P", "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_supported_exports_load_from_declared_private_owners(self) -> None:
        for facade_name, export_name, owner_name in self.LAZY_EXPORT_CASES:
            with self.subTest(facade=facade_name, export=export_name):
                script = f"""
import importlib
import json
import sys

facade = importlib.import_module({facade_name!r})
owner_loaded_before = {owner_name!r} in sys.modules
value = getattr(facade, {export_name!r})
owner = importlib.import_module({owner_name!r})
value_owner = getattr(value, "__module__", None)
owner_matches = (
    value_owner == {owner_name!r}
    if value_owner is not None
    else getattr(owner, {export_name!r}) == value
)
print(json.dumps({{
    "owner_loaded_before": owner_loaded_before,
    "owner_loaded_after": {owner_name!r} in sys.modules,
    "owner_matches": owner_matches,
}}))
"""
                completed = subprocess.run(
                    [sys.executable, "-P", "-c", script],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    msg=(f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"),
                )
                self.assertEqual(
                    json.loads(completed.stdout),
                    {
                        "owner_loaded_before": False,
                        "owner_loaded_after": True,
                        "owner_matches": True,
                    },
                )


if __name__ == "__main__":
    unittest.main()
