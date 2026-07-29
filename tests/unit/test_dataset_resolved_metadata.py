from __future__ import annotations

import ast
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from emperor.datasets.multimodal.vqa._vqa_v2 import VQAv2
from emperor.datasets.text.language_modeling import PennTreebank

_MUTABLE_CATALOG_FIELDS = {
    "flattened_input_dim",
    "num_classes",
    "vocab_size",
}


class _InMemoryPennTreebank(PennTreebank):
    def __init__(self, corpus: tuple[str, ...]) -> None:
        super().__init__(sequence_length=2, num_workers=0, drop_last=False)
        self._corpus = corpus

    def _dataset(self, split: str):
        del split
        return iter(self._corpus)


class _FakeVocabulary:
    def __init__(self, tokens) -> None:
        unique_tokens = {token for token_group in tokens for token in token_group}
        self._tokens = ("<unk>", "<pad>", *sorted(unique_tokens))

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, token: str) -> int:
        return self._tokens.index(token)

    def set_default_index(self, index: int) -> None:
        self.default_index = index


def _build_fake_vocabulary(tokens, *, specials):
    del specials
    return _FakeVocabulary(tokens)


def _catalog_metadata(dataset_type: type) -> dict[str, int]:
    return {
        field: getattr(dataset_type, field)
        for field in _MUTABLE_CATALOG_FIELDS
        if hasattr(dataset_type, field)
    }


class DatasetResolvedMetadataTests(unittest.TestCase):
    def test_causal_instances_resolve_different_schemas_without_catalog_mutation(
        self,
    ) -> None:
        expected_catalog = _catalog_metadata(PennTreebank)
        first = _InMemoryPennTreebank(("red green blue",))
        second = _InMemoryPennTreebank(("one two three four five",))

        first._build_vocab()
        second._build_vocab()

        self.assertEqual(_catalog_metadata(PennTreebank), expected_catalog)
        self.assertEqual(first.resolved_metadata.vocab_size, 4)
        self.assertEqual(first.resolved_metadata.flattened_input_dim, 4)
        self.assertEqual(first.resolved_metadata.num_classes, 4)
        self.assertEqual(second.resolved_metadata.vocab_size, 6)
        self.assertEqual(second.resolved_metadata.flattened_input_dim, 6)
        self.assertEqual(second.resolved_metadata.num_classes, 6)
        self.assertIsNot(first.resolved_metadata, second.resolved_metadata)

    def test_parallel_causal_resolution_does_not_share_state(self) -> None:
        expected_catalog = _catalog_metadata(PennTreebank)
        corpora = (("alpha beta",), ("one two three four",))

        def resolve(corpus: tuple[str, ...]):
            dataset = _InMemoryPennTreebank(corpus)
            dataset._build_vocab()
            return dataset.resolved_metadata

        with ThreadPoolExecutor(max_workers=2) as executor:
            first, second = executor.map(resolve, corpora)

        self.assertEqual(_catalog_metadata(PennTreebank), expected_catalog)
        self.assertEqual(first.vocab_size, 3)
        self.assertEqual(second.vocab_size, 5)

    def test_vqa_resolves_question_and_answer_dimensions_on_its_instance(self) -> None:
        expected_catalog = _catalog_metadata(VQAv2)
        questions = {
            1: {"question": "red shape"},
            2: {"question": "blue shape"},
        }
        annotations = {
            1: {"multiple_choice_answer": "circle"},
            2: {"multiple_choice_answer": "square"},
        }
        dataset = VQAv2(num_answer_classes=2)

        with patch(
            "emperor.datasets.multimodal.vqa._vqa_v2.build_vocab_from_iterator",
            side_effect=_build_fake_vocabulary,
        ):
            dataset._build_vocabs(questions, annotations)

        self.assertEqual(_catalog_metadata(VQAv2), expected_catalog)
        self.assertEqual(dataset.resolved_metadata.vocab_size, 5)
        self.assertEqual(dataset.resolved_metadata.num_classes, 2)
        self.assertIsNone(dataset.resolved_metadata.flattened_input_dim)

    def test_dataset_sources_do_not_assign_resolved_values_to_catalog_classes(
        self,
    ) -> None:
        datasets_root = Path(__file__).parents[2] / "src" / "emperor" / "datasets"
        violations = []
        for source_path in datasets_root.rglob("*.py"):
            tree = ast.parse(source_path.read_text(), filename=str(source_path))
            for node in ast.walk(tree):
                targets = []
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                for target in targets:
                    if not isinstance(target, ast.Attribute):
                        continue
                    if target.attr not in _MUTABLE_CATALOG_FIELDS:
                        continue
                    if isinstance(target.value, ast.Name) and target.value.id == "self":
                        continue
                    violations.append(
                        f"{source_path.relative_to(datasets_root)}:{node.lineno}"
                    )

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
