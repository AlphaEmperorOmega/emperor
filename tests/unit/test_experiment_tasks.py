import unittest

from emperor.experiments import (
    ExperimentTask,
    experiment_task_label,
    experiment_task_name,
    resolve_experiment_task,
)


class TestExperimentTaskResolution(unittest.TestCase):
    def test_every_cli_label_resolves_to_the_exact_declared_owner(self) -> None:
        expected = (
            (
                ExperimentTask.IMAGE_CLASSIFICATION,
                1,
                "IMAGE_CLASSIFICATION",
                "image-classification",
                "Image Classification",
            ),
            (
                ExperimentTask.BERT_PRETRAINING,
                2,
                "BERT_PRETRAINING",
                "bert-pretraining",
                "Bert Pretraining",
            ),
            (
                ExperimentTask.TEXT_TRANSLATION,
                3,
                "TEXT_TRANSLATION",
                "text-translation",
                "Text Translation",
            ),
            (
                ExperimentTask.CAUSAL_LANGUAGE_MODELING,
                4,
                "CAUSAL_LANGUAGE_MODELING",
                "causal-language-modeling",
                "Causal Language Modeling",
            ),
        )

        self.assertEqual(tuple(ExperimentTask), tuple(row[0] for row in expected))
        for task, value, enum_name, cli_name, label in expected:
            with self.subTest(task=enum_name):
                self.assertEqual(task.value, value)
                self.assertEqual(task.name, enum_name)
                self.assertEqual(experiment_task_name(task), cli_name)
                self.assertEqual(experiment_task_label(task), label)
                for literal in (task, cli_name, enum_name, enum_name.title()):
                    self.assertIs(resolve_experiment_task(literal), task)
                    self.assertIs(resolve_experiment_task(literal), task)

        self.assertIsNone(resolve_experiment_task(None))
        for invalid in (" image-classification", "image-classification ", "unknown"):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "does not exist"),
            ):
                resolve_experiment_task(invalid)


if __name__ == "__main__":
    unittest.main()
