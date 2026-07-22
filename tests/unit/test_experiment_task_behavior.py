from __future__ import annotations

import unittest
from types import SimpleNamespace

from emperor.experiments import ExperimentTask
from model_runtime.task_behavior import (
    EXPERIMENT_TASK_BEHAVIORS,
    experiment_task_behavior,
)


class ExperimentTaskBehaviorTests(unittest.TestCase):
    def test_registry_is_exhaustive_and_unique(self) -> None:
        self.assertEqual(set(EXPERIMENT_TASK_BEHAVIORS), set(ExperimentTask))
        self.assertEqual(
            [behavior.task for behavior in EXPERIMENT_TASK_BEHAVIORS.values()],
            list(EXPERIMENT_TASK_BEHAVIORS),
        )
        with self.assertRaises(TypeError):
            EXPERIMENT_TASK_BEHAVIORS[ExperimentTask.IMAGE_CLASSIFICATION] = None  # type: ignore[index]

    def test_dataset_arguments_are_owned_by_task_policy(self) -> None:
        token_configuration = SimpleNamespace(batch_size=7, sequence_length=19)
        translation_configuration = SimpleNamespace(
            batch_size=5,
            experiment_config=SimpleNamespace(
                source_sequence_length=31,
                target_sequence_length=29,
            ),
        )

        self.assertEqual(
            experiment_task_behavior(
                ExperimentTask.IMAGE_CLASSIFICATION
            ).dataset_constructor_kwargs(token_configuration),
            {"batch_size": 7},
        )
        self.assertEqual(
            experiment_task_behavior(
                ExperimentTask.BERT_PRETRAINING
            ).dataset_constructor_kwargs(token_configuration),
            {"batch_size": 7},
        )
        self.assertEqual(
            experiment_task_behavior(
                ExperimentTask.CAUSAL_LANGUAGE_MODELING
            ).dataset_constructor_kwargs(token_configuration),
            {"batch_size": 7, "sequence_length": 19},
        )
        self.assertEqual(
            experiment_task_behavior(
                ExperimentTask.TEXT_TRANSLATION
            ).dataset_constructor_kwargs(translation_configuration),
            {
                "batch_size": 5,
                "source_sequence_length": 31,
                "target_sequence_length": 29,
            },
        )

    def test_ranking_metric_preference_and_direction_are_task_owned(self) -> None:
        classification = experiment_task_behavior(ExperimentTask.IMAGE_CLASSIFICATION)
        language_model = experiment_task_behavior(
            ExperimentTask.CAUSAL_LANGUAGE_MODELING
        )
        translation = experiment_task_behavior(ExperimentTask.TEXT_TRANSLATION)

        self.assertGreater(
            classification.ranking_score({"metrics": {"validation_accuracy": 0.9}}),
            classification.ranking_score({"metrics": {"validation/accuracy": 0.7}}),
        )
        self.assertGreater(
            language_model.ranking_score({"metrics": {"validation/loss": 0.2}}),
            language_model.ranking_score({"metrics": {"validation_loss": 0.8}}),
        )
        self.assertGreater(
            translation.ranking_score({"metrics": {"validation/bleu": 1.0}}),
            translation.ranking_score({"metrics": {"validation/loss": 0.1}}),
        )


if __name__ == "__main__":
    unittest.main()
