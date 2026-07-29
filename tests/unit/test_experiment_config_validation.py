import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from emperor.config import ModelConfig
from emperor.experiments.bert_pretraining import BertPretrainingExperiment
from emperor.experiments.classifier import ClassifierExperiment
from emperor.experiments.language_model import LanguageModelExperiment
from emperor.experiments.masked_language_model import MaskedLanguageModelExperiment
from emperor.experiments.sequence_classifier import SequenceClassifierExperiment
from emperor.experiments.translation import TranslationExperiment

COMMON_EXPERIMENT_CASES = (
    ("BERT-pretraining", BertPretrainingExperiment, "vocab_size"),
    ("Language-model", LanguageModelExperiment, "vocab_size"),
    ("Masked-language-model", MaskedLanguageModelExperiment, "vocab_size"),
    ("Sequence-classifier", SequenceClassifierExperiment, "num_classes"),
    ("Classifier", ClassifierExperiment, "num_classes"),
    ("Translation", TranslationExperiment, "vocab_size"),
)


def _config(
    *,
    learning_rate: object = 1e-3,
    output_dim: object = 4,
    hidden_dim: object = 8,
    experiment_config: object = None,
) -> ModelConfig:
    return ModelConfig(
        learning_rate=learning_rate,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        experiment_config=experiment_config,
    )


class TestSharedExperimentConfigValidation(unittest.TestCase):
    def test_all_experiments_accept_zero_learning_rate_and_resolved_output_dim(self):
        for label, experiment_type, output_attribute in COMMON_EXPERIMENT_CASES:
            with self.subTest(experiment=label):
                experiment = experiment_type(_config(learning_rate=0, output_dim=4))

                self.assertEqual(experiment.learning_rate, 0)
                self.assertEqual(getattr(experiment, output_attribute), 4)

    def test_zero_learning_rate_reaches_each_optimizer_unchanged(self):
        for label, experiment_type, _ in COMMON_EXPERIMENT_CASES:
            with self.subTest(experiment=label):
                experiment = experiment_type(_config(learning_rate=0))
                experiment.probe = torch.nn.Parameter(torch.ones(()))

                configured = experiment.configure_optimizers()
                optimizer = (
                    configured["optimizer"]
                    if isinstance(configured, dict)
                    else configured
                )

                self.assertEqual(optimizer.param_groups[0]["lr"], 0)

    def test_token_experiments_accept_the_positive_output_boundary(self):
        token_experiment_cases = (
            BertPretrainingExperiment,
            LanguageModelExperiment,
            MaskedLanguageModelExperiment,
            TranslationExperiment,
        )

        for experiment_type in token_experiment_cases:
            with self.subTest(experiment=experiment_type.__name__):
                experiment = experiment_type(_config(output_dim=1))

                self.assertEqual(experiment.vocab_size, 1)

    def test_all_experiments_reject_invalid_learning_rates(self):
        invalid_values = (None, True, "1e-3", -1e-3, float("nan"), float("inf"))

        for label, experiment_type, _ in COMMON_EXPERIMENT_CASES:
            for value in invalid_values:
                with (
                    self.subTest(experiment=label, learning_rate=value),
                    self.assertRaisesRegex(
                        ValueError,
                        rf"{label} config\.learning_rate",
                    ),
                ):
                    experiment_type(_config(learning_rate=value))

    def test_all_experiments_reject_invalid_output_dimensions(self):
        invalid_values = (None, True, "4", 4.0, 0, -1)

        for label, experiment_type, _ in COMMON_EXPERIMENT_CASES:
            for value in invalid_values:
                with (
                    self.subTest(experiment=label, output_dim=value),
                    self.assertRaisesRegex(
                        ValueError,
                        rf"{label} config\.output_dim",
                    ),
                ):
                    experiment_type(_config(output_dim=value))

    def test_classifier_experiments_require_at_least_two_classes(self):
        classifier_cases = (
            ("Sequence-classifier", SequenceClassifierExperiment),
            ("Classifier", ClassifierExperiment),
        )

        for label, experiment_type in classifier_cases:
            with (
                self.subTest(experiment=label),
                self.assertRaisesRegex(
                    ValueError,
                    rf"{label} config\.output_dim",
                ),
            ):
                experiment_type(_config(output_dim=1))

    def test_common_validation_precedes_task_dependency_construction(self):
        random_state = torch.random.get_rng_state()

        with (
            patch(
                "emperor.experiments.classifier._experiment.nn.CrossEntropyLoss"
            ) as loss_type,
            patch(
                "emperor.experiments.classifier._experiment.ClassifierMetricsLogger"
            ) as metrics_type,
            self.assertRaisesRegex(ValueError, r"Classifier config\.output_dim"),
        ):
            ClassifierExperiment(_config(output_dim=None))

        loss_type.assert_not_called()
        metrics_type.assert_not_called()
        torch.testing.assert_close(torch.random.get_rng_state(), random_state)

    def test_common_validation_has_stable_field_precedence(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Language-model config\.learning_rate",
        ):
            LanguageModelExperiment(_config(learning_rate=None, output_dim=None))


class TestTranslationExperimentConfigValidation(unittest.TestCase):
    def test_translation_defaults_and_boundary_values_remain_valid(self):
        default_experiment = TranslationExperiment(_config(learning_rate=0))
        self.assertEqual(default_experiment.pad_token_id, 0)
        self.assertEqual(default_experiment.label_smoothing, 0.1)
        self.assertEqual(default_experiment.warmup_steps, 4_000)
        self.assertIs(default_experiment.generation_metrics_flag, True)

        for label_smoothing in (0, 1):
            with self.subTest(label_smoothing=label_smoothing):
                experiment = TranslationExperiment(
                    _config(
                        experiment_config=SimpleNamespace(
                            pad_token_id=-100,
                            label_smoothing=label_smoothing,
                            warmup_steps=1,
                            generation_metrics_flag=False,
                        )
                    )
                )

                self.assertEqual(experiment.pad_token_id, -100)
                self.assertEqual(experiment.label_smoothing, float(label_smoothing))
                self.assertEqual(experiment.warmup_steps, 1)
                self.assertIs(experiment.generation_metrics_flag, False)

    def test_translation_rejects_invalid_task_specific_values(self):
        invalid_cases = (
            ("hidden_dim", None, r"Translation config\.hidden_dim"),
            ("hidden_dim", True, r"Translation config\.hidden_dim"),
            ("hidden_dim", 0, r"Translation config\.hidden_dim"),
            ("pad_token_id", 0.0, r"Translation config\.pad_token_id"),
            ("pad_token_id", True, r"Translation config\.pad_token_id"),
            ("label_smoothing", None, r"Translation config\.label_smoothing"),
            ("label_smoothing", -0.1, r"Translation config\.label_smoothing"),
            ("label_smoothing", 1.1, r"Translation config\.label_smoothing"),
            (
                "label_smoothing",
                float("nan"),
                r"Translation config\.label_smoothing",
            ),
            (
                "label_smoothing",
                float("inf"),
                r"Translation config\.label_smoothing",
            ),
            ("warmup_steps", None, r"Translation config\.warmup_steps"),
            ("warmup_steps", True, r"Translation config\.warmup_steps"),
            ("warmup_steps", 0, r"Translation config\.warmup_steps"),
            (
                "generation_metrics_flag",
                "false",
                r"Translation config\.generation_metrics_flag",
            ),
            (
                "generation_metrics_flag",
                1,
                r"Translation config\.generation_metrics_flag",
            ),
        )

        for field_name, value, message in invalid_cases:
            with self.subTest(field=field_name, value=value):
                if field_name == "hidden_dim":
                    config = _config(hidden_dim=value)
                else:
                    config = _config(
                        experiment_config=SimpleNamespace(**{field_name: value})
                    )
                with self.assertRaisesRegex(ValueError, message):
                    TranslationExperiment(config)

    def test_translation_validation_has_stable_field_precedence(self):
        all_invalid = _config(
            hidden_dim=0,
            experiment_config=SimpleNamespace(
                pad_token_id=True,
                label_smoothing=None,
                warmup_steps=0,
                generation_metrics_flag="false",
            ),
        )
        with self.assertRaisesRegex(ValueError, r"Translation config\.hidden_dim"):
            TranslationExperiment(all_invalid)

        invalid_task_config = _config(
            experiment_config=SimpleNamespace(
                pad_token_id=True,
                label_smoothing=None,
                warmup_steps=0,
                generation_metrics_flag="false",
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            r"Translation config\.pad_token_id",
        ):
            TranslationExperiment(invalid_task_config)

    def test_translation_validation_precedes_loss_construction_and_rng_use(self):
        random_state = torch.random.get_rng_state()
        config = _config(
            experiment_config=SimpleNamespace(generation_metrics_flag="false")
        )

        with (
            patch(
                "emperor.experiments.translation._experiment.nn.CrossEntropyLoss"
            ) as loss_type,
            self.assertRaisesRegex(
                ValueError,
                r"Translation config\.generation_metrics_flag",
            ),
        ):
            TranslationExperiment(config)

        loss_type.assert_not_called()
        torch.testing.assert_close(torch.random.get_rng_state(), random_state)


if __name__ == "__main__":
    unittest.main()
