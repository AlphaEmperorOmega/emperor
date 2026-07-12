import unittest

import torch
import torch.nn as nn

from emperor.config import ModelConfig
from emperor.experiments.classifier import ClassifierExperiment, ClassifierMetricsLogger


class HealthProbeClassifier(ClassifierExperiment):
    def __init__(self):
        super().__init__(ModelConfig(input_dim=1, output_dim=2))
        self.probe = nn.Parameter(torch.tensor([3.0, 4.0]))
        self.bad_probe = nn.Parameter(torch.tensor([0.0, 0.0]))

    def forward(self, X):
        return torch.zeros(X.size(0), self.num_classes, device=X.device)


class AuxiliaryLossProbeClassifier(ClassifierExperiment):
    def __init__(self):
        super().__init__(ModelConfig(input_dim=1, output_dim=2))
        self.auxiliary_loss = torch.tensor(0.25)

        def fail_item():
            raise AssertionError("auxiliary loss should stay as a tensor")

        self.auxiliary_loss.item = fail_item

    def forward(self, X):
        return (
            torch.zeros(X.size(0), self.num_classes, device=X.device),
            self.auxiliary_loss.to(X.device),
        )


class FakeTensorBoardExperiment:
    def __init__(self):
        self.images = []
        self.text = []

    def add_image(self, tag, image, global_step):
        self.images.append((tag, image, global_step))

    def add_text(self, tag, text_string, global_step):
        self.text.append((tag, text_string, global_step))


class FakeLogger:
    def __init__(self):
        self.experiment = FakeTensorBoardExperiment()


class TestClassifierMetricsLogger(unittest.TestCase):
    def test_epoch_metrics_and_gap_are_sample_weighted(self):
        logger = ClassifierMetricsLogger(num_classes=2)

        logger.log_training_step(
            self._discard_log,
            torch.tensor(0.2),
            torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
            torch.tensor([0, 1]),
        )
        logger.log_training_step(
            self._discard_log,
            torch.tensor(0.6),
            torch.tensor([[4.0, 0.0], [4.0, 0.0]]),
            torch.tensor([0, 1]),
        )
        logger.log_validation_step(
            self._discard_log,
            torch.tensor(0.7),
            torch.tensor([[4.0, 0.0], [4.0, 0.0]]),
            torch.tensor([0, 1]),
        )

        train_metrics = logger.train_epoch_metrics()
        validation_metrics = logger.validation_epoch_metrics()
        gap_metrics = logger.train_validation_gap_metrics()

        torch.testing.assert_close(
            train_metrics["train/loss_epoch"],
            torch.tensor(0.4),
        )
        torch.testing.assert_close(
            train_metrics["train/accuracy_epoch"],
            torch.tensor(0.75),
        )
        torch.testing.assert_close(
            validation_metrics["validation/loss_epoch"],
            torch.tensor(0.7),
        )
        torch.testing.assert_close(
            validation_metrics["validation/accuracy_epoch"],
            torch.tensor(0.5),
        )
        torch.testing.assert_close(gap_metrics["gap/accuracy"], torch.tensor(0.25))
        torch.testing.assert_close(gap_metrics["gap/loss"], torch.tensor(0.3))

    def test_gap_metrics_are_skipped_until_both_sides_exist(self):
        logger = ClassifierMetricsLogger(num_classes=2)

        logger.log_validation_step(
            self._discard_log,
            torch.tensor(0.7),
            torch.tensor([[4.0, 0.0], [4.0, 0.0]]),
            torch.tensor([0, 1]),
        )

        self.assertEqual(logger.train_validation_gap_metrics(), {})

    def test_epoch_resets_clear_gap_inputs(self):
        logger = ClassifierMetricsLogger(num_classes=2)

        logger.log_training_step(
            self._discard_log,
            torch.tensor(0.2),
            torch.tensor([[4.0, 0.0]]),
            torch.tensor([0]),
        )
        logger.log_validation_step(
            self._discard_log,
            torch.tensor(0.7),
            torch.tensor([[4.0, 0.0]]),
            torch.tensor([1]),
        )
        self.assertIn("gap/loss", logger.train_validation_gap_metrics())

        logger.reset_train_epoch()

        self.assertEqual(logger.train_epoch_metrics(), {})
        self.assertEqual(logger.train_validation_gap_metrics(), {})

    def test_epoch_logging_uses_expected_tags_and_lightning_options(self):
        logger = ClassifierMetricsLogger(num_classes=2)
        calls = []

        def log_fn(payload, **kwargs):
            calls.append((payload, kwargs))

        logger.log_training_step(
            self._discard_log,
            torch.tensor(0.2),
            torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
            torch.tensor([0, 1]),
        )
        logger.log_validation_step(
            self._discard_log,
            torch.tensor(0.7),
            torch.tensor([[4.0, 0.0], [4.0, 0.0]]),
            torch.tensor([0, 1]),
        )

        logger.log_train_epoch(log_fn)
        logger.log_validation_epoch_and_gap(log_fn)

        self.assertEqual(
            calls[0][1],
            {"prog_bar": True, "on_step": False, "on_epoch": True},
        )
        self.assertIn("train/loss_epoch", calls[0][0])
        self.assertIn("train/accuracy_epoch", calls[0][0])
        self.assertEqual(
            calls[1][1],
            {"prog_bar": False, "on_step": False, "on_epoch": True},
        )
        self.assertIn("train/per_class/class_0/accuracy", calls[1][0])
        self.assertIn(
            "train/confusion_matrix/true_class_0/predicted_class_0/count",
            calls[1][0],
        )
        self.assertIn("train/confidence/mean", calls[1][0])
        self.assertEqual(
            calls[2][1],
            {"prog_bar": True, "on_step": False, "on_epoch": True},
        )
        self.assertIn("validation/loss_epoch", calls[2][0])
        self.assertIn("validation/accuracy_epoch", calls[2][0])
        self.assertIn("gap/loss", calls[2][0])
        self.assertIn("gap/accuracy", calls[2][0])
        self.assertEqual(
            calls[3][1],
            {"prog_bar": False, "on_step": False, "on_epoch": True},
        )
        self.assertIn("validation/per_class/class_0/accuracy", calls[3][0])
        self.assertIn(
            "validation/confusion_matrix/true_class_0/predicted_class_0/count",
            calls[3][0],
        )
        self.assertIn("validation/confidence/mean", calls[3][0])

    def test_per_class_confusion_and_confidence_metrics_are_reported(self):
        logger = ClassifierMetricsLogger(num_classes=4)
        probabilities = torch.tensor(
            [
                [0.90, 0.05, 0.03, 0.02],
                [0.10, 0.80, 0.05, 0.05],
                [0.10, 0.70, 0.10, 0.10],
                [0.05, 0.05, 0.85, 0.05],
                [0.60, 0.20, 0.10, 0.10],
            ]
        )
        logits = probabilities.log()
        targets = torch.tensor([0, 0, 1, 1, 2])

        logger.log_validation_step(
            self._discard_log,
            torch.tensor(0.5),
            logits,
            targets,
        )

        per_class = logger.validation_per_class_epoch_metrics()
        confusion = logger.validation_confusion_matrix_epoch_metrics()
        confidence = logger.validation_confidence_epoch_metrics()

        torch.testing.assert_close(
            per_class["validation/per_class/class_0/precision"],
            torch.tensor(0.5),
        )
        torch.testing.assert_close(
            per_class["validation/per_class/class_0/recall"],
            torch.tensor(0.5),
        )
        torch.testing.assert_close(
            per_class["validation/per_class/class_0/f1_score"],
            torch.tensor(0.5),
        )
        torch.testing.assert_close(
            per_class["validation/per_class/class_2/precision"],
            torch.tensor(0.0),
        )
        torch.testing.assert_close(
            per_class["validation/per_class/class_3/accuracy"],
            torch.tensor(0.0),
        )
        torch.testing.assert_close(
            confusion[
                "validation/confusion_matrix/true_class_0"
                "/predicted_class_1/count"
            ],
            torch.tensor(1.0),
        )
        torch.testing.assert_close(
            confusion[
                "validation/confusion_matrix/true_class_0"
                "/predicted_class_1/rate"
            ],
            torch.tensor(0.5),
        )
        torch.testing.assert_close(
            confusion[
                "validation/confusion_matrix/true_class_2"
                "/predicted_class_0/rate"
            ],
            torch.tensor(1.0),
        )
        torch.testing.assert_close(
            confidence["validation/confidence/mean"],
            torch.tensor(0.77),
        )
        torch.testing.assert_close(
            confidence["validation/calibration/ece"],
            torch.tensor(0.53),
        )

    def test_large_class_confusion_metrics_emit_bounded_top_pairs(self):
        logger = ClassifierMetricsLogger(
            num_classes=100,
            full_confusion_matrix_class_limit=20,
            top_confused_pair_limit=3,
        )
        logits = torch.full((6, 100), -8.0)
        targets = torch.tensor([0, 0, 1, 2, 3, 4])
        predictions = torch.tensor([1, 1, 2, 3, 4, 5])
        logits[torch.arange(targets.numel()), predictions] = 8.0

        logger.log_validation_step(
            self._discard_log,
            torch.tensor(0.5),
            logits,
            targets,
        )

        confusion = logger.validation_confusion_matrix_epoch_metrics()

        self.assertNotIn(
            "validation/confusion_matrix/true_class_0"
            "/predicted_class_1/count",
            confusion,
        )
        self.assertIn("validation/confusion_top_pairs/rank_1/count", confusion)
        self.assertIn("validation/confusion_top_pairs/rank_1/true_class", confusion)
        self.assertIn(
            "validation/confusion_top_pairs/rank_1/predicted_class",
            confusion,
        )
        self.assertLessEqual(len(confusion), 12)

    def test_best_validation_tracks_best_accuracy_and_loss_epochs(self):
        logger = ClassifierMetricsLogger(num_classes=2)

        logger.log_validation_step(
            self._discard_log,
            torch.tensor(0.8),
            torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
            torch.tensor([0, 1]),
        )
        first_best = logger.update_best_validation_metrics(epoch=3)
        logger.reset_validation_epoch()
        logger.log_validation_step(
            self._discard_log,
            torch.tensor(0.4),
            torch.tensor([[4.0, 0.0], [4.0, 0.0]]),
            torch.tensor([0, 1]),
        )
        second_best = logger.update_best_validation_metrics(epoch=4)

        torch.testing.assert_close(
            first_best["best_validation/accuracy"],
            torch.tensor(1.0),
        )
        torch.testing.assert_close(
            second_best["best_validation/accuracy"],
            torch.tensor(1.0),
        )
        torch.testing.assert_close(
            second_best["best_validation/loss"],
            torch.tensor(0.4),
        )
        torch.testing.assert_close(
            second_best["best_validation/epoch"],
            torch.tensor(3.0),
        )
        torch.testing.assert_close(
            second_best["best_validation/loss_epoch"],
            torch.tensor(4.0),
        )

    def test_validation_examples_log_most_confident_wrong_images_and_labels(self):
        logger = ClassifierMetricsLogger(num_classes=2, validation_example_limit=2)
        tensorboard_logger = FakeLogger()
        examples = torch.arange(12, dtype=torch.float32).reshape(3, 1, 2, 2)
        logits = torch.tensor(
            [
                [0.90, 0.10],
                [0.05, 0.95],
                [0.30, 0.70],
            ]
        ).log()
        targets = torch.tensor([0, 0, 0])

        logger.log_validation_step(
            self._discard_log,
            torch.tensor(0.5),
            logits,
            targets,
            examples,
        )
        logger.log_validation_examples(tensorboard_logger, epoch=7)

        image_tag, image, image_step = tensorboard_logger.experiment.images[0]
        text_tag, text, text_step = tensorboard_logger.experiment.text[0]

        self.assertEqual(image_tag, "validation/examples/most_confident_wrong")
        self.assertEqual(image_step, 7)
        self.assertEqual(image.shape, torch.Size([3, 2, 4]))
        self.assertEqual(
            text_tag,
            "validation/examples/most_confident_wrong_labels",
        )
        self.assertEqual(text_step, 7)
        self.assertIn("true=0 predicted=1 confidence=0.9500", text)
        self.assertIn("true=0 predicted=1 confidence=0.7000", text)

    def test_optimizer_health_metrics_report_norm_ratio_and_bad_gradient_counts(self):
        model = HealthProbeClassifier()
        model.probe.grad = torch.tensor([6.0, 8.0])
        model.bad_probe.grad = torch.tensor([float("nan"), float("inf")])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        metrics = model.optimizer_health_metrics(optimizer)

        torch.testing.assert_close(
            metrics["parameters/global_norm"],
            torch.tensor(5.0),
        )
        torch.testing.assert_close(
            metrics["gradients/global_norm"],
            torch.tensor(10.0),
        )
        torch.testing.assert_close(
            metrics["updates/update_to_weight_ratio"],
            torch.tensor(0.2),
        )
        torch.testing.assert_close(
            metrics["gradients/nan_count"],
            torch.tensor(1.0),
        )
        torch.testing.assert_close(
            metrics["gradients/inf_count"],
            torch.tensor(1.0),
        )

    def test_model_step_adds_auxiliary_loss_without_scalar_sync(self):
        model = AuxiliaryLossProbeClassifier()
        loss, logits, targets = model._model_step(
            (
                torch.ones(2, 1),
                torch.tensor([0, 1]),
            )
        )

        torch.testing.assert_close(
            loss,
            torch.tensor(0.9431471824645996),
        )
        torch.testing.assert_close(logits, torch.zeros(2, 2))
        torch.testing.assert_close(targets, torch.tensor([0, 1]))

    def _discard_log(self, payload, **kwargs):
        pass


if __name__ == "__main__":
    unittest.main()
