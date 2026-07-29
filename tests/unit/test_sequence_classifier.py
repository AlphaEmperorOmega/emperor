import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from emperor.config import ModelConfig
from emperor.experiments.sequence_classifier import (
    SequenceClassifierExperiment,
)
from emperor.experiments.sequence_classifier._metrics import (
    SequenceClassifierMetricsLogger,
)


class StaticSequenceClassifier(SequenceClassifierExperiment):
    def __init__(
        self,
        cfg: ModelConfig,
        logits: torch.Tensor,
        auxiliary_loss: torch.Tensor | None = None,
        *,
        tuple_output: bool = True,
    ) -> None:
        super().__init__(cfg)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("fixed_logits", logits)
        self.register_buffer("fixed_auxiliary_loss", auxiliary_loss)
        self.tuple_output = tuple_output
        self.forward_calls = []

    def forward(self, tokens: torch.Tensor):
        self.forward_calls.append(tokens)
        logits = self.fixed_logits * self.scale
        if not self.tuple_output:
            return logits
        return logits, self.fixed_auxiliary_loss


class GraphAuxiliarySequenceClassifier(StaticSequenceClassifier):
    def __init__(self, cfg: ModelConfig, logits: torch.Tensor) -> None:
        super().__init__(cfg, logits, None)
        self.auxiliary_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, tokens: torch.Tensor):
        logits = self.fixed_logits * self.scale
        auxiliary_loss = self.auxiliary_scale - self.auxiliary_scale.detach()
        return logits, auxiliary_loss


class InvalidAuxiliarySequenceClassifier(StaticSequenceClassifier):
    def __init__(
        self,
        cfg: ModelConfig,
        logits: torch.Tensor,
        auxiliary_loss: object,
    ) -> None:
        super().__init__(cfg, logits, None)
        self.invalid_auxiliary_loss = auxiliary_loss

    def forward(self, tokens: torch.Tensor):
        return self.fixed_logits * self.scale, self.invalid_auxiliary_loss


class TestSequenceClassifierExperiment(unittest.TestCase):
    def preset(self, output_dim: int = 3) -> ModelConfig:
        return ModelConfig(learning_rate=3e-4, output_dim=output_dim)

    def deterministic_batch(self):
        logits = torch.tensor(
            [
                [4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [4.0, 0.0, 0.0],
            ]
        )
        tokens = torch.tensor([[2, 5, 3], [2, 6, 3], [2, 7, 3]])
        labels = torch.tensor([0, 2, 1])
        return logits, tokens, labels

    def test_initialization_and_metric_module_topology_are_stable(self) -> None:
        cfg = self.preset()
        logits, _tokens, _labels = self.deterministic_batch()
        model = StaticSequenceClassifier(cfg, logits, torch.tensor(0.25))

        self.assertIs(model.cfg, cfg)
        self.assertEqual(model.learning_rate, cfg.learning_rate)
        self.assertEqual(model.num_classes, cfg.output_dim)
        self.assertIsInstance(model.loss_fn, nn.CrossEntropyLoss)
        self.assertIsInstance(model.metrics, SequenceClassifierMetricsLogger)
        self.assertEqual(
            tuple(dict(model.named_children())),
            ("loss_fn", "metrics"),
        )
        self.assertEqual(
            tuple(dict(model.metrics.named_children())),
            (
                "train_accuracy",
                "train_f1_score",
                "validation_accuracy",
                "validation_f1_score",
                "test_accuracy",
                "test_f1_score",
            ),
        )
        self.assertEqual(
            tuple(name for name, _module in model.metrics.named_modules()),
            (
                "",
                "train_accuracy",
                "train_f1_score",
                "validation_accuracy",
                "validation_f1_score",
                "test_accuracy",
                "test_f1_score",
            ),
        )
        self.assertEqual(tuple(model.metrics.state_dict()), ())

    def test_state_dict_topology_supports_strict_roundtrip(self) -> None:
        cfg = self.preset()
        logits, _tokens, _labels = self.deterministic_batch()
        model = StaticSequenceClassifier(cfg, logits, torch.tensor(0.25))
        state = model.state_dict()

        self.assertEqual(
            tuple(state),
            ("scale", "fixed_logits", "fixed_auxiliary_loss"),
        )

        restored = StaticSequenceClassifier(cfg, logits, torch.tensor(0.25))
        incompatible = restored.load_state_dict(state, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        for name, value in state.items():
            torch.testing.assert_close(restored.state_dict()[name], value)

    def test_logits_and_auxiliary_paths_match_cross_entropy(self) -> None:
        logits, tokens, labels = self.deterministic_batch()
        expected_cross_entropy = F.cross_entropy(logits, labels)
        cases = (
            (False, None, expected_cross_entropy),
            (True, None, expected_cross_entropy),
            (True, torch.tensor(0.0), expected_cross_entropy),
            (True, torch.tensor(0.5), expected_cross_entropy + 0.5),
        )

        for tuple_output, auxiliary_loss, expected_loss in cases:
            with self.subTest(
                tuple_output=tuple_output,
                auxiliary_loss=auxiliary_loss,
            ):
                model = StaticSequenceClassifier(
                    self.preset(),
                    logits,
                    auxiliary_loss,
                    tuple_output=tuple_output,
                )
                loss, observed_logits, observed_labels = model._model_step(
                    (tokens, labels)
                )

                torch.testing.assert_close(loss, expected_loss)
                torch.testing.assert_close(observed_logits, logits)
                torch.testing.assert_close(observed_labels, labels)
                torch.testing.assert_close(model.forward_calls[0], tokens)

    def test_gradient_and_optimizer_cover_trainable_model_state(self) -> None:
        logits, tokens, labels = self.deterministic_batch()
        model = StaticSequenceClassifier(
            self.preset(),
            logits,
            torch.tensor(0.1),
        )

        loss, _logits, _labels = model._model_step((tokens, labels))
        loss.backward()
        optimizer = model.configure_optimizers()

        self.assertIsNotNone(model.scale.grad)
        self.assertTrue(torch.any(model.scale.grad.abs() > 0))
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.param_groups[0]["lr"], model.learning_rate)
        self.assertEqual(
            [id(parameter) for parameter in optimizer.param_groups[0]["params"]],
            [id(parameter) for parameter in model.parameters()],
        )

    def test_zero_valued_auxiliary_loss_preserves_its_gradient_path(self) -> None:
        logits, tokens, labels = self.deterministic_batch()
        model = GraphAuxiliarySequenceClassifier(self.preset(), logits)

        loss, _logits, _labels = model._model_step((tokens, labels))
        loss.backward()

        torch.testing.assert_close(model.auxiliary_scale.grad, torch.tensor(1.0))

    def test_non_scalar_or_non_tensor_auxiliary_loss_is_rejected(self) -> None:
        logits, tokens, labels = self.deterministic_batch()
        for auxiliary_loss in (torch.ones(2), 0.5):
            with self.subTest(auxiliary_loss=auxiliary_loss):
                model = InvalidAuxiliarySequenceClassifier(
                    self.preset(),
                    logits,
                    auxiliary_loss,
                )
                with self.assertRaisesRegex(ValueError, "auxiliary loss"):
                    model._model_step((tokens, labels))

    def test_invalid_model_output_contract_is_rejected_at_the_sequence_seam(
        self,
    ) -> None:
        logits, tokens, labels = self.deterministic_batch()
        model = StaticSequenceClassifier(self.preset(), logits)
        invalid_outputs = (
            ((logits,), "two-item tuple"),
            ((logits, torch.tensor(0.0), torch.tensor(0.0)), "two-item tuple"),
            ("not logits", "rank-2 tensor"),
            ((logits.unsqueeze(0), torch.ones(2)), "rank-2 tensor"),
            (logits[:2], "batch dimension"),
            (logits[:, :2], "class dimension"),
        )

        for output, message in invalid_outputs:
            with (
                self.subTest(message=message),
                patch.object(model, "forward", return_value=output),
                patch.object(
                    model.loss_fn,
                    "forward",
                    wraps=model.loss_fn.forward,
                ) as loss,
                self.assertRaisesRegex(ValueError, message),
            ):
                model._model_step((tokens, labels))
            loss.assert_not_called()

    def test_invalid_sequence_batch_geometry_is_rejected_before_forward(self) -> None:
        logits, tokens, labels = self.deterministic_batch()
        model = StaticSequenceClassifier(self.preset(), logits)
        invalid_batches = (
            ((tokens,), "must contain"),
            (("not tokens", labels), "rank-2 tensor"),
            ((tokens, labels.unsqueeze(1)), "rank-1 tensor"),
            ((tokens[:2], labels), "batch dimension"),
        )

        for batch, message in invalid_batches:
            with (
                self.subTest(message=message),
                patch.object(model, "forward", wraps=model.forward) as forward,
                self.assertRaisesRegex(ValueError, message),
            ):
                model._model_step(batch)
            forward.assert_not_called()

    def test_stage_steps_delegate_to_private_metrics_logger(self) -> None:
        logits, tokens, labels = self.deterministic_batch()
        model = StaticSequenceClassifier(self.preset(), logits)
        loss = torch.tensor(0.5)

        for step_name, log_name in (
            ("training_step", "log_training_step"),
            ("validation_step", "log_validation_step"),
            ("test_step", "log_test_step"),
        ):
            with (
                self.subTest(step=step_name),
                patch.object(
                    model,
                    "_model_step",
                    return_value=(loss, logits, labels),
                ),
                patch.object(model.metrics, log_name) as log_method,
            ):
                result = getattr(model, step_name)((tokens, labels), 0)

                self.assertIs(result, loss)
                log_method.assert_called_once()
                logged_loss, logged_logits, logged_labels = log_method.call_args.args[
                    1:
                ]
                self.assertIs(logged_loss, loss)
                self.assertIs(logged_logits, logits)
                self.assertIs(logged_labels, labels)

    def test_metric_payloads_have_exact_stage_keys_and_values(self) -> None:
        logits, _tokens, labels = self.deterministic_batch()
        loss = torch.tensor(0.5)

        for stage, method_name, expected_kwargs in (
            ("train", "log_training_step", {"prog_bar": True}),
            ("validation", "log_validation_step", {"prog_bar": True}),
            ("test", "log_test_step", {}),
        ):
            with self.subTest(stage=stage):
                logger = SequenceClassifierMetricsLogger(num_classes=3)
                calls = []
                getattr(logger, method_name)(
                    lambda payload, calls=calls, **kwargs: calls.append(
                        (payload, kwargs)
                    ),
                    loss,
                    logits,
                    labels,
                )

                payload, kwargs = calls[0]
                self.assertEqual(
                    set(payload),
                    {
                        f"{stage}/loss",
                        f"{stage}/accuracy",
                        f"{stage}/f1_score",
                    },
                )
                self.assertIs(payload[f"{stage}/loss"], loss)
                torch.testing.assert_close(
                    payload[f"{stage}/accuracy"],
                    torch.tensor(1.0 / 3.0),
                )
                torch.testing.assert_close(
                    payload[f"{stage}/f1_score"],
                    torch.tensor(2.0 / 9.0),
                )
                self.assertEqual(kwargs, expected_kwargs)


if __name__ == "__main__":
    unittest.main()
