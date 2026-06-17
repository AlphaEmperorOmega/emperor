import math
import torch
import unittest
import torch.nn as nn
import torch.nn.functional as F

from emperor.config import ModelConfig
from emperor.experiments.masked_language_model import (
    MaskedLanguageModelExperiment,
    MaskedLanguageModelMetricsLogger,
)


class StaticMaskedLanguageModel(MaskedLanguageModelExperiment):
    def __init__(
        self,
        cfg: ModelConfig,
        logits: torch.Tensor,
        auxiliary_loss: torch.Tensor | None = None,
    ):
        super().__init__(cfg)
        self.probe = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("logits", logits.clone())
        self.register_buffer("auxiliary_loss", auxiliary_loss)
        self.forward_calls = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ):
        self.forward_calls.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        if self.auxiliary_loss is None:
            return self.logits
        return self.logits, self.auxiliary_loss


class TestMaskedLanguageModelExperiment(unittest.TestCase):
    def preset(
        self,
        learning_rate: float = 1e-3,
        output_dim: int = 4,
    ) -> ModelConfig:
        return ModelConfig(learning_rate=learning_rate, output_dim=output_dim)

    def test_initialization_stores_config_and_loss_settings(self):
        cfg = self.preset(learning_rate=2e-4, output_dim=7)
        logits = torch.zeros(2, 3, cfg.output_dim)
        model = StaticMaskedLanguageModel(cfg, logits)

        self.assertIs(model.cfg, cfg)
        self.assertEqual(model.learning_rate, cfg.learning_rate)
        self.assertEqual(model.vocab_size, cfg.output_dim)
        self.assertIsInstance(model.loss_fn, nn.CrossEntropyLoss)
        self.assertEqual(model.loss_fn.ignore_index, -100)
        self.assertIsInstance(model.metrics, MaskedLanguageModelMetricsLogger)

    def test_model_step_matches_manual_cross_entropy(self):
        cfg = self.preset(output_dim=4)
        logits = torch.tensor(
            [
                [[2.0, 0.5, -1.0, 0.0], [0.0, 1.0, 3.0, -2.0]],
                [[-1.0, 2.0, 0.0, 1.0], [0.5, -0.5, 1.5, 2.5]],
            ]
        )
        labels = torch.tensor([[0, -100], [1, 3]])
        input_ids = torch.tensor([[5, 6], [7, 8]])
        model = StaticMaskedLanguageModel(cfg, logits)

        loss = model._model_step((input_ids, labels))
        expected = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            ignore_index=-100,
        )

        torch.testing.assert_close(loss, expected)

    def test_ignore_index_labels_do_not_contribute_to_loss(self):
        cfg = self.preset(output_dim=3)
        logits = torch.tensor([[[6.0, 0.0, 0.0], [-20.0, 20.0, 0.0]]])
        labels = torch.tensor([[0, -100]])
        input_ids = torch.tensor([[1, 2]])
        model = StaticMaskedLanguageModel(cfg, logits)

        loss = model._model_step((input_ids, labels))
        expected = F.cross_entropy(logits[:, 0, :], torch.tensor([0]))
        unignored_loss = F.cross_entropy(
            logits.transpose(1, 2),
            torch.tensor([[0, 2]]),
        )

        torch.testing.assert_close(loss, expected)
        self.assertFalse(torch.allclose(loss, unignored_loss))

    def test_optional_attention_tensors_are_passed_to_forward(self):
        cfg = self.preset(output_dim=3)
        logits = torch.zeros(2, 2, cfg.output_dim)
        labels = torch.tensor([[0, -100], [2, -100]])
        input_ids = torch.tensor([[1, 2], [3, 4]])
        attention_mask = torch.tensor([[1, 1], [1, 0]])
        token_type_ids = torch.tensor([[0, 0], [0, 1]])
        model = StaticMaskedLanguageModel(cfg, logits)

        model._model_step((input_ids, labels, attention_mask))
        call = model.forward_calls[-1]
        torch.testing.assert_close(call["input_ids"], input_ids)
        torch.testing.assert_close(call["attention_mask"], attention_mask)
        self.assertIsNone(call["token_type_ids"])

        model._model_step((input_ids, labels, attention_mask, token_type_ids))
        call = model.forward_calls[-1]
        torch.testing.assert_close(call["input_ids"], input_ids)
        torch.testing.assert_close(call["attention_mask"], attention_mask)
        torch.testing.assert_close(call["token_type_ids"], token_type_ids)

    def test_tuple_output_auxiliary_loss_is_added_when_nonzero(self):
        cfg = self.preset(output_dim=3)
        logits = torch.tensor([[[1.0, 0.0, -1.0], [0.0, 2.0, -2.0]]])
        labels = torch.tensor([[0, 1]])
        input_ids = torch.tensor([[1, 2]])
        auxiliary_loss = torch.tensor(0.75)
        model = StaticMaskedLanguageModel(cfg, logits, auxiliary_loss)

        loss = model._model_step((input_ids, labels))
        expected = (
            F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100)
            + auxiliary_loss
        )

        torch.testing.assert_close(loss, expected)

    def test_zero_tuple_output_auxiliary_loss_is_not_added(self):
        cfg = self.preset(output_dim=3)
        logits = torch.tensor([[[1.0, 0.0, -1.0], [0.0, 2.0, -2.0]]])
        labels = torch.tensor([[0, 1]])
        input_ids = torch.tensor([[1, 2]])
        model = StaticMaskedLanguageModel(cfg, logits, torch.tensor(0.0))

        loss = model._model_step((input_ids, labels))
        expected = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            ignore_index=-100,
        )

        torch.testing.assert_close(loss, expected)

    def test_configure_optimizers_returns_adam_for_model_parameters(self):
        cfg = self.preset(learning_rate=3e-4, output_dim=3)
        logits = torch.zeros(1, 1, cfg.output_dim)
        model = StaticMaskedLanguageModel(cfg, logits)

        optimizer = model.configure_optimizers()
        optimizer_params = optimizer.param_groups[0]["params"]
        model_params = list(model.parameters())

        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.param_groups[0]["lr"], cfg.learning_rate)
        self.assertEqual(
            [id(param) for param in optimizer_params],
            [id(param) for param in model_params],
        )

    def test_metrics_logger_matches_language_model_logging_shape(self):
        logger = MaskedLanguageModelMetricsLogger()
        loss = torch.tensor(0.25)
        auxiliary_loss = torch.tensor(0.1)
        logits = torch.tensor(
            [
                [
                    [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.1, 0.2, 5.0, 4.0, -1.0],
                ]
            ]
        )
        labels = torch.tensor([[0, -100, 4]])
        calls = []

        def log_fn(payload, **kwargs):
            calls.append((payload, kwargs))

        logger.log_training_step(log_fn, loss, logits, labels, auxiliary_loss)
        logger.log_validation_step(log_fn, loss, logits, labels, auxiliary_loss)
        logger.log_test_step(log_fn, loss, logits, labels, auxiliary_loss)

        self.assertIs(calls[0][0]["train/loss"], loss)
        self.assertEqual(calls[0][0]["train/perplexity"], math.exp(loss.item()))
        torch.testing.assert_close(
            calls[0][0]["train/masked/accuracy"],
            torch.tensor(0.5),
        )
        torch.testing.assert_close(
            calls[0][0]["train/masked/top_5_accuracy"],
            torch.tensor(1.0),
        )
        self.assertIs(calls[0][0]["train/auxiliary/loss"], auxiliary_loss)
        self.assertEqual(calls[0][1], {"prog_bar": True})
        self.assertIs(calls[1][0]["validation/loss"], loss)
        self.assertEqual(
            calls[1][0]["validation/perplexity"],
            math.exp(loss.item()),
        )
        torch.testing.assert_close(
            calls[1][0]["validation/masked/accuracy"],
            torch.tensor(0.5),
        )
        torch.testing.assert_close(
            calls[1][0]["validation/masked/top_5_accuracy"],
            torch.tensor(1.0),
        )
        self.assertIs(calls[1][0]["validation/auxiliary/loss"], auxiliary_loss)
        self.assertEqual(calls[1][1], {"prog_bar": True})
        self.assertIs(calls[2][0]["test/loss"], loss)
        self.assertEqual(calls[2][0]["test/perplexity"], math.exp(loss.item()))
        torch.testing.assert_close(
            calls[2][0]["test/masked/accuracy"],
            torch.tensor(0.5),
        )
        torch.testing.assert_close(
            calls[2][0]["test/masked/top_5_accuracy"],
            torch.tensor(1.0),
        )
        self.assertIs(calls[2][0]["test/auxiliary/loss"], auxiliary_loss)
        self.assertEqual(calls[2][1], {})

    def test_metrics_logger_omits_auxiliary_loss_when_absent(self):
        logger = MaskedLanguageModelMetricsLogger()
        loss = torch.tensor(0.25)
        logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        labels = torch.tensor([[0, 1]])
        calls = []

        def log_fn(payload, **kwargs):
            calls.append((payload, kwargs))

        logger.log_training_step(log_fn, loss, logits, labels)

        self.assertNotIn("train/auxiliary/loss", calls[0][0])


if __name__ == "__main__":
    unittest.main()
