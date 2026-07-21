import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from emperor.config import ModelConfig
from emperor.experiments.language_model import LanguageModelExperiment
from emperor.experiments.language_model._metrics import LanguageModelMetricsLogger
from emperor.experiments.language_model._records import LanguageModelStepOutput


class StaticLanguageModel(LanguageModelExperiment):
    def __init__(
        self,
        cfg: ModelConfig,
        logits: torch.Tensor,
        auxiliary_loss: torch.Tensor | None = None,
        tuple_output: bool = True,
    ) -> None:
        super().__init__(cfg)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("fixed_logits", logits)
        self.register_buffer("fixed_auxiliary_loss", auxiliary_loss)
        self.tuple_output = tuple_output

    def forward(self, input_ids: torch.Tensor):
        logits = self.fixed_logits * self.scale
        if not self.tuple_output:
            return logits
        return logits, self.fixed_auxiliary_loss


class InvalidTupleLanguageModel(StaticLanguageModel):
    def forward(self, input_ids: torch.Tensor):
        return self.fixed_logits, torch.tensor(0.0), torch.tensor(0.0)


class TestLanguageModelExperiment(unittest.TestCase):
    def preset(self, output_dim: int = 3) -> ModelConfig:
        return ModelConfig(
            batch_size=2,
            learning_rate=2e-4,
            sequence_length=2,
            input_dim=output_dim,
            hidden_dim=4,
            output_dim=output_dim,
        )

    def deterministic_batch(self):
        logits = torch.tensor(
            [
                [[2.0, 0.0, -1.0], [0.0, 3.0, -2.0]],
                [[-1.0, 2.0, 0.0], [0.5, -0.5, 2.0]],
            ]
        )
        input_ids = torch.tensor([[0, 1], [1, 2]])
        labels = torch.tensor([[1, 2], [2, 0]])
        return logits, input_ids, labels

    def test_logits_only_and_tuple_outputs_share_cross_entropy(self):
        logits, input_ids, labels = self.deterministic_batch()
        expected = F.cross_entropy(logits.transpose(1, 2), labels)
        for tuple_output in (False, True):
            with self.subTest(tuple_output=tuple_output):
                model = StaticLanguageModel(
                    self.preset(),
                    logits,
                    auxiliary_loss=None,
                    tuple_output=tuple_output,
                )
                output = model._model_step_outputs((input_ids, labels))
                torch.testing.assert_close(output.cross_entropy, expected)
                torch.testing.assert_close(output.total_loss, expected)
                torch.testing.assert_close(output.auxiliary_loss, torch.tensor(0.0))

    def test_auxiliary_loss_is_added_only_to_total_loss(self):
        logits, input_ids, labels = self.deterministic_batch()
        auxiliary_loss = torch.tensor(0.75)
        model = StaticLanguageModel(self.preset(), logits, auxiliary_loss)
        output = model._model_step_outputs((input_ids, labels))
        expected_cross_entropy = F.cross_entropy(logits.transpose(1, 2), labels)

        torch.testing.assert_close(output.cross_entropy, expected_cross_entropy)
        torch.testing.assert_close(
            output.total_loss,
            expected_cross_entropy + auxiliary_loss,
        )

    def test_gradient_reaches_model_parameters(self):
        logits, input_ids, labels = self.deterministic_batch()
        model = StaticLanguageModel(self.preset(), logits, torch.tensor(0.1))
        loss = model._model_step((input_ids, labels))
        loss.backward()
        self.assertIsNotNone(model.scale.grad)
        self.assertTrue(torch.any(model.scale.grad.abs() > 0))

    def test_metrics_use_cross_entropy_for_perplexity(self):
        logger = LanguageModelMetricsLogger()
        output = LanguageModelStepOutput(
            total_loss=torch.tensor(2.0),
            cross_entropy=torch.tensor(1.0),
            logits=torch.zeros(1, 1, 3),
            labels=torch.zeros(1, 1, dtype=torch.long),
            auxiliary_loss=torch.tensor(1.0),
        )
        calls = []

        logger.log_validation_step(
            lambda payload, **kwargs: calls.append((payload, kwargs)),
            output,
        )

        payload, kwargs = calls[0]
        self.assertIs(payload["validation/loss"], output.total_loss)
        torch.testing.assert_close(
            payload["validation/perplexity"],
            torch.exp(output.cross_entropy),
        )
        self.assertIs(payload["validation/auxiliary_loss"], output.auxiliary_loss)
        self.assertEqual(kwargs, {"prog_bar": True})

    def test_invalid_batches_and_outputs_are_rejected(self):
        logits, input_ids, labels = self.deterministic_batch()
        model = StaticLanguageModel(self.preset(), logits)
        invalid_batches = (
            (input_ids,),
            (input_ids.unsqueeze(0), labels.unsqueeze(0)),
            (input_ids, labels[:, :1]),
        )
        for batch in invalid_batches:
            with self.subTest(shapes=[tuple(item.shape) for item in batch]):
                with self.assertRaises(ValueError):
                    model._model_step(batch)

    def test_invalid_model_outputs_are_rejected(self):
        logits, input_ids, labels = self.deterministic_batch()
        cases = (
            StaticLanguageModel(self.preset(), logits[:, 0], tuple_output=False),
            StaticLanguageModel(
                self.preset(),
                logits,
                auxiliary_loss=torch.ones(2),
            ),
            StaticLanguageModel(
                self.preset(output_dim=4),
                logits,
                tuple_output=False,
            ),
            InvalidTupleLanguageModel(self.preset(), logits),
        )
        for model in cases:
            with self.subTest(
                model=type(model).__name__,
                shape=model.fixed_logits.shape,
            ):
                with self.assertRaises(ValueError):
                    model._model_step((input_ids, labels))

    def test_configure_optimizers_returns_adam(self):
        logits, _input_ids, _labels = self.deterministic_batch()
        model = StaticLanguageModel(self.preset(), logits)
        optimizer = model.configure_optimizers()
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.param_groups[0]["lr"], model.cfg.learning_rate)


if __name__ == "__main__":
    unittest.main()
