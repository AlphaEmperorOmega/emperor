import unittest
from dataclasses import FrozenInstanceError, fields
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from emperor.config import ModelConfig
from emperor.experiments.masked_language_model import (
    MaskedLanguageModelExperiment,
)
from emperor.experiments.masked_language_model._metrics import (
    MaskedLanguageModelMetricsLogger,
)
from emperor.experiments.masked_language_model._records import (
    MaskedLanguageModelStepOutput,
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
        self.log_calls = []

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

    def log_dict(self, payload, **kwargs) -> None:
        self.log_calls.append((payload, kwargs))


class GraphAuxiliaryMaskedLanguageModel(StaticMaskedLanguageModel):
    def __init__(self, cfg: ModelConfig, logits: torch.Tensor) -> None:
        super().__init__(cfg, logits)
        self.auxiliary_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ):
        logits = super().forward(input_ids, attention_mask, token_type_ids)
        auxiliary_loss = self.auxiliary_scale - self.auxiliary_scale.detach()
        return logits, auxiliary_loss


class InvalidAuxiliaryMaskedLanguageModel(StaticMaskedLanguageModel):
    def __init__(
        self,
        cfg: ModelConfig,
        logits: torch.Tensor,
        auxiliary_loss: object,
    ) -> None:
        super().__init__(cfg, logits)
        self.invalid_auxiliary_loss = auxiliary_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ):
        return self.logits, self.invalid_auxiliary_loss


class TestMaskedLanguageModelExperiment(unittest.TestCase):
    def preset(
        self,
        learning_rate: float = 1e-3,
        output_dim: int = 4,
    ) -> ModelConfig:
        return ModelConfig(learning_rate=learning_rate, output_dim=output_dim)

    def test_step_output_is_a_frozen_named_record_of_current_step_facts(self) -> None:
        total_loss = torch.tensor(1.0, requires_grad=True)
        cross_entropy = torch.tensor(0.75, requires_grad=True)
        logits = torch.randn(2, 3, 4, requires_grad=True)
        labels = torch.randint(0, 4, (2, 3))
        auxiliary_loss = torch.tensor(0.25, requires_grad=True)

        output = MaskedLanguageModelStepOutput(
            total_loss=total_loss,
            cross_entropy=cross_entropy,
            logits=logits,
            labels=labels,
            auxiliary_loss=auxiliary_loss,
        )

        self.assertEqual(
            tuple(field.name for field in fields(output)),
            (
                "total_loss",
                "cross_entropy",
                "logits",
                "labels",
                "auxiliary_loss",
            ),
        )
        self.assertIs(output.total_loss, total_loss)
        self.assertIs(output.cross_entropy, cross_entropy)
        self.assertIs(output.logits, logits)
        self.assertIs(output.labels, labels)
        self.assertIs(output.auxiliary_loss, auxiliary_loss)
        with self.assertRaises(FrozenInstanceError):
            output.total_loss = torch.tensor(2.0)

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

        output = model._model_step_outputs((input_ids, labels))
        self.assertIs(output.total_loss, output.cross_entropy)
        torch.testing.assert_close(output.cross_entropy, expected)

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

        with patch.object(model.metrics, "log_training_step") as log_training_step:
            observed_loss = model.training_step((input_ids, labels), 0)

        output = log_training_step.call_args.args[1]
        self.assertIs(observed_loss, output.total_loss)
        torch.testing.assert_close(
            output.cross_entropy,
            expected - auxiliary_loss,
        )
        torch.testing.assert_close(
            output.total_loss,
            output.cross_entropy + auxiliary_loss,
        )

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

    def test_zero_valued_auxiliary_loss_preserves_its_gradient_path(self) -> None:
        cfg = self.preset(output_dim=3)
        logits = torch.tensor([[[1.0, 0.0, -1.0], [0.0, 2.0, -2.0]]])
        labels = torch.tensor([[0, 1]])
        input_ids = torch.tensor([[1, 2]])
        model = GraphAuxiliaryMaskedLanguageModel(cfg, logits)

        model._model_step((input_ids, labels)).backward()

        torch.testing.assert_close(model.auxiliary_scale.grad, torch.tensor(1.0))

    def test_non_scalar_or_non_tensor_auxiliary_loss_is_rejected(self) -> None:
        cfg = self.preset(output_dim=3)
        logits = torch.zeros(1, 2, cfg.output_dim)
        batch = torch.tensor([[1, 2]]), torch.tensor([[0, 1]])
        for auxiliary_loss in (torch.ones(2), 0.5):
            with self.subTest(auxiliary_loss=auxiliary_loss):
                model = InvalidAuxiliaryMaskedLanguageModel(
                    cfg,
                    logits,
                    auxiliary_loss,
                )
                with self.assertRaisesRegex(ValueError, "auxiliary loss"):
                    model._model_step(batch)

    def test_invalid_model_output_contract_is_rejected_at_the_mlm_seam(self) -> None:
        cfg = self.preset(output_dim=3)
        logits = torch.zeros(2, 3, cfg.output_dim)
        input_ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
        labels = torch.tensor([[0, 1, 2], [2, 1, 0]])
        model = StaticMaskedLanguageModel(cfg, logits)
        invalid_outputs = (
            ((logits,), "two-item tuple"),
            ((logits, torch.tensor(0.0), torch.tensor(0.0)), "two-item tuple"),
            ("not logits", "MLM logits"),
            ((logits[:, 0], torch.ones(2)), "rank-3 tensor"),
            (logits[:1], "logits and labels"),
            (logits[..., :-1], "vocabulary dimension"),
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
                model._model_step((input_ids, labels))
            loss.assert_not_called()

    def test_invalid_mlm_batch_geometry_is_rejected_before_forward(self) -> None:
        cfg = self.preset(output_dim=3)
        logits = torch.zeros(2, 3, cfg.output_dim)
        model = StaticMaskedLanguageModel(cfg, logits)
        input_ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
        labels = torch.tensor([[0, 1, 2], [2, 1, 0]])
        invalid_batches = (
            (("not input IDs", labels), "rank-2 tensors"),
            ((input_ids, labels[:, :2]), "equal shapes"),
            ((input_ids, labels, torch.ones(2, 2)), "attention mask"),
            ((input_ids, labels, None, "not token types"), "token-type IDs"),
            ((input_ids, labels, None, torch.ones(3)), "token-type IDs"),
            ((input_ids, labels, None, torch.ones(2, 2)), "token-type IDs"),
        )

        for batch, message in invalid_batches:
            with (
                self.subTest(message=message),
                patch.object(model, "forward", wraps=model.forward) as forward,
                self.assertRaisesRegex(ValueError, message),
            ):
                model._model_step(batch)
            forward.assert_not_called()

    def test_real_validation_and_test_stages_and_batch_arity_boundaries(self) -> None:
        cfg = self.preset(output_dim=3)
        logits = torch.tensor(
            [
                [[2.0, 0.0, -1.0], [0.0, 3.0, -2.0]],
                [[-1.0, 2.0, 0.0], [0.5, -0.5, 2.0]],
            ]
        )
        input_ids = torch.tensor([[0, 1], [1, 2]])
        labels = torch.tensor([[1, -100], [2, 0]])
        attention_mask = torch.tensor([[1, 1], [1, 0]])
        token_type_ids = torch.tensor([[0, 0], [0, 1]])
        auxiliary_loss = torch.tensor(0.25)
        batch = input_ids, labels, attention_mask, token_type_ids
        expected_loss = (
            F.cross_entropy(
                logits.transpose(1, 2),
                labels,
                ignore_index=-100,
            )
            + auxiliary_loss
        )

        for stage, step_name, expected_kwargs in (
            ("validation", "validation_step", {"prog_bar": True}),
            ("test", "test_step", {}),
        ):
            with self.subTest(stage=stage):
                model = StaticMaskedLanguageModel(cfg, logits, auxiliary_loss)

                observed_loss = getattr(model, step_name)(batch, 0)

                torch.testing.assert_close(observed_loss, expected_loss)
                self.assertIs(model.forward_calls[0]["input_ids"], input_ids)
                self.assertIs(
                    model.forward_calls[0]["token_type_ids"],
                    token_type_ids,
                )
                payload, kwargs = model.log_calls[0]
                self.assertEqual(
                    set(payload),
                    {
                        f"{stage}/loss",
                        f"{stage}/perplexity",
                        f"{stage}/masked/accuracy",
                        f"{stage}/masked/top_5_accuracy",
                        f"{stage}/auxiliary/loss",
                    },
                )
                self.assertEqual(kwargs, expected_kwargs)

        model = StaticMaskedLanguageModel(cfg, logits)
        for invalid_batch in ((input_ids,), (*batch, input_ids)):
            with (
                self.subTest(arity=len(invalid_batch)),
                self.assertRaisesRegex(ValueError, "batches must contain"),
            ):
                model._unpack_batch(invalid_batch)

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
        token_loss = torch.tensor(0.25)
        auxiliary_loss = torch.tensor(1.0)
        loss = token_loss + auxiliary_loss
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

        output = MaskedLanguageModelStepOutput(
            total_loss=loss,
            cross_entropy=token_loss,
            logits=logits,
            labels=labels,
            auxiliary_loss=auxiliary_loss,
        )
        logger.log_training_step(
            log_fn,
            output,
        )
        logger.log_validation_step(
            log_fn,
            output,
        )
        logger.log_test_step(
            log_fn,
            output,
        )

        self.assertIs(calls[0][0]["train/loss"], loss)
        torch.testing.assert_close(
            calls[0][0]["train/perplexity"],
            torch.exp(token_loss),
        )
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
        torch.testing.assert_close(
            calls[1][0]["validation/perplexity"],
            torch.exp(token_loss),
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
        torch.testing.assert_close(
            calls[2][0]["test/perplexity"],
            torch.exp(token_loss),
        )
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

        logger.log_training_step(
            log_fn,
            MaskedLanguageModelStepOutput(loss, loss, logits, labels),
        )

        self.assertNotIn("train/auxiliary/loss", calls[0][0])

    def test_metrics_logger_returns_zero_when_no_masked_tokens_exist(self):
        logger = MaskedLanguageModelMetricsLogger()
        loss = torch.tensor(0.25)
        logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        labels = torch.full((1, 2), -100)
        calls = []

        def log_fn(payload, **kwargs):
            calls.append((payload, kwargs))

        logger.log_training_step(
            log_fn,
            MaskedLanguageModelStepOutput(loss, loss, logits, labels),
        )

        torch.testing.assert_close(
            calls[0][0]["train/masked/accuracy"],
            torch.tensor(0.0),
        )
        torch.testing.assert_close(
            calls[0][0]["train/masked/top_5_accuracy"],
            torch.tensor(0.0),
        )


if __name__ == "__main__":
    unittest.main()
