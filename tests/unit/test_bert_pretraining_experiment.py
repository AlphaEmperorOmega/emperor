import math
import unittest
from dataclasses import FrozenInstanceError, fields, replace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from emperor.config import ModelConfig
from emperor.experiments.bert_pretraining import BertPretrainingExperiment
from emperor.experiments.bert_pretraining._metrics import (
    BertPretrainingMetricsLogger,
)
from emperor.experiments.bert_pretraining._records import (
    BertPretrainingStepOutput,
)


class StaticBertPretrainingExperiment(BertPretrainingExperiment):
    def __init__(
        self,
        cfg: ModelConfig,
        mlm_logits: torch.Tensor,
        nsp_logits: torch.Tensor,
        auxiliary_loss: torch.Tensor | None,
    ) -> None:
        super().__init__(cfg)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("fixed_mlm_logits", mlm_logits)
        self.register_buffer("fixed_nsp_logits", nsp_logits)
        self.register_buffer("fixed_auxiliary_loss", auxiliary_loss)
        self.forward_calls = []

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ):
        self.forward_calls.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        return (
            self.fixed_mlm_logits * self.scale,
            self.fixed_nsp_logits * self.scale,
            self.fixed_auxiliary_loss,
        )


class TestBertPretrainingExperiment(unittest.TestCase):
    def preset(self, output_dim: int = 6) -> ModelConfig:
        return ModelConfig(learning_rate=2e-4, output_dim=output_dim)

    def deterministic_inputs(self):
        mlm_logits = torch.tensor(
            [
                [
                    [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.1, 0.2, 5.0, 4.0, -1.0],
                ],
                [
                    [0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
                ],
            ]
        )
        nsp_logits = torch.tensor([[4.0, 0.0], [3.0, 1.0]])
        batch = (
            torch.tensor([[2, 5, 3], [2, 6, 3]]),
            torch.tensor([[0, -100, 4], [1, 2, 3]]),
            torch.tensor([[1, 1, 1], [1, 1, 1]]),
            torch.tensor([[0, 0, 1], [0, 1, 1]]),
            torch.tensor([0, 1]),
        )
        return mlm_logits, nsp_logits, batch

    def test_initialization_and_module_topology_are_stable(self) -> None:
        cfg = self.preset()
        mlm_logits, nsp_logits, _batch = self.deterministic_inputs()
        model = StaticBertPretrainingExperiment(
            cfg,
            mlm_logits,
            nsp_logits,
            torch.tensor(0.25),
        )

        self.assertIs(model.cfg, cfg)
        self.assertEqual(model.learning_rate, cfg.learning_rate)
        self.assertEqual(model.vocab_size, cfg.output_dim)
        self.assertIsInstance(model.mlm_loss_fn, nn.CrossEntropyLoss)
        self.assertEqual(model.mlm_loss_fn.ignore_index, -100)
        self.assertIsInstance(model.nsp_loss_fn, nn.CrossEntropyLoss)
        self.assertIsInstance(model.metrics, BertPretrainingMetricsLogger)
        self.assertEqual(
            tuple(dict(model.named_children())),
            ("mlm_loss_fn", "nsp_loss_fn", "metrics"),
        )
        self.assertEqual(tuple(model.metrics.state_dict()), ())
        self.assertEqual(
            tuple(name for name, _module in model.metrics.named_modules()),
            ("",),
        )

    def test_step_output_schema_is_exact_and_frozen(self) -> None:
        self.assertEqual(
            tuple(field.name for field in fields(BertPretrainingStepOutput)),
            (
                "total_loss",
                "mlm_loss",
                "nsp_loss",
                "mlm_logits",
                "mlm_labels",
                "nsp_logits",
                "next_sentence_labels",
                "auxiliary_loss",
            ),
        )
        output = self._step_output()
        with self.assertRaises(FrozenInstanceError):
            output.total_loss = torch.tensor(0.0)

    def test_state_dict_topology_supports_strict_roundtrip(self) -> None:
        cfg = self.preset()
        mlm_logits, nsp_logits, _batch = self.deterministic_inputs()
        model = StaticBertPretrainingExperiment(
            cfg,
            mlm_logits,
            nsp_logits,
            torch.tensor(0.25),
        )
        state = model.state_dict()

        self.assertEqual(
            tuple(state),
            (
                "scale",
                "fixed_mlm_logits",
                "fixed_nsp_logits",
                "fixed_auxiliary_loss",
            ),
        )

        restored = StaticBertPretrainingExperiment(
            cfg,
            mlm_logits,
            nsp_logits,
            torch.tensor(0.25),
        )
        incompatible = restored.load_state_dict(state, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        for name, value in state.items():
            torch.testing.assert_close(restored.state_dict()[name], value)

    def test_model_step_matches_manual_losses_and_forwards_masks(self) -> None:
        mlm_logits, nsp_logits, batch = self.deterministic_inputs()
        auxiliary_loss = torch.tensor(0.25)
        model = StaticBertPretrainingExperiment(
            self.preset(),
            mlm_logits,
            nsp_logits,
            auxiliary_loss,
        )

        output = model._model_step_outputs(batch)

        expected_mlm = F.cross_entropy(
            mlm_logits.transpose(1, 2),
            batch[1],
            ignore_index=-100,
        )
        expected_nsp = F.cross_entropy(nsp_logits, batch[4])
        torch.testing.assert_close(output.mlm_loss, expected_mlm)
        torch.testing.assert_close(output.nsp_loss, expected_nsp)
        torch.testing.assert_close(
            output.total_loss,
            expected_mlm + expected_nsp + auxiliary_loss,
        )
        torch.testing.assert_close(output.mlm_logits, mlm_logits)
        torch.testing.assert_close(output.mlm_labels, batch[1])
        torch.testing.assert_close(output.nsp_logits, nsp_logits)
        torch.testing.assert_close(output.next_sentence_labels, batch[4])
        self.assertIs(output.auxiliary_loss, model.fixed_auxiliary_loss)

        call = model.forward_calls[0]
        torch.testing.assert_close(call["input_ids"], batch[0])
        torch.testing.assert_close(call["attention_mask"], batch[2])
        torch.testing.assert_close(call["token_type_ids"], batch[3])

    def test_auxiliary_loss_none_zero_and_nonzero_paths(self) -> None:
        mlm_logits, nsp_logits, batch = self.deterministic_inputs()
        expected_base = F.cross_entropy(
            mlm_logits.transpose(1, 2),
            batch[1],
            ignore_index=-100,
        ) + F.cross_entropy(nsp_logits, batch[4])
        cases = (
            (None, expected_base),
            (torch.tensor(0.0), expected_base),
            (torch.tensor(0.5), expected_base + 0.5),
        )

        for auxiliary_loss, expected in cases:
            with self.subTest(auxiliary_loss=auxiliary_loss):
                model = StaticBertPretrainingExperiment(
                    self.preset(),
                    mlm_logits,
                    nsp_logits,
                    auxiliary_loss,
                )
                output = model._model_step_outputs(batch)
                torch.testing.assert_close(output.total_loss, expected)

    def test_gradient_and_optimizer_cover_trainable_model_state(self) -> None:
        mlm_logits, nsp_logits, batch = self.deterministic_inputs()
        model = StaticBertPretrainingExperiment(
            self.preset(),
            mlm_logits,
            nsp_logits,
            torch.tensor(0.1),
        )

        loss = model._model_step(batch)
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

    def test_invalid_batch_arity_is_rejected(self) -> None:
        mlm_logits, nsp_logits, batch = self.deterministic_inputs()
        model = StaticBertPretrainingExperiment(
            self.preset(),
            mlm_logits,
            nsp_logits,
            None,
        )

        with self.assertRaisesRegex(ValueError, "must contain"):
            model._model_step(batch[:4])

    def test_stage_steps_delegate_to_private_metrics_logger(self) -> None:
        mlm_logits, nsp_logits, batch = self.deterministic_inputs()
        model = StaticBertPretrainingExperiment(
            self.preset(),
            mlm_logits,
            nsp_logits,
            None,
        )
        output = self._step_output()

        for step_name, log_name in (
            ("training_step", "log_training_step"),
            ("validation_step", "log_validation_step"),
            ("test_step", "log_test_step"),
        ):
            with (
                self.subTest(step=step_name),
                patch.object(model, "_model_step_outputs", return_value=output),
                patch.object(model.metrics, log_name) as log_method,
            ):
                result = getattr(model, step_name)(batch, 0)

                self.assertIs(result, output.total_loss)
                log_method.assert_called_once()
                self.assertIs(log_method.call_args.args[1], output)

    def _step_output(self) -> BertPretrainingStepOutput:
        mlm_logits = torch.tensor(
            [
                [
                    [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.1, 0.2, 5.0, 4.0, -1.0],
                ]
            ]
        )
        return BertPretrainingStepOutput(
            total_loss=torch.tensor(1.5),
            mlm_loss=torch.tensor(1.0),
            nsp_loss=torch.tensor(0.5),
            mlm_logits=mlm_logits,
            mlm_labels=torch.tensor([[0, -100, 4]]),
            nsp_logits=torch.tensor([[4.0, 0.0], [3.0, 1.0]]),
            next_sentence_labels=torch.tensor([0, 1]),
            auxiliary_loss=torch.tensor(0.25),
        )

    def test_metrics_payload_has_exact_masked_top_k_and_nsp_values(self) -> None:
        logger = BertPretrainingMetricsLogger()
        output = self._step_output()
        calls = []

        for stage, log_method, expected_kwargs in (
            ("train", logger.log_training_step, {"prog_bar": True}),
            ("validation", logger.log_validation_step, {"prog_bar": True}),
            ("test", logger.log_test_step, {}),
        ):
            with self.subTest(stage=stage):
                calls.clear()
                log_method(
                    lambda payload, **kwargs: calls.append((payload, kwargs)),
                    output,
                )
                payload, kwargs = calls[0]
                self.assertEqual(
                    set(payload),
                    {
                        f"{stage}/loss",
                        f"{stage}/mlm/loss",
                        f"{stage}/mlm/perplexity",
                        f"{stage}/mlm/masked_accuracy",
                        f"{stage}/mlm/masked_top_5_accuracy",
                        f"{stage}/nsp/loss",
                        f"{stage}/nsp/accuracy",
                        f"{stage}/auxiliary/loss",
                    },
                )
                torch.testing.assert_close(
                    payload[f"{stage}/mlm/masked_accuracy"],
                    torch.tensor(0.5),
                )
                torch.testing.assert_close(
                    payload[f"{stage}/mlm/masked_top_5_accuracy"],
                    torch.tensor(1.0),
                )
                torch.testing.assert_close(
                    payload[f"{stage}/nsp/accuracy"],
                    torch.tensor(0.5),
                )
                self.assertEqual(
                    payload[f"{stage}/mlm/perplexity"],
                    math.exp(output.mlm_loss.item()),
                )
                self.assertIs(payload[f"{stage}/loss"], output.total_loss)
                self.assertIs(
                    payload[f"{stage}/auxiliary/loss"],
                    output.auxiliary_loss,
                )
                self.assertEqual(kwargs, expected_kwargs)

    def test_metrics_handle_empty_masks_small_vocabularies_and_absent_auxiliary_loss(
        self,
    ) -> None:
        logger = BertPretrainingMetricsLogger()
        logits = torch.tensor([[[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]])
        empty_labels = torch.full((1, 2), -100)

        torch.testing.assert_close(
            logger._masked_accuracy(logits, empty_labels),
            torch.tensor(0.0),
        )
        torch.testing.assert_close(
            logger._masked_top_k_accuracy(logits, empty_labels, k=5),
            torch.tensor(0.0),
        )
        torch.testing.assert_close(
            logger._masked_top_k_accuracy(
                logits[:, :1],
                torch.tensor([[0]]),
                k=5,
            ),
            torch.tensor(1.0),
        )

        payload = logger._payload(
            "validation",
            replace(self._step_output(), auxiliary_loss=None),
        )
        self.assertNotIn("validation/auxiliary/loss", payload)

    def test_metrics_legacy_positional_contract_is_preserved(self) -> None:
        logger = BertPretrainingMetricsLogger()
        output = self._step_output()
        calls = []

        logger.log_training_step(
            lambda payload, **kwargs: calls.append((payload, kwargs)),
            output.total_loss,
            output.mlm_loss,
            output.nsp_loss,
            output.mlm_logits,
            output.mlm_labels,
            output.nsp_logits,
            output.next_sentence_labels,
            output.auxiliary_loss,
        )

        payload, kwargs = calls[0]
        self.assertIs(payload["train/loss"], output.total_loss)
        self.assertIs(payload["train/auxiliary/loss"], output.auxiliary_loss)
        torch.testing.assert_close(
            payload["train/mlm/masked_accuracy"],
            torch.tensor(0.5),
        )
        self.assertEqual(kwargs, {"prog_bar": True})

        with self.assertRaises(TypeError):
            logger.log_training_step(lambda *_args, **_kwargs: None, output.total_loss)


if __name__ == "__main__":
    unittest.main()
