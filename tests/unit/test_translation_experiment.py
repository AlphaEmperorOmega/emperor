import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from emperor.config import ModelConfig
from emperor.experiments import (
    ExperimentTask,
    experiment_task_name,
)
from emperor.experiments.translation import TranslationExperiment
from emperor.experiments.translation._metrics import _translation_step_metrics
from emperor.experiments.translation._records import TranslationStepOutput
from model_runtime.task_behavior import experiment_task_behavior
from models.catalog import model_package
from models.training_test_utils import RandomTranslationDataModule
from models.transformer.linear.config_builder import TransformerLinearConfigBuilder
from models.transformer.linear.model import Model
from models.transformer.linear.presets import Experiment


class StaticTranslationExperiment(TranslationExperiment):
    def __init__(self, auxiliary_loss: torch.Tensor | None = None) -> None:
        super().__init__(
            ModelConfig(
                learning_rate=1e-3,
                hidden_dim=4,
                output_dim=5,
            )
        )
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("fixed_logits", torch.zeros(1, 3, self.vocab_size))
        self.register_buffer("fixed_auxiliary_loss", auxiliary_loss)
        self.forward_calls = []
        self.generate_calls = []
        self.log_dict_calls = []
        self.log_calls = []

    def forward(
        self,
        source_ids: torch.Tensor,
        target_input_ids: torch.Tensor,
    ):
        self.forward_calls.append((source_ids, target_input_ids))
        logits = (
            self.fixed_logits.expand(
                source_ids.size(0),
                target_input_ids.size(1),
                -1,
            )
            * self.scale
        )
        return logits, self.fixed_auxiliary_loss

    def generate(self, source_ids: torch.Tensor, *, max_length: int) -> torch.Tensor:
        self.generate_calls.append((source_ids, max_length))
        return source_ids

    def log_dict(self, payload, **kwargs) -> None:
        self.log_dict_calls.append((payload, kwargs))

    def log(self, name, value, **kwargs) -> None:
        self.log_calls.append((name, value, kwargs))


class NumericTranslationDataModule:
    @staticmethod
    def decode_batch(token_ids: torch.Tensor) -> list[str]:
        return [" ".join(str(value) for value in row.tolist()) for row in token_ids]


class TestTranslationExperiment(unittest.TestCase):
    def preset(self):
        runtime = model_package("transformer/linear").bind_runtime_defaults(
            {
                "batch_size": 2,
                "model_dim": 16,
                "source_sequence_length": 6,
                "target_sequence_length": 6,
                "encoder_num_layers": 1,
                "decoder_num_layers": 1,
                "attn_num_heads": 2,
                "ff_stack_hidden_dim": 32,
                "dropout_probability": 0.0,
            }
        )
        config = TransformerLinearConfigBuilder(runtime=runtime).build()
        return Model(config)

    def batch(self):
        source_ids = torch.tensor([[2, 8, 3, 0], [2, 9, 10, 3]])
        target_ids = torch.tensor([[2, 11, 12, 3, 0], [2, 13, 3, 0, 0]])
        return source_ids, target_ids

    def test_teacher_forcing_loss_nll_and_pad_ignoring(self):
        model = self.preset()
        source_ids, target_ids = self.batch()
        observed = {}

        def forward(source, target_input):
            observed["source"] = source
            observed["target_input"] = target_input
            logits = torch.zeros(
                target_input.size(0),
                target_input.size(1),
                model.vocab_size,
            )
            logits[..., 3] = 1.0
            return logits, logits.new_tensor(0.25)

        with patch.object(model, "forward", side_effect=forward):
            output = model._model_step_outputs((source_ids, target_ids))

        torch.testing.assert_close(observed["source"], source_ids)
        torch.testing.assert_close(
            observed["target_input"],
            target_ids[:, :-1],
        )
        torch.testing.assert_close(output.labels, target_ids[:, 1:])
        expected_nll = model.nll_fn(
            output.logits.reshape(-1, model.vocab_size),
            output.labels.reshape(-1),
        )
        torch.testing.assert_close(output.nll, expected_nll)
        self.assertTrue(torch.isfinite(output.total_loss))
        self.assertGreater(output.total_loss.item(), output.nll.item())

    def test_model_step_rejects_non_scalar_or_non_tensor_auxiliary_loss(self):
        model = self.preset()
        source_ids, target_ids = self.batch()
        logits = torch.zeros(
            source_ids.size(0),
            target_ids.size(1) - 1,
            model.vocab_size,
        )

        for auxiliary_loss in (torch.ones(2), 0.5):
            with (
                self.subTest(auxiliary_loss=auxiliary_loss),
                patch.object(
                    model,
                    "forward",
                    return_value=(logits, auxiliary_loss),
                ),
                self.assertRaisesRegex(ValueError, "auxiliary loss"),
            ):
                model._model_step_outputs((source_ids, target_ids))

    def test_invalid_model_output_contract_is_rejected_at_translation_seam(self):
        model = self.preset()
        source_ids, target_ids = self.batch()
        logits = torch.zeros(
            source_ids.size(0),
            target_ids.size(1) - 1,
            model.vocab_size,
        )
        invalid_outputs = (
            (logits, "two-item tuple"),
            ((logits,), "two-item tuple"),
            ((logits, None, None), "two-item tuple"),
            (("not logits", None), "rank-3 tensor"),
            ((logits[:, 0], torch.ones(2)), "rank-3 tensor"),
            ((logits[:1], None), "logits and labels"),
            ((logits[:, :-1], None), "logits and labels"),
            ((logits[..., :-1], None), "vocabulary dimension"),
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
                model._model_step_outputs((source_ids, target_ids))
            loss.assert_not_called()

    def test_invalid_translation_batch_geometry_is_rejected_before_forward(self):
        model = self.preset()
        source_ids, target_ids = self.batch()
        invalid_batches = (
            ((source_ids,), "must contain"),
            ((source_ids, target_ids, target_ids), "must contain"),
            (("not source IDs", target_ids), "rank-2 tensors"),
            ((source_ids, "not target IDs"), "rank-2 tensors"),
            ((source_ids.unsqueeze(0), target_ids), "rank-2 tensors"),
            ((source_ids, target_ids.unsqueeze(0)), "rank-2 tensors"),
            ((source_ids, target_ids[:, :1]), "at least 2 IDs"),
            ((source_ids[:1], target_ids), "batch dimension"),
        )

        for batch, message in invalid_batches:
            with (
                self.subTest(message=message),
                patch.object(model, "forward", wraps=model.forward) as forward,
                self.assertRaisesRegex(ValueError, message),
            ):
                model._model_step_outputs(batch)
            forward.assert_not_called()

    def test_metric_logging_exposes_canonical_translation_metrics(self):
        model = self.preset()
        output = model._model_step_outputs(self.batch())

        with patch.object(model, "log_dict") as log_dict:
            model._log_step("validation", output, prog_bar=True)

        metrics = log_dict.call_args.args[0]
        self.assertEqual(
            set(metrics),
            {
                "validation/loss",
                "validation/nll",
                "validation/perplexity",
                "validation/token_accuracy",
                "validation/auxiliary_loss",
            },
        )
        torch.testing.assert_close(
            metrics["validation/perplexity"],
            torch.exp(output.nll.detach().clamp(max=20.0)),
        )

    def test_all_pad_targets_report_zero_token_accuracy(self) -> None:
        logits = torch.tensor([[[3.0, 0.0], [0.0, 3.0]]], dtype=torch.float64)
        output = TranslationStepOutput(
            total_loss=torch.tensor(float("nan"), dtype=torch.float64),
            nll=torch.tensor(0.5, dtype=torch.float64),
            logits=logits,
            labels=torch.zeros(1, 2, dtype=torch.long),
            auxiliary_loss=torch.tensor(0.0, dtype=torch.float64),
        )

        metrics = _translation_step_metrics("validation", output, pad_token_id=0)

        torch.testing.assert_close(
            metrics["validation/token_accuracy"],
            torch.tensor(0.0, dtype=torch.float64),
        )
        self.assertEqual(metrics["validation/token_accuracy"].device, logits.device)

    def test_corpus_sacrebleu_and_generation_disable_switch(self):
        model = self.preset()
        predictions = ["the cat sat here", "a red house stands"]
        references = list(predictions)

        with patch.object(model, "log") as log:
            model._log_corpus_bleu("validation", predictions, references)

        self.assertEqual(log.call_args.args[0], "validation/bleu")
        torch.testing.assert_close(log.call_args.args[1], torch.tensor(1.0))

        model.generation_metrics_flag = False
        with patch.object(model, "log") as disabled_log:
            model._log_corpus_bleu("validation", predictions, references)
        disabled_log.assert_not_called()

    def test_real_stages_generation_lifecycle_and_missing_decode_no_ops(self) -> None:
        model = StaticTranslationExperiment(auxiliary_loss=torch.tensor(0.25))
        model._trainer = SimpleNamespace(datamodule=NumericTranslationDataModule())
        first_batch = (
            torch.tensor([[2, 4, 3]]),
            torch.tensor([[2, 1, 3, 0]]),
        )
        second_batch = (
            torch.tensor([[2, 3, 0]]),
            torch.tensor([[2, 4, 3, 0]]),
        )
        expected_task_loss = F.cross_entropy(
            torch.zeros(3, model.vocab_size),
            first_batch[1][:, 1:].reshape(-1),
            ignore_index=model.pad_token_id,
            label_smoothing=model.label_smoothing,
        )
        expected_total_loss = expected_task_loss + 0.25

        private_step_loss = model._model_step(first_batch)
        training_loss = model.training_step(first_batch, 0)
        model.on_validation_epoch_start()
        validation_losses = (
            model.validation_step(first_batch, 0),
            model.validation_step(second_batch, 1),
        )

        torch.testing.assert_close(private_step_loss, expected_total_loss)
        torch.testing.assert_close(training_loss, expected_total_loss)
        self.assertTrue(training_loss.requires_grad)
        for validation_loss in validation_losses:
            torch.testing.assert_close(validation_loss, expected_total_loss)
        self.assertIs(model.forward_calls[1][0], first_batch[0])
        torch.testing.assert_close(model.forward_calls[1][1], first_batch[1][:, :-1])
        self.assertEqual(
            model.forward_calls[1][1].data_ptr(),
            first_batch[1].data_ptr(),
        )
        self.assertEqual(
            model._validation_predictions,
            ["2 4 3", "2 3 0"],
        )
        self.assertEqual(
            model._validation_references,
            ["2 1 3 0", "2 4 3 0"],
        )
        model.on_validation_epoch_end()
        self.assertEqual(model.log_calls[0][0], "validation/bleu")

        model.on_test_epoch_start()
        test_losses = (
            model.test_step(first_batch, 0),
            model.test_step(second_batch, 1),
        )
        for test_loss in test_losses:
            torch.testing.assert_close(test_loss, expected_total_loss)
        self.assertEqual(model._test_predictions, ["2 4 3", "2 3 0"])
        self.assertEqual(model._test_references, ["2 1 3 0", "2 4 3 0"])
        model.on_test_epoch_end()
        self.assertEqual(model.log_calls[1][0], "test/bleu")

        expected_stages = ("train", "validation", "validation", "test", "test")
        expected_prog_bars = (True, True, True, False, False)
        for (payload, kwargs), stage, prog_bar in zip(
            model.log_dict_calls,
            expected_stages,
            expected_prog_bars,
            strict=True,
        ):
            self.assertEqual(
                set(payload),
                {
                    f"{stage}/loss",
                    f"{stage}/nll",
                    f"{stage}/perplexity",
                    f"{stage}/token_accuracy",
                    f"{stage}/auxiliary_loss",
                },
            )
            self.assertEqual(
                kwargs,
                {
                    "prog_bar": prog_bar,
                    "on_step": stage == "train",
                    "on_epoch": True,
                    "batch_size": 1,
                },
            )

        model.on_validation_epoch_start()
        model.on_test_epoch_start()
        self.assertEqual(model._validation_predictions, [])
        self.assertEqual(model._validation_references, [])
        self.assertEqual(model._test_predictions, [])
        self.assertEqual(model._test_references, [])

        no_op_model = StaticTranslationExperiment()
        no_op_model._collect_generation("validation", first_batch)
        no_op_model._trainer = SimpleNamespace(datamodule=None)
        no_op_model._collect_generation("validation", first_batch)
        no_op_model._trainer = SimpleNamespace(datamodule=SimpleNamespace())
        no_op_model._collect_generation("validation", first_batch)
        no_op_model._trainer = SimpleNamespace(
            datamodule=SimpleNamespace(decode_batch=lambda _tokens: [])
        )
        no_op_model._collect_generation("validation", first_batch)
        no_op_model._log_corpus_bleu("validation", [], [])
        self.assertEqual(no_op_model._validation_predictions, [])
        self.assertEqual(no_op_model._validation_references, [])
        self.assertEqual(no_op_model.log_calls, [])
        self.assertEqual(len(no_op_model.generate_calls), 1)

        no_op_model.generation_metrics_flag = False
        no_op_model._collect_generation("validation", first_batch)
        self.assertEqual(len(no_op_model.generate_calls), 1)

    def test_optimizer_and_inverse_square_root_scheduler_defaults(self):
        model = self.preset()

        configured = model.configure_optimizers()
        optimizer = configured["optimizer"]
        scheduler = configured["lr_scheduler"]["scheduler"]

        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.98))
        self.assertEqual(optimizer.defaults["eps"], 1e-9)
        self.assertEqual(configured["lr_scheduler"]["interval"], "step")
        expected_initial_factor = 16**-0.5 * 4_000**-1.5
        self.assertAlmostEqual(
            scheduler.get_last_lr()[0],
            model.learning_rate * expected_initial_factor,
        )
        schedule = scheduler.lr_lambdas[0]
        for step in (-2, 0, model.warmup_steps - 1, 4 * model.warmup_steps - 1):
            with self.subTest(step=step):
                safe_step = max(1, step + 1)
                expected_factor = model.model_dim**-0.5 * min(
                    safe_step**-0.5,
                    safe_step * model.warmup_steps**-1.5,
                )
                self.assertAlmostEqual(schedule(step), expected_factor)

    def test_random_translation_data_is_deterministic_and_decodes_numerically(self):
        model = self.preset()
        data = RandomTranslationDataModule(
            model.cfg,
            batch_size=2,
            num_batches=1,
            seed=7,
        )

        first_source, first_target = next(iter(data.train_dataloader()))
        second_source, second_target = next(iter(data.train_dataloader()))

        torch.testing.assert_close(first_source, second_source)
        torch.testing.assert_close(first_target, second_target)
        self.assertEqual(first_source.shape, (2, 6))
        self.assertEqual(first_target.shape, (2, 6))
        self.assertTrue(torch.all(first_source[:, 0] == 2))
        self.assertTrue(torch.all(first_target[:, 0] == 2))
        self.assertEqual(
            data.decode_batch(torch.tensor([[2, 14, 15, 3, 0], [2, 9, 3, 0, 0]])),
            ["14 15", "9"],
        )

    def test_task_ranking_and_dataset_length_hook(self):
        package = model_package("transformer/linear")
        experiment = Experiment(model_package=package)
        self.assertEqual(experiment.experiment_task, ExperimentTask.TEXT_TRANSLATION)
        self.assertEqual(
            experiment_task_name(experiment.experiment_task), "text-translation"
        )
        behavior = experiment_task_behavior(experiment.experiment_task)
        self.assertGreater(
            behavior.ranking_score({"metrics": {"validation/bleu": 12.0}}),
            behavior.ranking_score({"metrics": {"validation/loss": 1.0}}),
        )
        self.assertGreater(
            behavior.ranking_score({"metrics": {"validation/loss": 1.0}}),
            behavior.ranking_score({"metrics": {"validation/loss": 2.0}}),
        )

        runtime = package.bind_runtime_defaults(
            {
                "batch_size": 7,
                "source_sequence_length": 31,
                "target_sequence_length": 29,
            }
        )
        config = TransformerLinearConfigBuilder(runtime=runtime).build()
        training_run = SimpleNamespace(config=config)
        self.assertEqual(
            experiment._dataset_constructor_kwargs(training_run),
            {
                "batch_size": 7,
                "source_sequence_length": 31,
                "target_sequence_length": 29,
            },
        )
        runtime_config = experiment._load_runtime_config({})
        self.assertEqual(runtime_config["seed"], 0)
        dataset = SimpleNamespace(num_workers=4, seed=99)
        experiment._configure_dataset(dataset, runtime_config)
        self.assertEqual(dataset.seed, 0)


if __name__ == "__main__":
    unittest.main()
