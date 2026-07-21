import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from emperor.experiments import ExperimentTask, experiment_task_name
from models.training_test_utils import RandomTranslationDataModule
from models.transformer.linear.config_builder import TransformerLinearConfigBuilder
from models.transformer.linear.model import Model
from models.transformer.linear.presets import Experiment


class TestTranslationExperiment(unittest.TestCase):
    def preset(self):
        config = TransformerLinearConfigBuilder(
            batch_size=2,
            model_dim=16,
            source_sequence_length=6,
            target_sequence_length=6,
            encoder_num_layers=1,
            decoder_num_layers=1,
            attn_num_heads=2,
            feed_forward_hidden_dim=32,
            dropout_probability=0.0,
        ).build()
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
            torch.exp(output.nll.detach().clamp(max=math.log(1e9))),
        )

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
        experiment = Experiment()
        self.assertEqual(experiment.experiment_task, ExperimentTask.TEXT_TRANSLATION)
        self.assertEqual(
            experiment_task_name(experiment.experiment_task), "text-translation"
        )
        self.assertGreater(
            experiment._result_ranking_score({"metrics": {"validation/bleu": 12.0}}),
            experiment._result_ranking_score({"metrics": {"validation/loss": 1.0}}),
        )
        self.assertGreater(
            experiment._result_ranking_score({"metrics": {"validation/loss": 1.0}}),
            experiment._result_ranking_score({"metrics": {"validation/loss": 2.0}}),
        )

        config = TransformerLinearConfigBuilder(
            batch_size=7,
            source_sequence_length=31,
            target_sequence_length=29,
        ).build()
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
