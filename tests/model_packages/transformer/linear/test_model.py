import importlib
import runpy
import sys
import unittest
from unittest.mock import patch

import torch

from emperor.augmentations.adaptive_parameters import AdaptiveLinearLayerConfig
from emperor.experiments.translation import TranslationExperiment
from emperor.linears import LinearLayer
from model_runtime.packages import GridSearch, PresetLock
from models.catalog import catalog_entry
from models.training_test_utils import (
    RandomTranslationDataModule,
    tiny_cpu_trainer,
)
from models.transformer.linear import dataset_options, search_space
from models.transformer.linear.config_builder import TransformerLinearConfigBuilder
from models.transformer.linear.model import Model
from models.transformer.linear.presets import (
    Experiment,
    ExperimentPreset,
    ExperimentPresets,
)

_ADAPTIVE_LINEAR_LAYER_TYPE = AdaptiveLinearLayerConfig().registry_owner()


class TestTransformerLinearModel(unittest.TestCase):
    def _overrides(self, **overrides):
        return {
            "batch_size": 2,
            "vocab_size": 32,
            "model_dim": 8,
            "source_sequence_length": 4,
            "target_sequence_length": 4,
            "encoder_num_layers": 2,
            "decoder_num_layers": 2,
            "attn_num_heads": 2,
            "ff_stack_hidden_dim": 8,
            "dropout_probability": 0.0,
            "recurrent_max_steps": 2,
            **overrides,
        }

    def _datasets(self):
        return dataset_options.DATASET_OPTIONS_BY_TASK[
            dataset_options.DEFAULT_EXPERIMENT_TASK
        ]

    def _config(self, preset=ExperimentPreset.BASELINE, dataset=None, **overrides):
        return ExperimentPresets().get_config(
            preset,
            dataset or self._datasets()[0],
            config_overrides=self._overrides(**overrides),
        )[0]

    def _ids(self):
        source = torch.tensor([[2, 8, 3, 0], [2, 9, 10, 3]])
        target = torch.tensor([[2, 11, 3], [2, 12, 3]])
        return source, target

    @staticmethod
    def _layer_configs(cfg):
        encoder = cfg.experiment_config.encoder_config
        decoder = cfg.experiment_config.decoder_config
        encoder = getattr(encoder, "block_config", encoder)
        decoder = getattr(decoder, "block_config", decoder)
        return (
            encoder.layer_config.layer_model_config,
            decoder.layer_config.layer_model_config,
        )

    def test_public_surface_catalog_identity_and_translation_task(self):
        package = importlib.import_module("models.transformer.linear")
        self.assertEqual(
            package.__all__,
            [
                "Experiment",
                "ExperimentConfig",
                "ExperimentPreset",
                "ExperimentPresets",
                "Model",
                "TransformerLinearConfigBuilder",
            ],
        )
        self.assertTrue(issubclass(Model, TranslationExperiment))
        self.assertEqual(Experiment()._public_model_id(), "transformer/linear")
        self.assertEqual(
            catalog_entry("transformer/linear").module_path,
            "models.transformer.linear",
        )
        self.assertEqual(
            [dataset.language_pair for dataset in self._datasets()],
            [("de", "en"), ("en", "de")],
        )
        for dataset in self._datasets():
            with self.subTest(dataset=dataset.__name__):
                self.assertEqual(dataset.flattened_input_dim, 8192)
                self.assertEqual(dataset.num_classes, 8192)

    def test_module_entrypoint_resolves_cli_without_training(self):
        with (
            patch.object(
                sys,
                "argv",
                ["models.transformer.linear", "--preset", "baseline"],
            ),
            patch(
                "models.package_cli.execute_runs",
                return_value=(),
            ) as execute_runs,
        ):
            runpy.run_module(
                "models.transformer.linear.__main__",
                run_name="__main__",
            )

        execute_runs.assert_called_once()
        package, plan = execute_runs.call_args.args
        self.assertEqual(package.catalog_key, "transformer/linear")
        self.assertEqual(plan.presets, ("baseline",))
        self.assertIsNone(plan.search)
        self.assertEqual(
            plan.datasets,
            tuple(dataset.__name__ for dataset in self._datasets()),
        )

    def test_all_presets_build_forward_and_keep_attention_roles(self):
        self.assertEqual(
            [preset.value for preset in ExperimentPreset],
            list(range(1, 28)),
        )
        source, target = self._ids()
        presets = ExperimentPresets()
        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = self._config(preset)
                model = Model(cfg).eval()
                with torch.no_grad():
                    logits, auxiliary_loss = model(source, target)
                encoder, decoder = self._layer_configs(cfg)

                self.assertEqual(logits.shape, (2, 3, 32))
                self.assertEqual(auxiliary_loss.shape, ())
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(torch.isfinite(auxiliary_loss))
                self.assertFalse(encoder.attention_config.causal_attention_mask_flag)
                self.assertTrue(
                    decoder.self_attention_config.causal_attention_mask_flag
                )
                self.assertFalse(
                    decoder.cross_attention_config.causal_attention_mask_flag
                )
                locks = presets.locks_for_preset(preset)
                for lock in locks.values():
                    if isinstance(lock, PresetLock):
                        self.assertIn(preset.name, lock.reason)

    def test_baseline_builds_for_both_multi30k_directions(self):
        for dataset in self._datasets():
            with self.subTest(dataset=dataset.__name__):
                cfg = ExperimentPresets().get_config(
                    ExperimentPreset.BASELINE,
                    dataset,
                    config_overrides={
                        "batch_size": 2,
                        "model_dim": 8,
                        "encoder_num_layers": 1,
                        "decoder_num_layers": 1,
                        "attn_num_heads": 2,
                        "ff_stack_hidden_dim": 8,
                        "dropout_probability": 0.0,
                    },
                )[0]
                self.assertEqual(cfg.input_dim, 8192)
                self.assertEqual(cfg.output_dim, 8192)
                self.assertEqual(cfg.experiment_config.vocab_size, 8192)

    def test_construction_validation_rejects_invalid_local_options(self):
        cases = (
            ("batch_size", {"batch_size": 0}, ValueError),
            ("learning_rate", {"learning_rate": 0.0}, ValueError),
            ("vocab_size", {"vocab_size": 3}, ValueError),
            ("model_dim", {"model_dim": 0}, ValueError),
            ("source_sequence_length", {"source_sequence_length": 1}, ValueError),
            ("target_sequence_length", {"target_sequence_length": 1}, ValueError),
            ("encoder_num_layers", {"encoder_num_layers": 0}, ValueError),
            ("decoder_num_layers", {"decoder_num_layers": 0}, ValueError),
            ("recurrent_max_steps", {"recurrent_max_steps": 0}, ValueError),
            ("attn_num_heads", {"model_dim": 8, "attn_num_heads": 3}, ValueError),
            ("attn_num_heads", {"attn_num_heads": 0}, ValueError),
            ("ff_stack_hidden_dim", {"ff_stack_hidden_dim": 0}, ValueError),
            ("ff_num_layers", {"ff_num_layers": 0}, ValueError),
            ("dropout_probability", {"dropout_probability": -0.1}, ValueError),
            ("dropout_probability", {"dropout_probability": 1.1}, ValueError),
            ("batch_size", {"batch_size": True}, TypeError),
            ("learning_rate", {"learning_rate": "fast"}, TypeError),
        )
        for field, overrides, error in cases:
            with self.subTest(field=field, overrides=overrides):
                with self.assertRaisesRegex(error, field):
                    TransformerLinearConfigBuilder(**overrides)

    def test_search_axes_apply_and_unknown_axes_fail(self):
        configs = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            self._datasets()[0],
            GridSearch(),
            search_keys=["attn_num_heads"],
            config_overrides=self._overrides(model_dim=8),
        )
        self.assertEqual(
            {self._layer_configs(cfg)[0].attention_config.num_heads for cfg in configs},
            set(search_space.SEARCH_SPACE_ATTN_NUM_HEADS),
        )
        with self.assertRaises((KeyError, ValueError)):
            ExperimentPresets().get_config(
                ExperimentPreset.BASELINE,
                self._datasets()[0],
                GridSearch(),
                search_keys=["not_an_axis"],
                config_overrides=self._overrides(),
            )

    def test_linear_backend_tying_independence_and_gradients(self):
        model = Model(self._config()).eval()
        source, target = self._ids()
        logits, auxiliary_loss = model(source, target)
        (logits.square().mean() + auxiliary_loss).backward()

        modules = tuple(model.modules())
        self.assertTrue(any(isinstance(module, LinearLayer) for module in modules))
        self.assertFalse(
            any(isinstance(module, _ADAPTIVE_LINEAR_LAYER_TYPE) for module in modules)
        )
        self.assertIs(model.source_embedding, model.target_embedding)
        self.assertIs(model.output_projection.weight, model.shared_embedding.weight)
        self.assertIsNot(
            next(model.encoder.parameters()),
            next(model.decoder.parameters()),
        )
        self.assertIsNotNone(model.shared_embedding.weight.grad)
        self.assertGreater(model.shared_embedding.weight.grad.abs().sum().item(), 0)

    def test_baseline_lifecycle_and_recurrent_signature_train(self):
        baseline_cfg = self._config()
        baseline = Model(baseline_cfg)
        data = RandomTranslationDataModule(
            baseline_cfg,
            batch_size=2,
            num_batches=1,
        )
        trainer = tiny_cpu_trainer()
        trainer.fit(baseline, datamodule=data)
        validation = trainer.validate(baseline, datamodule=data)
        testing = trainer.test(baseline, datamodule=data)
        self.assertTrue(torch.isfinite(torch.tensor(validation[0]["validation/loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(testing[0]["test/loss"])))

        signature_cfg = self._config(ExperimentPreset.RECURRENT_GATING_HALTING)
        tiny_cpu_trainer().fit(
            Model(signature_cfg),
            datamodule=RandomTranslationDataModule(
                signature_cfg,
                batch_size=2,
                num_batches=1,
            ),
        )


if __name__ == "__main__":
    unittest.main()
