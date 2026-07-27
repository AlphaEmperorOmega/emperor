import importlib
import runpy
import sys
import unittest
from unittest.mock import patch

import torch

from emperor.augmentations.adaptive_parameters import AdaptiveLinearLayerConfig
from emperor.experiments.translation import TranslationExperiment
from emperor.layers import (
    HierarchicalReasoningModelRecurrentConfig,
    LayerNormPositionOptions,
    RecurrentCompositionConfig,
    RecurrentLayerConfig,
    TinyRecursiveModelRecurrentConfig,
    WeightedBlendResidualConfig,
)
from emperor.linears import LinearLayer
from model_runtime.packages import PresetLock
from models.catalog import model_package
from models.training_test_utils import (
    RandomTranslationDataModule,
    tiny_cpu_trainer,
)
from models.transformer.linear import dataset_options
from models.transformer.linear.config_builder import TransformerLinearConfigBuilder
from models.transformer.linear.model import Model
from models.transformer.linear.presets import (
    Experiment,
    ExperimentPreset,
)

_ADAPTIVE_LINEAR_LAYER_TYPE = AdaptiveLinearLayerConfig().registry_owner()


def _build_config(**runtime_defaults):
    runtime = model_package("transformer/linear").bind_runtime_defaults(
        runtime_defaults
    )
    return TransformerLinearConfigBuilder(runtime=runtime).build()


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
        return model_package("transformer/linear").presets.get_config(
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
        self.assertEqual(package.__all__, ["MODEL_PACKAGE"])
        self.assertTrue(issubclass(Model, TranslationExperiment))
        experiment = Experiment(model_package=model_package("transformer/linear"))
        self.assertEqual(
            experiment.model_package.identity.catalog_key,
            "transformer/linear",
        )
        self.assertIs(package.MODEL_PACKAGE, model_package("transformer/linear"))
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
            self.assertRaises(SystemExit) as exit_context,
        ):
            runpy.run_module(
                "models.transformer.linear.__main__",
                run_name="__main__",
            )

        self.assertEqual(exit_context.exception.code, 0)
        execute_runs.assert_called_once()
        package, plan = execute_runs.call_args.args
        self.assertEqual(package.catalog_key, "transformer/linear")
        self.assertEqual(plan.presets, ("baseline",))
        self.assertIsNone(plan.search)
        self.assertEqual(
            plan.datasets,
            (self._datasets()[0].__name__,),
        )

    def test_all_presets_build_forward_and_keep_attention_roles(self):
        self.assertEqual(
            [preset.value for preset in ExperimentPreset],
            list(range(1, 28)),
        )
        source, target = self._ids()
        presets = model_package("transformer/linear").presets
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
                cfg = model_package("transformer/linear").presets.get_config(
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
                    _build_config(**overrides)

    def test_recurrent_composition_selector_keeps_standard_default(self):
        cfg = _build_config(
            batch_size=2,
            vocab_size=32,
            model_dim=8,
            source_sequence_length=4,
            target_sequence_length=4,
            encoder_num_layers=1,
            decoder_num_layers=1,
            attn_num_heads=2,
            ff_stack_hidden_dim=8,
            dropout_probability=0.0,
            recurrent_flag=True,
        )

        self.assertIsInstance(
            cfg.experiment_config.encoder_config,
            RecurrentLayerConfig,
        )

    def test_updated_residual_config_builds_for_stack_and_recurrence(self):
        cfg = _build_config(
            batch_size=2,
            vocab_size=32,
            model_dim=8,
            source_sequence_length=4,
            target_sequence_length=4,
            encoder_num_layers=1,
            decoder_num_layers=1,
            attn_num_heads=2,
            ff_stack_hidden_dim=8,
            dropout_probability=0.0,
            recurrent_flag=True,
            stack_residual_connection_option=WeightedBlendResidualConfig,
            recurrent_residual_connection_option=WeightedBlendResidualConfig,
        )

        encoder = cfg.experiment_config.encoder_config
        self.assertIsInstance(encoder, RecurrentLayerConfig)
        self.assertIsInstance(
            encoder.residual_config,
            WeightedBlendResidualConfig,
        )
        self.assertIsInstance(
            encoder.block_config.layer_config.residual_config,
            WeightedBlendResidualConfig,
        )

        model = Model(cfg).eval()
        source, target = self._ids()
        with torch.no_grad():
            logits, auxiliary_loss = model(source, target)

        self.assertEqual(logits.shape, (2, 3, 32))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(auxiliary_loss))

    def test_standard_recurrence_reinjection_builds_for_every_transformer_scope(self):
        cfg = _build_config(
            batch_size=2,
            vocab_size=32,
            model_dim=8,
            source_sequence_length=4,
            target_sequence_length=4,
            encoder_num_layers=1,
            decoder_num_layers=1,
            attn_num_heads=2,
            ff_stack_hidden_dim=8,
            dropout_probability=0.0,
            recurrent_flag=True,
            recurrent_reinject_original_hidden_flag=True,
            attn_recurrent_flag=True,
            attn_recurrent_reinject_original_hidden_flag=True,
            ff_recurrent_flag=True,
            ff_recurrent_reinject_original_hidden_flag=True,
        )

        encoder = cfg.experiment_config.encoder_config
        decoder = cfg.experiment_config.decoder_config
        encoder_layer, decoder_layer = self._layer_configs(cfg)

        for scope, recurrent_config in (
            ("encoder", encoder),
            ("decoder", decoder),
            (
                "encoder attention",
                encoder_layer.attention_config.projection_model_config,
            ),
            (
                "decoder self attention",
                decoder_layer.self_attention_config.projection_model_config,
            ),
            (
                "decoder cross attention",
                decoder_layer.cross_attention_config.projection_model_config,
            ),
            (
                "encoder feed forward",
                encoder_layer.feed_forward_config.stack_config,
            ),
            (
                "decoder feed forward",
                decoder_layer.feed_forward_config.stack_config,
            ),
        ):
            with self.subTest(scope=scope):
                self.assertIsInstance(recurrent_config, RecurrentLayerConfig)
                self.assertIs(recurrent_config.reinject_original_hidden_flag, True)

    def test_tiny_recursive_model_is_selectable_for_stack_and_submodule_recurrence(
        self,
    ):
        cfg = _build_config(
            batch_size=2,
            vocab_size=32,
            model_dim=8,
            source_sequence_length=4,
            target_sequence_length=4,
            encoder_num_layers=1,
            decoder_num_layers=1,
            attn_num_heads=2,
            ff_stack_hidden_dim=8,
            dropout_probability=0.0,
            recurrent_flag=True,
            recurrent_composition_option=TinyRecursiveModelRecurrentConfig,
            recurrent_stack_gate_flag=True,
            recurrent_no_gradient_transition_count=1,
            recurrent_latent_updates_per_answer_update=1,
            recurrent_answer_update_count=2,
            recurrent_initialization_standard_deviation=0.0,
            attn_recurrent_flag=True,
            attn_recurrent_composition_option=TinyRecursiveModelRecurrentConfig,
            attn_recurrent_stack_gate_flag=True,
            attn_recurrent_no_gradient_transition_count=0,
            attn_recurrent_latent_updates_per_answer_update=1,
            attn_recurrent_answer_update_count=1,
            attn_recurrent_initialization_standard_deviation=0.0,
            ff_recurrent_flag=True,
            ff_recurrent_composition_option=TinyRecursiveModelRecurrentConfig,
            ff_recurrent_no_gradient_transition_count=1,
            ff_recurrent_latent_updates_per_answer_update=1,
            ff_recurrent_answer_update_count=1,
            ff_recurrent_initialization_standard_deviation=0.0,
        )

        encoder = cfg.experiment_config.encoder_config
        self.assertIsInstance(encoder, TinyRecursiveModelRecurrentConfig)
        self.assertEqual(
            (
                encoder.latent_updates_per_answer_update,
                encoder.answer_update_count,
            ),
            (1, 2),
        )
        self.assertEqual(encoder.no_gradient_transition_count, 1)
        self.assertIsNotNone(encoder.gate_config)
        encoder_layer = encoder.block_config.layer_config.layer_model_config
        attention_recurrence = encoder_layer.attention_config.projection_model_config
        self.assertIsInstance(
            attention_recurrence,
            TinyRecursiveModelRecurrentConfig,
        )
        self.assertIsNotNone(attention_recurrence.gate_config)
        feed_forward_recurrence = encoder_layer.feed_forward_config.stack_config
        self.assertIsInstance(
            feed_forward_recurrence,
            TinyRecursiveModelRecurrentConfig,
        )
        self.assertEqual(feed_forward_recurrence.no_gradient_transition_count, 1)

        model = Model(cfg).eval()
        source, target = self._ids()
        logits, auxiliary_loss = model(source, target)
        (logits.square().mean() + auxiliary_loss).backward()

        self.assertEqual(logits.shape, (2, 3, 32))
        self.assertTrue(torch.isfinite(logits).all())
        transition_parameters = tuple(model.encoder.block_model.parameters())
        self.assertTrue(transition_parameters)
        self.assertTrue(
            any(parameter.grad is not None for parameter in transition_parameters)
        )
        self.assertIsNotNone(model.encoder.recurrent_gate)
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.encoder.recurrent_gate.parameters()
            )
        )

        checkpoint = model.state_dict()
        self.assertIn("encoder.answer_initial", checkpoint)
        self.assertIn("encoder.latent_initial", checkpoint)
        self.assertFalse(any("reasoning_process" in key for key in checkpoint))
        restored = Model(cfg).eval()
        restored.load_state_dict(checkpoint, strict=True)
        with torch.no_grad():
            restored_logits, restored_auxiliary_loss = restored(source, target)
        torch.testing.assert_close(restored_logits, logits.detach())
        torch.testing.assert_close(restored_auxiliary_loss, auxiliary_loss.detach())

    def test_hierarchical_reasoning_model_is_selectable_as_a_recurrent_stack(self):
        cfg = _build_config(
            batch_size=2,
            vocab_size=32,
            model_dim=8,
            source_sequence_length=4,
            target_sequence_length=4,
            encoder_num_layers=1,
            decoder_num_layers=1,
            attn_num_heads=2,
            ff_stack_hidden_dim=8,
            dropout_probability=0.0,
            recurrent_flag=True,
            recurrent_composition_option=HierarchicalReasoningModelRecurrentConfig,
            recurrent_stack_halting_flag=True,
            recurrent_no_gradient_transition_count=3,
            recurrent_high_cycles=2,
            recurrent_low_cycles=2,
            recurrent_initialization_standard_deviation=0.0,
        )

        encoder = cfg.experiment_config.encoder_config
        self.assertIsInstance(encoder, HierarchicalReasoningModelRecurrentConfig)
        self.assertEqual((encoder.high_cycles, encoder.low_cycles), (2, 2))
        self.assertEqual(encoder.no_gradient_transition_count, 3)
        self.assertIsNotNone(encoder.halting_config)

        model = Model(cfg).eval()
        source, target = self._ids()
        logits, auxiliary_loss = model(source, target)
        (logits.square().mean() + auxiliary_loss).backward()

        self.assertEqual(logits.shape, (2, 3, 32))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertIsNot(model.encoder.high_model, model.encoder.low_model)
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.encoder.high_model.parameters()
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.encoder.low_model.parameters()
            )
        )
        self.assertIsNotNone(model.encoder.halting_model)
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.encoder.halting_model.parameters()
            )
        )

        checkpoint = model.state_dict()
        self.assertIn("encoder.high_initial", checkpoint)
        self.assertIn("encoder.low_initial", checkpoint)
        self.assertTrue(
            any(key.startswith("encoder.high_model.") for key in checkpoint)
        )
        self.assertTrue(any(key.startswith("encoder.low_model.") for key in checkpoint))
        self.assertFalse(any("reasoning_process" in key for key in checkpoint))
        restored = Model(cfg).eval()
        restored.load_state_dict(checkpoint, strict=True)
        with torch.no_grad():
            restored_logits, restored_auxiliary_loss = restored(source, target)
        torch.testing.assert_close(restored_logits, logits.detach())
        torch.testing.assert_close(restored_auxiliary_loss, auxiliary_loss.detach())

    def test_hierarchical_reasoning_model_is_selectable_for_feed_forward_recurrence(
        self,
    ):
        cfg = _build_config(
            batch_size=2,
            vocab_size=32,
            model_dim=8,
            source_sequence_length=4,
            target_sequence_length=4,
            encoder_num_layers=1,
            decoder_num_layers=1,
            attn_num_heads=2,
            ff_stack_hidden_dim=8,
            dropout_probability=0.0,
            ff_recurrent_flag=True,
            ff_recurrent_composition_option=HierarchicalReasoningModelRecurrentConfig,
            ff_recurrent_layer_norm_position=LayerNormPositionOptions.BEFORE,
            ff_recurrent_high_cycles=1,
            ff_recurrent_low_cycles=1,
            ff_recurrent_initialization_standard_deviation=0.0,
        )

        encoder_stack = cfg.experiment_config.encoder_config
        encoder_layer = encoder_stack.layer_config.layer_model_config
        feed_forward_recurrence = encoder_layer.feed_forward_config.stack_config
        self.assertIsInstance(
            feed_forward_recurrence, HierarchicalReasoningModelRecurrentConfig
        )
        self.assertIs(
            feed_forward_recurrence.recurrent_layer_norm_position,
            LayerNormPositionOptions.BEFORE,
        )

        model = Model(cfg).eval()
        source, target = self._ids()
        logits, auxiliary_loss = model(source, target)
        (logits.square().mean() + auxiliary_loss).backward()

        self.assertEqual(logits.shape, (2, 3, 32))
        self.assertTrue(torch.isfinite(logits).all())

    def test_recurrent_composition_selector_rejects_invalid_choices(self):
        cases = (
            (
                {"recurrent_composition_option": object},
                TypeError,
                "RecurrentCompositionConfig",
            ),
            (
                {"recurrent_composition_option": RecurrentCompositionConfig},
                ValueError,
                "concrete recurrent config",
            ),
            (
                {
                    "recurrent_composition_option": TinyRecursiveModelRecurrentConfig,
                    "recurrent_latent_updates_per_answer_update": 0,
                },
                ValueError,
                "recurrent_latent_updates_per_answer_update",
            ),
            (
                {
                    "recurrent_composition_option": TinyRecursiveModelRecurrentConfig,
                    "recurrent_answer_update_count": True,
                },
                TypeError,
                "recurrent_answer_update_count",
            ),
            (
                {
                    "recurrent_composition_option": TinyRecursiveModelRecurrentConfig,
                    "recurrent_initialization_standard_deviation": -0.1,
                },
                ValueError,
                "initialization_standard_deviation",
            ),
            (
                {"recurrent_no_gradient_transition_count": True},
                TypeError,
                "recurrent_no_gradient_transition_count",
            ),
            (
                {"recurrent_no_gradient_transition_count": -1},
                ValueError,
                "recurrent_no_gradient_transition_count",
            ),
            (
                {"recurrent_reinject_original_hidden_flag": 1},
                TypeError,
                "recurrent_reinject_original_hidden_flag",
            ),
            (
                {
                    "recurrent_composition_option": TinyRecursiveModelRecurrentConfig,
                    "recurrent_latent_updates_per_answer_update": 1,
                    "recurrent_answer_update_count": 2,
                    "recurrent_no_gradient_transition_count": 4,
                },
                ValueError,
                "recurrent_no_gradient_transition_count",
            ),
            (
                {
                    "recurrent_composition_option": HierarchicalReasoningModelRecurrentConfig,
                    "recurrent_high_cycles": 0,
                },
                ValueError,
                "recurrent_high_cycles",
            ),
            (
                {
                    "recurrent_composition_option": HierarchicalReasoningModelRecurrentConfig,
                    "recurrent_low_cycles": True,
                },
                TypeError,
                "recurrent_low_cycles",
            ),
            (
                {
                    "recurrent_composition_option": TinyRecursiveModelRecurrentConfig,
                    "recurrent_reinject_original_hidden_flag": True,
                },
                ValueError,
                "fixed-input reinjection",
            ),
            (
                {
                    "ff_recurrent_composition_option": (
                        HierarchicalReasoningModelRecurrentConfig
                    ),
                    "ff_recurrent_reinject_original_hidden_flag": True,
                },
                ValueError,
                "fixed-input reinjection",
            ),
        )
        for overrides, error, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(error, message):
                    _build_config(**overrides)

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
