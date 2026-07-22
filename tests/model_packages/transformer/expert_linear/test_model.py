import importlib
import runpy
import sys
import unittest
from unittest.mock import patch

import torch

from emperor.attention import MixtureOfAttentionHeadsConfig
from emperor.experiments.translation import TranslationExperiment
from emperor.experts import MixtureOfExpertsConfig, MixtureOfExpertsModelConfig
from emperor.linears import LinearLayer
from emperor.sampler import RouterModel
from emperor.transformer import (
    TransformerDecoderLayerConfig,
    TransformerEncoderLayerConfig,
)
from model_runtime.packages import PresetLock
from models.catalog import model_package
from models.training_test_utils import (
    RandomTranslationDataModule,
    tiny_cpu_trainer,
)
from models.transformer.expert_linear import dataset_options
from models.transformer.expert_linear.config_builder import (
    TransformerExpertLinearConfigBuilder,
)
from models.transformer.expert_linear.model import Model
from models.transformer.expert_linear.presets import (
    Experiment,
    ExperimentPreset,
)

_MIXTURE_ATTENTION_TYPE = MixtureOfAttentionHeadsConfig().registry_owner()
_MIXTURE_OF_EXPERTS_LAYER_TYPE = MixtureOfExpertsConfig().registry_owner()
_MIXTURE_OF_EXPERTS_TYPE = MixtureOfExpertsModelConfig().registry_owner()
_TRANSFORMER_DECODER_LAYER_TYPE = TransformerDecoderLayerConfig().registry_owner()
_TRANSFORMER_ENCODER_LAYER_TYPE = TransformerEncoderLayerConfig().registry_owner()


def _build_config(**runtime_defaults):
    runtime = model_package("transformer/expert_linear").bind_runtime_defaults(
        runtime_defaults
    )
    return TransformerExpertLinearConfigBuilder(runtime=runtime).build()


class TestTransformerExpertLinearModel(unittest.TestCase):
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
        return model_package("transformer/expert_linear").presets.get_config(
            preset,
            dataset or self._datasets()[0],
            config_overrides=self._overrides(**overrides),
        )[0]

    @staticmethod
    def _ids():
        return (
            torch.tensor([[2, 8, 3, 0], [2, 9, 10, 3]]),
            torch.tensor([[2, 11, 3], [2, 12, 3]]),
        )

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
        package = importlib.import_module("models.transformer.expert_linear")
        self.assertEqual(package.__all__, ["MODEL_PACKAGE"])
        self.assertTrue(issubclass(Model, TranslationExperiment))
        experiment = Experiment(
            model_package=model_package("transformer/expert_linear")
        )
        self.assertEqual(
            experiment.model_package.identity.catalog_key,
            "transformer/expert_linear",
        )
        self.assertIs(
            package.MODEL_PACKAGE,
            model_package("transformer/expert_linear"),
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
                ["models.transformer.expert_linear", "--preset", "baseline"],
            ),
            patch(
                "models.package_cli.execute_runs",
                return_value=(),
            ) as execute_runs,
            self.assertRaises(SystemExit) as exit_context,
        ):
            runpy.run_module(
                "models.transformer.expert_linear.__main__",
                run_name="__main__",
            )

        self.assertEqual(exit_context.exception.code, 0)
        execute_runs.assert_called_once()
        package, plan = execute_runs.call_args.args
        self.assertEqual(package.catalog_key, "transformer/expert_linear")
        self.assertEqual(plan.presets, ("baseline",))
        self.assertIsNone(plan.search)
        self.assertEqual(
            plan.datasets,
            (self._datasets()[0].__name__,),
        )

    def test_all_presets_build_forward_and_keep_attention_roles(self):
        self.assertEqual(
            [preset.value for preset in ExperimentPreset],
            list(range(1, 29)),
        )
        source, target = self._ids()
        presets = model_package("transformer/expert_linear").presets
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
                for lock in presets.locks_for_preset(preset).values():
                    if isinstance(lock, PresetLock):
                        self.assertIn(preset.name, lock.reason)

    def test_baseline_builds_for_both_multi30k_directions(self):
        for dataset in self._datasets():
            with self.subTest(dataset=dataset.__name__):
                cfg = model_package("transformer/expert_linear").presets.get_config(
                    ExperimentPreset.BASELINE,
                    dataset,
                    config_overrides={
                        "batch_size": 2,
                        "model_dim": 8,
                        "encoder_num_layers": 1,
                        "decoder_num_layers": 1,
                        "attn_num_heads": 2,
                        "ff_stack_hidden_dim": 8,
                        "num_experts": 4,
                        "top_k": 2,
                        "dropout_probability": 0.0,
                    },
                )[0]
                self.assertEqual(cfg.input_dim, 8192)
                self.assertEqual(cfg.output_dim, 8192)
                self.assertEqual(cfg.experiment_config.vocab_size, 8192)

    def test_construction_validation_rejects_dimensions_and_routing(self):
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
            ("num_experts", {"num_experts": 0}, ValueError),
            ("top_k", {"top_k": 0}, ValueError),
            (
                "top_k",
                {"num_experts": 2, "top_k": 3},
                ValueError,
            ),
            (
                "switch_loss_weight",
                {"switch_loss_weight": -0.1},
                ValueError,
            ),
            (
                "capacity_factor",
                {"capacity_factor": -0.1},
                ValueError,
            ),
            ("batch_size", {"batch_size": True}, TypeError),
            ("learning_rate", {"learning_rate": "fast"}, TypeError),
        )
        for field, overrides, error in cases:
            with self.subTest(field=field, overrides=overrides):
                with self.assertRaisesRegex(error, field):
                    _build_config(**overrides)

    def test_mixture_attention_kv_roles_router_isolation_and_routed_gradients(self):
        cfg = self._config(ExperimentPreset.TOP1_SWITCH_AUX)
        model = Model(cfg).eval()
        source, target = self._ids()
        logits, auxiliary_loss = model(source, target)
        (logits.square().mean() + auxiliary_loss).backward()

        encoder_layer = next(
            module
            for module in model.encoder.modules()
            if isinstance(module, _TRANSFORMER_ENCODER_LAYER_TYPE)
        )
        decoder_layer = next(
            module
            for module in model.decoder.modules()
            if isinstance(module, _TRANSFORMER_DECODER_LAYER_TYPE)
        )
        self_attention = encoder_layer.self_attention_model
        cross_attention = decoder_layer.cross_attention_model
        self.assertIsInstance(self_attention, _MIXTURE_ATTENTION_TYPE)
        self.assertIsInstance(cross_attention, _MIXTURE_ATTENTION_TYPE)
        self.assertIsInstance(
            self_attention.projector.key_model,
            _MIXTURE_OF_EXPERTS_LAYER_TYPE,
        )
        self.assertIsInstance(
            self_attention.projector.value_model,
            _MIXTURE_OF_EXPERTS_LAYER_TYPE,
        )
        self.assertNotIsInstance(
            cross_attention.projector.key_model,
            _MIXTURE_OF_EXPERTS_LAYER_TYPE,
        )
        self.assertNotIsInstance(
            cross_attention.projector.value_model,
            _MIXTURE_OF_EXPERTS_LAYER_TYPE,
        )
        self.assertTrue(
            any(
                isinstance(module, _MIXTURE_OF_EXPERTS_TYPE)
                for module in encoder_layer.feed_forward_model.modules()
            )
        )
        routers = [
            module for module in model.modules() if isinstance(module, RouterModel)
        ]
        self.assertGreater(len(routers), 3)
        self.assertEqual(
            len({id(next(router.parameters())) for router in routers}), len(routers)
        )
        self.assertTrue(auxiliary_loss.requires_grad)
        self.assertTrue(
            any(
                parameter.grad is not None and parameter.grad.abs().sum().item() > 0
                for router in routers
                for parameter in router.parameters()
            )
        )
        self.assertTrue(
            any(isinstance(module, LinearLayer) for module in model.modules())
        )
        self.assertIs(model.output_projection.weight, model.shared_embedding.weight)

    def test_baseline_lifecycle_and_top1_signature_train(self):
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

        signature_cfg = self._config(ExperimentPreset.TOP1_SWITCH_AUX)
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
