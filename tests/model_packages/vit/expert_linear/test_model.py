import importlib
import inspect
import unittest

import torch

import models.vit.expert_linear.config as config
import models.vit.expert_linear.dataset_options as dataset_options
from emperor.attention import (
    MixtureOfAttentionHeadsConfig,
    SelfAttentionConfig,
)
from emperor.experts import MixtureOfExpertsConfig, MixtureOfExpertsModelConfig
from emperor.layers import RecurrentLayerConfig
from emperor.linears import LinearLayerConfig
from models.catalog import model_package
from models.vit.expert_linear.config_builder import VitExpertLinearConfigBuilder
from models.vit.expert_linear.model import Model
from models.vit.expert_linear.presets import (
    Experiment,
    ExperimentPreset,
)

_MIXTURE_ATTENTION_TYPE = MixtureOfAttentionHeadsConfig().registry_owner()
_MIXTURE_OF_EXPERTS_TYPE = MixtureOfExpertsModelConfig().registry_owner()
_MIXTURE_OF_EXPERTS_LAYER_TYPE = MixtureOfExpertsConfig().registry_owner()
_SELF_ATTENTION_TYPE = SelfAttentionConfig().registry_owner()


class TestVitExpertLinearModel(unittest.TestCase):
    def test_public_surface_and_catalog_id(self):
        for module_name in (
            "models.vit.expert_linear.config",
            "models.vit.expert_linear.presets",
            "models.vit.expert_linear.model",
            "models.vit.expert_linear.config_builder",
            "models.vit.expert_linear.experiment_config",
        ):
            with self.subTest(module_name=module_name):
                module = importlib.import_module(module_name)

                self.assertEqual(module.__name__, module_name)

        experiment = Experiment(model_package=model_package("vit/expert_linear"))
        self.assertEqual(
            experiment.model_package.identity.catalog_key,
            "vit/expert_linear",
        )
        self.assertIsNotNone(model_package("vit/expert_linear"))

    def test_attention_mode_switch_is_not_part_of_the_public_contract(self):
        self.assertFalse(hasattr(config, "EXPERT_ATTENTION_FLAG"))
        self.assertFalse(hasattr(ExperimentPreset, "EXPERT_ATTENTION"))
        self.assertNotIn(
            "expert_attention_flag",
            inspect.signature(VitExpertLinearConfigBuilder).parameters,
        )
        with self.assertRaises(ValueError):
            self._config(
                ExperimentPreset.BASELINE,
                {
                    **self._runtime_overrides(),
                    "expert_attention_flag": False,
                },
            )

    def test_feed_forward_stack_is_expert_backed(self):
        cfg = self._config(ExperimentPreset.TOP1_SWITCH_AUX)
        feed_forward_stack_config = self._encoder_layer_config(
            cfg
        ).feed_forward_config.stack_config
        expert_core_config = (
            feed_forward_stack_config.stack_config.layer_config.layer_model_config
        )

        self.assertIsInstance(feed_forward_stack_config, MixtureOfExpertsModelConfig)
        self.assertEqual(feed_forward_stack_config.top_k, 1)
        self.assertEqual(
            feed_forward_stack_config.sampler_config.switch_loss_weight,
            0.1,
        )
        self.assertIsInstance(
            expert_core_config.expert_model_config.layer_config.layer_model_config,
            LinearLayerConfig,
        )

    def test_ff_controls_apply_to_outer_moe_slot_not_expert_internals(self):
        cfg = self._config(
            ExperimentPreset.TOP1_SWITCH_AUX,
            config_overrides={
                **self._test_overrides(),
                "ff_stack_hidden_dim": 17,
                "ff_stack_gate_flag": True,
                "expert_stack_hidden_dim": 11,
            },
        )
        feed_forward_stack_config = self._encoder_layer_config(
            cfg
        ).feed_forward_config.stack_config
        outer_stack_config = feed_forward_stack_config.stack_config
        expert_stack_config = (
            outer_stack_config.layer_config.layer_model_config.expert_model_config
        )

        self.assertIsInstance(feed_forward_stack_config, MixtureOfExpertsModelConfig)
        self.assertEqual(outer_stack_config.hidden_dim, 17)
        self.assertIsNotNone(outer_stack_config.layer_config.gate_config)
        self.assertEqual(expert_stack_config.hidden_dim, 11)
        self.assertIsNone(expert_stack_config.layer_config.gate_config)

    def test_ff_recurrent_flag_wraps_outer_moe_feed_forward_slot(self):
        cfg = self._config(
            ExperimentPreset.TOP1_SWITCH_AUX,
            config_overrides={
                **self._test_overrides(),
                "ff_recurrent_flag": True,
                "ff_recurrent_max_steps": 2,
            },
        )
        feed_forward_stack_config = self._encoder_layer_config(
            cfg
        ).feed_forward_config.stack_config
        output = Model(cfg)(self._fake_batch(cfg))
        logits = output[0] if isinstance(output, tuple) else output

        self.assertIsInstance(feed_forward_stack_config, RecurrentLayerConfig)
        self.assertIsInstance(
            feed_forward_stack_config.block_config,
            MixtureOfExpertsModelConfig,
        )
        self.assertEqual(logits.shape, (2, cfg.output_dim))

    def test_attention_projection_controls_apply_to_regular_projection_path_only(self):
        cfg = self._config(
            ExperimentPreset.BASELINE,
            config_overrides={
                **self._test_overrides(),
                "attn_stack_hidden_dim": 17,
                "attn_stack_gate_flag": True,
                "expert_stack_hidden_dim": 11,
            },
        )
        attention_config = self._encoder_layer_config(cfg).attention_config
        projection_stack_config = attention_config.projection_model_config
        expert_stack_config = attention_config.experts_config.expert_model_config

        self.assertIsInstance(attention_config, MixtureOfAttentionHeadsConfig)
        self.assertEqual(projection_stack_config.hidden_dim, 17)
        self.assertIsNotNone(projection_stack_config.layer_config.gate_config)
        self.assertEqual(expert_stack_config.hidden_dim, 11)
        self.assertIsNone(expert_stack_config.layer_config.gate_config)

    def test_all_presets_build_mixture_of_attention_heads_runtime(self):
        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = self._config(preset, self._test_overrides())
                model = Model(cfg)
                modules = tuple(model.modules())

                self.assertTrue(
                    any(
                        isinstance(module, _MIXTURE_ATTENTION_TYPE)
                        for module in modules
                    )
                )
                self.assertFalse(
                    any(isinstance(module, _SELF_ATTENTION_TYPE) for module in modules)
                )

    def test_encoder_layers_own_separate_attention_and_feed_forward_experts(self):
        cfg = self._config(
            ExperimentPreset.BASELINE,
            {
                **self._test_overrides(),
                "stack_num_layers": 2,
            },
        )
        model = Model(cfg)
        encoder_layers = [layer.model for layer in model.transformer.layers]
        attentions = [layer.self_attention_model for layer in encoder_layers]
        feed_forwards = [layer.feed_forward_model for layer in encoder_layers]

        self.assertEqual(len(encoder_layers), 2)
        self.assertTrue(
            all(
                isinstance(attention, _MIXTURE_ATTENTION_TYPE)
                for attention in attentions
            )
        )

        for left, right in (
            (attentions[0], attentions[1]),
            (feed_forwards[0], feed_forwards[1]),
            (attentions[0], feed_forwards[0]),
            (attentions[1], feed_forwards[1]),
        ):
            with self.subTest(
                left=type(left).__name__,
                right=type(right).__name__,
            ):
                self._assert_parameter_sets_are_disjoint(left, right)

        self.assertIsNot(
            attentions[0].projector.sampler.router,
            attentions[1].projector.sampler.router,
        )
        self.assertIsNot(
            attentions[0].projector.query_model.expert_modules[0],
            attentions[1].projector.query_model.expert_modules[0],
        )

    def test_classifier_step_adds_routed_auxiliary_loss(self):
        torch.manual_seed(0)
        cfg = self._config(
            ExperimentPreset.TOP1_SWITCH_AUX,
            self._test_overrides(),
        )
        model = Model(cfg)
        images = self._fake_batch(cfg)
        labels = torch.randint(0, cfg.output_dim, (cfg.batch_size,))
        captured_outputs = []
        hook = model.register_forward_hook(
            lambda _model, _args, output: captured_outputs.append(output)
        )

        try:
            loss, logits, returned_labels = model._model_step((images, labels))
        finally:
            hook.remove()

        self.assertEqual(len(captured_outputs), 1)
        self.assertIsInstance(captured_outputs[0], tuple)
        auxiliary_loss = captured_outputs[0][-1]
        self.assertGreater(auxiliary_loss.item(), 0.0)
        torch.testing.assert_close(
            loss,
            model.loss_fn(logits, labels) + auxiliary_loss,
        )
        torch.testing.assert_close(returned_labels, labels)

    def test_attention_kv_modes_forward_backward(self):
        cases = (
            ("regular_kv", ExperimentPreset.BASELINE, False, False),
            ("expert_kv", ExperimentPreset.BASELINE, True, False),
            ("expert_kv_with_bias", ExperimentPreset.ATTENTION_BIAS, True, True),
        )
        for name, preset, use_kv_experts, expect_bias in cases:
            with self.subTest(name=name):
                torch.manual_seed(0)
                cfg = self._config(
                    preset,
                    {
                        **self._test_overrides(),
                        "expert_attention_use_kv_expert_models_flag": use_kv_experts,
                    },
                )
                model = Model(cfg)
                images = self._fake_batch(cfg).requires_grad_()

                output = model(images)

                self.assertIsInstance(output, tuple)
                logits, auxiliary_loss = output
                self.assertEqual(logits.shape, (2, cfg.output_dim))
                self.assertEqual(auxiliary_loss.dim(), 0)
                self.assertGreater(auxiliary_loss.item(), 0.0)
                self.assertTrue(auxiliary_loss.requires_grad)

                attention = next(
                    module
                    for module in model.modules()
                    if isinstance(module, _MIXTURE_ATTENTION_TYPE)
                )
                projector = attention.projector
                self.assertEqual(
                    attention.cfg.use_kv_expert_models_flag,
                    use_kv_experts,
                )
                self.assertEqual(attention.cfg.add_key_value_bias_flag, expect_bias)

                loss = logits.square().mean() + auxiliary_loss
                loss.backward()
                self.assertIsNotNone(images.grad)
                self.assertGreater(images.grad.abs().sum().item(), 0.0)

                if use_kv_experts:
                    expert_models = {
                        "query": projector.query_model,
                        "key": projector.key_model,
                        "value": projector.value_model,
                        "output": projector.output_model,
                    }
                    for role, expert_model in expert_models.items():
                        with self.subTest(name=name, role=role):
                            self.assertIsInstance(
                                expert_model, _MIXTURE_OF_EXPERTS_LAYER_TYPE
                            )
                            self._assert_nonzero_parameter_gradients(
                                expert_model,
                                role,
                            )

                if expect_bias:
                    self.assertIsNotNone(attention.bias.key_bias_vector.grad)
                    self.assertIsNotNone(attention.bias.value_bias_vector.grad)
                    self.assertGreater(
                        attention.bias.key_bias_vector.grad.abs().sum().item(),
                        0.0,
                    )
                    self.assertGreater(
                        attention.bias.value_bias_vector.grad.abs().sum().item(),
                        0.0,
                    )

    def test_all_presets_forward_one_batch(self):
        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = self._config(preset)
                output = Model(cfg)(self._fake_batch(cfg))
                logits = output[0] if isinstance(output, tuple) else output

                self.assertEqual(logits.shape, (2, cfg.output_dim))

    def _config(self, preset: ExperimentPreset, config_overrides: dict | None = None):
        return model_package("vit/expert_linear").presets.get_config(
            preset,
            dataset_options.DATASET_OPTIONS_BY_TASK[
                dataset_options.DEFAULT_EXPERIMENT_TASK
            ][0],
            config_overrides=config_overrides or self._runtime_overrides(),
        )[0]

    def _runtime_overrides(self) -> dict:
        return {"batch_size": 2}

    def _test_overrides(self) -> dict:
        return {
            "batch_size": 2,
            "hidden_dim": 16,
            "stack_num_layers": 1,
            "stack_dropout_probability": 0.0,
            "attn_num_heads": 4,
        }

    def _fake_batch(self, cfg):
        patch_config = cfg.experiment_config.patch_config
        return torch.randn(
            2,
            patch_config.num_input_channels,
            patch_config.patch_size * 7,
            patch_config.patch_size * 7,
        )

    def _encoder_layer_config(self, cfg):
        return cfg.experiment_config.encoder_config.layer_config.layer_model_config

    def _assert_nonzero_parameter_gradients(self, model, role: str) -> None:
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients, f"{role} expert model had no gradients")
        self.assertTrue(
            any(gradient.abs().sum().item() > 0.0 for gradient in gradients),
            f"{role} expert model gradients were all zero",
        )

    def _assert_parameter_sets_are_disjoint(self, left, right) -> None:
        left_parameter_ids = {id(parameter) for parameter in left.parameters()}
        right_parameter_ids = {id(parameter) for parameter in right.parameters()}

        self.assertTrue(left_parameter_ids)
        self.assertTrue(right_parameter_ids)
        self.assertTrue(left_parameter_ids.isdisjoint(right_parameter_ids))


if __name__ == "__main__":
    unittest.main()
