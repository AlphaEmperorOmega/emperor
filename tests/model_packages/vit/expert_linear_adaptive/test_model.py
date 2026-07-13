import importlib
import inspect
import unittest
from dataclasses import fields, replace

import models.vit.expert_linear_adaptive.config as config
import models.vit.expert_linear_adaptive.dataset_options as dataset_options
import torch
from emperor.attention.core.variants.mixture_of_attention_heads.config import (
    MixtureOfAttentionHeadsConfig,
)
from emperor.attention.core.variants.mixture_of_attention_heads.layer import (
    MixtureOfAttentionHeads,
)
from emperor.attention.core.variants.self_attention.layer import SelfAttention
from emperor.experts.config import MixtureOfExpertsModelConfig
from emperor.experts.core.layers import MixtureOfExperts
from emperor.linears.core.config import AdaptiveLinearLayerConfig
from models.catalog import catalog_entry
from models.vit.expert_linear_adaptive import _config_defaults as config_defaults
from models.vit.expert_linear_adaptive._expert_config_factory import (
    ExpertAdaptiveConfigDependencies,
)
from models.vit.expert_linear_adaptive._vit_expert_config_factory import (
    VitExpertAdaptiveConfigDependencies,
)
from models.vit.expert_linear_adaptive.config_builder import (
    VitExpertLinearAdaptiveConfigBuilder,
)
from models.vit.expert_linear_adaptive.model import Model
from models.vit.expert_linear_adaptive.presets import (
    Experiment,
    ExperimentPreset,
    ExperimentPresets,
)


class TestVitExpertLinearAdaptiveModel(unittest.TestCase):
    def test_public_surface_and_catalog_id(self):
        for module_name in (
            "models.vit.expert_linear_adaptive.config",
            "models.vit.expert_linear_adaptive.presets",
            "models.vit.expert_linear_adaptive.model",
            "models.vit.expert_linear_adaptive.config_builder",
            "models.vit.expert_linear_adaptive.experiment_config",
        ):
            with self.subTest(module_name=module_name):
                module = importlib.import_module(module_name)

                self.assertEqual(module.__name__, module_name)

        self.assertEqual(
            Experiment()._public_model_id(),
            "vit/expert_linear_adaptive",
        )
        self.assertIsNotNone(catalog_entry("vit/expert_linear_adaptive"))

    def test_attention_mode_switch_is_removed(self):
        self.assertFalse(hasattr(config, "EXPERT_ATTENTION_FLAG"))
        self.assertFalse(hasattr(ExperimentPreset, "EXPERT_ATTENTION"))
        self.assertEqual(
            [preset.value for preset in ExperimentPreset],
            list(range(1, 7)),
        )
        parameters = inspect.signature(VitExpertLinearAdaptiveConfigBuilder).parameters
        self.assertNotIn("expert_attention_flag", parameters)
        self.assertIn("expert_attention_use_kv_expert_models_flag", parameters)
        for dependencies_type in (
            ExpertAdaptiveConfigDependencies,
            VitExpertAdaptiveConfigDependencies,
        ):
            with self.subTest(dependencies_type=dependencies_type.__name__):
                self.assertNotIn(
                    "expert_attention_flag",
                    {field.name for field in fields(dependencies_type)},
                )

        with self.assertRaises(TypeError):
            VitExpertLinearAdaptiveConfigBuilder(expert_attention_flag=False)
        with self.assertRaises(TypeError):
            self._config(
                ExperimentPreset.BASELINE,
                {
                    **self._runtime_overrides(),
                    "expert_attention_flag": False,
                },
            )

    def test_low_rank_expert_preset_uses_adaptive_expert_layers(self):
        cfg = self._config(ExperimentPreset.LOW_RANK_EXPERT_WEIGHT)
        feed_forward_stack_config = self._encoder_layer_config(
            cfg
        ).feed_forward_config.stack_config
        expert_core_config = (
            feed_forward_stack_config.stack_config.layer_config.layer_model_config
        )
        expert_layer_config = (
            expert_core_config.expert_model_config.layer_config.layer_model_config
        )

        self.assertIsInstance(feed_forward_stack_config, MixtureOfExpertsModelConfig)
        self.assertIsInstance(expert_layer_config, AdaptiveLinearLayerConfig)
        self.assertIsNotNone(
            expert_layer_config.adaptive_augmentation_config.weight_config
        )

    def test_ff_controls_apply_to_outer_slot_with_adaptive_experts_preserved(self):
        cfg = self._config(
            ExperimentPreset.LOW_RANK_EXPERT_WEIGHT,
            config_overrides={
                **self._test_overrides(),
                "feed_forward_stack_options": replace(
                    self._default_builder_kwargs()["feed_forward_stack_options"],
                    hidden_dim=17,
                ),
                "feed_forward_layer_controller_options": replace(
                    self._default_builder_kwargs()[
                        "feed_forward_layer_controller_options"
                    ],
                    stack_gate_flag=True,
                ),
                "expert_stack_options": replace(
                    self._default_builder_kwargs()["expert_stack_options"],
                    hidden_dim=11,
                ),
            },
        )
        feed_forward_stack_config = self._encoder_layer_config(
            cfg
        ).feed_forward_config.stack_config
        outer_stack_config = feed_forward_stack_config.stack_config
        expert_stack_config = (
            outer_stack_config.layer_config.layer_model_config.expert_model_config
        )
        expert_layer_config = expert_stack_config.layer_config.layer_model_config

        self.assertIsInstance(feed_forward_stack_config, MixtureOfExpertsModelConfig)
        self.assertEqual(outer_stack_config.hidden_dim, 17)
        self.assertIsNotNone(outer_stack_config.layer_config.gate_config)
        self.assertEqual(expert_stack_config.hidden_dim, 11)
        self.assertIsInstance(expert_layer_config, AdaptiveLinearLayerConfig)
        self.assertIsNotNone(
            expert_layer_config.adaptive_augmentation_config.weight_config
        )

    def test_attention_projection_controls_preserve_adaptive_expert_attention(self):
        cfg = self._config(
            ExperimentPreset.BASELINE,
            config_overrides={
                **self._test_overrides(),
                "hidden_adaptive_weight_options": replace(
                    self._default_builder_kwargs()["hidden_adaptive_weight_options"],
                    option_flag=True,
                    option=config.LowRankDynamicWeightConfig,
                ),
                "attention_projection_stack_options": replace(
                    self._default_builder_kwargs()[
                        "attention_projection_stack_options"
                    ],
                    hidden_dim=17,
                ),
                "attention_projection_layer_controller_options": replace(
                    self._default_builder_kwargs()[
                        "attention_projection_layer_controller_options"
                    ],
                    stack_gate_flag=True,
                ),
                "expert_stack_options": replace(
                    self._default_builder_kwargs()["expert_stack_options"],
                    hidden_dim=11,
                ),
            },
        )
        attention_config = self._encoder_layer_config(cfg).attention_config
        projection_stack_config = attention_config.projection_model_config
        expert_stack_config = attention_config.experts_config.expert_model_config
        expert_layer_config = expert_stack_config.layer_config.layer_model_config

        self.assertIsInstance(attention_config, MixtureOfAttentionHeadsConfig)
        self.assertEqual(projection_stack_config.hidden_dim, 17)
        self.assertIsNotNone(projection_stack_config.layer_config.gate_config)
        self.assertEqual(expert_stack_config.hidden_dim, 11)
        self.assertIsNone(expert_stack_config.layer_config.gate_config)
        self.assertIsInstance(expert_layer_config, AdaptiveLinearLayerConfig)
        self.assertIsNotNone(
            expert_layer_config.adaptive_augmentation_config.weight_config
        )

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
                        "expert_attention_use_kv_expert_models_flag": (use_kv_experts),
                    },
                )
                model = Model(cfg)
                images = self._fake_batch(cfg).requires_grad_()
                logits, auxiliary_loss = model(images)
                attention = next(
                    module
                    for module in model.modules()
                    if isinstance(module, MixtureOfAttentionHeads)
                )

                self.assertEqual(
                    attention.cfg.use_kv_expert_models_flag,
                    use_kv_experts,
                )
                self.assertEqual(
                    attention.cfg.add_key_value_bias_flag,
                    expect_bias,
                )
                self.assertEqual(logits.shape, (2, cfg.output_dim))
                self.assertEqual(auxiliary_loss.dim(), 0)
                self.assertTrue(torch.isfinite(auxiliary_loss))

                (logits.square().mean() + auxiliary_loss).backward()
                self.assertIsNotNone(images.grad)
                self.assertGreater(images.grad.abs().sum().item(), 0.0)

                projector = attention.projector
                if use_kv_experts:
                    for role, expert_model in {
                        "key": projector.key_model,
                        "value": projector.value_model,
                    }.items():
                        with self.subTest(name=name, role=role):
                            self.assertIsInstance(expert_model, MixtureOfExperts)
                            self._assert_nonzero_parameter_gradients(
                                expert_model,
                                role,
                            )
                else:
                    self._assert_nonzero_parameter_gradients(
                        projector.key_model,
                        "regular key projection",
                    )
                    self._assert_nonzero_parameter_gradients(
                        projector.value_model,
                        "regular value projection",
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
                attention_config = self._encoder_layer_config(cfg).attention_config
                model = Model(cfg)
                output = model(self._fake_batch(cfg))
                logits = output[0] if isinstance(output, tuple) else output
                modules = tuple(model.modules())

                self.assertIsInstance(
                    attention_config,
                    MixtureOfAttentionHeadsConfig,
                )
                self.assertTrue(
                    any(
                        isinstance(module, MixtureOfAttentionHeads)
                        for module in modules
                    )
                )
                self.assertFalse(
                    any(isinstance(module, SelfAttention) for module in modules)
                )
                self.assertEqual(logits.shape, (2, cfg.output_dim))

    def _config(self, preset: ExperimentPreset, config_overrides: dict | None = None):
        return ExperimentPresets().get_config(
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
            "encoder_options": replace(
                self._default_builder_kwargs()["encoder_options"],
                hidden_dim=16,
                num_layers=1,
                dropout_probability=0.0,
            ),
            "attention_options": replace(
                self._default_builder_kwargs()["attention_options"],
                num_heads=4,
            ),
        }

    def _default_builder_kwargs(self) -> dict:
        return {
            "feed_forward_stack_options": (
                config_defaults.linears_submodule_stack_options(
                    config,
                    "FF_STACK",
                    num_layers_key="FF_NUM_LAYERS",
                    bias_key="FF_BIAS_FLAG",
                )
            ),
            "feed_forward_layer_controller_options": (
                config_defaults.linears_layer_controller_options(
                    config,
                    gate_prefix="FF_GATE",
                    gate_stack_prefix="FF_GATE_STACK",
                    halting_prefix="FF_HALTING",
                    halting_stack_prefix="FF_HALTING_STACK",
                )
            ),
            "attention_projection_stack_options": (
                config_defaults.linears_submodule_stack_options(
                    config,
                    "ATTN_STACK",
                    num_layers_key="ATTN_NUM_LAYERS",
                    bias_key="ATTN_BIAS_FLAG",
                )
            ),
            "attention_projection_layer_controller_options": (
                config_defaults.linears_layer_controller_options(
                    config,
                    gate_prefix="ATTN_GATE",
                    gate_stack_prefix="ATTN_GATE_STACK",
                    halting_prefix="ATTN_HALTING",
                    halting_stack_prefix="ATTN_HALTING_STACK",
                )
            ),
            "expert_stack_options": config_defaults.experts_submodule_stack_options(
                config,
                "EXPERT_STACK",
                bias_key="EXPERT_BIAS_FLAG",
            ),
            "hidden_adaptive_weight_options": (
                config_defaults.hidden_adaptive_weight_options(config)
            ),
            "encoder_options": config_defaults.vit_encoder_options(config),
            "attention_options": config_defaults.vit_attention_options(config),
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
        self.assertTrue(gradients, f"{role} had no gradients")
        self.assertTrue(
            any(gradient.abs().sum().item() > 0.0 for gradient in gradients),
            f"{role} gradients were all zero",
        )


if __name__ == "__main__":
    unittest.main()
