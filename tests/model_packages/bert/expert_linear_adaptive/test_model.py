import importlib
import inspect
import unittest

import torch

import models.bert.expert_linear_adaptive.config as config
import models.bert.expert_linear_adaptive.dataset_options as dataset_options
from emperor.attention import (
    MixtureOfAttentionHeadsConfig,
    SelfAttentionConfig,
)
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    CombinedDynamicDiagonalConfig,
    DynamicDepthOptions,
    GeneratorDynamicBiasConfig,
    SingleModelDynamicWeightConfig,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.experts import MixtureOfExpertsConfig, MixtureOfExpertsModelConfig
from emperor.halting import StickBreakingConfig
from emperor.layers import (
    ActivationOptions,
    AttentionResidualConfig,
    LayerNormPositionOptions,
    RecurrentLayerConfig,
)
from models.bert.expert_linear_adaptive.config_builder import (
    BertExpertLinearAdaptiveConfigBuilder,
)
from models.bert.expert_linear_adaptive.model import Model
from models.bert.expert_linear_adaptive.presets import (
    Experiment,
    ExperimentPreset,
    ExperimentPresets,
)
from models.catalog import model_package
from models.training_test_utils import (
    RandomBertPretrainingDataModule,
    tiny_cpu_trainer,
)

_MIXTURE_ATTENTION_TYPE = MixtureOfAttentionHeadsConfig().registry_owner()
_MIXTURE_OF_EXPERTS_TYPE = MixtureOfExpertsModelConfig().registry_owner()
_MIXTURE_OF_EXPERTS_LAYER_TYPE = MixtureOfExpertsConfig().registry_owner()
_SELF_ATTENTION_TYPE = SelfAttentionConfig().registry_owner()


class TestBertExpertLinearAdaptiveModel(unittest.TestCase):
    def test_runtime_defaults_describe_a_conventional_bert_block(self):
        self.assertEqual(
            config.LAYER_NORM_POSITION,
            LayerNormPositionOptions.AFTER,
        )
        self.assertEqual(config.FF_NUM_LAYERS, 1)
        self.assertEqual(config.ATTN_STACK_ACTIVATION, ActivationOptions.DISABLED)
        self.assertFalse(config.ATTN_STACK_APPLY_OUTPUT_PIPELINE_FLAG)
        self.assertEqual(
            config.FF_STACK_LAYER_NORM_POSITION,
            LayerNormPositionOptions.DISABLED,
        )
        self.assertFalse(config.FF_STACK_APPLY_OUTPUT_PIPELINE_FLAG)

    def test_post_normalized_profile_has_no_extra_final_encoder_norm(self):
        model = Model(self._config(ExperimentPreset.BASELINE))

        self.assertFalse(
            any(name.startswith("encoder_layer_norm.") for name in model.state_dict())
        )

    def test_default_encoder_uses_future_context(self):
        torch.manual_seed(17)
        model = Model(self._config(ExperimentPreset.BASELINE)).eval()
        original_ids = torch.tensor([[2, 3, 4, 5]])
        changed_ids = torch.tensor([[2, 3, 9, 5]])
        attention_mask = torch.ones_like(original_ids)
        token_type_ids = torch.zeros_like(original_ids)

        with torch.no_grad():
            original_mlm, _, _ = model(
                original_ids,
                attention_mask,
                token_type_ids,
            )
            changed_mlm, _, _ = model(
                changed_ids,
                attention_mask,
                token_type_ids,
            )

        self.assertGreater(
            torch.max(torch.abs(changed_mlm[:, 0] - original_mlm[:, 0])).item(),
            1e-6,
        )

    def test_padding_token_values_do_not_change_visible_outputs(self):
        torch.manual_seed(19)
        model = Model(self._config(ExperimentPreset.BASELINE)).eval()
        original_ids = torch.tensor([[2, 3, 4, 5]])
        changed_ids = torch.tensor([[2, 3, 11, 12]])
        attention_mask = torch.tensor([[1, 1, 0, 0]])
        token_type_ids = torch.zeros_like(original_ids)

        with torch.no_grad():
            original_mlm, original_nsp, _ = model(
                original_ids,
                attention_mask,
                token_type_ids,
            )
            changed_mlm, changed_nsp, _ = model(
                changed_ids,
                attention_mask,
                token_type_ids,
            )

        torch.testing.assert_close(
            changed_mlm[:, :2],
            original_mlm[:, :2],
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            changed_nsp,
            original_nsp,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_public_surface_and_catalog_id(self):
        for module_name in (
            "models.bert.expert_linear_adaptive.config",
            "models.bert.expert_linear_adaptive.presets",
            "models.bert.expert_linear_adaptive.model",
            "models.bert.expert_linear_adaptive.config_builder",
            "models.bert.expert_linear_adaptive.experiment_config",
        ):
            with self.subTest(module_name=module_name):
                module = importlib.import_module(module_name)

                self.assertEqual(module.__name__, module_name)

        experiment = Experiment(
            model_package=model_package("bert/expert_linear_adaptive")
        )
        self.assertEqual(
            experiment.model_package.identity.catalog_key,
            "bert/expert_linear_adaptive",
        )
        self.assertIsNotNone(model_package("bert/expert_linear_adaptive"))

    def test_attention_mode_switch_is_removed(self):
        self.assertFalse(hasattr(config, "EXPERT_ATTENTION_FLAG"))
        self.assertFalse(hasattr(ExperimentPreset, "EXPERT_ATTENTION"))
        self.assertEqual(
            [preset.value for preset in ExperimentPreset],
            list(range(1, 25)),
        )
        parameters = inspect.signature(BertExpertLinearAdaptiveConfigBuilder).parameters
        self.assertNotIn("expert_attention_flag", parameters)
        self.assertEqual(tuple(parameters), ("runtime",))
        package = model_package("bert/expert_linear_adaptive")
        package.bind_runtime_defaults(
            {"expert_attention_use_kv_expert_models_flag": False}
        )

        with self.assertRaises(TypeError):
            BertExpertLinearAdaptiveConfigBuilder(expert_attention_flag=False)
        with self.assertRaisesRegex(ValueError, "expert_attention_flag"):
            self._config(
                ExperimentPreset.BASELINE,
                {"expert_attention_flag": False},
            )

    def test_feed_forward_expert_internals_are_adaptive(self):
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

    def test_single_layer_recurrent_attention_residual_preset(self):
        cfg = self._config(ExperimentPreset.SINGLE_LAYER_RECURRENT_ATTENTION_RESIDUAL)

        recurrent_config = cfg.experiment_config.encoder_config
        self.assertIsInstance(recurrent_config, RecurrentLayerConfig)
        self.assertEqual(recurrent_config.input_dim, 32)
        self.assertEqual(cfg.sequence_length, 35)
        self.assertEqual(recurrent_config.max_steps, 10)
        self.assertEqual(recurrent_config.block_config.num_layers, 1)
        self.assertEqual(recurrent_config.halting_config.threshold, 0.99)
        self.assertEqual(
            recurrent_config.halting_config.hidden_state_mode,
            config.HaltingHiddenStateModeOptions.RAW,
        )
        self.assertIsInstance(
            recurrent_config.residual_config,
            AttentionResidualConfig,
        )
        self.assertIsInstance(recurrent_config.halting_config, StickBreakingConfig)

        encoder_layer = recurrent_config.block_config.layer_config.layer_model_config
        feed_forward_mixture = encoder_layer.feed_forward_config.stack_config
        expert_core = feed_forward_mixture.stack_config.layer_config.layer_model_config
        expert_layer = expert_core.expert_model_config.layer_config.layer_model_config
        adaptive_config = expert_layer.adaptive_augmentation_config

        self.assertEqual(encoder_layer.dropout_probability, 0.0)
        self.assertEqual(encoder_layer.attention_config.num_heads, 4)
        self.assertFalse(encoder_layer.attention_config.causal_attention_mask_flag)
        self.assertFalse(encoder_layer.attention_config.use_kv_expert_models_flag)
        self.assertEqual(
            encoder_layer.attention_config.experts_config.capacity_factor,
            0.0,
        )
        self.assertEqual(feed_forward_mixture.stack_config.hidden_dim, 32)
        self.assertEqual(expert_core.num_experts, 12)
        self.assertEqual(expert_core.top_k, 2)
        self.assertEqual(expert_core.capacity_factor, 0.0)
        self.assertEqual(expert_core.sampler_config.switch_loss_weight, 0.01)
        self.assertEqual(expert_core.sampler_config.zero_centred_loss_weight, 0.001)
        self.assertEqual(encoder_layer.attention_config.experts_config.num_experts, 12)
        self.assertEqual(encoder_layer.attention_config.experts_config.top_k, 2)
        self.assertTrue(expert_layer.bias_flag)
        self.assertIsInstance(
            adaptive_config.weight_config,
            SingleModelDynamicWeightConfig,
        )
        self.assertEqual(
            adaptive_config.weight_config.generator_depth,
            DynamicDepthOptions.DEPTH_OF_FIVE,
        )
        self.assertEqual(
            adaptive_config.weight_config.normalization_option,
            WeightNormalizationOptions.L2_SCALE,
        )
        self.assertEqual(
            adaptive_config.weight_config.normalization_position_option,
            WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT,
        )
        self.assertEqual(
            adaptive_config.weight_config.decay_schedule,
            WeightDecayScheduleOptions.EXPONENTIAL,
        )
        self.assertEqual(adaptive_config.weight_config.decay_rate, 1e-4)
        self.assertEqual(adaptive_config.weight_config.decay_warmup_batches, 5_000)
        self.assertIsInstance(
            adaptive_config.bias_config,
            GeneratorDynamicBiasConfig,
        )
        self.assertIsInstance(
            adaptive_config.diagonal_config,
            CombinedDynamicDiagonalConfig,
        )
        self.assertIsNone(adaptive_config.mask_config)

        model = Model(cfg).eval()
        self.assertIsInstance(
            model.transformer.residual_connection,
            recurrent_config.residual_config.registry_owner(),
        )
        self.assertIsInstance(
            model.transformer.halting_model,
            recurrent_config.halting_config.registry_owner(),
        )
        mlm_logits, nsp_logits, auxiliary_loss = model(*self._fake_bert_inputs(cfg))
        self.assertEqual(
            tuple(mlm_logits.shape),
            (2, cfg.sequence_length, cfg.output_dim),
        )
        self.assertEqual(tuple(nsp_logits.shape), (2, 2))
        self.assertEqual(tuple(auxiliary_loss.shape), ())

    def test_single_layer_recurrent_attention_residual_preset_trains_one_batch(self):
        torch.manual_seed(449)
        cfg = self._config(ExperimentPreset.SINGLE_LAYER_RECURRENT_ATTENTION_RESIDUAL)
        model = Model(cfg)
        parameters_before_training = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        trainer = tiny_cpu_trainer()

        trainer.fit(
            model,
            datamodule=RandomBertPretrainingDataModule(
                cfg,
                batch_size=2,
                num_batches=1,
            ),
        )

        self.assertEqual(trainer.global_step, 1)
        self.assertTrue(torch.isfinite(trainer.callback_metrics["train/loss"]).item())
        self.assertTrue(
            any(
                not torch.equal(parameter.detach(), parameters_before_training[name])
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            )
        )
        self.assertTrue(
            all(
                torch.isfinite(parameter).all().item()
                for parameter in model.parameters()
            )
        )

    def test_expert_attention_uses_adaptive_expert_internals(self):
        cfg = self._config(ExperimentPreset.BASELINE)
        attention_config = self._encoder_layer_config(cfg).attention_config
        attention_expert_stack_config = (
            attention_config.experts_config.expert_model_config
        )
        attention_expert_layer_config = (
            attention_expert_stack_config.layer_config.layer_model_config
        )

        self.assertIsInstance(attention_config, MixtureOfAttentionHeadsConfig)
        self.assertIsInstance(attention_expert_layer_config, AdaptiveLinearLayerConfig)

    def test_all_presets_forward_one_batch(self):
        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = self._config(preset)
                attention_config = self._encoder_layer_config(cfg).attention_config
                model = Model(cfg)
                mlm_logits, nsp_logits, auxiliary_loss = model(
                    *self._fake_bert_inputs(cfg)
                )
                modules = tuple(model.modules())

                self.assertIsInstance(
                    attention_config,
                    MixtureOfAttentionHeadsConfig,
                )
                self.assertTrue(
                    any(
                        isinstance(module, _MIXTURE_ATTENTION_TYPE)
                        for module in modules
                    )
                )
                self.assertFalse(
                    any(isinstance(module, _SELF_ATTENTION_TYPE) for module in modules)
                )
                self.assertEqual(
                    mlm_logits.shape,
                    (2, cfg.sequence_length, cfg.output_dim),
                )
                self.assertEqual(nsp_logits.shape, (2, 2))
                self.assertEqual(auxiliary_loss.dim(), 0)
                self.assertTrue(torch.isfinite(auxiliary_loss))

    def test_attention_kv_modes_forward_backward(self):
        for use_kv_experts in (False, True):
            with self.subTest(use_kv_experts=use_kv_experts):
                torch.manual_seed(0)
                cfg = self._config(
                    ExperimentPreset.BASELINE,
                    {"expert_attention_use_kv_expert_models_flag": (use_kv_experts)},
                )
                model = Model(cfg)
                mlm_logits, nsp_logits, auxiliary_loss = model(
                    *self._fake_bert_inputs(cfg)
                )
                attention = next(
                    module
                    for module in model.modules()
                    if isinstance(module, _MIXTURE_ATTENTION_TYPE)
                )

                self.assertEqual(
                    attention.cfg.use_kv_expert_models_flag,
                    use_kv_experts,
                )
                self.assertEqual(
                    mlm_logits.shape,
                    (2, cfg.sequence_length, cfg.output_dim),
                )
                self.assertEqual(nsp_logits.shape, (2, 2))
                self.assertEqual(auxiliary_loss.dim(), 0)
                self.assertTrue(torch.isfinite(auxiliary_loss))

                loss = (
                    mlm_logits.square().mean()
                    + nsp_logits.square().mean()
                    + auxiliary_loss
                )
                loss.backward()
                self.assertIsNotNone(model.token_embedding.weight.grad)
                self.assertGreater(
                    model.token_embedding.weight.grad.abs().sum().item(),
                    0.0,
                )

                projector = attention.projector
                if use_kv_experts:
                    for role, expert_model in {
                        "key": projector.key_model,
                        "value": projector.value_model,
                    }.items():
                        with self.subTest(
                            use_kv_experts=use_kv_experts,
                            role=role,
                        ):
                            self.assertIsInstance(
                                expert_model, _MIXTURE_OF_EXPERTS_LAYER_TYPE
                            )
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

    def test_causal_mask_is_propagated_and_blocks_future_tokens(self):
        cfg = self._config(ExperimentPreset.CAUSAL)
        encoder_layer_config = self._encoder_layer_config(cfg)

        self.assertTrue(
            encoder_layer_config.attention_config.causal_attention_mask_flag
        )

        model = Model(cfg).eval()
        input_ids, attention_mask, token_type_ids = self._fake_bert_inputs(cfg)
        changed = input_ids.clone()
        changed[:, -2] = (changed[:, -2] + 17) % cfg.input_dim
        with torch.no_grad():
            original_mlm, original_nsp, _ = model(
                input_ids,
                attention_mask,
                token_type_ids,
            )
            changed_mlm, changed_nsp, _ = model(
                changed,
                attention_mask,
                token_type_ids,
            )

        torch.testing.assert_close(original_mlm[:, 0], changed_mlm[:, 0])
        torch.testing.assert_close(original_nsp, changed_nsp)

    def test_representative_presets_train_one_tiny_epoch(self):
        for preset in (
            ExperimentPreset.BASELINE,
            ExperimentPreset.LOW_RANK_EXPERT_WEIGHT,
        ):
            with self.subTest(preset=preset.name):
                cfg = self._config(preset)
                model = Model(cfg)
                datamodule = RandomBertPretrainingDataModule(cfg, batch_size=2)

                tiny_cpu_trainer().fit(model, datamodule=datamodule)

    def _config(
        self,
        preset: ExperimentPreset,
        config_overrides: dict | None = None,
    ):
        test_overrides = self._test_overrides()
        if preset is ExperimentPreset.SINGLE_LAYER_RECURRENT_ATTENTION_RESIDUAL:
            for preset_field in (
                "hidden_dim",
                "sequence_length",
                "stack_num_layers",
                "attn_num_heads",
                "stack_dropout_probability",
                "num_experts",
                "recurrent_max_steps",
            ):
                test_overrides.pop(preset_field, None)
            test_overrides.update({"input_dim": 32, "output_dim": 32})
        return ExperimentPresets().get_config(
            preset,
            dataset_options.DATASET_OPTIONS_BY_TASK[
                dataset_options.DEFAULT_EXPERIMENT_TASK
            ][0],
            config_overrides={
                **test_overrides,
                **(config_overrides or {}),
            },
        )[0]

    def _test_overrides(self) -> dict:
        return {
            "batch_size": 2,
            "hidden_dim": 16,
            "sequence_length": 8,
            "stack_num_layers": 2,
            "attn_num_heads": 4,
            "stack_dropout_probability": 0.0,
            "recurrent_max_steps": 2,
        }

    def _fake_bert_inputs(self, cfg):
        input_ids = torch.randint(5, cfg.input_dim, (2, cfg.sequence_length))
        input_ids[:, 0] = 2
        input_ids[:, -1] = 0
        attention_mask = (input_ids != 0).long()
        token_type_ids = torch.zeros_like(input_ids)
        token_type_ids[:, cfg.sequence_length // 2 : -1] = 1
        return input_ids, attention_mask, token_type_ids

    def _encoder_layer_config(self, cfg):
        encoder_config = cfg.experiment_config.encoder_config
        if hasattr(encoder_config, "block_config"):
            encoder_config = encoder_config.block_config
        return encoder_config.layer_config.layer_model_config

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
