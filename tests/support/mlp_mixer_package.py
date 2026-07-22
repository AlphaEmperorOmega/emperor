from __future__ import annotations

import io
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import torch
from lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    Layer,
    LayerNormPositionOptions,
    RecurrentLayerConfig,
    ResidualConnectionOptions,
)
from models.catalog import model_package

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class MlpMixerPackageContractMixin:
    MODEL_ID = ""
    ADAPTIVE = False
    EXPERT = False
    FIT_PRESETS: tuple[str, ...] = ()

    def _package(self):
        package = model_package(self.MODEL_ID)
        self.assertIsNotNone(package)
        return package

    def _small_overrides(self, **updates: object) -> dict[str, object]:
        values: dict[str, object] = {
            "batch_size": 2,
            "learning_rate": 0.01,
            "input_dim": 64,
            "hidden_dim": 8,
            "output_dim": 3,
            "image_patch_size": 4,
            "input_channels": 1,
            "image_height": 8,
            "stack_num_layers": 1,
            "stack_dropout_probability": 0.0,
            "token_mixer_stack_hidden_dim": 6,
            "token_mixer_num_layers": 2,
            "token_mixer_stack_dropout_probability": 0.0,
            "channel_mixer_stack_hidden_dim": 12,
            "channel_mixer_num_layers": 2,
            "channel_mixer_stack_dropout_probability": 0.0,
            "submodule_stack_hidden_dim": 4,
            "submodule_stack_num_layers": 1,
            "recurrent_max_steps": 2,
        }
        if self.ADAPTIVE:
            values.update(
                adaptive_generator_stack_hidden_dim=4,
                adaptive_generator_stack_num_layers=2,
            )
        if self.EXPERT:
            values.update(
                num_experts=3,
                router_stack_hidden_dim=4,
                router_stack_num_layers=2,
                expert_stack_hidden_dim=8,
                expert_stack_num_layers=1,
            )
        values.update(updates)
        return values

    def _configuration(self, preset=None, **updates: object):
        package = self._package()
        if preset is None:
            preset = package.preset_type.BASELINE
        return package.build_configuration(
            preset,
            config_overrides=self._small_overrides(**updates),
        )

    def _model(self, preset=None, **updates: object):
        package = self._package()
        return package.build_model(self._configuration(preset, **updates))

    @staticmethod
    def _logits_and_loss(output):
        if isinstance(output, tuple):
            return output[0], output[-1]
        return output, output.new_zeros(())

    @staticmethod
    def _block(model, index: int = 0):
        return model.transformer.layers[index].model

    @staticmethod
    def _stack_layers(stack):
        owned_stack = getattr(stack, "expert_stack", stack)
        return owned_stack.layers

    def test_catalog_metadata_runtime_defaults_search_and_cli(self) -> None:
        package = self._package()
        runtime = package.bind_runtime_defaults()

        self.assertEqual(package.catalog_key, self.MODEL_ID)
        self.assertEqual(package.identity.model_type, "mlp_mixer")
        self.assertEqual(runtime.image_height, 224)
        self.assertEqual(runtime.image_patch_size, 16)
        self.assertEqual(runtime.stack_num_layers, 8)
        self.assertEqual(runtime.hidden_dim, 32)
        self.assertEqual(runtime.token_mixer_stack_hidden_dim, 64)
        self.assertEqual(runtime.channel_mixer_stack_hidden_dim, 128)
        self.assertEqual(runtime.token_mixer_num_layers, 2)
        self.assertEqual(runtime.channel_mixer_num_layers, 2)
        self.assertIs(runtime.layer_norm_position, LayerNormPositionOptions.BEFORE)

        runtime_fields = set(package.runtime_options_type.__dataclass_fields__)
        self.assertGreaterEqual(
            runtime_fields,
            {
                "hidden_dim",
                "image_patch_size",
                "stack_num_layers",
                "token_mixer_stack_hidden_dim",
                "channel_mixer_stack_hidden_dim",
                "stack_gate_flag",
                "stack_halting_flag",
                "memory_flag",
                "recurrent_flag",
                "gate_stack_independent_flag",
                "recurrent_halting_stack_independent_flag",
                "token_mixer_stack_gate_flag",
                "token_mixer_stack_halting_flag",
                "token_mixer_memory_flag",
                "token_mixer_recurrent_flag",
                "channel_mixer_stack_gate_flag",
                "channel_mixer_stack_halting_flag",
                "channel_mixer_memory_flag",
                "channel_mixer_recurrent_flag",
            },
        )
        self.assertFalse(
            runtime_fields
            & {
                "class_token_flag",
                "positional_embedding_option",
                "num_heads",
                "attn_num_heads",
                "query_dim",
                "key_dim",
                "value_dim",
                "causal_attention_mask_flag",
                "decoder_num_layers",
            }
        )
        self.assertEqual(
            [
                dataset.__name__
                for dataset in package.dataset_metadata[package.default_experiment_task]
            ],
            ["Mnist", "FashionMNIST", "Cifar10", "Cifar100"],
        )

        monitor_names = {monitor.name for monitor in package.monitor_metadata}
        self.assertGreaterEqual(monitor_names, {"mixer", "layer-controller"})
        self.assertEqual("adaptive" in monitor_names, self.ADAPTIVE)
        self.assertEqual("weight-bank" in monitor_names, self.ADAPTIVE)
        self.assertEqual("sampler" in monitor_names, self.EXPERT)

        search_keys = set(package.search_metadata)
        self.assertGreaterEqual(
            search_keys,
            {
                "SEARCH_SPACE_LEARNING_RATE",
                "SEARCH_SPACE_HIDDEN_DIM",
                "SEARCH_SPACE_IMAGE_PATCH_SIZE",
                "SEARCH_SPACE_STACK_NUM_LAYERS",
                "SEARCH_SPACE_TOKEN_MIXER_STACK_HIDDEN_DIM",
                "SEARCH_SPACE_CHANNEL_MIXER_STACK_HIDDEN_DIM",
            },
        )
        self.assertEqual(
            "SEARCH_SPACE_ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM" in search_keys,
            self.ADAPTIVE,
        )
        self.assertEqual("SEARCH_SPACE_TOP_K" in search_keys, self.EXPERT)
        self.assertEqual("SEARCH_SPACE_NUM_EXPERTS" in search_keys, self.EXPERT)
        self.assertEqual("routing_initialization_mode" in runtime_fields, self.EXPERT)
        self.assertEqual("compute_expert_mixture_flag" in runtime_fields, self.EXPERT)
        self.assertEqual("weight_option_flag" in runtime_fields, self.ADAPTIVE)
        self.assertEqual("bias_option_flag" in runtime_fields, self.ADAPTIVE)
        self.assertEqual("diagonal_option_flag" in runtime_fields, self.ADAPTIVE)
        self.assertEqual("mask_option_flag" in runtime_fields, self.ADAPTIVE)
        self.assertEqual(
            "weight_generator_stack_independent_flag" in runtime_fields,
            self.ADAPTIVE,
        )
        self.assertEqual("expert_stack_gate_flag" in runtime_fields, self.EXPERT)
        self.assertEqual("expert_stack_halting_flag" in runtime_fields, self.EXPERT)
        self.assertEqual("expert_memory_flag" in runtime_fields, self.EXPERT)
        self.assertEqual("expert_recurrent_flag" in runtime_fields, self.EXPERT)
        self.assertEqual("router_bias_flag" in runtime_fields, self.EXPERT)

        preset_names = {preset.name for preset in package.preset_type}
        self.assertGreaterEqual(
            preset_names,
            {
                "BASELINE",
                "POST_NORM",
                "RECURRENT",
                "RECURRENT_CONTROLLER",
                "GATING",
                "HALTING",
                "MEMORY",
                "GATING_HALTING_MEMORY",
            },
        )
        self.assertEqual("ADAPTIVE" in preset_names, self.ADAPTIVE)
        self.assertEqual("TOP_1_EXPERT" in preset_names, self.EXPERT)
        self.assertEqual("EXPERT_AUXILIARY_LOSS" in preset_names, self.EXPERT)

        module_name = "models." + self.MODEL_ID.replace("/", ".")
        environment = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
        completed = subprocess.run(
            [sys.executable, "-m", module_name, "--help"],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--all-presets", completed.stdout)
        self.assertIn("image-classification", completed.stdout)

        listed = subprocess.run(
            [sys.executable, "-m", module_name, "--list-config"],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertIn("--hidden-dim", listed.stdout)
        self.assertIn("--token-mixer-stack-hidden-dim", listed.stdout)
        self.assertIn("--channel-mixer-stack-hidden-dim", listed.stdout)
        self.assertNotIn("--class-token", listed.stdout)
        self.assertNotIn("--positional", listed.stdout)
        self.assertNotIn("--num-heads", listed.stdout)

    def test_config_includes_every_applicable_reference_backend_field(self) -> None:
        backend = "linear_adaptive" if self.ADAPTIVE else "linear"
        reference_family = "experts" if self.EXPERT else "linears"
        reference_config = import_module(f"models.{reference_family}.{backend}.config")
        target_config = import_module(
            "models." + self.MODEL_ID.replace("/", ".") + ".config"
        )
        reference_fields = set(reference_config.__annotations__)
        target_fields = set(target_config.__annotations__)

        exclusions = {
            "HALTING_OUTPUT_DIM",
            "EXPERT_HALTING_OUTPUT_DIM",
        }
        if self.ADAPTIVE:
            exclusions.update(
                field
                for field in reference_fields
                if field.startswith(("INPUT_LAYER_", "OUTPUT_LAYER_"))
            )
        if self.EXPERT and self.ADAPTIVE:
            ordinary_router_fields = {
                "ROUTER_NOISY_TOPK_FLAG",
                "ROUTER_STACK_HIDDEN_DIM",
                "ROUTER_STACK_NUM_LAYERS",
                "ROUTER_STACK_ACTIVATION",
                "ROUTER_STACK_RESIDUAL_CONNECTION_OPTION",
                "ROUTER_STACK_DROPOUT_PROBABILITY",
                "ROUTER_STACK_LAYER_NORM_POSITION",
                "ROUTER_STACK_LAST_LAYER_BIAS_OPTION",
                "ROUTER_STACK_APPLY_OUTPUT_PIPELINE_FLAG",
                "ROUTER_BIAS_FLAG",
            }
            exclusions.update(
                field
                for field in reference_fields
                if field.startswith("ROUTER_") and field not in ordinary_router_fields
            )

        missing_fields = reference_fields - exclusions - target_fields
        self.assertEqual(missing_fields, set())

    def test_reference_stack_defaults_are_live(self) -> None:
        config = self._configuration(
            stack_activation=ActivationOptions.MISH,
            stack_bias_flag=False,
            submodule_stack_hidden_dim=5,
            submodule_stack_num_layers=2,
            submodule_stack_activation=ActivationOptions.TANH,
            submodule_stack_layer_norm_position=LayerNormPositionOptions.AFTER,
            submodule_stack_dropout_probability=0.1,
            submodule_stack_last_layer_bias_option=(LastLayerBiasOptions.DISABLED),
            submodule_stack_apply_output_pipeline_flag=True,
            submodule_stack_bias_flag=False,
            stack_gate_flag=True,
            token_mixer_stack_gate_flag=True,
        )
        encoder_stack = config.experiment_config.encoder_config
        transformer_config = encoder_stack.layer_config.layer_model_config
        token_model = transformer_config.attention_config.mixing_model_config
        channel_model = transformer_config.feed_forward_config.stack_config
        token_stack = getattr(token_model, "stack_config", token_model)
        channel_stack = getattr(channel_model, "stack_config", channel_model)

        self.assertIs(token_stack.layer_config.activation, ActivationOptions.MISH)
        self.assertIs(channel_stack.layer_config.activation, ActivationOptions.MISH)

        outer_gate_stack = encoder_stack.layer_config.gate_config.model_config
        self.assertEqual(outer_gate_stack.hidden_dim, 5)
        self.assertEqual(outer_gate_stack.num_layers, 2)
        self.assertIs(
            outer_gate_stack.layer_config.activation,
            ActivationOptions.TANH,
        )
        self.assertIs(
            outer_gate_stack.layer_config.layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )
        self.assertEqual(outer_gate_stack.layer_config.dropout_probability, 0.1)
        self.assertIs(
            outer_gate_stack.last_layer_bias_option,
            LastLayerBiasOptions.DISABLED,
        )
        self.assertTrue(outer_gate_stack.apply_output_pipeline_flag)
        self.assertFalse(outer_gate_stack.layer_config.layer_model_config.bias_flag)

        token_gate_stack = token_stack.layer_config.gate_config.model_config
        self.assertFalse(token_gate_stack.layer_config.layer_model_config.bias_flag)

        runtime_defaults = import_module(
            "models." + self.MODEL_ID.replace("/", ".") + ".runtime_defaults"
        )
        compatibility_runtime = runtime_defaults.runtime_from_flat(
            {"controller_stack_hidden_dim": 7}
        )
        self.assertEqual(compatibility_runtime.controller_stack_hidden_dim, 7)
        self.assertEqual(compatibility_runtime.submodule_stack_hidden_dim, 7)

        specialized_config = self._configuration(
            stack_activation=ActivationOptions.MISH,
            token_mixer_stack_activation=ActivationOptions.RELU,
        )
        specialized_transformer = specialized_config.experiment_config.encoder_config.layer_config.layer_model_config
        specialized_token_model = (
            specialized_transformer.attention_config.mixing_model_config
        )
        specialized_channel_model = (
            specialized_transformer.feed_forward_config.stack_config
        )
        specialized_token_stack = getattr(
            specialized_token_model,
            "stack_config",
            specialized_token_model,
        )
        specialized_channel_stack = getattr(
            specialized_channel_model,
            "stack_config",
            specialized_channel_model,
        )
        self.assertIs(
            specialized_token_stack.layer_config.activation,
            ActivationOptions.RELU,
        )
        self.assertIs(
            specialized_channel_stack.layer_config.activation,
            ActivationOptions.MISH,
        )

    def test_construction_has_exact_patch_token_and_channel_dimensions(self) -> None:
        config = self._configuration()
        experiment_config = config.experiment_config
        model = self._package().build_model(config).eval()
        inputs = torch.randn(2, 1, 8, 8)

        patch_tokens = model.patch(inputs)
        logits, auxiliary_loss = self._logits_and_loss(model(inputs))
        block = self._block(model)
        token_stack = block.self_attention_model.mixing_model
        channel_stack = block.feed_forward_model.model
        token_layers = self._stack_layers(token_stack)
        channel_layers = self._stack_layers(channel_stack)
        patch_projection = model.patch.embedding_model.layers[0].model

        self.assertEqual(config.sequence_length, 4)
        self.assertFalse(experiment_config.patch_config.class_token_flag)
        self.assertEqual(patch_tokens.shape, (2, 4, 8))
        self.assertEqual(logits.shape, (2, 3))
        self.assertTrue(torch.isfinite(auxiliary_loss).item())
        self.assertFalse(hasattr(model.patch, "class_token"))
        self.assertFalse(
            any(
                "class_token" in name or "position" in name
                for name in model.state_dict()
            )
        )
        self.assertEqual((token_stack.input_dim, token_stack.output_dim), (4, 4))
        self.assertEqual((channel_stack.input_dim, channel_stack.output_dim), (8, 8))
        self.assertEqual(
            [(layer.input_dim, layer.output_dim) for layer in token_layers],
            [(4, 6), (6, 4)],
        )
        self.assertEqual(
            [(layer.input_dim, layer.output_dim) for layer in channel_layers],
            [(8, 12), (12, 8)],
        )
        self.assertEqual(
            (patch_projection.input_dim, patch_projection.output_dim),
            (16, 8),
        )
        self.assertEqual((model.output.input_dim, model.output.output_dim), (8, 3))
        self.assertEqual(len(model.transformer.layers), 1)

    def test_block_matches_pre_normalized_mlp_mixer_equations(self) -> None:
        torch.manual_seed(401)
        model = self._model().eval()
        block = self._block(model)
        values = torch.randn(2, 4, 8)

        token_normalized = block.self_attention_layer.layer_norm_module(values)
        token_mixed, _, _ = block.self_attention_model(
            token_normalized,
            token_normalized,
            token_normalized,
        )
        after_token_mixing = values + token_mixed
        channel_normalized = block.feed_forward_layer.layer_norm_module(
            after_token_mixing
        )
        channel_mixed, _ = block.feed_forward_model(channel_normalized)
        expected = after_token_mixing + channel_mixed

        actual, auxiliary_loss = block(values)

        self.assertIs(
            block.self_attention_layer.layer_norm_position,
            LayerNormPositionOptions.BEFORE,
        )
        self.assertIs(
            block.feed_forward_layer.layer_norm_position,
            LayerNormPositionOptions.BEFORE,
        )
        self.assertIsNotNone(block.self_attention_layer.residual_connection)
        self.assertIsNotNone(block.feed_forward_layer.residual_connection)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.isfinite(auxiliary_loss).item())

    def test_token_and_channel_control_surfaces_are_live(self) -> None:
        recurrent_config = self._configuration(
            token_mixer_recurrent_flag=True,
            token_mixer_recurrent_max_steps=2,
            token_mixer_recurrent_stack_gate_flag=True,
            token_mixer_recurrent_gate_stack_independent_flag=True,
            token_mixer_recurrent_gate_stack_hidden_dim=5,
            channel_mixer_recurrent_flag=True,
            channel_mixer_recurrent_max_steps=2,
            channel_mixer_recurrent_stack_halting_flag=True,
            channel_mixer_recurrent_halting_stack_independent_flag=True,
            channel_mixer_recurrent_halting_stack_hidden_dim=5,
        )
        transformer_config = recurrent_config.experiment_config.encoder_config.layer_config.layer_model_config
        self.assertIsInstance(
            transformer_config.attention_config.mixing_model_config,
            RecurrentLayerConfig,
        )
        self.assertIsInstance(
            transformer_config.feed_forward_config.stack_config,
            RecurrentLayerConfig,
        )
        recurrent_model = self._package().build_model(recurrent_config)
        recurrent_output = recurrent_model(torch.randn(2, 1, 8, 8))
        recurrent_logits, recurrent_loss = self._logits_and_loss(recurrent_output)
        (recurrent_logits.square().mean() + recurrent_loss).backward()
        recurrent_module_names = {
            type(module).__name__ for module in recurrent_model.modules()
        }
        self.assertIn("RecurrentLayer", recurrent_module_names)
        self.assertIn("LayerGate", recurrent_module_names)
        self.assertIn("StickBreaking", recurrent_module_names)

        stable_model = self._model(
            hidden_dim=4,
            token_mixer_stack_hidden_dim=4,
            channel_mixer_stack_hidden_dim=4,
            token_mixer_stack_gate_flag=True,
            token_mixer_stack_halting_flag=True,
            token_mixer_memory_flag=True,
            channel_mixer_stack_gate_flag=True,
            channel_mixer_stack_halting_flag=True,
            channel_mixer_memory_flag=True,
            **(
                {
                    "top_k": 1,
                    "sampler_normalize_probabilities_flag": False,
                    "expert_stack_hidden_dim": 4,
                }
                if self.EXPERT
                else {}
            ),
        )
        stable_output = stable_model(torch.randn(2, 1, 8, 8))
        stable_logits, stable_loss = self._logits_and_loss(stable_output)
        (stable_logits.square().mean() + stable_loss).backward()
        stable_module_names = {
            type(module).__name__ for module in stable_model.modules()
        }
        self.assertGreaterEqual(
            stable_module_names,
            {"LayerGate", "StickBreaking", "GatedResidualDynamicMemory"},
        )

    def test_expert_and_router_control_surfaces_are_live(self) -> None:
        if not self.EXPERT:
            self.skipTest("expert controls only apply to expert packages")

        config = self._configuration(
            hidden_dim=4,
            token_mixer_stack_hidden_dim=4,
            channel_mixer_stack_hidden_dim=4,
            top_k=1,
            sampler_normalize_probabilities_flag=False,
            expert_stack_hidden_dim=4,
            expert_stack_num_layers=2,
            expert_stack_gate_flag=True,
            expert_stack_halting_flag=True,
            expert_memory_flag=True,
            expert_recurrent_flag=True,
            expert_recurrent_max_steps=2,
            expert_recurrent_stack_gate_flag=True,
            expert_recurrent_stack_halting_flag=True,
            router_stack_num_layers=3,
            router_stack_activation=ActivationOptions.TANH,
            router_stack_layer_norm_position=LayerNormPositionOptions.AFTER,
            router_stack_residual_connection_option=(
                ResidualConnectionOptions.RESIDUAL
            ),
            router_stack_last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            router_stack_apply_output_pipeline_flag=True,
            router_bias_flag=False,
        )
        token_mixture = config.experiment_config.encoder_config.layer_config.layer_model_config.attention_config.mixing_model_config
        mixture_config = token_mixture.stack_config.layer_config.layer_model_config
        router_stack = mixture_config.sampler_config.router_config.model_config
        expert_config = mixture_config.expert_model_config

        self.assertEqual(router_stack.num_layers, 3)
        self.assertIs(
            router_stack.layer_config.activation,
            ActivationOptions.TANH,
        )
        self.assertIs(
            router_stack.layer_config.layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )
        self.assertIsNotNone(router_stack.layer_config.residual_config)
        self.assertIs(
            router_stack.last_layer_bias_option,
            LastLayerBiasOptions.DISABLED,
        )
        self.assertTrue(router_stack.apply_output_pipeline_flag)
        self.assertFalse(router_stack.layer_config.layer_model_config.bias_flag)
        self.assertIsInstance(expert_config, RecurrentLayerConfig)
        self.assertIsNotNone(expert_config.gate_config)
        self.assertIsNotNone(expert_config.halting_config)
        self.assertIsNotNone(expert_config.block_config.layer_config.gate_config)
        self.assertIsNotNone(expert_config.block_config.layer_config.halting_config)
        self.assertIsNotNone(expert_config.block_config.layer_config.memory_config)

        model = self._package().build_model(config)
        logits, auxiliary_loss = self._logits_and_loss(model(torch.randn(2, 1, 8, 8)))
        (logits.square().mean() + auxiliary_loss).backward()
        module_names = {type(module).__name__ for module in model.modules()}
        self.assertGreaterEqual(
            module_names,
            {
                "LayerGate",
                "StickBreaking",
                "GatedResidualDynamicMemory",
                "RecurrentLayer",
            },
        )

    def test_token_channel_sharing_and_batch_isolation_are_exact(self) -> None:
        torch.manual_seed(409)
        model = self._model().eval()
        block = self._block(model)
        token_mixer = block.self_attention_model
        channel_mixer = block.feed_forward_model

        token_values = torch.randn(2, 4, 8)
        token_values[..., 1] = token_values[..., 0]
        token_output, _, _ = token_mixer(
            token_values,
            token_values,
            token_values,
        )
        torch.testing.assert_close(token_output[..., 0], token_output[..., 1])

        changed_tokens = token_values.clone()
        changed_tokens[0, :, 0] += 5.0
        changed_token_output, _, _ = token_mixer(
            changed_tokens,
            changed_tokens,
            changed_tokens,
        )
        torch.testing.assert_close(changed_token_output[1], token_output[1])
        torch.testing.assert_close(
            changed_token_output[0, :, 1:],
            token_output[0, :, 1:],
        )

        channel_values = torch.randn(2, 4, 8)
        channel_values[:, 1] = channel_values[:, 0]
        channel_output, _ = channel_mixer(channel_values)
        torch.testing.assert_close(channel_output[:, 0], channel_output[:, 1])

        changed_channels = channel_values.clone()
        changed_channels[0, 0] += 5.0
        changed_channel_output, _ = channel_mixer(changed_channels)
        torch.testing.assert_close(changed_channel_output[1], channel_output[1])
        torch.testing.assert_close(
            changed_channel_output[0, 1:],
            channel_output[0, 1:],
        )

    def test_final_normalization_mean_pooling_and_classifier_are_exact(self) -> None:
        torch.manual_seed(419)
        model = self._model().eval()
        inputs = torch.randn(2, 1, 8, 8)

        patch_tokens = model.patch(inputs)
        encoder_state = Layer.run_model_returning_state(
            model.transformer,
            patch_tokens,
        )
        normalized_tokens = model.encoder_layer_norm(encoder_state.hidden)
        expected_pool = normalized_tokens.mean(dim=1)
        expected_logits = Layer.run_model_returning_hidden(
            model.output,
            expected_pool,
        )
        actual_logits, _ = self._logits_and_loss(model(inputs))

        torch.testing.assert_close(actual_logits, expected_logits)
        self.assertEqual(expected_pool.shape, (2, 8))

    def test_gradients_independence_and_strict_checkpoint_round_trip(self) -> None:
        torch.manual_seed(421)
        config = self._configuration(stack_num_layers=2)
        model = self._package().build_model(config).eval()
        inputs = torch.randn(2, 1, 8, 8, requires_grad=True)

        logits, auxiliary_loss = self._logits_and_loss(model(inputs))
        loss = logits.square().mean() + auxiliary_loss
        loss.backward()

        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all().item())
        self.assertGreater(inputs.grad.abs().sum().item(), 0.0)
        gradients = {
            name: parameter.grad
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        for role, marker in (
            ("patch", "patch."),
            ("token", ".self_attention_layer."),
            ("channel", ".feed_forward_layer."),
            ("classifier", "output."),
        ):
            role_gradients = [
                gradient
                for name, gradient in gradients.items()
                if marker in name or name.startswith(marker)
            ]
            self.assertTrue(role_gradients, role)
            self.assertTrue(
                any(gradient.abs().sum().item() > 0.0 for gradient in role_gradients),
                role,
            )
        if self.ADAPTIVE:
            generator_gradients = [
                gradient
                for name, gradient in gradients.items()
                if "adaptive_behaviour" in name
            ]
            self.assertTrue(generator_gradients)
            self.assertTrue(
                any(
                    gradient.abs().sum().item() > 0.0
                    for gradient in generator_gradients
                )
            )
        if self.EXPERT:
            router_gradients = [
                gradient
                for name, gradient in gradients.items()
                if ".sampler.router." in name
            ]
            expert_gradients = [
                gradient
                for name, gradient in gradients.items()
                if ".expert_modules." in name
            ]
            self.assertTrue(router_gradients)
            self.assertTrue(expert_gradients)
            self.assertTrue(
                any(gradient.abs().sum().item() > 0.0 for gradient in router_gradients)
            )
            self.assertTrue(
                any(gradient.abs().sum().item() > 0.0 for gradient in expert_gradients)
            )

        first_block = self._block(model, 0)
        second_block = self._block(model, 1)
        first_parameters = {id(parameter) for parameter in first_block.parameters()}
        second_parameters = {id(parameter) for parameter in second_block.parameters()}
        token_parameters = {
            id(parameter) for parameter in first_block.self_attention_model.parameters()
        }
        channel_parameters = {
            id(parameter) for parameter in first_block.feed_forward_model.parameters()
        }
        self.assertTrue(first_parameters)
        self.assertTrue(second_parameters)
        self.assertTrue(token_parameters)
        self.assertTrue(channel_parameters)
        self.assertFalse(first_parameters & second_parameters)
        self.assertFalse(token_parameters & channel_parameters)

        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = self._package().build_model(config).eval()
        incompatible = restored.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True),
            strict=True,
        )
        restored_logits, restored_auxiliary_loss = self._logits_and_loss(
            restored(inputs.detach())
        )
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        torch.testing.assert_close(restored_logits, logits.detach())
        torch.testing.assert_close(restored_auxiliary_loss, auxiliary_loss.detach())

    def test_every_preset_runs_reduced_forward_and_backward(self) -> None:
        package = self._package()
        for preset in package.preset_type:
            with self.subTest(preset=preset.name):
                torch.manual_seed(431)
                model = self._model(preset, stack_num_layers=2)
                inputs = torch.randn(2, 1, 8, 8, requires_grad=True)

                logits, auxiliary_loss = self._logits_and_loss(model(inputs))
                loss = logits.square().mean() + auxiliary_loss
                loss.backward()

                self.assertEqual(logits.shape, (2, 3))
                self.assertTrue(torch.isfinite(loss.detach()).item())
                self.assertIsNotNone(inputs.grad)
                self.assertTrue(torch.isfinite(inputs.grad).all().item())
                self.assertTrue(
                    any(
                        parameter.grad is not None
                        and torch.isfinite(parameter.grad).all().item()
                        for parameter in model.parameters()
                    )
                )

    def test_representative_presets_fit_with_the_cpu_trainer(self) -> None:
        package = self._package()
        for preset_name in self.FIT_PRESETS:
            with self.subTest(preset=preset_name):
                torch.manual_seed(439)
                preset = package.resolve_preset(preset_name)
                model = self._model(preset)
                images = torch.randn(4, 1, 8, 8)
                labels = torch.tensor([0, 1, 2, 1])
                loader = DataLoader(
                    TensorDataset(images, labels),
                    batch_size=2,
                    shuffle=False,
                )
                before = {
                    name: parameter.detach().clone()
                    for name, parameter in model.named_parameters()
                }
                trainer = Trainer(
                    max_epochs=1,
                    accelerator="cpu",
                    devices=1,
                    limit_train_batches=1,
                    limit_val_batches=1,
                    num_sanity_val_steps=0,
                    logger=False,
                    enable_checkpointing=False,
                    enable_progress_bar=False,
                    enable_model_summary=False,
                )

                trainer.fit(
                    model,
                    train_dataloaders=loader,
                    val_dataloaders=loader,
                )

                self.assertEqual(trainer.global_step, 1)
                self.assertTrue(
                    any(
                        not torch.equal(parameter.detach(), before[name])
                        for name, parameter in model.named_parameters()
                    )
                )
