"""Non-training regression coverage for GPT's hierarchical embedding boundary."""

import io
import unittest
from dataclasses import fields
from types import SimpleNamespace

import torch
from lightning import LightningDataModule, Trainer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from emperor.attention import MixtureOfAttentionHeadsConfig, SelfAttentionConfig
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    CombinedDynamicDiagonalConfig,
    DualModelDynamicWeightConfig,
    GeneratorDynamicBiasConfig,
    WeightDecayScheduleOptions,
)
from emperor.config import ConfigBase
from emperor.embedding.hierarchical import HierarchicalByteEmbeddingConfig
from emperor.experts import MixtureOfExpertsConfig
from emperor.layers import LayerStackConfig
from model_runtime.inspection import configuration_schema
from models.catalog import model_package
from models.cli_selection import resolve_cli_selection
from models.experiment_cli_parser import get_experiment_parser
from models.gpt.expert_linear_adaptive.config_builder import (
    GptExpertLinearAdaptiveConfigBuilder,
)
from models.gpt.expert_linear_adaptive.model import Model
from models.gpt.expert_linear_adaptive.presets import ExperimentPreset

TOKENS = (
    "<unk>",
    "apple",
    "bravo",
    "cider",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
    "mango",
    "novel",
    "omega",
)


def configuration(**overrides):
    runtime = model_package("gpt/expert_linear_adaptive").bind_runtime_defaults(
        {
            "input_dim": len(TOKENS),
            "output_dim": len(TOKENS),
            "sequence_length": 6,
            "batch_size": 2,
            "hidden_dim": 8,
            "attn_num_heads": 2,
            "num_experts": 3,
            "top_k": 2,
            "capacity_factor": 0.0,
            "expert_stack_hidden_dim": 8,
            "ff_stack_hidden_dim": 8,
            "router_stack_hidden_dim": 8,
            "adaptive_generator_stack_hidden_dim": 8,
            "stack_dropout_probability": 0.0,
            "hierarchical_embedding_flag": True,
            "hierarchical_embedding_max_token_bytes": 8,
            "lm_head_weight_tying_flag": False,
            "bias_option_flag": True,
            "bias_option": GeneratorDynamicBiasConfig,
            **overrides,
        }
    )
    return GptExpertLinearAdaptiveConfigBuilder(runtime=runtime).build()


def model(**overrides):
    result = Model(configuration(**overrides))
    result.set_token_vocabulary(TOKENS)
    return result


def nested_configs(config):
    if isinstance(config, ConfigBase):
        yield config
        for config_field in fields(config):
            yield from nested_configs(getattr(config, config_field.name))


def encoder_layer(embedding):
    return embedding.encoder_config.encoder_stack_config.layer_config.layer_model_config


class GptHierarchicalEmbeddingTests(unittest.TestCase):
    def test_cli_and_inspection_expose_hierarchical_options(self):
        package = model_package("gpt/expert_linear_adaptive")
        parser = get_experiment_parser(package)
        args = parser.parse_args(
            [
                "--preset",
                "baseline",
                "--config",
                "--hierarchical-embedding-flag",
                "true",
                "--lm-head-weight-tying-flag",
                "false",
                "--hierarchical-embedding-max-token-bytes",
                "12",
                "--hierarchical-embedding-encoder-num-layers",
                "2",
            ]
        )
        selection = resolve_cli_selection(args, package, ExperimentPreset)
        overrides = selection.config_overrides
        self.assertTrue(overrides["hierarchical_embedding_flag"])
        self.assertEqual(overrides["hierarchical_embedding_max_token_bytes"], 12)
        self.assertEqual(overrides["hierarchical_embedding_encoder_num_layers"], 2)
        schema_keys = {field.key for field in configuration_schema(package).fields}
        self.assertIn("HIERARCHICAL_EMBEDDING_FLAG", schema_keys)
        self.assertIn("HIERARCHICAL_EMBEDDING_ENCODER_NUM_LAYERS", schema_keys)
        embedding = configuration(
            hierarchical_embedding_max_token_bytes=12,
            hierarchical_embedding_encoder_num_layers=2,
        ).experiment_config.hierarchical_embedding_config
        self.assertEqual(embedding.max_token_bytes, 12)
        self.assertEqual(embedding.byte_position_config.num_embeddings, 13)
        self.assertEqual(embedding.encoder_config.encoder_stack_config.num_layers, 2)
        attention = encoder_layer(embedding).attention_config
        self.assertEqual(attention.source_sequence_length, 13)
        self.assertEqual(attention.target_sequence_length, 13)

    def test_lookup_default_has_no_hierarchical_embedding(self):
        config = configuration(
            hierarchical_embedding_flag=False, lm_head_weight_tying_flag=True
        )
        result = Model(config)
        self.assertIsNone(config.experiment_config.hierarchical_embedding_config)
        self.assertIsInstance(result.token_embedding, nn.Embedding)
        self.assertIsNone(result.token_text_adapter)

    def test_encoder_reuses_adaptive_layers_without_experts_or_decay(self):
        config = configuration(
            weight_option_flag=True,
            weight_option=DualModelDynamicWeightConfig,
            weight_decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
            weight_decay_rate=1e-4,
            diagonal_option_flag=True,
            diagonal_option=CombinedDynamicDiagonalConfig,
        )
        embedding = config.experiment_config.hierarchical_embedding_config
        self.assertIsInstance(embedding, HierarchicalByteEmbeddingConfig)
        layer = encoder_layer(embedding)
        attention = layer.attention_config
        self.assertIs(type(attention), SelfAttentionConfig)
        self.assertFalse(attention.causal_attention_mask_flag)
        self.assertTrue(attention.batch_first_flag)
        self.assertEqual(attention.batch_size, 2 * 6)
        self.assertIsInstance(layer.feed_forward_config.stack_config, LayerStackConfig)
        for stack in [
            attention.projection_model_config,
            layer.feed_forward_config.stack_config,
            embedding.projection_config,
        ]:
            adaptive = stack.layer_config.layer_model_config
            self.assertIsInstance(adaptive, AdaptiveLinearLayerConfig)
            augmentation = adaptive.adaptive_augmentation_config
            self.assertIsInstance(
                augmentation.weight_config, DualModelDynamicWeightConfig
            )
            self.assertIsInstance(augmentation.bias_config, GeneratorDynamicBiasConfig)
            self.assertIsInstance(
                augmentation.diagonal_config, CombinedDynamicDiagonalConfig
            )
        children = list(nested_configs(embedding))
        self.assertFalse(
            any(isinstance(child, MixtureOfAttentionHeadsConfig) for child in children)
        )
        self.assertFalse(
            any(isinstance(child, MixtureOfExpertsConfig) for child in children)
        )
        self.assertTrue(
            all(getattr(child, "grouping_config", None) is None for child in children)
        )
        schedules = [
            child.decay_schedule for child in children if hasattr(child, "decay_schedule")
        ]
        self.assertTrue(schedules)
        self.assertTrue(
            all(schedule == WeightDecayScheduleOptions.DISABLED for schedule in schedules)
        )
        # The transformer keeps its own experts and decay schedule.
        transformer_children = list(nested_configs(config.experiment_config.decoder_config))
        self.assertTrue(
            any(
                isinstance(child, MixtureOfAttentionHeadsConfig)
                for child in transformer_children
            )
        )
        self.assertIn(
            WeightDecayScheduleOptions.EXPONENTIAL,
            [getattr(child, "decay_schedule", None) for child in transformer_children],
        )
        built = Model(config).token_embedding
        module_types = {type(module).__name__ for module in built.modules()}
        self.assertIn("SelfAttention", module_types)
        self.assertIn("AdaptiveLinearLayer", module_types)
        self.assertNotIn("MixtureOfAttentionHeads", module_types)
        self.assertNotIn("MixtureOfExperts", module_types)

    def test_incompatible_head_and_embedding_configs_fail_clearly(self):
        for overrides, error, pattern in [
            ({"lm_head_weight_tying_flag": True}, ValueError, "weight_tying_flag=False"),
            ({"contextual_embedding_flag": True}, ValueError, "cannot both be True"),
            (
                {"hierarchical_embedding_max_token_bytes": 0},
                ValueError,
                "max_token_bytes",
            ),
            (
                {"hierarchical_embedding_encoder_num_layers": 0},
                ValueError,
                "encoder_num_layers",
            ),
        ]:
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(error, pattern),
            ):
                configuration(**overrides)
        config = configuration()
        config.experiment_config.hierarchical_embedding_config = None
        with self.assertRaisesRegex(TypeError, "HierarchicalByteEmbeddingConfig"):
            Model(config)
        config = configuration()
        config.experiment_config.hierarchical_embedding_config.output_dim = 99
        with self.assertRaisesRegex(ValueError, "hidden_dim"):
            Model(config)
        config = configuration(hierarchical_embedding_flag=False)
        config.experiment_config.hierarchical_embedding_config = (
            configuration().experiment_config.hierarchical_embedding_config
        )
        with self.assertRaisesRegex(ValueError, "hierarchical_embedding_flag=True"):
            Model(config)

    def test_vocabulary_longer_than_byte_limit_is_rejected_with_the_needed_limit(self):
        result = Model(configuration(hierarchical_embedding_max_token_bytes=5))
        with self.assertRaisesRegex(ValueError, "'foxtrot' has 7 UTF-8 bytes.*at least 7"):
            result.set_token_vocabulary(TOKENS)

    def test_sentence_positions_masking_and_losses_reach_the_decoder(self):
        result = model()
        result.eval()
        self.assertNotIsInstance(result.positional_embedding, nn.Identity)
        captured = {}
        result.token_embedding.register_forward_hook(
            lambda _module, _args, output: captured.update(embedding=output)
        )

        class Decoder(nn.Module):
            def forward(self, state):
                captured["decoder_input"] = state.hidden.clone()
                state.loss = state.hidden.new_tensor(2.0)
                return state

        result.transformer = Decoder()
        ids = torch.tensor([[1, 1, 1, 4], [5, 6, 7, 8]])
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
        logits, loss = result(ids, mask)
        embedding = captured["embedding"]
        expected = (embedding.hidden + result.positional_embedding(ids)) * mask.unsqueeze(
            -1
        )
        torch.testing.assert_close(captured["decoder_input"], expected)
        torch.testing.assert_close(
            captured["decoder_input"][0, 3], torch.zeros_like(expected[0, 3])
        )
        # The same token at different sentence positions reaches the decoder differently.
        torch.testing.assert_close(embedding.hidden[0, 0], embedding.hidden[0, 1])
        self.assertFalse(
            torch.allclose(captured["decoder_input"][0, 0], captured["decoder_input"][0, 1])
        )
        torch.testing.assert_close(loss, embedding.loss + 2)
        self.assertEqual(tuple(logits.shape), (2, 4, len(TOKENS)))

    def test_batch_isolation_causality_and_generation(self):
        result = model()
        result.eval()
        ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        logits, _ = result(ids)
        alone, _ = result(ids[:1])
        torch.testing.assert_close(logits[:1], alone, rtol=1e-5, atol=1e-6)
        changed = ids.clone()
        changed[:, 2:] = torch.tensor([[9, 10], [11, 12]])
        altered, _ = result(changed)
        torch.testing.assert_close(logits[:, :2], altered[:, :2], rtol=1e-5, atol=1e-6)
        result.train()
        generated = result.generate(ids[:, :2], max_new_tokens=2)
        self.assertTrue(result.training)
        self.assertEqual(tuple(generated.shape), (2, 4))
        torch.testing.assert_close(generated[:, :2], ids[:, :2])

    def test_full_batch_of_equal_length_tokens_fits_the_encoder_bound(self):
        result = model()
        result.eval()
        four_byte_ids = [TOKENS.index(token) for token in ("echo", "golf", "kilo", "lima")]
        ids = torch.tensor(four_byte_ids + four_byte_ids[:2]).repeat(2, 1)
        logits, loss = result(ids)
        self.assertEqual(tuple(logits.shape), (2, 6, len(TOKENS)))
        self.assertTrue(torch.isfinite(loss))

    def test_task_loss_backpropagates_through_the_byte_encoder(self):
        result = model()
        ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        output = result._model_step_outputs((ids, torch.roll(ids, -1, dims=1)))
        torch.testing.assert_close(
            output.total_loss, output.cross_entropy + output.auxiliary_loss
        )
        output.total_loss.backward()
        for module in [
            result.token_embedding.byte_embedding,
            result.token_embedding.byte_position,
            result.token_embedding.encoder,
            result.token_embedding.projection,
            result.positional_embedding,
            result.lm_head,
        ]:
            gradients = [
                parameter.grad
                for parameter in module.parameters()
                if parameter.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(
                all(torch.isfinite(gradient).all() for gradient in gradients)
            )
            self.assertGreater(
                sum(gradient.abs().sum().item() for gradient in gradients), 0
            )

    def test_checkpoint_roundtrip_restores_vocabulary_and_outputs(self):
        result = model()
        result.eval()
        ids = torch.tensor([[1, 2, 3]])
        expected = result(ids)[0]
        buffer = io.BytesIO()
        torch.save(result.state_dict(), buffer)
        buffer.seek(0)
        restored = Model(configuration())
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        restored.eval()
        self.assertEqual(restored.token_text_adapter.token_texts, TOKENS)
        torch.testing.assert_close(restored(ids)[0], expected)

    def test_trainer_setup_binds_vocabulary_and_checks_the_byte_limit(self):
        class Data(LightningDataModule):
            def setup(self, stage):
                self.vocab = SimpleNamespace(get_itos=lambda: list(TOKENS))

            def val_dataloader(self):
                ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
                return DataLoader(
                    TensorDataset(ids, torch.roll(ids, -1, dims=1)), batch_size=2
                )

        def trainer():
            return Trainer(
                accelerator="cpu",
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                limit_val_batches=1,
            )

        result = Model(configuration())
        trainer().validate(result, datamodule=Data(), verbose=False)
        self.assertEqual(result.token_text_adapter.token_texts, TOKENS)
        too_short = Model(configuration(hierarchical_embedding_max_token_bytes=5))
        with self.assertRaisesRegex(ValueError, "at least 7"):
            trainer().validate(too_short, datamodule=Data(), verbose=False)


if __name__ == "__main__":
    unittest.main()
