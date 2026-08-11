import unittest

from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.layers import ActivationOptions
from model_runtime.inspection import configuration_schema
from model_runtime.packages.configuration import iter_supported_config_keys
from models.catalog import model_package
from models.transformer.expert_linear._building import (
    build_experiment_config as build_expert_linear_experiment_config,
)
from models.transformer.expert_linear_adaptive._building import (
    build_experiment_config as build_expert_linear_adaptive_experiment_config,
)

_TRANSFORMER_PACKAGES = (
    "transformer/linear",
    "transformer/linear_adaptive",
    "transformer/expert_linear",
    "transformer/expert_linear_adaptive",
)

_EXPERT_TRANSFORMER_PACKAGES = (
    (
        "transformer/expert_linear",
        build_expert_linear_experiment_config,
    ),
    (
        "transformer/expert_linear_adaptive",
        build_expert_linear_adaptive_experiment_config,
    ),
)


class TestTransformerConfigurationFlexibility(unittest.TestCase):
    def test_every_package_exposes_independent_transformer_path_defaults(self):
        expected_fields = {
            "ENCODER_ATTN_STACK_HIDDEN_DIM",
            "ENCODER_ATTN_GATE_STACK_INDEPENDENT_FLAG",
            "DECODER_SELF_ATTN_STACK_HIDDEN_DIM",
            "DECODER_SELF_ATTN_MEMORY_FLAG",
            "DECODER_CROSS_ATTN_STACK_HIDDEN_DIM",
            "DECODER_CROSS_ATTN_RECURRENT_STACK_GATE_FLAG",
            "ENCODER_FF_STACK_HIDDEN_DIM",
            "ENCODER_FF_GATE_STACK_INDEPENDENT_FLAG",
            "DECODER_FF_STACK_HIDDEN_DIM",
            "DECODER_FF_MEMORY_FLAG",
        }
        overrides = {
            "encoder_attn_stack_hidden_dim": 11,
            "decoder_self_attn_stack_hidden_dim": 12,
            "decoder_cross_attn_stack_hidden_dim": 13,
            "encoder_ff_stack_hidden_dim": 14,
            "decoder_ff_stack_hidden_dim": 15,
        }

        for package_key in _TRANSFORMER_PACKAGES:
            with self.subTest(package=package_key):
                package = model_package(package_key)
                self.assertTrue(
                    expected_fields.issubset(
                        iter_supported_config_keys(package.runtime_defaults)
                    )
                )
                self.assertTrue(
                    expected_fields.issubset(
                        {field.key for field in configuration_schema(package).fields}
                    )
                )

                runtime = package.bind_runtime_defaults(overrides)
                self.assertEqual(
                    runtime.encoder_attention_options.stack_options.hidden_dim,
                    11,
                )
                self.assertEqual(
                    runtime.decoder_self_attention_options.stack_options.hidden_dim,
                    12,
                )
                self.assertEqual(
                    runtime.decoder_cross_attention_options.stack_options.hidden_dim,
                    13,
                )
                self.assertEqual(
                    runtime.encoder_feed_forward_options.stack_options.hidden_dim,
                    14,
                )
                self.assertEqual(
                    runtime.decoder_feed_forward_options.stack_options.hidden_dim,
                    15,
                )

    def test_expert_packages_apply_router_sampler_and_mixture_defaults(self):
        overrides = {
            "batch_size": 2,
            "vocab_size": 32,
            "model_dim": 8,
            "source_sequence_length": 4,
            "target_sequence_length": 4,
            "encoder_num_layers": 1,
            "decoder_num_layers": 1,
            "attn_num_heads": 2,
            "ff_stack_hidden_dim": 8,
            "dropout_probability": 0.0,
            "expert_attention_use_kv_expert_models_flag": False,
            "dropped_token_behavior": DroppedTokenOptions.IDENTITY,
            "compute_expert_mixture_flag": False,
            "weighted_parameters_flag": True,
            "weighting_position_option": (ExpertWeightingPositionOptions.AFTER_EXPERTS),
            "routing_initialization_mode": RoutingInitializationMode.SHARED,
            "sampler_threshold": 0.25,
            "sampler_filter_above_threshold": True,
            "sampler_num_topk_samples": 1,
            "sampler_noisy_topk_flag": True,
            "coefficient_of_variation_loss_weight": 0.2,
            "zero_centred_loss_weight": 0.3,
            "mutual_information_loss_weight": 0.4,
            "router_noisy_topk_flag": True,
            "router_stack_hidden_dim": 19,
            "router_stack_num_layers": 2,
            "router_stack_activation": ActivationOptions.GELU,
            "expert_stack_hidden_dim": 17,
            "expert_stack_num_layers": 3,
            "expert_stack_activation": ActivationOptions.TANH,
        }

        for package_key, build_experiment_config in _EXPERT_TRANSFORMER_PACKAGES:
            with self.subTest(package=package_key):
                runtime = model_package(package_key).bind_runtime_defaults(overrides)
                self.assertEqual(
                    runtime.attention_expert_options.router_path_options.stack_options.hidden_dim,
                    19,
                )
                self.assertEqual(
                    runtime.feed_forward_expert_options.expert_path_options.stack_options.hidden_dim,
                    17,
                )

                experiment = build_experiment_config(runtime)
                encoder = getattr(
                    experiment.encoder_config,
                    "block_config",
                    experiment.encoder_config,
                )
                attention = encoder.layer_config.layer_model_config.attention_config
                experts = attention.experts_config
                sampler = experts.sampler_config

                self.assertFalse(attention.use_kv_expert_models_flag)
                self.assertIs(
                    experts.dropped_token_behavior,
                    DroppedTokenOptions.IDENTITY,
                )
                self.assertFalse(experts.compute_expert_mixture_flag)
                self.assertTrue(experts.weighted_parameters_flag)
                self.assertIs(
                    experts.weighting_position_option,
                    ExpertWeightingPositionOptions.AFTER_EXPERTS,
                )
                self.assertIs(
                    experts.routing_initialization_mode,
                    RoutingInitializationMode.SHARED,
                )
                self.assertEqual(sampler.threshold, 0.25)
                self.assertTrue(sampler.filter_above_threshold)
                self.assertEqual(sampler.num_topk_samples, 1)
                self.assertTrue(sampler.noisy_topk_flag)
                self.assertEqual(
                    sampler.coefficient_of_variation_loss_weight,
                    0.2,
                )
                self.assertEqual(sampler.zero_centred_loss_weight, 0.3)
                self.assertEqual(sampler.mutual_information_loss_weight, 0.4)
                self.assertTrue(sampler.router_config.noisy_topk_flag)
                self.assertEqual(sampler.router_config.model_config.hidden_dim, 19)
                self.assertEqual(experts.expert_model_config.hidden_dim, 17)


if __name__ == "__main__":
    unittest.main()
