import unittest

from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterGroupingScopeOptions,
    BankExpansionFactorOptions,
    DynamicDepthOptions,
    MaskDimensionOptions,
    SingleModelDynamicWeightConfig,
    StandardDynamicDiagonalConfig,
    TopSliceAxisMaskConfig,
    WeightDecayScheduleOptions,
    WeightedBankDynamicBiasConfig,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.layers import ActivationOptions
from model_runtime.inspection import configuration_schema
from models.catalog import model_package

_TRANSFORMER_PACKAGES = (
    "transformer/linear",
    "transformer/linear_adaptive",
    "transformer/expert_linear",
    "transformer/expert_linear_adaptive",
)

_EXPERT_TRANSFORMER_PACKAGES = (
    "transformer/expert_linear",
    "transformer/expert_linear_adaptive",
)

_ADAPTIVE_TRANSFORMER_PACKAGES = (
    (
        "transformer/linear_adaptive",
        "projection_adaptive_",
    ),
    (
        "transformer/expert_linear_adaptive",
        "attention_projection_adaptive_",
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
                self.assertIsNotNone(package)
                assert package is not None
                self.assertTrue(
                    expected_fields.issubset(
                        package.runtime_defaults_spec.supported_keys
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

        for package_key in _EXPERT_TRANSFORMER_PACKAGES:
            with self.subTest(package=package_key):
                package = model_package(package_key)
                self.assertIsNotNone(package)
                assert package is not None
                runtime = package.bind_runtime_defaults(overrides)
                self.assertEqual(
                    runtime.attention_expert_options.router_path_options.stack_options.hidden_dim,
                    19,
                )
                self.assertEqual(
                    runtime.feed_forward_expert_options.expert_path_options.stack_options.hidden_dim,
                    17,
                )

                experiment = package.build_configuration(
                    config_overrides=overrides
                ).experiment_config
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

    def test_adaptive_packages_apply_every_generator_control_per_path(self):
        common = {
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
            "decoder_cross_attn_adaptive_generator_stack_hidden_dim": 29,
        }

        for (
            package_key,
            prefix,
        ) in _ADAPTIVE_TRANSFORMER_PACKAGES:
            overrides = {
                **common,
                f"{prefix}grouping_scope": (AdaptiveParameterGroupingScopeOptions.ROWS),
                f"{prefix}group_count": 2,
                f"{prefix}weight_option": SingleModelDynamicWeightConfig,
                f"{prefix}generator_depth": DynamicDepthOptions.DEPTH_OF_THREE,
                f"{prefix}weight_decay_schedule": (WeightDecayScheduleOptions.LINEAR),
                f"{prefix}weight_decay_rate": 0.2,
                f"{prefix}weight_decay_warmup_batches": 7,
                f"{prefix}weight_normalization_option": (
                    WeightNormalizationOptions.CLAMP
                ),
                f"{prefix}weight_normalization_position_option": (
                    WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT
                ),
                f"{prefix}bias_option": WeightedBankDynamicBiasConfig,
                f"{prefix}bias_bank_expansion_factor": (
                    BankExpansionFactorOptions.FACTOR_OF_THREE
                ),
                f"{prefix}diagonal_option": StandardDynamicDiagonalConfig,
                f"{prefix}row_mask_option": TopSliceAxisMaskConfig,
                f"{prefix}mask_threshold": 0.6,
                f"{prefix}mask_surrogate_scale": 9.0,
                f"{prefix}mask_floor": 0.1,
                f"{prefix}mask_dimension_option": MaskDimensionOptions.COLUMN,
                f"{prefix}mask_transition_width": 0.2,
                f"{prefix}generator_stack_hidden_dim": 23,
                f"{prefix}generator_stack_num_layers": 2,
                f"{prefix}generator_stack_activation": ActivationOptions.GELU,
                f"{prefix}weight_generator_stack_independent_flag": True,
                f"{prefix}weight_generator_stack_hidden_dim": 31,
                f"{prefix}weight_generator_stack_num_layers": 3,
                f"{prefix}weight_generator_stack_activation": (ActivationOptions.TANH),
            }
            with self.subTest(package=package_key):
                package = model_package(package_key)
                self.assertIsNotNone(package)
                assert package is not None
                schema_keys = {
                    field.key for field in configuration_schema(package).fields
                }
                self.assertIn(
                    f"{prefix}weight_generator_stack_hidden_dim".upper(),
                    schema_keys,
                )
                self.assertEqual(
                    package.runtime_defaults_spec.parse_value(
                        f"{prefix}weight_generator_stack_hidden_dim".upper(),
                        "31",
                    ),
                    31,
                )
                self.assertIs(
                    package.runtime_defaults_spec.parse_value(
                        f"{prefix}weight_generator_stack_independent_flag".upper(),
                        "true",
                    ),
                    True,
                )
                runtime = package.bind_runtime_defaults(overrides)
                encoder_options = runtime.encoder_attention_adaptive_options
                cross_options = runtime.decoder_cross_attention_adaptive_options
                self.assertEqual(encoder_options.generator_stack_options.hidden_dim, 23)
                self.assertEqual(cross_options.generator_stack_options.hidden_dim, 29)
                self.assertEqual(
                    encoder_options.weight_generator_stack_options.hidden_dim,
                    31,
                )

                experiment = package.build_configuration(
                    config_overrides=overrides
                ).experiment_config
                encoder = getattr(
                    experiment.encoder_config,
                    "block_config",
                    experiment.encoder_config,
                ).layer_config.layer_model_config
                decoder = getattr(
                    experiment.decoder_config,
                    "block_config",
                    experiment.decoder_config,
                ).layer_config.layer_model_config
                augmentation = encoder.attention_config.projection_model_config.layer_config.layer_model_config.adaptive_augmentation_config
                cross_augmentation = decoder.cross_attention_config.projection_model_config.layer_config.layer_model_config.adaptive_augmentation_config

                self.assertIs(
                    augmentation.grouping_scope,
                    AdaptiveParameterGroupingScopeOptions.ROWS,
                )
                self.assertEqual(augmentation.group_count, 2)
                self.assertEqual(augmentation.model_config.hidden_dim, 23)
                self.assertEqual(cross_augmentation.model_config.hidden_dim, 29)
                self.assertIsInstance(
                    augmentation.weight_config,
                    SingleModelDynamicWeightConfig,
                )
                self.assertIs(
                    augmentation.weight_config.generator_depth,
                    DynamicDepthOptions.DEPTH_OF_THREE,
                )
                self.assertIs(
                    augmentation.weight_config.decay_schedule,
                    WeightDecayScheduleOptions.LINEAR,
                )
                self.assertEqual(augmentation.weight_config.decay_rate, 0.2)
                self.assertEqual(augmentation.weight_config.decay_warmup_batches, 7)
                self.assertIs(
                    augmentation.weight_config.normalization_option,
                    WeightNormalizationOptions.CLAMP,
                )
                self.assertIs(
                    augmentation.weight_config.normalization_position_option,
                    WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT,
                )
                self.assertEqual(augmentation.weight_config.model_config.hidden_dim, 31)
                self.assertIsInstance(
                    augmentation.bias_config,
                    WeightedBankDynamicBiasConfig,
                )
                self.assertIs(
                    augmentation.bias_config.bank_expansion_factor,
                    BankExpansionFactorOptions.FACTOR_OF_THREE,
                )
                self.assertIsInstance(
                    augmentation.diagonal_config,
                    StandardDynamicDiagonalConfig,
                )
                self.assertIsInstance(
                    augmentation.mask_config,
                    TopSliceAxisMaskConfig,
                )
                self.assertEqual(augmentation.mask_config.mask_threshold, 0.6)
                self.assertEqual(augmentation.mask_config.mask_surrogate_scale, 9.0)
                self.assertEqual(augmentation.mask_config.mask_floor, 0.1)
                self.assertIs(
                    augmentation.mask_config.mask_dimension_option,
                    MaskDimensionOptions.COLUMN,
                )
                self.assertEqual(augmentation.mask_config.mask_transition_width, 0.2)

    def test_expert_adaptive_router_and_nonexpert_kv_paths_stay_adaptive(self):
        overrides = {
            "batch_size": 2,
            "vocab_size": 32,
            "model_dim": 8,
            "source_sequence_length": 4,
            "target_sequence_length": 4,
            "encoder_num_layers": 1,
            "decoder_num_layers": 1,
            "encoder_attn_num_heads": 2,
            "encoder_ff_stack_hidden_dim": 8,
            "decoder_ff_stack_hidden_dim": 8,
            "dropout_probability": 0.0,
            "expert_attention_use_kv_expert_models_flag": False,
            "encoder_attn_adaptive_weight_option": (SingleModelDynamicWeightConfig),
            "encoder_attn_adaptive_generator_stack_hidden_dim": 27,
            "router_adaptive_weight_option": SingleModelDynamicWeightConfig,
            "router_adaptive_group_count": 2,
            "router_adaptive_generator_stack_hidden_dim": 33,
        }
        package = model_package("transformer/expert_linear_adaptive")
        self.assertIsNotNone(package)
        assert package is not None
        package.bind_runtime_defaults(overrides)

        experiment = package.build_configuration(
            config_overrides=overrides
        ).experiment_config
        encoder = getattr(
            experiment.encoder_config,
            "block_config",
            experiment.encoder_config,
        ).layer_config.layer_model_config
        attention = encoder.attention_config
        projection_augmentation = attention.projection_model_config.layer_config.layer_model_config.adaptive_augmentation_config
        router_augmentation = attention.experts_config.sampler_config.router_config.model_config.layer_config.layer_model_config.adaptive_augmentation_config

        self.assertFalse(attention.use_kv_expert_models_flag)
        self.assertEqual(projection_augmentation.model_config.hidden_dim, 27)
        self.assertIsInstance(
            projection_augmentation.weight_config,
            SingleModelDynamicWeightConfig,
        )
        self.assertEqual(router_augmentation.group_count, 2)
        self.assertEqual(router_augmentation.model_config.hidden_dim, 33)
        self.assertIsInstance(
            router_augmentation.weight_config,
            SingleModelDynamicWeightConfig,
        )


if __name__ == "__main__":
    unittest.main()
