from __future__ import annotations

import unittest
from dataclasses import fields, replace

import torch
import torch.nn as nn

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AdaptiveParameterGroupingScopeOptions,
    AdditiveDynamicBiasConfig,
    WeightDecayScheduleOptions,
)
from emperor.embedding.absolute import TextLearnedPositionalEmbeddingConfig
from emperor.embedding.contextual import (
    ByteContextualEmbeddingConfig,
    ByteContextualEmbeddingState,
    CausalPrefixKernelConfig,
)
from emperor.embedding.contextual._encoding import Utf8BitEncoder
from emperor.embedding.contextual._validation import CausalPrefixKernelValidator
from emperor.embedding.relative import DynamicPositionalBiasConfig
from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    MixtureOfExpertsConfig,
    RoutingInitializationMode,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.sampler import RouterConfig, SamplerConfig
from support.adaptive_grouping import grouping_value


def _linear_stack(
    *,
    hidden_dim: int,
    num_layers: int = 1,
    adaptive: bool = False,
) -> LayerStackConfig:
    layer_model_config = LinearLayerConfig(bias_flag=True)
    if adaptive:
        layer_model_config = AdaptiveLinearLayerConfig(
            bias_flag=True,
            adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                bias_config=AdditiveDynamicBiasConfig(
                    decay_schedule=WeightDecayScheduleOptions.DISABLED,
                    decay_rate=0.0,
                    decay_warmup_batches=0,
                    model_config=_linear_stack(
                        hidden_dim=hidden_dim,
                        num_layers=1,
                    ),
                ),
                grouping_config=None,
            ),
        )
    return LayerStackConfig(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_postprocessing_flag=False,
        shared_gate_config=None,
        shared_halting_config=None,
        shared_memory_config=None,
        layer_config=LayerConfig(
            activation=ActivationOptions.RELU,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=layer_model_config,
        ),
    )


def _moe_config(
    input_dim: int,
    output_dim: int,
    *,
    num_experts: int = 3,
    top_k: int = 2,
    capacity_factor: float = 0.0,
    routing_mode: RoutingInitializationMode = RoutingInitializationMode.LAYER,
    normalize_probabilities: bool = True,
    weighted_parameters: bool = True,
    weighting_position: ExpertWeightingPositionOptions = (
        ExpertWeightingPositionOptions.AFTER_EXPERTS
    ),
    compute_mixture: bool = True,
    auxiliary_loss_weight: float = 0.1,
    adaptive: bool = True,
) -> MixtureOfExpertsConfig:
    router_output_dim = num_experts
    router_stack = _linear_stack(hidden_dim=max(input_dim, router_output_dim))
    router_stack.input_dim = input_dim
    router_stack.output_dim = router_output_dim
    return MixtureOfExpertsConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        top_k=top_k,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        dropped_token_behavior=DroppedTokenOptions.ZEROS,
        compute_expert_mixture_flag=compute_mixture,
        weighted_parameters_flag=weighted_parameters,
        weighting_position_option=weighting_position,
        routing_initialization_mode=routing_mode,
        sampler_config=SamplerConfig(
            top_k=top_k,
            threshold=0.0,
            filter_above_threshold=False,
            num_topk_samples=0,
            normalize_probabilities_flag=normalize_probabilities,
            noisy_topk_flag=False,
            num_experts=num_experts,
            coefficient_of_variation_loss_weight=auxiliary_loss_weight,
            switch_loss_weight=0.0,
            zero_centred_loss_weight=0.0,
            mutual_information_loss_weight=0.0,
            router_config=RouterConfig(
                input_dim=input_dim,
                num_experts=num_experts,
                noisy_topk_flag=False,
                model_config=router_stack,
            ),
        ),
        expert_model_config=_linear_stack(
            hidden_dim=output_dim,
            num_layers=2,
            adaptive=adaptive,
        ),
    )


def contextual_config(
    *,
    max_token_bytes: int = 4,
    hidden_dim: int = 8,
    maximum_sequence_length: int = 6,
    kernel_dim: int = 4,
    num_experts: int = 3,
    top_k: int = 2,
    residual_scale: float = 1e-3,
) -> ByteContextualEmbeddingConfig:
    return ByteContextualEmbeddingConfig(
        max_token_bytes=max_token_bytes,
        hidden_dim=hidden_dim,
        byte_moe_config=_moe_config(
            max_token_bytes * 9,
            hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
        ),
        positional_embedding_config=TextLearnedPositionalEmbeddingConfig(
            num_embeddings=maximum_sequence_length,
            embedding_dim=hidden_dim,
            padding_idx=0,
        ),
        prefix_kernel_config=CausalPrefixKernelConfig(
            hidden_dim=hidden_dim,
            kernel_dim=kernel_dim,
            relative_position_config=DynamicPositionalBiasConfig(
                num_heads=1,
                embedding_dim=kernel_dim,
                max_positions=maximum_sequence_length - 1,
            ),
        ),
        context_moe_config=_moe_config(
            hidden_dim * 2,
            hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
        ),
        residual_scale_initial_value=residual_scale,
    )


def _parameter_ids(module: nn.Module) -> set[int]:
    return {id(parameter) for parameter in module.parameters()}


class Utf8BitEncoderTests(unittest.TestCase):
    def test_constructor_rejects_non_positive_exact_integer_widths(self) -> None:
        invalid_cases = (
            (
                True,
                TypeError,
                "max_token_bytes must be int for Utf8BitEncoder, got bool",
            ),
            (
                0,
                ValueError,
                "max_token_bytes must be greater than 0 for Utf8BitEncoder, received 0",
            ),
        )
        for value, exception, expected_message in invalid_cases:
            with self.subTest(value=value), self.assertRaises(exception) as raised:
                Utf8BitEncoder(value)
            self.assertEqual(str(raised.exception), expected_message)

    def test_exact_msb_first_bits_truncation_padding_and_byte_validity(self) -> None:
        encoder = Utf8BitEncoder(max_token_bytes=4)
        token_texts = [["A", "€", "", "abc€", "\0", "<pad>"]]

        bits, byte_mask = encoder(token_texts, device=torch.device("cpu"))

        expected_bytes = torch.tensor(
            [
                [
                    [0x41, 0x00, 0x00, 0x00],
                    [0xE2, 0x82, 0xAC, 0x00],
                    [0x00, 0x00, 0x00, 0x00],
                    [0x61, 0x62, 0x63, 0xE2],
                    [0x00, 0x00, 0x00, 0x00],
                    [0x3C, 0x70, 0x61, 0x64],
                ]
            ],
            dtype=torch.uint8,
        )
        shifts = torch.arange(7, -1, -1, dtype=torch.uint8)
        expected_bits = (
            expected_bytes.unsqueeze(-1).bitwise_right_shift(shifts).bitwise_and(1)
        ).bool()
        expected_bits = expected_bits.reshape(1, 6, 32)
        expected_mask = torch.tensor(
            [
                [
                    [True, False, False, False],
                    [True, True, True, False],
                    [False, False, False, False],
                    [True, True, True, True],
                    [True, False, False, False],
                    [True, True, True, True],
                ]
            ]
        )

        self.assertEqual(bits.dtype, torch.bool)
        self.assertEqual(byte_mask.dtype, torch.bool)
        self.assertEqual(bits.shape, (1, 6, 32))
        torch.testing.assert_close(bits, expected_bits, rtol=0, atol=0)
        torch.testing.assert_close(byte_mask, expected_mask, rtol=0, atol=0)
        self.assertTrue(torch.equal(bits[0, 2], bits[0, 4]))
        self.assertFalse(torch.equal(byte_mask[0, 2], byte_mask[0, 4]))
        self.assertEqual(tuple(encoder.parameters()), ())
        self.assertEqual(tuple(encoder.buffers()), ())

        one_byte_encoder = Utf8BitEncoder(max_token_bytes=1)
        one_byte_bits, one_byte_mask = one_byte_encoder(
            [["A"]],
            device=torch.device("cpu"),
        )
        self.assertEqual(one_byte_bits.shape, (1, 1, 8))
        self.assertEqual(one_byte_mask.shape, (1, 1, 1))

        meta_bits, meta_mask = one_byte_encoder(
            [["A"]],
            device=torch.device("meta"),
        )
        self.assertEqual(meta_bits.device.type, "meta")
        self.assertEqual(meta_mask.device.type, "meta")

    def test_unicode_encoding_failure_reports_the_token_coordinate(self) -> None:
        encoder = Utf8BitEncoder(max_token_bytes=4)

        with self.assertRaises(ValueError) as raised:
            encoder([["valid", "\ud800"]], device=torch.device("cpu"))
        self.assertEqual(
            str(raised.exception),
            "token_texts[0][1] cannot be encoded as UTF-8",
        )


class ContextualConfigurationTests(unittest.TestCase):
    def assert_build_error(
        self,
        config,
        exception_type: type[Exception],
        expected_message: str,
    ) -> None:
        with self.assertRaises(exception_type) as raised:
            config.build()
        self.assertEqual(str(raised.exception), expected_message)

    def test_public_contract_defaults_fields_registry_and_state_are_exact(self) -> None:
        empty = ByteContextualEmbeddingConfig()
        kernel = CausalPrefixKernelConfig()

        self.assertEqual(empty.max_token_bytes, 4)
        self.assertEqual(empty.residual_scale_initial_value, 1e-3)
        self.assertIsNone(empty.hidden_dim)
        self.assertEqual(kernel.kernel_dim, 32)
        self.assertEqual(
            tuple(field.name for field in fields(ByteContextualEmbeddingConfig)),
            (
                "max_token_bytes",
                "hidden_dim",
                "byte_moe_config",
                "positional_embedding_config",
                "prefix_kernel_config",
                "context_moe_config",
                "residual_scale_initial_value",
            ),
        )
        self.assertEqual(
            tuple(field.name for field in fields(CausalPrefixKernelConfig)),
            ("hidden_dim", "kernel_dim", "relative_position_config"),
        )
        state = ByteContextualEmbeddingState(
            hidden=torch.zeros(1, 1, 1),
            byte_moe_auxiliary_loss=torch.tensor(1.0),
            context_moe_auxiliary_loss=torch.tensor(2.0),
            loss=torch.tensor(3.0),
        )
        self.assertEqual(
            tuple(field.name for field in fields(ByteContextualEmbeddingState)),
            (
                "hidden",
                "byte_moe_auxiliary_loss",
                "context_moe_auxiliary_loss",
                "loss",
            ),
        )
        self.assertEqual(state.loss.item(), 3.0)
        config = contextual_config()
        self.assertIsInstance(config.build(), config.registry_owner())
        self.assertIsInstance(
            config.prefix_kernel_config.build(),
            config.prefix_kernel_config.registry_owner(),
        )

    def test_required_types_dimensions_and_nested_compatibility_fail_closed(
        self,
    ) -> None:
        base = contextual_config()
        invalid_cases: tuple[
            tuple[str, ByteContextualEmbeddingConfig, type[Exception]], ...
        ] = (
            (
                "max_token_bytes type",
                replace(base, max_token_bytes=True),
                TypeError,
            ),
            (
                "max_token_bytes positive",
                replace(base, max_token_bytes=0),
                ValueError,
            ),
            ("hidden_dim type", replace(base, hidden_dim=3.5), TypeError),
            ("hidden_dim positive", replace(base, hidden_dim=0), ValueError),
            (
                "byte config type",
                replace(base, byte_moe_config=object()),
                TypeError,
            ),
            (
                "position config type",
                replace(base, positional_embedding_config=object()),
                TypeError,
            ),
            (
                "prefix config type",
                replace(base, prefix_kernel_config=object()),
                TypeError,
            ),
            (
                "context config type",
                replace(base, context_moe_config=object()),
                TypeError,
            ),
            (
                "residual bool",
                replace(base, residual_scale_initial_value=True),
                TypeError,
            ),
            (
                "residual nonfinite",
                replace(base, residual_scale_initial_value=float("inf")),
                ValueError,
            ),
            (
                "byte input",
                replace(
                    base,
                    byte_moe_config=replace(base.byte_moe_config, input_dim=35),
                ),
                ValueError,
            ),
            (
                "byte output",
                replace(
                    base,
                    byte_moe_config=replace(base.byte_moe_config, output_dim=7),
                ),
                ValueError,
            ),
            (
                "context input",
                replace(
                    base,
                    context_moe_config=replace(base.context_moe_config, input_dim=15),
                ),
                ValueError,
            ),
            (
                "context output",
                replace(
                    base,
                    context_moe_config=replace(base.context_moe_config, output_dim=7),
                ),
                ValueError,
            ),
            (
                "position dimension",
                replace(
                    base,
                    positional_embedding_config=replace(
                        base.positional_embedding_config,
                        embedding_dim=7,
                    ),
                ),
                ValueError,
            ),
            (
                "position dimension type",
                replace(
                    base,
                    positional_embedding_config=replace(
                        base.positional_embedding_config,
                        embedding_dim=True,
                    ),
                ),
                TypeError,
            ),
            (
                "position count",
                replace(
                    base,
                    positional_embedding_config=replace(
                        base.positional_embedding_config,
                        num_embeddings=0,
                    ),
                ),
                ValueError,
            ),
            (
                "position padding",
                replace(
                    base,
                    positional_embedding_config=replace(
                        base.positional_embedding_config,
                        padding_idx=None,
                    ),
                ),
                ValueError,
            ),
            (
                "prefix dimension",
                replace(
                    base,
                    prefix_kernel_config=replace(
                        base.prefix_kernel_config,
                        hidden_dim=7,
                    ),
                ),
                ValueError,
            ),
        )

        for name, config, exception in invalid_cases:
            with self.subTest(name=name), self.assertRaises(exception):
                config.build()

    def test_component_configuration_errors_are_exact(self) -> None:
        base = contextual_config()
        config_cases = (
            (
                replace(base, max_token_bytes=True),
                TypeError,
                "max_token_bytes must be int, got bool",
            ),
            (
                replace(base, max_token_bytes=0),
                ValueError,
                "max_token_bytes must be greater than 0, received 0",
            ),
            (
                replace(base, hidden_dim=3.5),
                TypeError,
                "hidden_dim must be int, got float",
            ),
            (
                replace(base, hidden_dim=0),
                ValueError,
                "hidden_dim must be greater than 0, received 0",
            ),
            (
                replace(base, residual_scale_initial_value=True),
                TypeError,
                "residual_scale_initial_value must be a real scalar, got bool",
            ),
            (
                replace(base, residual_scale_initial_value=float("inf")),
                ValueError,
                "residual_scale_initial_value must be finite, received inf",
            ),
            (
                replace(base, byte_moe_config=object()),
                TypeError,
                "byte_moe_config must be MixtureOfExpertsConfig, got object",
            ),
            (
                replace(base, positional_embedding_config=object()),
                TypeError,
                "positional_embedding_config must be "
                "TextLearnedPositionalEmbeddingConfig, got object",
            ),
            (
                replace(base, prefix_kernel_config=object()),
                TypeError,
                "prefix_kernel_config must be CausalPrefixKernelConfig, got object",
            ),
            (
                replace(base, context_moe_config=object()),
                TypeError,
                "context_moe_config must be MixtureOfExpertsConfig, got object",
            ),
            (
                replace(
                    base,
                    byte_moe_config=replace(base.byte_moe_config, input_dim=35),
                ),
                ValueError,
                "byte_moe_config.input_dim must equal 36, received 35",
            ),
            (
                replace(
                    base,
                    byte_moe_config=replace(base.byte_moe_config, output_dim=7),
                ),
                ValueError,
                "byte_moe_config.output_dim must equal 8, received 7",
            ),
            (
                replace(
                    base,
                    context_moe_config=replace(base.context_moe_config, input_dim=15),
                ),
                ValueError,
                "context_moe_config.input_dim must equal 16, received 15",
            ),
            (
                replace(
                    base,
                    context_moe_config=replace(base.context_moe_config, output_dim=7),
                ),
                ValueError,
                "context_moe_config.output_dim must equal 8, received 7",
            ),
        )
        for config, exception_type, message in config_cases:
            with self.subTest(message=message):
                self.assert_build_error(config, exception_type, message)

    def test_moe_routing_and_mixture_safety_constraints_fail_before_build(self) -> None:
        base = contextual_config()
        unsafe_variants = (
            {"capacity_factor": 0.5},
            {"routing_initialization_mode": RoutingInitializationMode.SHARED},
            {"compute_expert_mixture_flag": False},
            {"weighted_parameters_flag": False},
            {
                "weighting_position_option": (
                    ExpertWeightingPositionOptions.BEFORE_EXPERTS
                )
            },
            {
                "sampler_config": replace(
                    base.byte_moe_config.sampler_config,
                    normalize_probabilities_flag=False,
                )
            },
        )
        for overrides in unsafe_variants:
            with self.subTest(overrides=overrides):
                unsafe_byte = replace(base.byte_moe_config, **overrides)
                with self.assertRaises(ValueError):
                    replace(base, byte_moe_config=unsafe_byte).build()

        with self.assertRaises(TypeError):
            replace(
                base,
                byte_moe_config=replace(base.byte_moe_config, sampler_config=None),
            ).build()

        grouped_augmentation = AdaptiveParameterAugmentationConfig(
            grouping_config=grouping_value(
                AdaptiveParameterGroupingScopeOptions.ROWS, 1, input_order="BATCH_FIRST"
            ),
        )
        grouped_layer = replace(
            base.byte_moe_config.expert_model_config.layer_config.layer_model_config,
            adaptive_augmentation_config=grouped_augmentation,
        )
        grouped_layer_config = replace(
            base.byte_moe_config.expert_model_config.layer_config,
            layer_model_config=grouped_layer,
        )
        grouped_stack = replace(
            base.byte_moe_config.expert_model_config,
            layer_config=grouped_layer_config,
        )
        grouped_moe = replace(
            base.byte_moe_config,
            expert_model_config=grouped_stack,
        )
        with self.assertRaisesRegex(ValueError, "grouping"):
            replace(base, byte_moe_config=grouped_moe).build()

    def test_moe_configuration_errors_are_exact(self) -> None:
        base = contextual_config()
        byte_moe = base.byte_moe_config
        invalid_moes = (
            (
                replace(byte_moe, capacity_factor=0.5),
                ValueError,
                "byte_moe_config.capacity_factor must equal 0.0, received 0.5",
            ),
            (
                replace(
                    byte_moe,
                    routing_initialization_mode=RoutingInitializationMode.SHARED,
                ),
                ValueError,
                "byte_moe_config.routing_initialization_mode must be "
                "RoutingInitializationMode.LAYER",
            ),
            (
                replace(byte_moe, compute_expert_mixture_flag=False),
                ValueError,
                "byte_moe_config.compute_expert_mixture_flag must be True",
            ),
            (
                replace(byte_moe, weighted_parameters_flag=False),
                ValueError,
                "byte_moe_config.weighted_parameters_flag must be True",
            ),
            (
                replace(
                    byte_moe,
                    weighting_position_option=(
                        ExpertWeightingPositionOptions.BEFORE_EXPERTS
                    ),
                ),
                ValueError,
                "byte_moe_config.weighting_position_option must be "
                "ExpertWeightingPositionOptions.AFTER_EXPERTS",
            ),
            (
                replace(byte_moe, sampler_config=None),
                TypeError,
                "byte_moe_config.sampler_config must be SamplerConfig, got NoneType",
            ),
            (
                replace(byte_moe, sampler_config=object()),
                TypeError,
                "byte_moe_config.sampler_config must be SamplerConfig, got object",
            ),
            (
                replace(
                    byte_moe,
                    sampler_config=replace(
                        byte_moe.sampler_config,
                        normalize_probabilities_flag=False,
                    ),
                ),
                ValueError,
                "byte_moe_config.sampler_config.normalize_probabilities_flag must be "
                "True",
            ),
        )
        for invalid_moe, exception_type, message in invalid_moes:
            with self.subTest(message=message):
                self.assert_build_error(
                    replace(base, byte_moe_config=invalid_moe),
                    exception_type,
                    message,
                )

        context_capacity = replace(base.context_moe_config, capacity_factor=0.5)
        self.assert_build_error(
            replace(base, context_moe_config=context_capacity),
            ValueError,
            "context_moe_config.capacity_factor must equal 0.0, received 0.5",
        )

    def test_expert_stack_configuration_errors_are_exact(self) -> None:
        base = contextual_config()
        byte_moe = base.byte_moe_config
        stack = byte_moe.expert_model_config
        layer = stack.layer_config
        adaptive_linear = layer.layer_model_config
        disabled_empty_augmentation = AdaptiveParameterAugmentationConfig(
            grouping_config=None,
        )
        grouped_augmentation = replace(
            adaptive_linear.adaptive_augmentation_config,
            grouping_config=grouping_value(
                AdaptiveParameterGroupingScopeOptions.ROWS, 1
            ),
        )
        invalid_stacks = (
            (
                object(),
                TypeError,
                "byte_moe_config.expert_model_config must be LayerStackConfig, "
                "got object",
            ),
            (
                replace(stack, num_layers=True),
                TypeError,
                "byte_moe_config.expert_model_config.num_layers must be int, got bool",
            ),
            (
                replace(stack, num_layers=1),
                ValueError,
                "byte_moe_config.expert_model_config.num_layers must equal 2, received 1",
            ),
            (
                replace(stack, layer_config=object()),
                TypeError,
                "byte_moe_config.expert_model_config.layer_config must be LayerConfig, "
                "got object",
            ),
            (
                replace(
                    stack,
                    layer_config=replace(
                        layer,
                        layer_model_config=LinearLayerConfig(bias_flag=True),
                    ),
                ),
                TypeError,
                "byte_moe_config.expert layer model must be "
                "AdaptiveLinearLayerConfig, got LinearLayerConfig",
            ),
            (
                replace(
                    stack,
                    layer_config=replace(
                        layer,
                        layer_model_config=replace(
                            adaptive_linear,
                            adaptive_augmentation_config=None,
                        ),
                    ),
                ),
                TypeError,
                "byte_moe_config.expert adaptive augmentation must be "
                "AdaptiveParameterAugmentationConfig, got NoneType",
            ),
            (
                replace(
                    stack,
                    layer_config=replace(
                        layer,
                        layer_model_config=replace(
                            adaptive_linear,
                            adaptive_augmentation_config=object(),
                        ),
                    ),
                ),
                TypeError,
                "byte_moe_config.expert adaptive augmentation must be "
                "AdaptiveParameterAugmentationConfig, got object",
            ),
            (
                replace(
                    stack,
                    layer_config=replace(
                        layer,
                        layer_model_config=replace(
                            adaptive_linear,
                            adaptive_augmentation_config=grouped_augmentation,
                        ),
                    ),
                ),
                ValueError,
                "byte_moe_config.expert adaptive grouping must be absent (grouping_config=None)",
            ),
            (
                replace(
                    stack,
                    layer_config=replace(
                        layer,
                        layer_model_config=replace(
                            adaptive_linear,
                            adaptive_augmentation_config=disabled_empty_augmentation,
                        ),
                    ),
                ),
                ValueError,
                "byte_moe_config.expert adaptive augmentation must configure at least "
                "one adaptive parameter component",
            ),
        )
        for invalid_stack, exception_type, message in invalid_stacks:
            with self.subTest(message=message):
                invalid_moe = replace(byte_moe, expert_model_config=invalid_stack)
                self.assert_build_error(
                    replace(base, byte_moe_config=invalid_moe),
                    exception_type,
                    message,
                )

    def test_both_moes_require_two_layer_adaptive_row_local_experts(self) -> None:
        base = contextual_config()
        for moe_field in ("byte_moe_config", "context_moe_config"):
            base_moe = getattr(base, moe_field)
            base_stack = base_moe.expert_model_config
            base_layer = base_stack.layer_config
            base_linear = base_layer.layer_model_config
            invalid_stacks = (
                (object(), TypeError),
                (replace(base_stack, num_layers=True), TypeError),
                (replace(base_stack, num_layers=1), ValueError),
                (replace(base_stack, layer_config=object()), TypeError),
                (
                    replace(
                        base_stack,
                        layer_config=replace(
                            base_layer,
                            layer_model_config=LinearLayerConfig(bias_flag=True),
                        ),
                    ),
                    TypeError,
                ),
                (
                    replace(
                        base_stack,
                        layer_config=replace(
                            base_layer,
                            layer_model_config=replace(
                                base_linear,
                                adaptive_augmentation_config=None,
                            ),
                        ),
                    ),
                    TypeError,
                ),
                (
                    replace(
                        base_stack,
                        layer_config=replace(
                            base_layer,
                            layer_model_config=replace(
                                base_linear,
                                adaptive_augmentation_config=(
                                    AdaptiveParameterAugmentationConfig(
                                        grouping_config=None,
                                    )
                                ),
                            ),
                        ),
                    ),
                    ValueError,
                ),
            )
            for stack, exception in invalid_stacks:
                with self.subTest(moe=moe_field, stack=stack):
                    invalid_moe = replace(base_moe, expert_model_config=stack)
                    with self.assertRaises(exception):
                        replace(base, **{moe_field: invalid_moe}).build()

    def test_prefix_nested_configuration_is_strict(self) -> None:
        base = contextual_config().prefix_kernel_config
        invalid_cases = (
            (replace(base, hidden_dim=True), TypeError),
            (replace(base, hidden_dim=0), ValueError),
            (replace(base, kernel_dim=False), TypeError),
            (replace(base, kernel_dim=0), ValueError),
            (replace(base, relative_position_config=object()), TypeError),
            (
                replace(
                    base,
                    relative_position_config=replace(
                        base.relative_position_config,
                        num_heads=2,
                    ),
                ),
                ValueError,
            ),
            (
                replace(
                    base,
                    relative_position_config=replace(
                        base.relative_position_config,
                        num_heads=True,
                    ),
                ),
                TypeError,
            ),
            (
                replace(
                    base,
                    relative_position_config=replace(
                        base.relative_position_config,
                        embedding_dim=3,
                    ),
                ),
                ValueError,
            ),
            (
                replace(
                    base,
                    relative_position_config=replace(
                        base.relative_position_config,
                        embedding_dim="4",
                    ),
                ),
                TypeError,
            ),
            (
                replace(
                    base,
                    relative_position_config=replace(
                        base.relative_position_config,
                        max_positions=0,
                    ),
                ),
                ValueError,
            ),
        )
        for config, exception in invalid_cases:
            with self.subTest(config=config), self.assertRaises(exception):
                config.build()

        with self.assertRaises(TypeError):
            CausalPrefixKernelValidator.validate_config(object())

    def test_position_and_prefix_configuration_errors_are_exact(self) -> None:
        base = contextual_config()
        position = base.positional_embedding_config
        position_cases = (
            (
                replace(position, num_embeddings=True),
                TypeError,
                "positional_embedding_config.num_embeddings must be int, got bool",
            ),
            (
                replace(position, num_embeddings=0),
                ValueError,
                "positional_embedding_config.num_embeddings must be greater than 0, "
                "received 0",
            ),
            (
                replace(position, embedding_dim=True),
                TypeError,
                "positional_embedding_config.embedding_dim must be int, got bool",
            ),
            (
                replace(position, embedding_dim=0),
                ValueError,
                "positional_embedding_config.embedding_dim must be greater than 0, "
                "received 0",
            ),
            (
                replace(position, embedding_dim=7),
                ValueError,
                "positional_embedding_config.embedding_dim must equal hidden_dim, "
                "received 7 and 8",
            ),
            (
                replace(position, padding_idx=None),
                ValueError,
                "positional_embedding_config.padding_idx must equal 0, received None",
            ),
        )
        for invalid_position, exception_type, message in position_cases:
            with self.subTest(message=message):
                self.assert_build_error(
                    replace(base, positional_embedding_config=invalid_position),
                    exception_type,
                    message,
                )

        prefix = base.prefix_kernel_config
        self.assert_build_error(
            replace(base, prefix_kernel_config=replace(prefix, hidden_dim=7)),
            ValueError,
            "prefix_kernel_config.hidden_dim must equal hidden_dim, received 7 and 8",
        )

        kernel_cases = (
            (
                replace(prefix, hidden_dim=True),
                TypeError,
                "hidden_dim must be int, got bool",
            ),
            (
                replace(prefix, hidden_dim=0),
                ValueError,
                "hidden_dim must be greater than 0, received 0",
            ),
            (
                replace(prefix, kernel_dim=False),
                TypeError,
                "kernel_dim must be int, got bool",
            ),
            (
                replace(prefix, kernel_dim=0),
                ValueError,
                "kernel_dim must be greater than 0, received 0",
            ),
            (
                replace(prefix, relative_position_config=object()),
                TypeError,
                "relative_position_config must be DynamicPositionalBiasConfig, "
                "got object",
            ),
            (
                replace(
                    prefix,
                    relative_position_config=replace(
                        prefix.relative_position_config,
                        num_heads=True,
                    ),
                ),
                TypeError,
                "relative_position_config.num_heads must be int, got bool",
            ),
            (
                replace(
                    prefix,
                    relative_position_config=replace(
                        prefix.relative_position_config,
                        num_heads=0,
                    ),
                ),
                ValueError,
                "relative_position_config.num_heads must be greater than 0, received 0",
            ),
            (
                replace(
                    prefix,
                    relative_position_config=replace(
                        prefix.relative_position_config,
                        embedding_dim="4",
                    ),
                ),
                TypeError,
                "relative_position_config.embedding_dim must be int, got str",
            ),
            (
                replace(
                    prefix,
                    relative_position_config=replace(
                        prefix.relative_position_config,
                        embedding_dim=0,
                    ),
                ),
                ValueError,
                "relative_position_config.embedding_dim must be greater than 0, "
                "received 0",
            ),
            (
                replace(
                    prefix,
                    relative_position_config=replace(
                        prefix.relative_position_config,
                        max_positions=True,
                    ),
                ),
                TypeError,
                "relative_position_config.max_positions must be int, got bool",
            ),
            (
                replace(
                    prefix,
                    relative_position_config=replace(
                        prefix.relative_position_config,
                        max_positions=0,
                    ),
                ),
                ValueError,
                "relative_position_config.max_positions must be greater than 0, "
                "received 0",
            ),
            (
                replace(
                    prefix,
                    relative_position_config=replace(
                        prefix.relative_position_config,
                        num_heads=2,
                    ),
                ),
                ValueError,
                "relative_position_config.num_heads must equal 1, received 2",
            ),
            (
                replace(
                    prefix,
                    relative_position_config=replace(
                        prefix.relative_position_config,
                        embedding_dim=3,
                    ),
                ),
                ValueError,
                "relative_position_config.embedding_dim must equal kernel_dim, "
                "received 3 and 4",
            ),
        )
        for config, exception_type, message in kernel_cases:
            with self.subTest(message=message):
                self.assert_build_error(config, exception_type, message)

        with self.assertRaises(TypeError) as raised:
            CausalPrefixKernelValidator.validate_config(object())
        self.assertEqual(
            str(raised.exception),
            "cfg must be CausalPrefixKernelConfig, got object",
        )

    def test_config_build_rejects_wrong_root_and_override_types(self) -> None:
        config = contextual_config()
        component_type = config.registry_owner()
        kernel_config = config.prefix_kernel_config
        kernel_type = kernel_config.registry_owner()

        for constructor, cfg, overrides, expected_message in (
            (
                component_type,
                object(),
                None,
                "cfg must be ByteContextualEmbeddingConfig, got object",
            ),
            (
                component_type,
                config,
                object(),
                "overrides must be ByteContextualEmbeddingConfig or None, got object",
            ),
            (
                kernel_type,
                object(),
                None,
                "cfg must be CausalPrefixKernelConfig, got object",
            ),
            (
                kernel_type,
                kernel_config,
                object(),
                "overrides must be CausalPrefixKernelConfig or None, got object",
            ),
        ):
            with self.subTest(constructor=constructor, overrides=overrides):
                with self.assertRaises(TypeError) as raised:
                    constructor(cfg, overrides)
                self.assertEqual(str(raised.exception), expected_message)

    def test_partial_and_full_overrides_preserve_and_replace_values(self) -> None:
        base = contextual_config(
            max_token_bytes=2,
            hidden_dim=6,
            kernel_dim=3,
            residual_scale=0.2,
        )
        partial = ByteContextualEmbeddingConfig(hidden_dim=6)

        partially_overridden = base.build(partial)

        self.assertEqual(partially_overridden.cfg.max_token_bytes, 2)
        self.assertEqual(partially_overridden.cfg.hidden_dim, 6)
        self.assertEqual(partially_overridden.cfg.residual_scale_initial_value, 0.2)
        self.assertEqual(partially_overridden.cfg.byte_moe_config, base.byte_moe_config)
        self.assertIsNot(partially_overridden.cfg.byte_moe_config, base.byte_moe_config)

        replacement = contextual_config(
            max_token_bytes=3,
            hidden_dim=10,
            maximum_sequence_length=7,
            kernel_dim=5,
            residual_scale=0.25,
        )
        fully_overridden = base.build(replacement)

        self.assertEqual(fully_overridden.cfg.max_token_bytes, 3)
        self.assertEqual(fully_overridden.cfg.hidden_dim, 10)
        self.assertEqual(fully_overridden.cfg.residual_scale_initial_value, 0.25)
        self.assertIsNot(fully_overridden.cfg.byte_moe_config, base.byte_moe_config)
        self.assertEqual(base.max_token_bytes, 2)
        self.assertEqual(base.hidden_dim, 6)

        kernel_base = base.prefix_kernel_config
        default_overridden_kernel = kernel_base.build(CausalPrefixKernelConfig())
        self.assertEqual(default_overridden_kernel.hidden_dim, 6)
        self.assertEqual(default_overridden_kernel.kernel_dim, 3)
        kernel_partial = CausalPrefixKernelConfig(hidden_dim=6)
        partially_overridden_kernel = kernel_base.build(kernel_partial)
        self.assertEqual(partially_overridden_kernel.kernel_dim, 3)
        self.assertEqual(
            partially_overridden_kernel.cfg.relative_position_config,
            kernel_base.relative_position_config,
        )
        self.assertIsNot(
            partially_overridden_kernel.cfg.relative_position_config,
            kernel_base.relative_position_config,
        )

        kernel_replacement = replacement.prefix_kernel_config
        fully_overridden_kernel = kernel_base.build(kernel_replacement)
        self.assertEqual(fully_overridden_kernel.hidden_dim, 10)
        self.assertEqual(fully_overridden_kernel.kernel_dim, 5)
        self.assertIsNot(
            fully_overridden_kernel.cfg.relative_position_config,
            kernel_base.relative_position_config,
        )

        one_byte = contextual_config(max_token_bytes=1).build()
        self.assertEqual(one_byte([["A"]]).hidden.shape, (1, 1, 8))

    def test_baseline_32_dimensional_configuration_builds_independent_moes(
        self,
    ) -> None:
        config = contextual_config(
            hidden_dim=32,
            maximum_sequence_length=35,
            kernel_dim=32,
            num_experts=12,
            top_k=3,
        )
        model = config.build()

        self.assertEqual(model.byte_moe.input_dim, 36)
        self.assertEqual(model.byte_moe.output_dim, 32)
        self.assertEqual(model.context_moe.input_dim, 64)
        self.assertEqual(model.context_moe.output_dim, 32)
        self.assertEqual(model.prefix_kernel.kernel_dim, 32)
        self.assertEqual(model.prefix_kernel.relative_position.max_positions, 34)
        self.assertEqual(model.byte_moe.num_experts, 12)
        self.assertEqual(model.context_moe.top_k, 3)
        self.assertIsNot(model.byte_moe, model.context_moe)
        self.assertTrue(
            _parameter_ids(model.byte_moe).isdisjoint(_parameter_ids(model.context_moe))
        )
        self.assertIsNot(model.byte_moe.sampler, model.context_moe.sampler)
        self.assertIsNot(
            model.byte_moe.expert_modules, model.context_moe.expert_modules
        )
        module_names = dict(model.named_modules())
        self.assertIn("byte_moe.sampler", module_names)
        self.assertIn("context_moe.sampler", module_names)
        torch.testing.assert_close(
            model.gamma.detach(),
            torch.tensor(1e-3),
            rtol=0,
            atol=0,
        )
        for moe in (model.byte_moe, model.context_moe):
            self.assertEqual(len(moe.expert_modules), 12)
            for expert in moe.expert_modules:
                self.assertEqual(len(expert.layers), 2)
                for layer in expert.layers:
                    self.assertIsInstance(
                        layer.model,
                        AdaptiveLinearLayerConfig().registry_owner(),
                    )
                    self.assertIsNotNone(layer.model.adaptive_behaviour)

        state = model([["a", "b"], ["c", "d"]])
        self.assertEqual(state.hidden.shape, (2, 2, 32))
        (state.hidden.square().mean() + state.loss).backward()
        self.assertIsNotNone(model.gamma.grad)


class ContextualForwardContractTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260825)
        self.config = contextual_config()
        self.model = self.config.build().eval()

    def test_input_collection_and_attention_mask_validation_is_strict(self) -> None:
        invalid_texts: tuple[object, ...] = (
            [],
            "abc",
            ["abc"],
            [[]],
            [["a"], ["b", "c"]],
            [["a", 1]],
            [["a", "b"], ("c",)],
            [["a"], "b"],
        )
        for token_texts in invalid_texts:
            with (
                self.subTest(token_texts=token_texts),
                self.assertRaises((TypeError, ValueError)),
            ):
                self.model(token_texts)

        valid_texts = [["a", "b"], ["c", "d"]]
        invalid_masks: tuple[object, ...] = (
            [[True, True], [True, True]],
            torch.ones(2, 2),
            torch.ones(2, dtype=torch.bool),
            torch.ones(1, 2, dtype=torch.bool),
        )
        for attention_mask in invalid_masks:
            with (
                self.subTest(attention_mask=attention_mask),
                self.assertRaises((TypeError, ValueError)),
            ):
                self.model(valid_texts, attention_mask)

        with self.assertRaisesRegex(ValueError, "maximum sequence length"):
            self.model([["x"] * 7])

        self.assertEqual(self.model([["x"] * 6]).hidden.shape, (1, 6, 8))

    def test_forward_input_errors_are_exact(self) -> None:
        invalid_text_cases = (
            (
                "abc",
                TypeError,
                "token_texts must be a sequence of token-text sequences",
            ),
            (
                [],
                ValueError,
                "token_texts must contain at least one batch row",
            ),
            (
                ["abc"],
                TypeError,
                "each token_texts row must be a sequence",
            ),
            (
                [[]],
                ValueError,
                "token_texts rows must contain at least one token",
            ),
            (
                [["a"], "b"],
                TypeError,
                "token_texts[1] must be a sequence",
            ),
            (
                [["a"], ["b", "c"]],
                ValueError,
                "token_texts must be rectangular",
            ),
            (
                [["a", 1]],
                TypeError,
                "token_texts[0][1] must be exact str, got int",
            ),
            (
                [["x"] * 7],
                ValueError,
                "token_texts sequence length exceeds the configured maximum "
                "sequence length 6",
            ),
        )
        for token_texts, exception_type, message in invalid_text_cases:
            with (
                self.subTest(message=message),
                self.assertRaises(exception_type) as raised,
            ):
                self.model(token_texts)
            self.assertEqual(str(raised.exception), message)

        valid_texts = [["a", "b"], ["c", "d"]]
        mask_cases = (
            (
                [[True, True], [True, True]],
                TypeError,
                "attention_mask must be a Tensor or None, got list",
            ),
            (
                torch.ones(2, 2),
                TypeError,
                "attention_mask must use torch.bool, got torch.float32",
            ),
            (
                torch.ones(2, dtype=torch.bool),
                ValueError,
                "attention_mask must have shape (2, 2), got (2,)",
            ),
        )
        for attention_mask, exception_type, message in mask_cases:
            with (
                self.subTest(message=message),
                self.assertRaises(exception_type) as raised,
            ):
                self.model(valid_texts, attention_mask)
            self.assertEqual(str(raised.exception), message)

    def test_prefix_kernel_rejects_invalid_tensor_geometry_dtype_mask_and_device(
        self,
    ) -> None:
        kernel = self.model.prefix_kernel
        valid_inputs = torch.zeros(1, 2, 8)
        valid_mask = torch.ones(1, 2, dtype=torch.bool)
        invalid_cases: tuple[tuple[object, object, type[Exception], str], ...] = (
            (
                object(),
                valid_mask,
                TypeError,
                "inputs must be a Tensor, got object",
            ),
            (
                torch.zeros(2, 8),
                valid_mask,
                ValueError,
                "inputs must have shape (batch, sequence, hidden_dim), got (2, 8)",
            ),
            (
                torch.zeros(0, 2, 8),
                torch.zeros(0, 2, dtype=torch.bool),
                ValueError,
                "inputs batch and sequence dimensions must both be greater than 0, "
                "got (0, 2)",
            ),
            (
                torch.zeros(1, 0, 8),
                torch.zeros(1, 0, dtype=torch.bool),
                ValueError,
                "inputs batch and sequence dimensions must both be greater than 0, "
                "got (1, 0)",
            ),
            (
                torch.zeros(1, 2, 8, dtype=torch.long),
                valid_mask,
                TypeError,
                "inputs must be floating point, got torch.int64",
            ),
            (
                torch.zeros(1, 2, 7),
                valid_mask,
                ValueError,
                "inputs final dimension must be 8, got 7",
            ),
            (
                valid_inputs,
                object(),
                TypeError,
                "attention_mask must be a Tensor, got object",
            ),
            (
                valid_inputs,
                torch.ones(2, dtype=torch.bool),
                ValueError,
                "attention_mask must have shape (1, 2), got (2,)",
            ),
            (
                valid_inputs,
                torch.ones(1, 2),
                TypeError,
                "attention_mask must use torch.bool, got torch.float32",
            ),
            (
                valid_inputs,
                torch.ones(1, 2, dtype=torch.bool, device="meta"),
                ValueError,
                "attention_mask and inputs must be on the same device, received "
                "meta and cpu",
            ),
        )
        for inputs, mask, exception, expected_message in invalid_cases:
            with self.subTest(inputs=inputs, mask=mask):
                with self.assertRaises(exception) as raised:
                    kernel(inputs, mask)
                self.assertEqual(str(raised.exception), expected_message)

    def test_component_and_kernel_derive_all_tensor_devices_from_runtime_state(
        self,
    ) -> None:
        observed: dict[str, torch.device] = {}

        def capture_encoder_device(_module, _args, kwargs) -> None:
            observed["encoder"] = kwargs["device"]

        handle = self.model.encoder.register_forward_pre_hook(
            capture_encoder_device,
            with_kwargs=True,
        )
        try:
            self.model([["a"]])
        finally:
            handle.remove()
        self.assertEqual(observed["encoder"], self.model.gamma.device)

        meta_model = self.config.build().to("meta")
        resolve_mask = meta_model._ByteContextualEmbedding__resolve_attention_mask
        generated_mask = resolve_mask(None, batch_size=1, sequence_length=2)
        moved_mask = resolve_mask(
            torch.ones(1, 2, dtype=torch.bool),
            batch_size=1,
            sequence_length=2,
        )
        self.assertEqual(generated_mask.device.type, "meta")
        self.assertEqual(moved_mask.device.type, "meta")

        meta_kernel = self.config.prefix_kernel_config.build().to("meta")
        output = meta_kernel(
            torch.zeros(1, 2, 8, device="meta"),
            torch.ones(1, 2, dtype=torch.bool, device="meta"),
        )
        self.assertEqual(output.device.type, "meta")

    def test_prefix_kernel_exact_scaled_relative_causal_masked_equation(self) -> None:
        kernel = CausalPrefixKernelConfig(
            hidden_dim=2,
            kernel_dim=2,
            relative_position_config=DynamicPositionalBiasConfig(
                num_heads=1,
                embedding_dim=2,
                max_positions=2,
            ),
        ).build()
        for projection in (
            kernel.query_projection,
            kernel.key_projection,
            kernel.value_projection,
            kernel.output_projection,
        ):
            self.assertIs(projection.bias_flag, False)
            self.assertIsNone(projection.bias_params)
        with torch.no_grad():
            identity = torch.eye(2)
            kernel.query_projection.weight_params.copy_(identity)
            kernel.key_projection.weight_params.copy_(identity)
            kernel.value_projection.weight_params.copy_(identity)
            kernel.output_projection.weight_params.copy_(identity)
            kernel.relative_position.relative_positional_embeddings.zero_()
            kernel.relative_position.relative_positional_embeddings[0, :, 0] = (
                torch.tensor([1.0, 0.0])
            )
            kernel.relative_position.relative_positional_embeddings[0, :, 2] = (
                torch.tensor([0.0, 2.0])
            )
        inputs = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]]])
        mask = torch.tensor([[True, False, True]])

        output = kernel(inputs, mask)

        final_logits = torch.tensor([6.0, 12.0]) / (2.0**0.5)
        final_weights = torch.softmax(final_logits, dim=0)
        expected_final = (
            final_weights[0] * inputs[0, 0] + final_weights[1] * inputs[0, 2]
        )
        expected = torch.stack(
            (
                inputs[0, 0],
                torch.zeros(2),
                expected_final,
            )
        ).unsqueeze(0)
        torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-7)

    def test_prefix_kernel_handles_fully_masked_rows_inside_a_mixed_batch(self) -> None:
        kernel = self.config.prefix_kernel_config.build()
        inputs = torch.randn(2, 3, 8)
        mask = torch.tensor([[False, False, False], [True, False, True]])

        output = kernel(inputs, mask)

        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(
            output[0],
            torch.zeros_like(output[0]),
            rtol=0,
            atol=0,
        )

    def test_shape_padding_batch_isolation_dtype_and_loss_ownership(self) -> None:
        token_texts = [["A", "", "ignored"], ["€", "B", "C"]]
        mask = torch.tensor([[True, True, False], [True, False, True]])

        state = self.model(token_texts, mask)
        all_valid = self.model([["A", "", "replacement"]])
        first_batch_alone = self.model([["A", "", "ignored"]], mask[:1])

        self.assertIsInstance(state, ByteContextualEmbeddingState)
        self.assertEqual(state.hidden.shape, (2, 3, 8))
        self.assertEqual(state.hidden.dtype, torch.float32)
        self.assertEqual(state.byte_moe_auxiliary_loss.shape, ())
        self.assertEqual(state.context_moe_auxiliary_loss.shape, ())
        torch.testing.assert_close(
            state.loss,
            state.byte_moe_auxiliary_loss + state.context_moe_auxiliary_loss,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            state.hidden[~mask],
            torch.zeros_like(state.hidden[~mask]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            state.hidden[0],
            first_batch_alone.hidden[0],
            rtol=1e-6,
            atol=1e-7,
        )
        torch.testing.assert_close(
            state.hidden[0, :2],
            all_valid.hidden[0, :2],
            rtol=0,
            atol=0,
        )

        double_model = self.config.build().double().eval()
        double_model.load_state_dict(self.model.state_dict(), strict=True)
        double_state = double_model([["a", "b"]])
        self.assertEqual(double_state.hidden.dtype, torch.float64)
        self.assertEqual(double_state.loss.dtype, torch.float64)

    def test_context_moe_receives_exact_joint_and_losses_keep_stage_ownership(
        self,
    ) -> None:
        captured: dict[str, torch.Tensor] = {}

        def capture_encoding(_module, _inputs, output) -> None:
            captured["bits"] = output[0].detach().clone()
            captured["byte_mask"] = output[1].detach().clone()

        def capture_byte_input(_module, inputs) -> None:
            captured["byte_input"] = inputs[0].detach().clone()

        def capture_byte(_module, _inputs, output) -> None:
            captured["byte_loss"] = output[2].detach().clone()

        def capture_prefix_input(_module, inputs) -> None:
            captured["positioned_draft"] = inputs[0].detach().clone()

        def capture_prefix_output(_module, _inputs, output) -> None:
            captured["prefix_context"] = output.detach().clone()

        def capture_context_input(_module, inputs) -> None:
            captured["context_input"] = inputs[0].detach().clone()

        def capture_context(_module, _inputs, output) -> None:
            captured["context_loss"] = output[2].detach().clone()

        handles = (
            self.model.encoder.register_forward_hook(capture_encoding),
            self.model.byte_moe.register_forward_pre_hook(capture_byte_input),
            self.model.byte_moe.register_forward_hook(capture_byte),
            self.model.prefix_kernel.register_forward_pre_hook(capture_prefix_input),
            self.model.prefix_kernel.register_forward_hook(capture_prefix_output),
            self.model.context_moe.register_forward_pre_hook(capture_context_input),
            self.model.context_moe.register_forward_hook(capture_context),
        )
        mask = torch.tensor([[True, False, True], [True, True, False]])
        try:
            state = self.model([["a", "pad", "b"], ["c", "d", "pad"]], mask)
        finally:
            for handle in handles:
                handle.remove()

        expected_byte_features = torch.cat(
            (captured["bits"], captured["byte_mask"]),
            dim=-1,
        ).to(dtype=self.model.gamma.dtype)
        expected_valid_byte_features = expected_byte_features.reshape(-1, 36)[
            mask.reshape(-1)
        ]
        torch.testing.assert_close(
            captured["byte_input"],
            expected_valid_byte_features,
            rtol=0,
            atol=0,
        )
        expected_joint = torch.cat(
            (captured["positioned_draft"], captured["prefix_context"]),
            dim=-1,
        )
        expected_valid_joint = expected_joint.reshape(-1, 16)[mask.reshape(-1)]
        torch.testing.assert_close(
            captured["context_input"],
            expected_valid_joint,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            state.byte_moe_auxiliary_loss,
            captured["byte_loss"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            state.context_moe_auxiliary_loss,
            captured["context_loss"],
            rtol=0,
            atol=0,
        )

    def test_text_below_false_mask_is_observationally_irrelevant(self) -> None:
        mask = torch.tensor([[True, False, True, False]])
        first = self.model([["a", "future", "b", "tail"]], mask)
        second = self.model([["a", "changed", "b", "different"]], mask)

        torch.testing.assert_close(first.hidden, second.hidden, rtol=0, atol=0)
        torch.testing.assert_close(
            first.byte_moe_auxiliary_loss,
            second.byte_moe_auxiliary_loss,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            first.context_moe_auxiliary_loss,
            second.context_moe_auxiliary_loss,
            rtol=0,
            atol=0,
        )

    def test_position_lookup_occurs_once_after_byte_moe_and_before_prefix(self) -> None:
        observations: dict[str, object] = {"position_calls": 0}

        def capture_byte_output(_module, _inputs, output) -> None:
            observations["draft_valid"] = output[0].detach().clone()

        def capture_position(_module, inputs, kwargs, output) -> None:
            observations["position_calls"] = int(observations["position_calls"]) + 1
            observations["position_ids"] = kwargs["positions"].detach().clone()
            observations["position_values"] = output.detach().clone()
            observations["position_input"] = inputs[0].detach().clone()

        def capture_prefix_input(_module, inputs) -> None:
            observations["prefix_input"] = inputs[0].detach().clone()

        handles = (
            self.model.byte_moe.register_forward_hook(capture_byte_output),
            self.model.positional_embedding.register_forward_hook(
                capture_position,
                with_kwargs=True,
            ),
            self.model.prefix_kernel.register_forward_pre_hook(capture_prefix_input),
        )
        try:
            mask = torch.tensor([[True, False, True]])
            self.model([["a", "pad", "b"]], mask)
        finally:
            for handle in handles:
                handle.remove()

        expected_positions = torch.tensor([[1, 0, 2]])
        self.assertEqual(observations["position_calls"], 1)
        torch.testing.assert_close(
            observations["position_ids"], expected_positions, rtol=0, atol=0
        )
        torch.testing.assert_close(
            observations["position_input"], expected_positions, rtol=0, atol=0
        )
        draft = torch.zeros(1, 3, 8)
        draft[:, mask[0]] = observations["draft_valid"]
        expected_prefix_input = (
            draft + observations["position_values"]
        ) * mask.unsqueeze(-1)
        torch.testing.assert_close(
            observations["prefix_input"],
            expected_prefix_input,
            rtol=0,
            atol=0,
        )

    def test_prefix_is_inclusive_causal_and_future_text_cannot_leak(self) -> None:
        with torch.no_grad():
            for moe in (self.model.byte_moe, self.model.context_moe):
                for parameter in moe.sampler.router.parameters():
                    parameter.zero_()
        original = self.model([["a", "b", "c", "d"]]).hidden
        future_changed = self.model([["a", "b", "changed", "future"]]).hidden
        earlier_changed = self.model([["changed", "b", "c", "d"]]).hidden

        torch.testing.assert_close(
            original[:, :2],
            future_changed[:, :2],
            rtol=0,
            atol=0,
        )
        self.assertFalse(torch.equal(original[:, 2:], future_changed[:, 2:]))
        self.assertFalse(torch.equal(original[:, 1:], earlier_changed[:, 1:]))

    def test_fully_padded_batch_is_finite_zero_and_skips_both_moes(self) -> None:
        calls = {"byte": 0, "context": 0}

        def count_byte(_module, _inputs) -> None:
            calls["byte"] += 1

        def count_context(_module, _inputs) -> None:
            calls["context"] += 1

        handles = (
            self.model.byte_moe.register_forward_pre_hook(count_byte),
            self.model.context_moe.register_forward_pre_hook(count_context),
        )
        try:
            state = self.model(
                [["ignored", "also ignored"], ["x", "y"]],
                torch.zeros(2, 2, dtype=torch.bool),
            )
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(calls, {"byte": 0, "context": 0})
        self.assertTrue(torch.isfinite(state.hidden).all())
        torch.testing.assert_close(
            state.hidden, torch.zeros_like(state.hidden), rtol=0, atol=0
        )
        torch.testing.assert_close(
            state.byte_moe_auxiliary_loss,
            torch.zeros_like(state.byte_moe_auxiliary_loss),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            state.context_moe_auxiliary_loss,
            torch.zeros_like(state.context_moe_auxiliary_loss),
            rtol=0,
            atol=0,
        )

    def test_zero_gamma_is_exact_identity_and_initial_gate_reaches_moe2(self) -> None:
        zero_model = contextual_config(residual_scale=0.0).build().eval()
        captured: dict[str, torch.Tensor] = {}

        def capture_prefix_input(_module, inputs) -> None:
            captured["x"] = inputs[0].detach().clone()

        handle = zero_model.prefix_kernel.register_forward_pre_hook(
            capture_prefix_input
        )
        try:
            zero_state = zero_model([["a", "b", "c"]])
        finally:
            handle.remove()
        torch.testing.assert_close(zero_state.hidden, captured["x"], rtol=0, atol=0)

        initial_capture: dict[str, torch.Tensor] = {}

        def capture_positioned_draft(_module, inputs) -> None:
            initial_capture["x"] = inputs[0].detach().clone()

        def capture_context_delta(_module, _inputs, output) -> None:
            initial_capture["delta"] = output[0].detach().clone()

        handles = (
            self.model.prefix_kernel.register_forward_pre_hook(
                capture_positioned_draft
            ),
            self.model.context_moe.register_forward_hook(capture_context_delta),
        )
        try:
            state = self.model([["a", "b", "c"], ["d", "e", "f"]])
        finally:
            for handle in handles:
                handle.remove()
        expected_hidden = initial_capture["x"] + self.model.gamma.detach() * (
            initial_capture["delta"].reshape(2, 3, 8)
        )
        torch.testing.assert_close(state.hidden, expected_hidden, rtol=0, atol=0)
        torch.testing.assert_close(
            self.model.gamma.detach(),
            torch.tensor(1e-3),
            rtol=0,
            atol=0,
        )
        state.hidden.square().sum().backward()

        self.assertIsNotNone(self.model.gamma.grad)
        self.assertNotEqual(float(self.model.gamma.grad), 0.0)
        for moe_name, moe in (
            ("byte_moe", self.model.byte_moe),
            ("context_moe", self.model.context_moe),
        ):
            adaptive_parameters = tuple(
                parameter
                for name, parameter in moe.named_parameters()
                if ".adaptive_behaviour." in name
            )
            with self.subTest(moe=moe_name):
                self.assertTrue(adaptive_parameters)
                self.assertTrue(
                    any(
                        parameter.grad is not None
                        and torch.count_nonzero(parameter.grad).item() > 0
                        for parameter in adaptive_parameters
                    )
                )

    def test_gradients_reach_both_moes_positions_kernel_relative_bias_and_gate(
        self,
    ) -> None:
        state = self.model(
            [["a", "b", "c"], ["d", "e", "f"]],
            torch.tensor([[True, True, True], [True, False, True]]),
        )
        objective = state.hidden.square().mean() + state.loss
        objective.backward()

        for name, parameter in (
            ("gamma", self.model.gamma),
            (
                "position",
                self.model.positional_embedding.embedding_model.weight,
            ),
            ("query", self.model.prefix_kernel.query_projection.weight_params),
            ("key", self.model.prefix_kernel.key_projection.weight_params),
            ("value", self.model.prefix_kernel.value_projection.weight_params),
            ("output", self.model.prefix_kernel.output_projection.weight_params),
            (
                "relative",
                self.model.prefix_kernel.relative_position.relative_positional_embeddings,
            ),
        ):
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)

        for name, moe in (
            ("byte_moe", self.model.byte_moe),
            ("context_moe", self.model.context_moe),
        ):
            router_parameters = tuple(moe.sampler.router.parameters())
            expert_parameters = tuple(moe.expert_modules.parameters())
            with self.subTest(module=name):
                self.assertTrue(
                    any(parameter.grad is not None for parameter in router_parameters)
                )
                self.assertTrue(
                    any(parameter.grad is not None for parameter in expert_parameters)
                )

    def test_strict_state_round_trip_and_optimizer_contain_only_model_state(
        self,
    ) -> None:
        source = self.model
        target = self.config.build().eval()
        token_texts = [["a", "", "\0"], ["€", "b", "c"]]
        mask = torch.tensor([[True, True, False], [True, False, True]])
        expected = source(token_texts, mask)

        incompatible = target.load_state_dict(source.state_dict(), strict=True)
        actual = target(token_texts, mask)

        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        torch.testing.assert_close(actual.hidden, expected.hidden, rtol=0, atol=0)
        torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
        state_names = tuple(source.state_dict())
        self.assertEqual(
            sum(isinstance(module, nn.Embedding) for module in source.modules()),
            1,
        )
        self.assertFalse(
            any("token" in name or "byte_mask" in name for name in state_names)
        )
        config_fields = {field.name for field in fields(type(self.config))}
        self.assertNotIn("vocabulary_size", config_fields)
        self.assertNotIn("token_ids", config_fields)
        optimizer = torch.optim.SGD(source.parameters(), lr=0.01)
        optimizer_parameters = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(optimizer_parameters, _parameter_ids(source))


if __name__ == "__main__":
    unittest.main()
