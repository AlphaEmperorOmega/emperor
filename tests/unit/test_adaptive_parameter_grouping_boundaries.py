import unittest
from copy import deepcopy
from dataclasses import dataclass

import torch

from emperor.attention import (
    MixerAttentionConfig,
    MixtureOfAttentionHeadsConfig,
    SelfAttentionConfig,
)
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AdaptiveParameterGroupingScopeOptions,
    AdditiveDynamicBiasConfig,
    WeightDecayScheduleOptions,
)
from emperor.config import ConfigBase
from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    MixtureOfExpertsConfig,
    RoutingInitializationMode,
)
from emperor.experts._layers.mixture import MixtureOfExperts
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    Layer,
    LayerConfig,
    LayerGateOptions,
    LayerNormPositionOptions,
    LayerStack,
    LayerStackConfig,
    LayerState,
    RecurrentLayer,
    RecurrentLayerConfig,
    RowLayout,
)
from emperor.linears import LinearLayerConfig
from emperor.memory import (
    GatedResidualDynamicMemoryConfig,
    MemoryPositionOptions,
)
from emperor.transformer import (
    FeedForwardConfig,
    Transformer,
    TransformerConfig,
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayerConfig,
)
from support.attention import build_attention_config


@dataclass
class _NestedConfig(ConfigBase):
    nested: object


def linear_stack(input_dim: int, output_dim: int) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=max(input_dim, output_dim),
        output_dim=output_dim,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=True,
            ),
        ),
    )


def grouped_linear_config(
    dimension: int,
    scope: AdaptiveParameterGroupingScopeOptions,
) -> AdaptiveLinearLayerConfig:
    return AdaptiveLinearLayerConfig(
        input_dim=dimension,
        output_dim=dimension,
        bias_flag=True,
        adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
            bias_config=AdditiveDynamicBiasConfig(
                input_dim=dimension,
                output_dim=dimension,
                decay_schedule=WeightDecayScheduleOptions.DISABLED,
                decay_rate=0.0,
                decay_warmup_batches=0,
                model_config=linear_stack(dimension, dimension),
            ),
            grouping_scope=scope,
            group_count=2,
        ),
    )


def grouped_stack(
    dimension: int,
    scope: AdaptiveParameterGroupingScopeOptions,
) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=dimension,
        hidden_dim=dimension,
        output_dim=dimension,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            input_dim=dimension,
            output_dim=dimension,
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=grouped_linear_config(dimension, scope),
        ),
    )


def memory_config(dimension: int) -> GatedResidualDynamicMemoryConfig:
    return GatedResidualDynamicMemoryConfig(
        input_dim=dimension,
        output_dim=dimension,
        memory_position_option=MemoryPositionOptions.BEFORE_AFFINE,
        test_time_training_learning_rate=None,
        test_time_training_num_inner_steps=None,
        model_config=linear_stack(dimension, dimension),
    )


def grouped_encoder_block_config() -> TransformerEncoderBlockLayerConfig:
    attention_config = build_attention_config(
        config_class=SelfAttentionConfig,
        batch_size=2,
        num_heads=2,
        embedding_dim=4,
        target_sequence_length=4,
        source_sequence_length=4,
    )
    attention_config.batch_first_flag = True
    encoder_config = TransformerEncoderLayerConfig(
        embedding_dim=4,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        dropout_probability=0.0,
        residual_config=None,
        attention_config=attention_config,
        feed_forward_config=FeedForwardConfig(
            input_dim=4,
            output_dim=4,
            stack_config=grouped_stack(
                4,
                AdaptiveParameterGroupingScopeOptions.SEQUENCE,
            ),
        ),
    )
    return TransformerEncoderBlockLayerConfig(
        input_dim=4,
        output_dim=4,
        activation=ActivationOptions.DISABLED,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        residual_config=None,
        dropout_probability=0.0,
        gate_config=None,
        halting_config=None,
        memory_config=None,
        layer_model_config=encoder_config,
    )


class AdaptiveParameterGroupingBoundaryTests(unittest.TestCase):
    def test_grouping_policy_does_not_expand_the_config_base_interface(self):
        config = grouped_stack(
            2,
            AdaptiveParameterGroupingScopeOptions.ROWS,
        )
        augmentation_config = (
            config.layer_config.layer_model_config.adaptive_augmentation_config
        )

        self.assertFalse(hasattr(config, "capability_paths"))
        self.assertFalse(
            hasattr(
                augmentation_config,
                "adaptive_parameter_grouping_enabled",
            )
        )

    def test_augmentation_config_deepcopy_preserves_grouping_without_aliasing(self):
        original = grouped_stack(
            2,
            AdaptiveParameterGroupingScopeOptions.ROWS,
        )

        cloned = deepcopy(original)
        cloned_augmentation = (
            cloned.layer_config.layer_model_config.adaptive_augmentation_config
        )
        original_augmentation = (
            original.layer_config.layer_model_config.adaptive_augmentation_config
        )

        self.assertIsNot(cloned_augmentation, original_augmentation)
        self.assertEqual(
            cloned_augmentation.grouping_scope,
            original_augmentation.grouping_scope,
        )
        self.assertEqual(
            cloned_augmentation.group_count,
            original_augmentation.group_count,
        )
        cloned_augmentation.group_count = 1
        self.assertEqual(original_augmentation.group_count, 2)

    def test_layer_and_shared_stack_controllers_reject_grouping_at_build_time(self):
        grouped_model = grouped_linear_config(
            2,
            AdaptiveParameterGroupingScopeOptions.ROWS,
        )
        layer_config = LayerConfig(
            input_dim=2,
            output_dim=2,
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=memory_config(2),
            layer_model_config=grouped_model,
        )
        with self.assertRaisesRegex(ValueError, "context sharing is restricted"):
            Layer(layer_config)

        stack_config = grouped_stack(
            2,
            AdaptiveParameterGroupingScopeOptions.ROWS,
        )
        stack_config.shared_memory_config = memory_config(2)
        with self.assertRaisesRegex(ValueError, "context sharing is restricted"):
            LayerStack(stack_config)

    def test_rank_two_layer_main_and_gate_both_support_explicit_row_grouping(self):
        gate_config = GateConfig(
            gate_dim=2,
            option=LayerGateOptions.ADDITION,
            activation=ActivationOptions.DISABLED,
            model_config=grouped_stack(
                2,
                AdaptiveParameterGroupingScopeOptions.ROWS,
            ),
        )
        model = Layer(
            LayerConfig(
                input_dim=2,
                output_dim=2,
                activation=ActivationOptions.DISABLED,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                residual_config=None,
                dropout_probability=0.0,
                gate_config=gate_config,
                halting_config=None,
                memory_config=None,
                layer_model_config=grouped_linear_config(
                    2,
                    AdaptiveParameterGroupingScopeOptions.ROWS,
                ),
            )
        )
        generated_context_shapes = []
        hooks = []
        for adaptive_leaf in (
            child
            for child in model.modules()
            if hasattr(child, "adaptive_behaviour")
            and child.adaptive_behaviour is not None
        ):
            hooks.append(
                adaptive_leaf.adaptive_behaviour.bias_model.register_forward_hook(
                    lambda _module, args, _output: generated_context_shapes.append(
                        tuple(args[1].shape)
                    )
                )
            )
        inputs = torch.randn(4, 2, requires_grad=True)

        try:
            output_state = model(
                LayerState(
                    hidden=inputs,
                    row_layout=RowLayout.rows(
                        4,
                        context_sharing_restricted=False,
                    ),
                )
            )
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(tuple(output_state.hidden.shape), (4, 2))
        self.assertEqual(generated_context_shapes, [(2, 2), (2, 2)])
        output_state.hidden.square().mean().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_rank_two_recurrent_block_and_gate_recompute_grouped_contexts_each_step(
        self,
    ) -> None:
        model = RecurrentLayer(
            RecurrentLayerConfig(
                input_dim=2,
                output_dim=2,
                max_steps=2,
                recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                block_config=grouped_stack(
                    2,
                    AdaptiveParameterGroupingScopeOptions.ROWS,
                ),
                gate_config=GateConfig(
                    gate_dim=2,
                    option=LayerGateOptions.ADDITION,
                    activation=ActivationOptions.DISABLED,
                    model_config=grouped_stack(
                        2,
                        AdaptiveParameterGroupingScopeOptions.ROWS,
                    ),
                ),
                residual_config=None,
                halting_config=None,
                memory_config=None,
            )
        )
        generated_contexts = []
        hooks = []
        for adaptive_leaf in (
            child
            for child in model.modules()
            if hasattr(child, "adaptive_behaviour")
            and child.adaptive_behaviour is not None
        ):
            hooks.append(
                adaptive_leaf.adaptive_behaviour.bias_model.register_forward_hook(
                    lambda _module, args, _output: generated_contexts.append(
                        args[1].detach().clone()
                    )
                )
            )
        inputs = torch.randn(4, 2, requires_grad=True)

        try:
            output_state = model(
                LayerState(
                    hidden=inputs,
                    row_layout=RowLayout.rows(
                        4,
                        context_sharing_restricted=False,
                    ),
                )
            )
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(tuple(output_state.hidden.shape), (4, 2))
        self.assertEqual(len(generated_contexts), 4)
        self.assertTrue(
            all(tuple(context.shape) == (2, 2) for context in generated_contexts)
        )
        output_state.hidden.square().mean().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_outer_recurrent_transformer_controller_closes_fresh_layout_bypass(self):
        config = RecurrentLayerConfig(
            input_dim=4,
            output_dim=4,
            max_steps=2,
            recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
            block_config=grouped_encoder_block_config(),
            gate_config=None,
            residual_config=None,
            halting_config=None,
            memory_config=memory_config(4),
        )

        with self.assertRaisesRegex(
            ValueError,
            "RecurrentLayerConfig cannot combine enabled adaptive parameter grouping",
        ):
            RecurrentLayer(config)

    def test_outer_transformer_rank_three_gate_rejects_grouped_adaptive_leaf(self):
        block_config = grouped_encoder_block_config()
        block_config.gate_config = GateConfig(
            gate_dim=4,
            option=LayerGateOptions.ADDITION,
            activation=ActivationOptions.DISABLED,
            model_config=grouped_stack(
                4,
                AdaptiveParameterGroupingScopeOptions.ROWS,
            ),
        )
        stack_config = LayerStackConfig(
            input_dim=4,
            hidden_dim=4,
            output_dim=4,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            shared_gate_config=None,
            shared_halting_config=None,
            shared_memory_config=None,
            layer_config=block_config,
        )

        with self.assertRaisesRegex(ValueError, "outer Transformer gate"):
            Transformer(
                TransformerConfig(
                    encoder_stack_config=stack_config,
                    decoder_stack_config=None,
                )
            )

    def test_transformer_grouping_scan_does_not_mask_missing_layer_config(self):
        stack_config = LayerStackConfig(
            input_dim=4,
            hidden_dim=4,
            output_dim=4,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            shared_gate_config=None,
            shared_halting_config=None,
            shared_memory_config=None,
            layer_config=None,
        )

        with self.assertRaisesRegex(ValueError, "layer_config is required"):
            Transformer(
                TransformerConfig(
                    encoder_stack_config=stack_config,
                    decoder_stack_config=None,
                )
            )

    def test_routed_expert_config_rejects_grouped_adaptive_leaves(self):
        config = MixtureOfExpertsConfig(
            input_dim=2,
            output_dim=2,
            top_k=1,
            num_experts=2,
            capacity_factor=0.0,
            dropped_token_behavior=DroppedTokenOptions.ZEROS,
            compute_expert_mixture_flag=True,
            weighted_parameters_flag=True,
            weighting_position_option=(ExpertWeightingPositionOptions.AFTER_EXPERTS),
            routing_initialization_mode=RoutingInitializationMode.DISABLED,
            sampler_config=None,
            expert_model_config=grouped_stack(
                2,
                AdaptiveParameterGroupingScopeOptions.ROWS,
            ),
        )

        with self.assertRaisesRegex(ValueError, "inside routed expert models"):
            MixtureOfExperts(config)

    def test_mixer_attention_rejects_grouping_before_building_mixing_model(self):
        config = MixerAttentionConfig(
            embedding_dim=2,
            sequence_length=4,
            batch_first_flag=True,
            mixing_model_config=grouped_stack(
                4,
                AdaptiveParameterGroupingScopeOptions.ROWS,
            ),
        )

        with self.assertRaisesRegex(ValueError, "not supported by MixerAttention"):
            config.build()

    def test_outer_validation_finds_grouping_in_extensible_config_containers(self):
        grouped_augmentation = grouped_linear_config(
            4,
            AdaptiveParameterGroupingScopeOptions.ROWS,
        ).adaptive_augmentation_config
        mixing_model_config = linear_stack(4, 4)
        mixing_model_config.layer_config.layer_model_config = _NestedConfig(
            nested={"custom": [grouped_augmentation]},
        )
        config = MixerAttentionConfig(
            embedding_dim=2,
            sequence_length=4,
            batch_first_flag=True,
            mixing_model_config=mixing_model_config,
        )

        with self.assertRaisesRegex(
            ValueError,
            r"nested\['custom'\]\[0\]",
        ):
            config.build()

    def test_outer_grouping_scan_does_not_mask_a_malformed_adaptive_config(self):
        mixing_model_config = grouped_stack(
            4,
            AdaptiveParameterGroupingScopeOptions.ROWS,
        )
        adaptive_linear_config = mixing_model_config.layer_config.layer_model_config
        adaptive_linear_config.adaptive_augmentation_config = LinearLayerConfig(
            input_dim=4,
            output_dim=4,
            bias_flag=True,
        )
        config = MixerAttentionConfig(
            embedding_dim=2,
            sequence_length=4,
            batch_first_flag=True,
            mixing_model_config=mixing_model_config,
        )

        with self.assertRaisesRegex(
            TypeError,
            "adaptive_augmentation_config must be AdaptiveParameterAugmentationConfig",
        ):
            config.build()

    def test_mixture_attention_rejects_grouping_before_router_construction(self):
        config = build_attention_config(
            config_class=MixtureOfAttentionHeadsConfig,
            batch_size=2,
            num_heads=2,
            embedding_dim=4,
            target_sequence_length=4,
            source_sequence_length=4,
            use_kv_expert_models_flag=False,
            experts_top_k=1,
            experts_num_experts=2,
        )
        config.projection_model_config = grouped_stack(
            4,
            AdaptiveParameterGroupingScopeOptions.SEQUENCE,
        )

        with self.assertRaisesRegex(
            ValueError,
            "not supported by mixture-of-attention-heads projections",
        ):
            config.build()


if __name__ == "__main__":
    unittest.main()
