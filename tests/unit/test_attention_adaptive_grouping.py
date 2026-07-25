import unittest
from dataclasses import replace

import torch
import torch.nn as nn

from emperor.attention import (
    IndependentAttentionConfig,
    SelfAttentionConfig,
    SelfAttentionProjectionStrategy,
)
from emperor.attention._runtime import (
    AttentionRuntimeLayout,
    MultiHeadAttentionInputs,
)
from emperor.attention._variants.self_attention.projection import (
    SelfAttentionProjector,
)
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AdaptiveParameterGroupingScopeOptions,
    AdditiveDynamicBiasConfig,
    DualModelDynamicWeightConfig,
    DynamicDepthOptions,
    SingleModelDynamicWeightConfig,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.augmentations.adaptive_parameters._linear_adapter import (
    AdaptiveLinearLayer,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    RowLayout,
)
from support.attention import build_attention_config, make_projection_model_config


def grouped_projection_model_config(group_count: int = 2) -> LayerStackConfig:
    return LayerStackConfig(
        hidden_dim=4,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=AdaptiveLinearLayerConfig(
                bias_flag=True,
                adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                    grouping_scope=AdaptiveParameterGroupingScopeOptions.SEQUENCE,
                    group_count=group_count,
                    bias_config=AdditiveDynamicBiasConfig(
                        decay_schedule=WeightDecayScheduleOptions.DISABLED,
                        decay_rate=0.0,
                        decay_warmup_batches=0,
                        model_config=make_projection_model_config(),
                    ),
                ),
            ),
        ),
    )


def grouped_weight_projection_model_config(group_count: int = 2) -> LayerStackConfig:
    return LayerStackConfig(
        hidden_dim=4,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=AdaptiveLinearLayerConfig(
                bias_flag=False,
                adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                    grouping_scope=AdaptiveParameterGroupingScopeOptions.SEQUENCE,
                    group_count=group_count,
                    weight_config=DualModelDynamicWeightConfig(
                        generator_depth=DynamicDepthOptions.DEPTH_OF_ONE,
                        decay_schedule=WeightDecayScheduleOptions.DISABLED,
                        decay_rate=0.0,
                        decay_warmup_batches=0,
                        normalization_option=WeightNormalizationOptions.DISABLED,
                        normalization_position_option=(
                            WeightNormalizationPositionOptions.DISABLED
                        ),
                        model_config=make_projection_model_config(),
                    ),
                ),
            ),
        ),
    )


def grouped_single_weight_projection_model_config(
    group_count: int = 2,
) -> LayerStackConfig:
    config = grouped_weight_projection_model_config(group_count)
    adaptive_config = config.layer_config.layer_model_config
    adaptive_config.adaptive_augmentation_config.weight_config = (
        SingleModelDynamicWeightConfig(
            generator_depth=DynamicDepthOptions.DEPTH_OF_ONE,
            decay_schedule=WeightDecayScheduleOptions.DISABLED,
            decay_rate=0.0,
            decay_warmup_batches=0,
            normalization_option=WeightNormalizationOptions.DISABLED,
            normalization_position_option=(WeightNormalizationPositionOptions.DISABLED),
            model_config=make_projection_model_config(),
        )
    )
    return config


class ProjectionStateSpy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layouts = []

    def forward(self, state):
        self.layouts.append(state.row_layout)
        return replace(state, hidden=state.hidden)


def adaptive_leaf(projection_model):
    return projection_model[0].model


class AttentionAdaptiveGroupingTests(unittest.TestCase):
    def test_projector_extracts_layout_from_runtime_context_for_all_projections(self):
        config = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=2,
            num_heads=2,
            embedding_dim=4,
            target_sequence_length=4,
            source_sequence_length=4,
            self_attention_projection_strategy=(
                SelfAttentionProjectionStrategy.SEPARATE
            ),
        )
        projector = SelfAttentionProjector(config)
        query_spy = ProjectionStateSpy()
        key_spy = ProjectionStateSpy()
        value_spy = ProjectionStateSpy()
        output_spy = ProjectionStateSpy()
        projector.query_model = query_spy
        projector.key_model = key_spy
        projector.value_model = value_spy
        projector.output_model = output_spy
        layout = RowLayout.sequence(
            leading_shape=(4, 2),
            batch_axis=1,
            sequence_axis=0,
            context_sharing_restricted=False,
        )
        runtime_layout = AttentionRuntimeLayout(
            batch_size=2,
            target_sequence_length=4,
            source_sequence_length=4,
            row_layout=layout,
        )
        tensor = torch.randn(4, 2, 4)

        projector.compute_qkv_projections(
            MultiHeadAttentionInputs(
                query=tensor,
                key=tensor,
                value=tensor,
                runtime_layout=runtime_layout,
            )
        )
        projector.compute_output_projection(tensor, runtime_layout=runtime_layout)

        for spy in (query_spy, key_spy, value_spy, output_spy):
            self.assertEqual(spy.layouts, [layout])

    def test_self_attention_builds_sequence_major_layout_and_inverts_padding_mask(self):
        config = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=2,
            num_heads=2,
            embedding_dim=4,
            target_sequence_length=4,
            source_sequence_length=4,
            self_attention_projection_strategy=(
                SelfAttentionProjectionStrategy.SEPARATE
            ),
        )
        config.batch_first_flag = True
        model = config.build().eval()
        spies = [ProjectionStateSpy() for _ in range(4)]
        (
            model.projector.query_model,
            model.projector.key_model,
            model.projector.value_model,
            model.projector.output_model,
        ) = spies
        inputs = torch.randn(2, 4, 4)
        padding_mask = torch.tensor(
            [[False, False, True, True], [False, True, False, True]]
        )

        output, _weights, _loss = model(
            inputs,
            inputs,
            inputs,
            k_padding_mask=padding_mask,
        )

        self.assertEqual(tuple(output.shape), (2, 4, 4))
        expected_valid_rows = torch.tensor(
            [True, True, True, False, False, True, False, False]
        )
        for spy in spies:
            self.assertEqual(len(spy.layouts), 1)
            layout = spy.layouts[0]
            self.assertEqual(layout.leading_shape, (4, 2))
            self.assertEqual(layout.batch_axis, 1)
            self.assertEqual(layout.sequence_axis, 0)
            self.assertFalse(layout.context_sharing_restricted)
            torch.testing.assert_close(layout.valid_rows, expected_valid_rows)

    def test_float_padding_layout_excludes_only_hard_negative_infinity(self):
        config = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=1,
            num_heads=2,
            embedding_dim=4,
            target_sequence_length=4,
            source_sequence_length=4,
            self_attention_projection_strategy=(
                SelfAttentionProjectionStrategy.SEPARATE
            ),
        )
        config.batch_first_flag = True
        model = config.build().eval()
        spies = [ProjectionStateSpy() for _ in range(4)]
        (
            model.projector.query_model,
            model.projector.key_model,
            model.projector.value_model,
            model.projector.output_model,
        ) = spies
        inputs = torch.randn(1, 4, 4)
        padding_mask = torch.tensor([[0.0, -0.25, -torch.inf, 0.0]])

        output, _weights, _loss = model(
            inputs,
            inputs,
            inputs,
            k_padding_mask=padding_mask,
        )

        self.assertEqual(tuple(output.shape), (1, 4, 4))
        expected_valid_rows = torch.tensor([True, True, False, True])
        for spy in spies:
            self.assertEqual(len(spy.layouts), 1)
            torch.testing.assert_close(
                spy.layouts[0].valid_rows,
                expected_valid_rows,
            )

    def test_grouped_self_attention_generates_batch_times_group_count_contexts(self):
        config = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=2,
            num_heads=2,
            embedding_dim=4,
            target_sequence_length=4,
            source_sequence_length=4,
            self_attention_projection_strategy=(
                SelfAttentionProjectionStrategy.SEPARATE
            ),
        )
        config.batch_first_flag = True
        config.projection_model_config = grouped_projection_model_config(group_count=2)
        model = config.build()
        generator_shapes = []
        hooks = []
        for projection_model in (
            model.projector.query_model,
            model.projector.key_model,
            model.projector.value_model,
            model.projector.output_model,
        ):
            generator = (
                adaptive_leaf(projection_model)
                .adaptive_behaviour.bias_model.model[0]
                .model
            )
            hooks.append(
                generator.register_forward_hook(
                    lambda _module, args, _output: generator_shapes.append(
                        tuple(args[0].shape)
                    )
                )
            )
        inputs = torch.randn(2, 4, 4, requires_grad=True)

        try:
            output, _weights, loss = model(inputs, inputs, inputs)
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(tuple(output.shape), (2, 4, 4))
        self.assertEqual(generator_shapes, [(4, 4)] * 4)
        objective = output.square().mean()
        if loss is not None:
            objective = objective + loss
        objective.backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_grouped_self_attention_supports_sequence_major_and_unbatched_inputs(self):
        cases = (
            (False, torch.randn(4, 2, 4, requires_grad=True), (4, 2, 4), 4),
            (False, torch.randn(4, 4, requires_grad=True), (4, 4), 2),
        )

        for batch_first, inputs, expected_shape, context_count in cases:
            with self.subTest(expected_shape=expected_shape):
                config = build_attention_config(
                    config_class=SelfAttentionConfig,
                    batch_size=2,
                    num_heads=2,
                    embedding_dim=4,
                    target_sequence_length=4,
                    source_sequence_length=4,
                    self_attention_projection_strategy=(
                        SelfAttentionProjectionStrategy.SEPARATE
                    ),
                )
                config.batch_first_flag = batch_first
                config.projection_model_config = grouped_projection_model_config()
                model = config.build()
                generated_context_shapes = []
                hooks = []
                for projection_model in (
                    model.projector.query_model,
                    model.projector.key_model,
                    model.projector.value_model,
                    model.projector.output_model,
                ):
                    bias_model = adaptive_leaf(
                        projection_model
                    ).adaptive_behaviour.bias_model
                    hooks.append(
                        bias_model.register_forward_hook(
                            lambda _module, args, _output, generated_context_shapes=generated_context_shapes: (
                                generated_context_shapes.append(tuple(args[1].shape))
                            )
                        )
                    )

                try:
                    output, _weights, loss = model(inputs, inputs, inputs)
                finally:
                    for hook in hooks:
                        hook.remove()

                self.assertEqual(tuple(output.shape), expected_shape)
                self.assertEqual(
                    generated_context_shapes,
                    [(context_count, 4)] * 4,
                )
                objective = output.square().mean()
                if loss is not None:
                    objective = objective + loss
                objective.backward()
                self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_padding_values_do_not_enter_group_contexts_or_valid_outputs(self):
        config = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=2,
            num_heads=2,
            embedding_dim=4,
            target_sequence_length=4,
            source_sequence_length=4,
            self_attention_projection_strategy=(
                SelfAttentionProjectionStrategy.SEPARATE
            ),
        )
        config.batch_first_flag = True
        config.projection_model_config = grouped_projection_model_config()
        model = config.build().eval()
        padding_mask = torch.tensor(
            [[False, False, True, True], [False, True, False, True]]
        )
        base_values = torch.tensor(
            [
                [[1.0] * 4, [2.0] * 4, [0.0] * 4, [0.0] * 4],
                [[10.0] * 4, [0.0] * 4, [20.0] * 4, [0.0] * 4],
            ]
        )
        altered_values = base_values.clone()
        altered_values[padding_mask] = torch.tensor(
            [[1000.0] * 4, [-1000.0] * 4, [500.0] * 4, [-500.0] * 4]
        )
        altered_values.requires_grad_()
        observed_query_contexts = []
        query_bias_model = adaptive_leaf(
            model.projector.query_model
        ).adaptive_behaviour.bias_model
        hook = query_bias_model.register_forward_hook(
            lambda _module, args, _output: observed_query_contexts.append(
                args[1].detach().clone()
            )
        )

        try:
            base_output, _weights, _loss = model(
                base_values,
                base_values,
                base_values,
                k_padding_mask=padding_mask,
            )
            altered_output, _weights, loss = model(
                altered_values,
                altered_values,
                altered_values,
                k_padding_mask=padding_mask,
            )
        finally:
            hook.remove()

        expected_contexts = torch.tensor([[3.0] * 4, [0.0] * 4, [10.0] * 4, [20.0] * 4])
        self.assertEqual(len(observed_query_contexts), 2)
        torch.testing.assert_close(observed_query_contexts[0], expected_contexts)
        torch.testing.assert_close(observed_query_contexts[1], expected_contexts)
        torch.testing.assert_close(
            base_output[~padding_mask],
            altered_output[~padding_mask],
        )
        objective = altered_output[~padding_mask].square().mean()
        if loss is not None:
            objective = objective + loss
        objective.backward()
        self.assertTrue(torch.isfinite(altered_values.grad).all())

    def test_post_projection_key_value_extensions_preserve_grouping_contract(self):
        config = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=2,
            num_heads=2,
            embedding_dim=4,
            target_sequence_length=4,
            source_sequence_length=4,
            add_key_value_bias_flag=True,
            zero_attention_flag=True,
            self_attention_projection_strategy=(
                SelfAttentionProjectionStrategy.SEPARATE
            ),
        )
        config.batch_first_flag = True
        config.projection_model_config = grouped_projection_model_config()
        model = config.build()
        generated_context_shapes = []
        hooks = []
        for projection_model in (
            model.projector.query_model,
            model.projector.key_model,
            model.projector.value_model,
            model.projector.output_model,
        ):
            bias_model = adaptive_leaf(projection_model).adaptive_behaviour.bias_model
            hooks.append(
                bias_model.register_forward_hook(
                    lambda _module, args, _output: generated_context_shapes.append(
                        tuple(args[1].shape)
                    )
                )
            )
        inputs = torch.randn(2, 4, 4, requires_grad=True)

        try:
            output, _weights, loss = model(inputs, inputs, inputs)
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(tuple(output.shape), (2, 4, 4))
        self.assertEqual(generated_context_shapes, [(4, 4)] * 4)
        objective = output.square().mean()
        if loss is not None:
            objective = objective + loss
        objective.backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_all_projection_strategies_keep_dynamic_weights_context_batched(self):
        strategy_shapes = (
            (
                SelfAttentionProjectionStrategy.FUSED,
                [(4, 4, 12), (4, 4, 4)],
            ),
            (
                SelfAttentionProjectionStrategy.FUSED_KEY_VALUE,
                [(4, 4, 4), (4, 4, 8), (4, 4, 4)],
            ),
            (
                SelfAttentionProjectionStrategy.SEPARATE,
                [(4, 4, 4)] * 4,
            ),
        )

        for strategy, expected_shapes in strategy_shapes:
            with self.subTest(strategy=strategy):
                config = build_attention_config(
                    config_class=SelfAttentionConfig,
                    batch_size=2,
                    num_heads=2,
                    embedding_dim=4,
                    target_sequence_length=4,
                    source_sequence_length=4,
                    self_attention_projection_strategy=strategy,
                )
                config.batch_first_flag = True
                config.projection_model_config = grouped_weight_projection_model_config(
                    group_count=2
                )
                model = config.build()
                generated_weight_shapes = []
                hooks = []
                for leaf in model.projector.modules():
                    if not isinstance(leaf, AdaptiveLinearLayer):
                        continue
                    hooks.append(
                        leaf.adaptive_behaviour.weight_model.register_forward_hook(
                            lambda _module, _args, output, generated_weight_shapes=generated_weight_shapes: (
                                generated_weight_shapes.append(tuple(output.shape))
                            )
                        )
                    )
                inputs = torch.randn(2, 4, 4, requires_grad=True)

                try:
                    output, _weights, loss = model(inputs, inputs, inputs)
                finally:
                    for hook in hooks:
                        hook.remove()

                self.assertEqual(tuple(output.shape), (2, 4, 4))
                self.assertEqual(generated_weight_shapes, expected_shapes)
                objective = output.square().mean()
                if loss is not None:
                    objective = objective + loss
                objective.backward()
                self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_single_model_weight_rejects_rectangular_fused_projections(self):
        for strategy in (
            SelfAttentionProjectionStrategy.FUSED,
            SelfAttentionProjectionStrategy.FUSED_KEY_VALUE,
        ):
            with self.subTest(strategy=strategy):
                config = build_attention_config(
                    config_class=SelfAttentionConfig,
                    batch_size=2,
                    num_heads=2,
                    embedding_dim=4,
                    target_sequence_length=4,
                    source_sequence_length=4,
                    self_attention_projection_strategy=strategy,
                )
                config.projection_model_config = (
                    grouped_single_weight_projection_model_config()
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "requires input_dim == output_dim",
                ):
                    config.build()

    def test_restrictive_attention_paths_fail_before_grouped_generator(self):
        inputs = torch.randn(2, 4, 4)
        cases = []

        causal_config = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=2,
            num_heads=2,
            embedding_dim=4,
            target_sequence_length=4,
            source_sequence_length=4,
            causal_attention_mask_flag=True,
            self_attention_projection_strategy=(
                SelfAttentionProjectionStrategy.SEPARATE
            ),
        )
        causal_config.batch_first_flag = True
        causal_config.projection_model_config = grouped_projection_model_config()
        cases.append(("causal", causal_config.build(), inputs, inputs, None))

        masked_config = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=2,
            num_heads=2,
            embedding_dim=4,
            target_sequence_length=4,
            source_sequence_length=4,
            self_attention_projection_strategy=(
                SelfAttentionProjectionStrategy.SEPARATE
            ),
        )
        masked_config.batch_first_flag = True
        masked_config.projection_model_config = grouped_projection_model_config()
        cases.append(
            (
                "explicit mask",
                masked_config.build(),
                inputs,
                inputs,
                torch.zeros(4, 4),
            )
        )

        cross_config = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=2,
            num_heads=2,
            embedding_dim=4,
            target_sequence_length=4,
            source_sequence_length=4,
        )
        cross_config.batch_first_flag = True
        cross_config.projection_model_config = grouped_projection_model_config()
        cases.append(
            (
                "cross attention",
                cross_config.build(),
                inputs,
                torch.randn(2, 4, 4),
                None,
            )
        )

        for name, model, query, key_value, attention_mask in cases:
            calls = []
            generator = (
                adaptive_leaf(model.projector.query_model)
                .adaptive_behaviour.bias_model.model[0]
                .model
            )
            hook = generator.register_forward_hook(
                lambda *_args, calls=calls: calls.append(True)
            )
            try:
                with self.subTest(name=name):
                    with self.assertRaisesRegex(
                        ValueError,
                        "context sharing is restricted",
                    ):
                        model(
                            query,
                            key_value,
                            key_value,
                            attention_mask=attention_mask,
                        )
                    self.assertEqual(calls, [])
            finally:
                hook.remove()

    def test_static_and_fused_cross_inputs_reject_before_grouped_generators(self):
        inputs = torch.randn(2, 4, 4)
        cases = []

        static_config = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=2,
            num_heads=2,
            embedding_dim=4,
            target_sequence_length=4,
            source_sequence_length=4,
            self_attention_projection_strategy=(
                SelfAttentionProjectionStrategy.SEPARATE
            ),
        )
        static_config.batch_first_flag = True
        static_config.projection_model_config = grouped_projection_model_config()
        static_values = torch.randn(4, 4, 2)
        cases.append(
            (
                "static key/value",
                static_config.build(),
                inputs,
                inputs,
                {"static_k": static_values, "static_v": static_values.clone()},
            )
        )

        fused_cross_config = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=2,
            num_heads=2,
            embedding_dim=4,
            target_sequence_length=4,
            source_sequence_length=4,
            self_attention_projection_strategy=(
                SelfAttentionProjectionStrategy.FUSED_KEY_VALUE
            ),
        )
        fused_cross_config.batch_first_flag = True
        fused_cross_config.projection_model_config = grouped_projection_model_config()
        cases.append(
            (
                "fused key/value cross input",
                fused_cross_config.build(),
                inputs,
                torch.randn_like(inputs),
                {},
            )
        )

        for name, model, query, key_value, keywords in cases:
            calls = []
            hooks = []
            for leaf in model.projector.modules():
                if not isinstance(leaf, AdaptiveLinearLayer):
                    continue
                generator = leaf.adaptive_behaviour.bias_model.model[0].model
                hooks.append(
                    generator.register_forward_hook(
                        lambda *_args, calls=calls: calls.append(True)
                    )
                )
            try:
                with self.subTest(name=name):
                    with self.assertRaisesRegex(
                        ValueError,
                        "context sharing is restricted",
                    ):
                        model(query, key_value, key_value, **keywords)
                    self.assertEqual(calls, [])
            finally:
                for hook in hooks:
                    hook.remove()


if __name__ == "__main__":
    unittest.main()
