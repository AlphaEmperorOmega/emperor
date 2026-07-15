import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch
from emperor.attention import (
    IndependentAttention,
    IndependentAttentionConfig,
    MixtureOfAttentionHeads,
    MixtureOfAttentionHeadsConfig,
    SelfAttention,
    SelfAttentionConfig,
)
from emperor.attention.core.config import MultiHeadAttentionConfig
from emperor.attention.core.handlers.batch import BatchDimensionManager
from emperor.attention.core.handlers.bias import KeyValueBias
from emperor.attention.core.handlers.mask import Mask
from emperor.attention.core.handlers.processor import ProcessorBase
from emperor.attention.core.handlers.projector import ProjectorBase
from emperor.attention.core.variants.independent_attention.validator import (
    IndependentAttentionValidator,
)

from support.attention import build_attention_config


@dataclass(frozen=True, slots=True)
class ForwardCase:
    label: str
    config_class: type[MultiHeadAttentionConfig]
    query_key_projection_dim: int = 4
    value_projection_dim: int = 4
    target_sequence_length: int = 3
    source_sequence_length: int = 4
    projection_kind: str = "base"
    mask_kind: str = "none"
    add_key_value_bias_flag: bool = False
    zero_attention_flag: bool = False
    causal_attention_mask_flag: bool = False
    return_attention_weights_flag: bool = False
    average_attention_weights_flag: bool = False


FORWARD_CASES = (
    ForwardCase(
        label="independent unequal projections",
        config_class=IndependentAttentionConfig,
        value_projection_dim=6,
    ),
    ForwardCase(
        label="independent additive mask and bias",
        config_class=IndependentAttentionConfig,
        mask_kind="additive",
        add_key_value_bias_flag=True,
    ),
    ForwardCase(
        label="independent padding mask and zero",
        config_class=IndependentAttentionConfig,
        mask_kind="padding",
        zero_attention_flag=True,
    ),
    ForwardCase(
        label="independent combined mask, bias, and zero",
        config_class=IndependentAttentionConfig,
        mask_kind="combined",
        add_key_value_bias_flag=True,
        zero_attention_flag=True,
    ),
    ForwardCase(
        label="self returned averaged weights",
        config_class=SelfAttentionConfig,
        source_sequence_length=3,
        return_attention_weights_flag=True,
        average_attention_weights_flag=True,
    ),
    ForwardCase(
        label="self returned per-head weights with bias",
        config_class=SelfAttentionConfig,
        source_sequence_length=3,
        add_key_value_bias_flag=True,
        return_attention_weights_flag=True,
    ),
    ForwardCase(
        label="self causal zero attention",
        config_class=SelfAttentionConfig,
        source_sequence_length=3,
        mask_kind="causal",
        zero_attention_flag=True,
        causal_attention_mask_flag=True,
    ),
    ForwardCase(
        label="self bias and zero attention",
        config_class=SelfAttentionConfig,
        source_sequence_length=3,
        add_key_value_bias_flag=True,
        zero_attention_flag=True,
    ),
    ForwardCase(
        label="mixture shared key/value",
        config_class=MixtureOfAttentionHeadsConfig,
    ),
    ForwardCase(
        label="mixture boolean mask and bias",
        config_class=MixtureOfAttentionHeadsConfig,
        mask_kind="boolean",
        add_key_value_bias_flag=True,
    ),
    ForwardCase(
        label="mixture padding mask and zero",
        config_class=MixtureOfAttentionHeadsConfig,
        mask_kind="padding",
        zero_attention_flag=True,
    ),
    ForwardCase(
        label="mixture combined mask, bias, and zero",
        config_class=MixtureOfAttentionHeadsConfig,
        mask_kind="combined",
        add_key_value_bias_flag=True,
        zero_attention_flag=True,
    ),
)


PROJECTION_CASES = (
    ForwardCase(
        label="self adaptive fused projection",
        config_class=SelfAttentionConfig,
        source_sequence_length=3,
        projection_kind="adaptive",
    ),
    ForwardCase(
        label="independent adaptive unequal projections",
        config_class=IndependentAttentionConfig,
        value_projection_dim=6,
        projection_kind="adaptive",
    ),
)


EXPECTED_LAYER_TYPES = {
    SelfAttentionConfig: SelfAttention,
    IndependentAttentionConfig: IndependentAttention,
    MixtureOfAttentionHeadsConfig: MixtureOfAttentionHeads,
}


class TestAttention(unittest.TestCase):
    batch_size = 2
    num_heads = 2
    embedding_dim = 4

    def config(self, case: ForwardCase):
        return build_attention_config(
            config_class=case.config_class,
            batch_size=self.batch_size,
            num_heads=self.num_heads,
            embedding_dim=self.embedding_dim,
            query_key_projection_dim=case.query_key_projection_dim,
            value_projection_dim=case.value_projection_dim,
            target_sequence_length=case.target_sequence_length,
            source_sequence_length=case.source_sequence_length,
            projection_kind=case.projection_kind,
            add_key_value_bias_flag=case.add_key_value_bias_flag,
            zero_attention_flag=case.zero_attention_flag,
            causal_attention_mask_flag=case.causal_attention_mask_flag,
            return_attention_weights_flag=case.return_attention_weights_flag,
            average_attention_weights_flag=case.average_attention_weights_flag,
            experts_top_k=1,
            experts_num_experts=2,
            experts_compute_expert_mixture_flag=False,
            experts_stack_num_layers=1,
        )

    def qkv(self, case: ForwardCase, *, requires_grad=False):
        query = torch.randn(
            case.target_sequence_length,
            self.batch_size,
            self.embedding_dim,
            requires_grad=requires_grad,
        )
        if case.config_class is SelfAttentionConfig:
            return query, query, query
        key = torch.randn(
            case.source_sequence_length,
            self.batch_size,
            self.embedding_dim,
            requires_grad=requires_grad,
        )
        value = torch.randn(
            case.source_sequence_length,
            self.batch_size,
            self.embedding_dim,
            requires_grad=requires_grad,
        )
        return query, key, value

    def masks(self, case: ForwardCase):
        key_padding_mask = None
        attention_mask = None
        if case.mask_kind in ("padding", "combined"):
            key_padding_mask = torch.zeros(
                self.batch_size,
                case.source_sequence_length,
                dtype=torch.bool,
            )
            key_padding_mask[:, -1] = True
        if case.mask_kind in ("boolean", "combined", "causal"):
            attention_mask = torch.zeros(
                case.target_sequence_length,
                case.source_sequence_length,
                dtype=torch.bool,
            )
            attention_mask[:, -1] = True
        elif case.mask_kind == "additive":
            attention_mask = torch.zeros(
                case.target_sequence_length,
                case.source_sequence_length,
            )
            attention_mask[:, -1] = float("-inf")
        return key_padding_mask, attention_mask

    def assert_config_matches_case(self, cfg, case: ForwardCase):
        self.assertIs(type(cfg), case.config_class)
        self.assertEqual(cfg.query_key_projection_dim, case.query_key_projection_dim)
        self.assertEqual(cfg.value_projection_dim, case.value_projection_dim)
        self.assertEqual(cfg.target_sequence_length, case.target_sequence_length)
        self.assertEqual(cfg.source_sequence_length, case.source_sequence_length)
        self.assertEqual(cfg.add_key_value_bias_flag, case.add_key_value_bias_flag)
        self.assertEqual(cfg.zero_attention_flag, case.zero_attention_flag)
        self.assertEqual(
            cfg.causal_attention_mask_flag,
            case.causal_attention_mask_flag,
        )
        self.assertEqual(
            cfg.return_attention_weights_flag,
            case.return_attention_weights_flag,
        )
        self.assertEqual(
            cfg.average_attention_weights_flag,
            case.average_attention_weights_flag,
        )

    def test_default_components_are_wired_to_the_validated_config(self):
        cfg = self.config(FORWARD_CASES[0])
        model = cfg.build()

        self.assertEqual(model.batch_size, cfg.batch_size)
        self.assertEqual(model.num_heads, cfg.num_heads)
        self.assertEqual(model.embedding_dim, cfg.embedding_dim)
        self.assertEqual(model.target_dtype, cfg.target_dtype)
        self.assertEqual(model.target_sequence_length, cfg.target_sequence_length)
        self.assertEqual(model.source_sequence_length, cfg.source_sequence_length)
        self.assertEqual(model.dropout_probability, cfg.dropout_probability)
        self.assertEqual(model.query_key_projection_dim, cfg.query_key_projection_dim)
        self.assertEqual(model.value_projection_dim, cfg.value_projection_dim)
        self.assertIs(model.zero_attention_flag, cfg.zero_attention_flag)
        self.assertIs(
            model.add_key_value_bias_flag,
            cfg.add_key_value_bias_flag,
        )
        self.assertIs(
            model.causal_attention_mask_flag,
            cfg.causal_attention_mask_flag,
        )
        self.assertIs(
            model.average_attention_weights_flag,
            cfg.average_attention_weights_flag,
        )
        self.assertIs(
            model.return_attention_weights_flag,
            cfg.return_attention_weights_flag,
        )
        self.assertIs(model.batch_first_flag, cfg.batch_first_flag)
        self.assertEqual(model.head_dim, cfg.embedding_dim // cfg.num_heads)
        self.assertIs(type(model.head_dim), int)
        self.assertIs(model.VALIDATOR, IndependentAttentionValidator)
        self.assertIsInstance(model.masks, Mask)
        self.assertIsInstance(model.projector, ProjectorBase)
        self.assertIsInstance(model.processor, ProcessorBase)
        self.assertIsInstance(model.bias, KeyValueBias)
        self.assertIsInstance(model.batch_manager, BatchDimensionManager)

    def test_model_config_wrapper_and_explicit_overrides_are_honoured(self):
        base = self.config(FORWARD_CASES[0])
        wrapper = SimpleNamespace(multi_head_attention_model_config=base)
        overrides = IndependentAttentionConfig(
            batch_size=3,
            dropout_probability=0.25,
        )

        model = IndependentAttention(wrapper, overrides)

        self.assertIsNot(model.cfg, base)
        self.assertEqual(model.batch_size, 3)
        self.assertEqual(model.dropout_probability, 0.25)
        self.assertEqual(model.embedding_dim, base.embedding_dim)

    def test_forward_forwards_static_and_runtime_contracts_to_extensions(self):
        cfg = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=5,
            num_heads=2,
            embedding_dim=4,
            query_key_projection_dim=4,
            value_projection_dim=4,
            target_sequence_length=3,
            source_sequence_length=5,
        )
        cfg.batch_first_flag = False
        model = cfg.build().eval()
        query = torch.randn(3, 2, 4)
        key = torch.randn(4, 2, 4)
        value = torch.randn(4, 2, 4)
        static_keys = torch.randn(4, 5, 2)
        static_values = torch.randn(4, 5, 2)

        with (
            patch.object(
                model.VALIDATOR,
                "validate_static_key_value_inputs",
                wraps=model.VALIDATOR.validate_static_key_value_inputs,
            ) as validate_static,
            patch.object(
                model.bias,
                "add_kv_learnable_bias_vectors",
                wraps=model.bias.add_kv_learnable_bias_vectors,
            ) as add_bias,
            patch.object(
                model.zero_attention,
                "add_zero_attention",
                wraps=model.zero_attention.add_zero_attention,
            ) as add_zero,
        ):
            output, _, _ = model(
                query,
                key,
                value,
                static_k=static_keys,
                static_v=static_values,
            )

        self.assertEqual(output.shape, query.shape)
        static_args = validate_static.call_args.args
        self.assertIs(static_args[0], model)
        self.assertIs(static_args[2], static_keys)
        self.assertIs(static_args[3], static_values)
        self.assertIsNotNone(static_args[4])
        self.assertEqual(static_args[4].batch_size, 2)

        bias_runtime_shape = add_bias.call_args.args[2]
        zero_runtime_shape = add_zero.call_args.args[2]
        self.assertIsNotNone(bias_runtime_shape)
        self.assertIsNotNone(zero_runtime_shape)
        self.assertEqual(bias_runtime_shape.batch_size, 2)
        self.assertEqual(bias_runtime_shape.source_sequence_length, 5)
        self.assertIs(zero_runtime_shape, bias_runtime_shape)

    def test_self_processor_reuses_the_layer_reshaper(self):
        cfg = self.config(
            ForwardCase(
                label="self wiring",
                config_class=SelfAttentionConfig,
                source_sequence_length=3,
            )
        )

        model = cfg.build()

        self.assertIs(model.processor.reshaper, model.reshaper)

    def test_config_build_dispatch_maps_every_leaf_to_its_layer(self):
        for config_class, expected_layer_type in EXPECTED_LAYER_TYPES.items():
            case = ForwardCase(
                label=config_class.__name__,
                config_class=config_class,
                source_sequence_length=(
                    3 if config_class is SelfAttentionConfig else 4
                ),
            )
            with self.subTest(case=case):
                cfg = self.config(case)

                self.assertIsInstance(cfg.build(), expected_layer_type)

    def test_orthogonal_forward_matrix_has_truthful_config_labels(self):
        for case in FORWARD_CASES:
            with self.subTest(case=case):
                torch.manual_seed(10)
                cfg = self.config(case)
                self.assert_config_matches_case(cfg, case)
                model = cfg.build().eval()
                query, key, value = self.qkv(case)
                key_padding_mask, attention_mask = self.masks(case)

                output, weights, auxiliary_loss = model(
                    query,
                    key,
                    value,
                    key_padding_mask,
                    attention_mask,
                )

                self.assertEqual(output.shape, query.shape)
                self.assertTrue(torch.isfinite(output).all())
                synthetic_positions = int(case.add_key_value_bias_flag) + int(
                    case.zero_attention_flag
                )
                if case.return_attention_weights_flag:
                    expected_source = case.source_sequence_length + synthetic_positions
                    expected_weight_shape = (
                        (
                            self.batch_size,
                            case.target_sequence_length,
                            expected_source,
                        )
                        if case.average_attention_weights_flag
                        else (
                            self.batch_size,
                            self.num_heads,
                            case.target_sequence_length,
                            expected_source,
                        )
                    )
                    self.assertEqual(weights.shape, expected_weight_shape)
                else:
                    self.assertIsNone(weights)
                if case.config_class is MixtureOfAttentionHeadsConfig:
                    self.assertIsInstance(auxiliary_loss, torch.Tensor)
                    self.assertTrue(torch.isfinite(auxiliary_loss).all())
                else:
                    self.assertIsNone(auxiliary_loss)

    def test_projection_kind_matrix_constructs_and_executes_each_label(self):
        for case in PROJECTION_CASES:
            with self.subTest(case=case):
                cfg = self.config(case)
                self.assert_config_matches_case(cfg, case)
                model = cfg.build()
                output, weights, auxiliary_loss = model(*self.qkv(case))

                self.assertEqual(output.shape, self.qkv(case)[0].shape)
                self.assertIsNone(weights)
                self.assertIsNone(auxiliary_loss)

    def test_returned_attention_weights_are_rejected_by_non_self_variants(self):
        for config_class in (
            IndependentAttentionConfig,
            MixtureOfAttentionHeadsConfig,
        ):
            case = ForwardCase(
                label=f"{config_class.__name__} returned weights",
                config_class=config_class,
                return_attention_weights_flag=True,
            )
            with self.subTest(case=case):
                model = self.config(case).build()

                with self.assertRaisesRegex(RuntimeError, "attention_weights"):
                    model(*self.qkv(case))

    def test_gradients_reach_inputs_and_trainable_attention_parameters(self):
        for config_class in EXPECTED_LAYER_TYPES:
            case = ForwardCase(
                label=f"{config_class.__name__} gradients",
                config_class=config_class,
                source_sequence_length=(
                    3 if config_class is SelfAttentionConfig else 4
                ),
                add_key_value_bias_flag=True,
            )
            with self.subTest(case=case):
                torch.manual_seed(20)
                model = self.config(case).build()
                query, key, value = self.qkv(case, requires_grad=True)

                output, _, auxiliary_loss = model(query, key, value)
                loss = output.square().mean()
                if auxiliary_loss is not None:
                    loss = loss + auxiliary_loss
                loss.backward()

                for label, tensor in zip(
                    ("query", "key", "value"),
                    (query, key, value),
                    strict=True,
                ):
                    with self.subTest(case=case, input=label):
                        self.assertIsNotNone(tensor.grad)
                        self.assertTrue(torch.any(tensor.grad.abs() > 0))
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                self.assertTrue(gradients)
                self.assertTrue(
                    any(torch.any(gradient.abs() > 0) for gradient in gradients)
                )


if __name__ == "__main__":
    unittest.main()
