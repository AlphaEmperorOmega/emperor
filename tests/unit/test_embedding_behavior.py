from __future__ import annotations

import json
import subprocess
import sys
import unittest
from dataclasses import dataclass

import torch
import torch.nn as nn

from emperor.attention import MultiHeadAttentionConfig
from emperor.config import ConfigBase, optional_field
from emperor.embedding.absolute import (
    AbsolutePositionalEmbeddingConfig,
    ImageLearnedPositionalEmbeddingConfig,
    ImageSinusoidalPositionalEmbeddingConfig,
    TextLearnedPositionalEmbeddingConfig,
    TextSinusoidalPositionalEmbeddingConfig,
)
from emperor.embedding.absolute._learned import LearnedPositionalEmbedding
from emperor.embedding.absolute._sinusoidal import TextSinusoidalPositionalEmbedding
from emperor.embedding.relative import (
    DynamicPositionalBiasConfig,
    RelativePositionalEmbeddingConfig,
)


@dataclass
class _InnerPositionalConfig(ConfigBase):
    positional_embedding_config: ConfigBase | None = optional_field(
        "Nested positional embedding configuration."
    )


@dataclass
class _OuterAbsoluteConfig(ConfigBase):
    absolute_positional_embedding_config: ConfigBase | None = optional_field(
        "Nested absolute positional embedding configuration."
    )


def text_learned_config(**overrides: object) -> TextLearnedPositionalEmbeddingConfig:
    values: dict[str, object] = {
        "num_embeddings": 4,
        "embedding_dim": 2,
        "init_size": 4,
        "padding_idx": 0,
        "auto_expand_flag": False,
    }
    values.update(overrides)
    return TextLearnedPositionalEmbeddingConfig(**values)


def relative_config(**overrides: object) -> DynamicPositionalBiasConfig:
    values: dict[str, object] = {
        "text_processing_flag": True,
        "num_heads": 2,
        "num_embeddings": 5,
        "embedding_dim": 4,
        "init_size": 5,
        "padding_idx": None,
        "auto_expand_flag": False,
        "max_positions": 2,
    }
    values.update(overrides)
    return DynamicPositionalBiasConfig(**values)


class EmbeddingConfigurationBehaviorTests(unittest.TestCase):
    def test_base_configs_have_exact_defaults_and_cannot_build(self) -> None:
        absolute = AbsolutePositionalEmbeddingConfig()
        relative = RelativePositionalEmbeddingConfig()

        self.assertEqual(
            (
                absolute.num_embeddings,
                absolute.embedding_dim,
                absolute.init_size,
                absolute.padding_idx,
                absolute.auto_expand_flag,
            ),
            (None, None, None, None, None),
        )
        self.assertEqual(
            (
                relative.text_processing_flag,
                relative.num_heads,
                relative.num_embeddings,
                relative.embedding_dim,
                relative.init_size,
                relative.padding_idx,
                relative.auto_expand_flag,
                relative.max_positions,
            ),
            (None, None, None, None, None, None, None, None),
        )
        self.assertEqual(absolute.get_custom_parameters(), {})
        self.assertEqual(relative.get_custom_parameters(), {})
        for config in (absolute, relative):
            with self.subTest(config=type(config).__name__):
                with self.assertRaises(NotImplementedError) as error:
                    config.build()
                self.assertEqual(
                    str(error.exception),
                    f"{type(config).__name__} must implement "
                    "`_registry_owner` or override `build`",
                )

    def test_concrete_registry_dispatch_and_partial_override_precedence(
        self,
    ) -> None:
        base = text_learned_config(
            embedding_dim=3,
            padding_idx=1,
            auto_expand_flag=True,
        )
        override = TextLearnedPositionalEmbeddingConfig(
            embedding_dim=2,
            auto_expand_flag=False,
        )

        model = base.build(override)

        self.assertIsInstance(model, base.registry_owner())
        self.assertEqual(model.embedding_dim, 2)
        self.assertEqual(model.padding_idx, 1)
        self.assertFalse(model.auto_expand_flag)
        self.assertEqual(model.embedding_model.num_embeddings, 6)
        self.assertEqual(
            (base.embedding_dim, base.padding_idx, base.auto_expand_flag),
            (3, 1, True),
        )
        self.assertEqual(
            (
                override.embedding_dim,
                override.padding_idx,
                override.auto_expand_flag,
            ),
            (2, None, False),
        )

    def test_absolute_nested_config_adapters_resolve_real_config(self) -> None:
        concrete = text_learned_config(embedding_dim=3)
        inner = _InnerPositionalConfig(positional_embedding_config=concrete)
        outer = _OuterAbsoluteConfig(absolute_positional_embedding_config=inner)

        model = concrete.registry_owner()(outer)

        self.assertIs(model.cfg, concrete)
        self.assertEqual(model.embedding_dim, 3)

    def test_image_nested_configs_and_partial_overrides_reach_base_models(
        self,
    ) -> None:
        learned_config = ImageLearnedPositionalEmbeddingConfig(
            num_embeddings=2,
            embedding_dim=2,
            init_size=2,
            padding_idx=0,
            auto_expand_flag=False,
            class_token_flag=True,
        )
        learned_wrapper = _OuterAbsoluteConfig(
            absolute_positional_embedding_config=_InnerPositionalConfig(
                positional_embedding_config=learned_config
            )
        )
        learned_override = ImageLearnedPositionalEmbeddingConfig(
            embedding_dim=4,
            class_token_flag=False,
        )

        learned = learned_config.registry_owner()(
            learned_wrapper,
            learned_override,
        )

        self.assertEqual(
            (
                learned.cfg.num_embeddings,
                learned.embedding_dim,
                learned.padding_idx,
                learned.auto_expand_flag,
                learned.class_token_flag,
            ),
            (2, 4, 0, False, False),
        )
        self.assertEqual(learned.embedding_model.weight.shape, (2, 4))

        sinusoidal_config = ImageSinusoidalPositionalEmbeddingConfig(
            num_embeddings=2,
            embedding_dim=2,
            init_size=2,
            padding_idx=None,
            auto_expand_flag=False,
            class_token_flag=True,
        )
        sinusoidal_wrapper = _OuterAbsoluteConfig(
            absolute_positional_embedding_config=_InnerPositionalConfig(
                positional_embedding_config=sinusoidal_config
            )
        )
        sinusoidal_override = ImageSinusoidalPositionalEmbeddingConfig(
            num_embeddings=3,
            embedding_dim=4,
            auto_expand_flag=True,
            class_token_flag=False,
        )

        sinusoidal = sinusoidal_config.registry_owner()(
            sinusoidal_wrapper,
            sinusoidal_override,
        )

        self.assertEqual(
            (
                sinusoidal.cfg.num_embeddings,
                sinusoidal.embedding_dim,
                sinusoidal.padding_idx,
                sinusoidal.auto_expand_flag,
                sinusoidal.class_token_flag,
                sinusoidal.position_offset,
                sinusoidal.init_size,
            ),
            (3, 4, None, True, False, 0, 3),
        )
        self.assertEqual(sinusoidal.weights.shape, (3, 4))

    def test_relative_attention_wrapper_and_false_overrides_are_preserved(
        self,
    ) -> None:
        concrete = relative_config(
            text_processing_flag=True,
            auto_expand_flag=True,
        )
        wrapper = MultiHeadAttentionConfig(
            relative_positional_embedding_config=concrete
        )
        wrapped_model = concrete.registry_owner()(wrapper)
        override = DynamicPositionalBiasConfig(
            text_processing_flag=False,
            auto_expand_flag=False,
        )
        overridden_model = concrete.build(override)

        self.assertIs(wrapped_model.cfg, concrete)
        self.assertTrue(wrapped_model.text_processing_flag)
        self.assertTrue(wrapped_model.auto_expand_flag)
        self.assertFalse(overridden_model.text_processing_flag)
        self.assertFalse(overridden_model.auto_expand_flag)
        self.assertTrue(concrete.text_processing_flag)
        self.assertTrue(concrete.auto_expand_flag)

    def test_all_concrete_absolute_config_types_dispatch_correctly(self) -> None:
        configs = (
            text_learned_config(),
            ImageLearnedPositionalEmbeddingConfig(
                num_embeddings=3,
                embedding_dim=2,
                init_size=3,
                padding_idx=0,
                auto_expand_flag=False,
                class_token_flag=True,
            ),
            TextSinusoidalPositionalEmbeddingConfig(
                num_embeddings=3,
                embedding_dim=2,
                init_size=3,
                padding_idx=0,
                auto_expand_flag=False,
            ),
            ImageSinusoidalPositionalEmbeddingConfig(
                num_embeddings=3,
                embedding_dim=2,
                init_size=3,
                padding_idx=None,
                auto_expand_flag=False,
                class_token_flag=True,
            ),
        )

        for config in configs:
            with self.subTest(config=type(config).__name__):
                model = config.build()
                self.assertIsInstance(model, config.registry_owner())


class LearnedEmbeddingBehaviorTests(unittest.TestCase):
    def test_generic_learned_embedding_uses_unadjusted_table_size(self) -> None:
        model = LearnedPositionalEmbedding(text_learned_config())

        self.assertEqual(model.num_embeddings, 4)
        self.assertEqual(model.embedding_model.weight.shape, (4, 2))

    def test_initialization_matches_scaled_normal_and_zero_padding_exactly(
        self,
    ) -> None:
        for padding_idx in (None, 1):
            with self.subTest(padding_idx=padding_idx), torch.random.fork_rng():
                config = text_learned_config(
                    num_embeddings=4,
                    embedding_dim=3,
                    padding_idx=padding_idx,
                )
                table_size = 4 if padding_idx is None else 6

                torch.manual_seed(8675309)
                model = config.build()
                torch.manual_seed(8675309)
                expected = nn.Embedding(table_size, 3, padding_idx)
                nn.init.normal_(expected.weight, std=3**-0.5)
                if padding_idx is not None:
                    nn.init.constant_(expected.weight[padding_idx], 0)

                torch.testing.assert_close(
                    model.embedding_model.weight,
                    expected.weight,
                    rtol=0,
                    atol=0,
                )

    def test_no_padding_table_positions_and_device_follow_input(self) -> None:
        model = text_learned_config(
            num_embeddings=4,
            padding_idx=None,
        ).build()
        weights = torch.tensor(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
            ]
        )
        with torch.no_grad():
            model.embedding_model.weight.copy_(weights)

        output = model(torch.tensor([[9, 8, 7], [6, 5, 4]]))

        self.assertEqual(model.embedding_model.num_embeddings, 4)
        torch.testing.assert_close(
            output,
            weights[torch.tensor([[0, 1, 2], [0, 1, 2]])],
            rtol=0,
            atol=0,
        )

        meta_tokens = torch.empty(
            (2, 3),
            dtype=torch.long,
            device="meta",
        )
        meta_positions = model._make_positions(meta_tokens)
        self.assertEqual(meta_positions.device.type, "meta")
        self.assertEqual(meta_positions.dtype, torch.long)
        self.assertEqual(meta_positions.shape, meta_tokens.shape)

    def test_text_learned_exact_positions_gradients_optimizer_and_state(
        self,
    ) -> None:
        config = text_learned_config(num_embeddings=4, padding_idx=0)
        source = config.build()
        target = config.build()
        weights = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
            ]
        )
        with torch.no_grad():
            source.embedding_model.weight.copy_(weights)
            target.embedding_model.weight.fill_(-1.0)
        tokens = torch.tensor([[5, 0, 7], [8, 9, 0]])
        expected_positions = torch.tensor([[1, 0, 2], [1, 2, 0]])

        output = source(tokens)

        torch.testing.assert_close(
            output,
            weights[expected_positions],
            rtol=0,
            atol=0,
        )
        output.sum().backward()
        gradient = source.embedding_model.weight.grad
        assert gradient is not None
        torch.testing.assert_close(
            gradient,
            torch.tensor(
                [
                    [0.0, 0.0],
                    [2.0, 2.0],
                    [2.0, 2.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ),
            rtol=0,
            atol=0,
        )
        optimizer = torch.optim.SGD(source.parameters(), lr=0.25)
        optimizer.step()
        expected_updated = weights.clone()
        expected_updated[1:3] -= 0.5
        torch.testing.assert_close(
            source.embedding_model.weight,
            expected_updated,
            rtol=0,
            atol=0,
        )

        incompatible = target.load_state_dict(
            source.state_dict(),
            strict=True,
        )
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        torch.testing.assert_close(target(tokens), source(tokens))

    def test_explicit_positions_override_tokens_and_preserve_batch_isolation(
        self,
    ) -> None:
        model = text_learned_config().build().double()
        with torch.no_grad():
            model.embedding_model.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0],
                        [1.0, -1.0],
                        [2.0, -2.0],
                        [3.0, -3.0],
                        [4.0, -4.0],
                    ],
                    dtype=torch.float64,
                )
            )
        tokens = torch.zeros(2, 2, dtype=torch.long)
        positions = torch.tensor([[0, 3], [4, 2]], dtype=torch.int32)

        output = model(tokens, positions=positions)

        self.assertEqual(output.dtype, torch.float64)
        torch.testing.assert_close(
            output,
            model.embedding_model.weight[positions.long()],
            rtol=0,
            atol=0,
        )
        self.assertFalse(torch.equal(output[0], output[1]))

    def test_image_learned_adds_exact_values_and_routes_gradients(self) -> None:
        config = ImageLearnedPositionalEmbeddingConfig(
            num_embeddings=2,
            embedding_dim=2,
            init_size=2,
            padding_idx=None,
            auto_expand_flag=False,
            class_token_flag=True,
        )
        model = config.build()
        weights = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        with torch.no_grad():
            model.embedding_model.weight.copy_(weights)
        patches = torch.tensor(
            [
                [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
                [[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0]],
            ],
            requires_grad=True,
        )

        output = model(patches)

        torch.testing.assert_close(
            output,
            patches.detach() + weights,
            rtol=0,
            atol=0,
        )
        output.sum().backward()
        torch.testing.assert_close(
            patches.grad,
            torch.ones_like(patches),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            model.embedding_model.weight.grad,
            torch.full_like(weights, 2.0),
            rtol=0,
            atol=0,
        )

    def test_image_learned_padding_row_stays_zero_after_optimizer_step(
        self,
    ) -> None:
        model = ImageLearnedPositionalEmbeddingConfig(
            num_embeddings=2,
            embedding_dim=2,
            init_size=2,
            padding_idx=0,
            auto_expand_flag=False,
            class_token_flag=True,
        ).build()
        initial_weights = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
        with torch.no_grad():
            model.embedding_model.weight.copy_(initial_weights)
        patches = torch.zeros(2, 3, 2, requires_grad=True)

        output = model(patches)
        output.sum().backward()

        torch.testing.assert_close(
            output,
            initial_weights.unsqueeze(0).expand(2, -1, -1),
            rtol=0,
            atol=0,
        )
        gradient = model.embedding_model.weight.grad
        assert gradient is not None
        torch.testing.assert_close(
            gradient,
            torch.tensor([[0.0, 0.0], [2.0, 2.0], [2.0, 2.0]]),
            rtol=0,
            atol=0,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
        optimizer.step()
        torch.testing.assert_close(
            model.embedding_model.weight,
            torch.tensor([[0.0, 0.0], [0.5, 1.5], [2.5, 3.5]]),
            rtol=0,
            atol=0,
        )

    def test_image_learned_indices_follow_input_not_global_default_device(
        self,
    ) -> None:
        model = ImageLearnedPositionalEmbeddingConfig(
            num_embeddings=2,
            embedding_dim=2,
            init_size=2,
            padding_idx=None,
            auto_expand_flag=False,
            class_token_flag=True,
        ).build()
        weights = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        with torch.no_grad():
            model.embedding_model.weight.copy_(weights)
        patches = torch.zeros(2, 3, 2)

        with torch.device("meta"):
            output = model(patches)

        self.assertEqual(output.device.type, "cpu")
        torch.testing.assert_close(
            output,
            weights.unsqueeze(0).expand(2, -1, -1),
            rtol=0,
            atol=0,
        )


class SinusoidalEmbeddingBehaviorTests(unittest.TestCase):
    def test_script_module_hybrid_rejects_nonpersistent_position_buffer(
        self,
    ) -> None:
        class ScriptSinusoidalEmbedding(
            TextSinusoidalPositionalEmbedding,
            torch.jit.ScriptModule,
        ):
            pass

        config = TextSinusoidalPositionalEmbeddingConfig(
            num_embeddings=4,
            embedding_dim=2,
            init_size=4,
            padding_idx=0,
            auto_expand_flag=False,
        )

        with self.assertRaises(RuntimeError) as error:
            ScriptSinusoidalEmbedding(config)

        self.assertEqual(
            str(error.exception),
            "ScriptModule does not support non-persistent buffers",
        )

    def test_sinusoidal_table_dtype_is_independent_of_global_default_dtype(
        self,
    ) -> None:
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            model = TextSinusoidalPositionalEmbeddingConfig(
                num_embeddings=3,
                embedding_dim=4,
                init_size=3,
                padding_idx=None,
                auto_expand_flag=False,
            ).build()
        finally:
            torch.set_default_dtype(previous_dtype)

        self.assertEqual(model.weights.dtype, torch.float32)
        expected = torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [
                    torch.sin(torch.tensor(1.0)),
                    torch.sin(torch.tensor(0.0001)),
                    torch.cos(torch.tensor(1.0)),
                    torch.cos(torch.tensor(0.0001)),
                ],
                [
                    torch.sin(torch.tensor(2.0)),
                    torch.sin(torch.tensor(0.0002)),
                    torch.cos(torch.tensor(2.0)),
                    torch.cos(torch.tensor(0.0002)),
                ],
            ]
        )
        torch.testing.assert_close(model.weights, expected)

    def test_text_table_sizes_follow_none_and_nonzero_padding_contracts(
        self,
    ) -> None:
        no_padding = TextSinusoidalPositionalEmbeddingConfig(
            num_embeddings=4,
            embedding_dim=4,
            init_size=4,
            padding_idx=None,
            auto_expand_flag=False,
        ).build()
        nonzero_padding = TextSinusoidalPositionalEmbeddingConfig(
            num_embeddings=4,
            embedding_dim=4,
            init_size=4,
            padding_idx=2,
            auto_expand_flag=False,
        ).build()

        self.assertEqual(
            (
                no_padding.position_offset,
                no_padding.init_size,
                no_padding.weights.shape,
            ),
            (0, 4, torch.Size((4, 4))),
        )
        self.assertEqual(
            (
                nonzero_padding.position_offset,
                nonzero_padding.init_size,
                nonzero_padding.weights.shape,
            ),
            (2, 7, torch.Size((7, 4))),
        )
        torch.testing.assert_close(
            nonzero_padding.weights[2],
            torch.zeros(4),
            rtol=0,
            atol=0,
        )

    def test_padding_positions_and_repeated_calls_are_exact_and_stateless(
        self,
    ) -> None:
        model = TextSinusoidalPositionalEmbeddingConfig(
            num_embeddings=5,
            embedding_dim=2,
            init_size=5,
            padding_idx=0,
            auto_expand_flag=False,
        ).build()
        tokens = torch.tensor([[7, 0, 8, 9], [0, 3, 0, 4]])
        expected_positions = torch.tensor([[1, 0, 2, 3], [0, 1, 0, 2]])

        first = model(tokens)
        second = model(tokens)

        torch.testing.assert_close(
            first,
            model.weights[expected_positions],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(second, first, rtol=0, atol=0)
        self.assertFalse(first.requires_grad)
        self.assertEqual(model.state_dict(), {})
        self.assertEqual(list(model.parameters()), [])

    def test_image_sinusoidal_adds_exact_values_and_only_patches_receive_gradients(
        self,
    ) -> None:
        model = ImageSinusoidalPositionalEmbeddingConfig(
            num_embeddings=2,
            embedding_dim=3,
            init_size=2,
            padding_idx=None,
            auto_expand_flag=False,
            class_token_flag=True,
        ).build()
        patches = (
            torch.arange(
                1.0,
                19.0,
            )
            .reshape(2, 3, 3)
            .transpose(1, 2)
            .detach()
            .requires_grad_()
        )
        self.assertFalse(patches.is_contiguous())

        output = model(patches)

        torch.testing.assert_close(
            output,
            patches.detach() + model.weights,
        )
        output.sum().backward()
        torch.testing.assert_close(
            patches.grad,
            torch.ones_like(patches),
            rtol=0,
            atol=0,
        )
        self.assertFalse(model.weights.requires_grad)

    def test_image_sinusoidal_class_token_has_zero_positional_offset(
        self,
    ) -> None:
        model = ImageSinusoidalPositionalEmbeddingConfig(
            num_embeddings=2,
            embedding_dim=2,
            init_size=2,
            padding_idx=None,
            auto_expand_flag=False,
            class_token_flag=True,
        ).build()
        patches = torch.tensor([[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]])

        output = model(patches)

        expected = patches.clone()
        expected[:, 1, 0] += torch.sin(torch.tensor(1.0))
        expected[:, 1, 1] += torch.cos(torch.tensor(1.0))
        expected[:, 2, 0] += torch.sin(torch.tensor(2.0))
        expected[:, 2, 1] += torch.cos(torch.tensor(2.0))
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(
            model.weights[0],
            torch.zeros(2),
            rtol=0,
            atol=0,
        )

    def test_auto_expansion_boundaries_disabled_mode_and_device_are_exact(
        self,
    ) -> None:
        boundary_model = TextSinusoidalPositionalEmbeddingConfig(
            num_embeddings=5,
            embedding_dim=2,
            init_size=5,
            padding_idx=0,
            auto_expand_flag=True,
        ).build()
        original_weights = boundary_model.weights.clone()
        original_pointer = boundary_model.weights.data_ptr()

        boundary_output = boundary_model(torch.ones(2, 5, dtype=torch.long))

        self.assertEqual(boundary_model.weights.data_ptr(), original_pointer)
        torch.testing.assert_close(
            boundary_model.weights,
            original_weights,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            boundary_output,
            original_weights[torch.arange(1, 6).reshape(1, -1).expand(2, -1)],
            rtol=0,
            atol=0,
        )

        disabled_model = TextSinusoidalPositionalEmbeddingConfig(
            num_embeddings=2,
            embedding_dim=2,
            init_size=2,
            padding_idx=0,
            auto_expand_flag=False,
        ).build()
        disabled_shape = disabled_model.weights.shape
        with self.assertRaises(IndexError) as error:
            disabled_model(torch.ones(1, 4, dtype=torch.long))
        self.assertEqual(str(error.exception), "index out of range in self")
        self.assertEqual(disabled_model.weights.shape, disabled_shape)

        meta_model = (
            TextSinusoidalPositionalEmbeddingConfig(
                num_embeddings=2,
                embedding_dim=2,
                init_size=2,
                padding_idx=0,
                auto_expand_flag=True,
            )
            .build()
            .to("meta")
        )
        expand_weights = (
            meta_model._TextSinusoidalPositionalEmbedding__maybe_expand_weights
        )
        expand_weights(
            torch.ones(1, 4, dtype=torch.long, device="meta"),
            incremental_state=None,
            timestep=None,
        )
        self.assertEqual(meta_model.weights.device.type, "meta")
        self.assertEqual(meta_model.weights.shape, (5, 2))


class RelativeEmbeddingBehaviorTests(unittest.TestCase):
    def test_exact_multihead_projection_preserves_batch_and_head_isolation(
        self,
    ) -> None:
        model = relative_config().build()
        with torch.no_grad():
            model.relative_positional_embeddings.copy_(
                torch.tensor(
                    [
                        [[1.0] * 5, [2.0] * 5],
                        [[3.0] * 5, [4.0] * 5],
                    ]
                )
            )
        query = torch.tensor(
            [
                [
                    [[1.0, 10.0], [1.0, 10.0]],
                    [[1.0, 10.0], [1.0, 10.0]],
                ],
                [
                    [[2.0, 20.0], [2.0, 20.0]],
                    [[2.0, 20.0], [2.0, 20.0]],
                ],
            ]
        )

        output = model(query, sequence_length=3)

        expected = torch.tensor(
            [
                [
                    [[21.0] * 3, [21.0] * 3],
                    [[43.0] * 3, [43.0] * 3],
                ],
                [
                    [[42.0] * 3, [42.0] * 3],
                    [[86.0] * 3, [86.0] * 3],
                ],
            ]
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertEqual(output.shape, (2, 2, 2, 3))

    def test_selected_offsets_receive_gradients_and_optimizer_updates(
        self,
    ) -> None:
        model = relative_config(
            num_heads=1,
            embedding_dim=1,
            max_positions=3,
        ).build()
        with torch.no_grad():
            model.relative_positional_embeddings.copy_(
                torch.arange(7, dtype=torch.float32).reshape(1, 1, 7)
            )
        query = torch.tensor([[[[2.0]]]], requires_grad=True)
        original = model.relative_positional_embeddings.detach().clone()

        output = model(query, sequence_length=1)
        output.sum().backward()

        gradient = model.relative_positional_embeddings.grad
        assert gradient is not None
        expected_gradient = torch.zeros_like(gradient)
        expected_gradient[..., 3] = 2.0
        torch.testing.assert_close(
            gradient,
            expected_gradient,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            query.grad,
            torch.tensor([[[[3.0]]]]),
            rtol=0,
            atol=0,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
        optimizer.step()
        expected_parameters = original.clone()
        expected_parameters[..., 3] -= 0.5
        torch.testing.assert_close(
            model.relative_positional_embeddings,
            expected_parameters,
            rtol=0,
            atol=0,
        )

    def test_float64_non_contiguous_query_and_strict_state_round_trip(
        self,
    ) -> None:
        config = relative_config()
        source = config.build().double()
        target = config.build().double()
        with torch.no_grad():
            source.relative_positional_embeddings.fill_(0.5)
            target.relative_positional_embeddings.zero_()
        query = (
            torch.arange(1.0, 9.0, dtype=torch.float64)
            .reshape(1, 2, 2, 2)
            .transpose(2, 3)
            .detach()
            .requires_grad_()
        )
        self.assertFalse(query.is_contiguous())

        output = source(query, sequence_length=3)
        incompatible = target.load_state_dict(
            source.state_dict(),
            strict=True,
        )

        self.assertEqual(output.dtype, torch.float64)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        torch.testing.assert_close(
            target(query.detach(), sequence_length=3),
            output.detach(),
            rtol=0,
            atol=0,
        )

    def test_relative_grid_helpers_preserve_meta_device_for_both_modes(
        self,
    ) -> None:
        model = relative_config(
            num_heads=1,
            embedding_dim=1,
        ).build()
        compute_grid = model._DynamicPositionalBias__compute_embedding_grid

        full_grid = compute_grid(
            torch.empty(1, 1, 3, 1, device="meta"),
            4,
            False,
        )
        last_grid = compute_grid(
            torch.empty(1, 1, 1, 1, device="meta"),
            4,
            True,
        )

        self.assertEqual(full_grid.device.type, "meta")
        self.assertEqual(last_grid.device.type, "meta")
        self.assertEqual(full_grid.dtype, torch.long)
        self.assertEqual(last_grid.dtype, torch.long)
        self.assertEqual(full_grid.shape, (3, 4))
        self.assertEqual(last_grid.shape, (1, 4))


class EmbeddingInterfaceBehaviorTests(unittest.TestCase):
    def test_interfaces_export_only_configuration_without_dynamic_shortcuts(
        self,
    ) -> None:
        import emperor.embedding as embedding
        import emperor.embedding.absolute as absolute
        import emperor.embedding.relative as relative

        self.assertEqual(embedding.__all__, ("absolute", "relative"))
        self.assertIs(embedding.absolute, absolute)
        self.assertIs(embedding.relative, relative)
        self.assertEqual(
            absolute.__all__,
            (
                "AbsolutePositionalEmbeddingConfig",
                "TextLearnedPositionalEmbeddingConfig",
                "ImageLearnedPositionalEmbeddingConfig",
                "TextSinusoidalPositionalEmbeddingConfig",
                "ImageSinusoidalPositionalEmbeddingConfig",
            ),
        )
        self.assertEqual(
            relative.__all__,
            (
                "RelativePositionalEmbeddingConfig",
                "DynamicPositionalBiasConfig",
            ),
        )
        for module in (embedding, absolute, relative):
            with self.subTest(module=module.__name__):
                self.assertFalse(hasattr(module, "__getattr__"))
                self.assertFalse(hasattr(module, "_LAZY_EXPORTS"))

    def test_explicit_interfaces_eagerly_load_only_configuration(
        self,
    ) -> None:
        script = """\
import json
import sys

import emperor.embedding as embedding

root_eager_modules = sorted(
    name for name in sys.modules if name.startswith("emperor.embedding.")
)
root_has_children = {
    "absolute": hasattr(embedding, "absolute"),
    "relative": hasattr(embedding, "relative"),
}
import emperor.embedding.absolute as absolute
import emperor.embedding.relative as relative

eager_modules = sorted(
    name for name in sys.modules if name.startswith("emperor.embedding.")
)
print(json.dumps({
    "root_all": embedding.__all__,
    "root_eager_modules": root_eager_modules,
    "root_has_children": root_has_children,
    "absolute_all": absolute.__all__,
    "relative_all": relative.__all__,
    "eager_modules": eager_modules,
    "heavy_modules": {
        name: name in sys.modules
        for name in (
            "emperor.embedding.absolute._base",
            "emperor.embedding.absolute._learned",
            "emperor.embedding.absolute._sinusoidal",
            "emperor.embedding.absolute._validation",
            "emperor.embedding.relative._bias",
            "emperor.embedding.relative._validation",
        )
    },
    "private_exports": {
        "AbsolutePositionalEmbeddingBase": hasattr(
            absolute, "AbsolutePositionalEmbeddingBase"
        ),
        "LearnedPositionalEmbedding": hasattr(
            absolute, "LearnedPositionalEmbedding"
        ),
        "AbsolutePositionalEmbeddingValidator": hasattr(
            absolute, "AbsolutePositionalEmbeddingValidator"
        ),
        "DynamicPositionalBias": hasattr(
            relative, "DynamicPositionalBias"
        ),
        "RelativePositionalEmbeddingValidator": hasattr(
            relative, "RelativePositionalEmbeddingValidator"
        ),
    },
    "runtime_loaded": {
        "torch": "torch" in sys.modules,
        "lightning": "lightning" in sys.modules,
    },
    "shortcut_attributes": {
        "root___getattr__": hasattr(embedding, "__getattr__"),
        "absolute___getattr__": hasattr(absolute, "__getattr__"),
        "relative___getattr__": hasattr(relative, "__getattr__"),
        "absolute__LAZY_EXPORTS": hasattr(absolute, "_LAZY_EXPORTS"),
        "relative__LAZY_EXPORTS": hasattr(relative, "_LAZY_EXPORTS"),
    },
}))
"""

        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            json.loads(completed.stdout),
            {
                "root_all": ["absolute", "relative"],
                "root_eager_modules": [
                    "emperor.embedding.absolute",
                    "emperor.embedding.absolute._config",
                    "emperor.embedding.relative",
                    "emperor.embedding.relative._config",
                ],
                "root_has_children": {
                    "absolute": True,
                    "relative": True,
                },
                "absolute_all": [
                    "AbsolutePositionalEmbeddingConfig",
                    "TextLearnedPositionalEmbeddingConfig",
                    "ImageLearnedPositionalEmbeddingConfig",
                    "TextSinusoidalPositionalEmbeddingConfig",
                    "ImageSinusoidalPositionalEmbeddingConfig",
                ],
                "relative_all": [
                    "RelativePositionalEmbeddingConfig",
                    "DynamicPositionalBiasConfig",
                ],
                "eager_modules": [
                    "emperor.embedding.absolute",
                    "emperor.embedding.absolute._config",
                    "emperor.embedding.relative",
                    "emperor.embedding.relative._config",
                ],
                "heavy_modules": {
                    "emperor.embedding.absolute._base": False,
                    "emperor.embedding.absolute._learned": False,
                    "emperor.embedding.absolute._sinusoidal": False,
                    "emperor.embedding.absolute._validation": False,
                    "emperor.embedding.relative._bias": False,
                    "emperor.embedding.relative._validation": False,
                },
                "private_exports": {
                    "AbsolutePositionalEmbeddingBase": False,
                    "LearnedPositionalEmbedding": False,
                    "AbsolutePositionalEmbeddingValidator": False,
                    "DynamicPositionalBias": False,
                    "RelativePositionalEmbeddingValidator": False,
                },
                "runtime_loaded": {
                    "torch": False,
                    "lightning": False,
                },
                "shortcut_attributes": {
                    "root___getattr__": False,
                    "absolute___getattr__": False,
                    "relative___getattr__": False,
                    "absolute__LAZY_EXPORTS": False,
                    "relative__LAZY_EXPORTS": False,
                },
            },
        )

    def test_removed_implementation_imports_fail(self) -> None:
        removed_exports = {
            "emperor.embedding.absolute": (
                "AbsolutePositionalEmbeddingBase",
                "LearnedPositionalEmbedding",
                "TextLearnedPositionalEmbedding",
                "ImageLearnedPositionalEmbedding",
                "SinusoidalPositionalEmbedding",
                "TextSinusoidalPositionalEmbedding",
                "ImageSinusoidalPositionalEmbedding",
                "AbsolutePositionalEmbeddingValidator",
            ),
            "emperor.embedding.relative": (
                "DynamicPositionalBias",
                "RelativePositionalEmbeddingValidator",
            ),
        }
        for module_name, export_names in removed_exports.items():
            for export_name in export_names:
                with self.subTest(module=module_name, export=export_name):
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-c",
                            f"from {module_name} import {export_name}",
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn("ImportError", completed.stderr)


if __name__ == "__main__":
    unittest.main()
