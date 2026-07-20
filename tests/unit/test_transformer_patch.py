import unittest

import torch
import torch.nn as nn
from torch import Tensor

from emperor.convs import Conv2dLayerConfig
from emperor.convs._layer import Conv2dLayer
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStack,
    LayerStackConfig,
)
from emperor.linears import LinearLayer, LinearLayerConfig
from emperor.patch import (
    ConvPatchEmbeddingConfig,
    LinearPatchEmbeddingConfig,
    PatchConfig,
    PatchEmbeddingConv,
    PatchEmbeddingLinear,
)


def expected_patch_count(
    image_height: int,
    image_width: int,
    patch_size: int,
    stride: int,
    padding: int,
) -> int:
    patch_rows = ((image_height + 2 * padding - patch_size) // stride) + 1
    patch_cols = ((image_width + 2 * padding - patch_size) // stride) + 1
    return patch_rows * patch_cols


def make_linear_embedding_stack_config(
    hidden_dim: int = 16,
    num_layers: int = 1,
    bias_flag: bool = True,
) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=1,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        num_layers=num_layers,
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
            layer_model_config=LinearLayerConfig(bias_flag=bias_flag),
        ),
    )


def make_conv_stack_config(
    embedding_dim: int = 16,
    kernel_size: int = 4,
    stride: int = 4,
    padding: int = 0,
    bias_flag: bool = True,
) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=1,
        hidden_dim=embedding_dim,
        output_dim=embedding_dim,
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
            layer_model_config=Conv2dLayerConfig(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias_flag=bias_flag,
            ),
        ),
    )


class TestPatchConfig(unittest.TestCase):
    def test_base_patch_config_cannot_build(self):
        cfg = PatchConfig(
            embedding_dim=8,
            num_input_channels=1,
            patch_size=4,
            dropout_probability=0.0,
        )

        with self.assertRaises(NotImplementedError):
            cfg.build()


class TestLinearPatchEmbedding(unittest.TestCase):
    def preset(
        self,
        embedding_dim: int = 16,
        num_input_channels: int = 1,
        patch_size: int = 4,
        stride: int = 4,
        padding: int = 0,
        dropout_probability: float = 0.0,
        embedding_stack_config: "LayerStackConfig | None" = None,
    ) -> LinearPatchEmbeddingConfig:
        if embedding_stack_config is None:
            embedding_stack_config = make_linear_embedding_stack_config(
                hidden_dim=embedding_dim
            )

        return LinearPatchEmbeddingConfig(
            embedding_dim=embedding_dim,
            num_input_channels=num_input_channels,
            patch_size=patch_size,
            dropout_probability=dropout_probability,
            stride=stride,
            padding=padding,
            embedding_stack_config=embedding_stack_config,
        )

    def test_init(self):
        cfg = self.preset(
            embedding_dim=12,
            num_input_channels=3,
            patch_size=2,
            stride=2,
            padding=1,
            dropout_probability=0.25,
        )
        model = PatchEmbeddingLinear(cfg)

        self.assertEqual(model.embedding_dim, cfg.embedding_dim)
        self.assertEqual(model.num_input_channels, cfg.num_input_channels)
        self.assertEqual(model.patch_size, cfg.patch_size)
        self.assertEqual(model.stride, cfg.stride)
        self.assertEqual(model.padding, cfg.padding)
        self.assertEqual(model.dropout_probability, cfg.dropout_probability)
        self.assertEqual(model.patch_dim, cfg.num_input_channels * cfg.patch_size**2)
        self.assertEqual(model.class_token.shape, (1, 1, cfg.embedding_dim))
        self.assertIsInstance(model.dropout, nn.Dropout)
        self.assertEqual(model.dropout.p, cfg.dropout_probability)
        self.assertIsInstance(model.patch_model, nn.Unfold)
        self.assertIsInstance(model.embedding_model, LayerStack)
        self.assertIsInstance(model.embedding_model[0].model, LinearLayer)
        self.assertEqual(model.embedding_model[0].model.input_dim, model.patch_dim)
        self.assertEqual(model.embedding_model[0].model.output_dim, cfg.embedding_dim)

    def test_forward_returns_class_token_plus_projected_patches(self):
        cfg = self.preset(embedding_dim=8, patch_size=4, stride=4, padding=0)
        model = PatchEmbeddingLinear(cfg)
        image_height = image_width = 8
        input_batch = torch.randn(2, cfg.num_input_channels, image_height, image_width)

        output = model(input_batch)

        expected_sequence_length = (
            expected_patch_count(
                image_height, image_width, cfg.patch_size, cfg.stride, cfg.padding
            )
            + 1
        )
        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.shape, (2, expected_sequence_length, cfg.embedding_dim))

    def test_output_matches_unfolded_patches_with_identity_projection(self):
        cfg = self.preset(embedding_dim=4, patch_size=2, stride=2, padding=0)
        model = PatchEmbeddingLinear(cfg)
        model.eval()

        with torch.no_grad():
            model.class_token.zero_()
            model.embedding_model[0].model.weight_params.copy_(torch.eye(4))
            model.embedding_model[0].model.bias_params.zero_()

        input_batch = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
        output = model(input_batch)

        expected_patches = torch.tensor(
            [
                [
                    [0.0, 1.0, 4.0, 5.0],
                    [2.0, 3.0, 6.0, 7.0],
                    [8.0, 9.0, 12.0, 13.0],
                    [10.0, 11.0, 14.0, 15.0],
                ]
            ]
        )
        expected = torch.cat([torch.zeros(1, 1, 4), expected_patches], dim=1)
        torch.testing.assert_close(output, expected)

    def test_gradients_flow_through_patch_embedding(self):
        cfg = self.preset(embedding_dim=6, patch_size=2, stride=2)
        model = PatchEmbeddingLinear(cfg)
        input_batch = torch.randn(2, cfg.num_input_channels, 4, 4, requires_grad=True)

        output = model(input_batch)
        output.sum().backward()

        self.assertIsNotNone(model.class_token.grad)
        self.assertEqual(model.class_token.grad.shape, model.class_token.shape)
        self.assertIsNotNone(model.embedding_model[0].model.weight_params.grad)
        self.assertIsNotNone(input_batch.grad)

    def test_config_build_returns_linear_patch_embedding(self):
        cfg = self.preset()
        model = cfg.build()

        self.assertIsInstance(model, PatchEmbeddingLinear)
        self.assertIsInstance(model, cfg._registry_owner())
        self.assertEqual(model.embedding_dim, cfg.embedding_dim)
        self.assertEqual(model.patch_size, cfg.patch_size)

    def test_config_build_applies_overrides(self):
        cfg = self.preset(embedding_dim=8, patch_size=2, stride=2)
        overrides = LinearPatchEmbeddingConfig(
            embedding_dim=10,
            num_input_channels=3,
            patch_size=3,
            dropout_probability=0.1,
            stride=1,
            padding=1,
            embedding_stack_config=make_linear_embedding_stack_config(hidden_dim=10),
        )
        model = cfg.build(overrides)

        self.assertIsInstance(model, PatchEmbeddingLinear)
        self.assertEqual(model.embedding_dim, overrides.embedding_dim)
        self.assertEqual(model.num_input_channels, overrides.num_input_channels)
        self.assertEqual(model.patch_size, overrides.patch_size)
        self.assertEqual(model.stride, overrides.stride)
        self.assertEqual(model.padding, overrides.padding)
        self.assertEqual(model.dropout_probability, overrides.dropout_probability)

    def test_partial_overrides_keep_unset_base_fields(self):
        cfg = self.preset(embedding_dim=8, patch_size=4, stride=4, padding=0)
        overrides = LinearPatchEmbeddingConfig(stride=2)
        model = PatchEmbeddingLinear(cfg, overrides)

        self.assertEqual(model.embedding_dim, cfg.embedding_dim)
        self.assertEqual(model.num_input_channels, cfg.num_input_channels)
        self.assertEqual(model.patch_size, cfg.patch_size)
        self.assertEqual(model.stride, overrides.stride)
        self.assertEqual(model.padding, cfg.padding)

    def test_init_raises_on_missing_required_fields(self):
        invalid_cases = [
            ("embedding_dim", LinearPatchEmbeddingConfig(
                num_input_channels=1,
                patch_size=4,
                dropout_probability=0.0,
                stride=4,
                padding=0,
                embedding_stack_config=make_linear_embedding_stack_config(),
            )),
            ("num_input_channels", LinearPatchEmbeddingConfig(
                embedding_dim=8,
                patch_size=4,
                dropout_probability=0.0,
                stride=4,
                padding=0,
                embedding_stack_config=make_linear_embedding_stack_config(),
            )),
            ("patch_size", LinearPatchEmbeddingConfig(
                embedding_dim=8,
                num_input_channels=1,
                dropout_probability=0.0,
                stride=4,
                padding=0,
                embedding_stack_config=make_linear_embedding_stack_config(),
            )),
            ("dropout_probability", LinearPatchEmbeddingConfig(
                embedding_dim=8,
                num_input_channels=1,
                patch_size=4,
                stride=4,
                padding=0,
                embedding_stack_config=make_linear_embedding_stack_config(),
            )),
            ("stride", LinearPatchEmbeddingConfig(
                embedding_dim=8,
                num_input_channels=1,
                patch_size=4,
                dropout_probability=0.0,
                padding=0,
                embedding_stack_config=make_linear_embedding_stack_config(),
            )),
            ("padding", LinearPatchEmbeddingConfig(
                embedding_dim=8,
                num_input_channels=1,
                patch_size=4,
                dropout_probability=0.0,
                stride=4,
                embedding_stack_config=make_linear_embedding_stack_config(),
            )),
            ("embedding_stack_config", LinearPatchEmbeddingConfig(
                embedding_dim=8,
                num_input_channels=1,
                patch_size=4,
                dropout_probability=0.0,
                stride=4,
                padding=0,
            )),
        ]

        for field_name, cfg in invalid_cases:
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError):
                    PatchEmbeddingLinear(cfg)

    def test_init_raises_on_invalid_dimensions_or_dropout(self):
        invalid_cases = [
            ("embedding_dim", {"embedding_dim": 0}),
            ("num_input_channels", {"num_input_channels": 0}),
            ("patch_size", {"patch_size": 0}),
            ("dropout_low", {"dropout_probability": -0.1}),
            ("dropout_high", {"dropout_probability": 1.1}),
        ]

        for case, kwargs in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    PatchEmbeddingLinear(self.preset(**kwargs))

    def test_forward_raises_on_invalid_input(self):
        model = PatchEmbeddingLinear(self.preset(num_input_channels=3))

        invalid_cases = [
            ("non_tensor", [1, 2, 3], TypeError),
            ("rank_3", torch.randn(3, 8, 8), ValueError),
            ("channel_mismatch", torch.randn(2, 1, 8, 8), ValueError),
        ]

        for case, input_batch, error_type in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(error_type):
                    model(input_batch)


class TestConvPatchEmbedding(unittest.TestCase):
    def preset(
        self,
        embedding_dim: int = 16,
        num_input_channels: int = 1,
        patch_size: int = 4,
        dropout_probability: float = 0.0,
        conv_stack_config: "LayerStackConfig | None" = None,
        stride: int = 4,
        padding: int = 0,
    ) -> ConvPatchEmbeddingConfig:
        if conv_stack_config is None:
            conv_stack_config = make_conv_stack_config(
                embedding_dim=embedding_dim,
                kernel_size=patch_size,
                stride=stride,
                padding=padding,
            )

        return ConvPatchEmbeddingConfig(
            embedding_dim=embedding_dim,
            num_input_channels=num_input_channels,
            patch_size=patch_size,
            dropout_probability=dropout_probability,
            conv_stack_config=conv_stack_config,
        )

    def test_init(self):
        cfg = self.preset(
            embedding_dim=12,
            num_input_channels=3,
            patch_size=2,
            dropout_probability=0.25,
            stride=2,
            padding=1,
        )
        model = PatchEmbeddingConv(cfg)

        self.assertEqual(model.embedding_dim, cfg.embedding_dim)
        self.assertEqual(model.num_input_channels, cfg.num_input_channels)
        self.assertEqual(model.patch_size, cfg.patch_size)
        self.assertEqual(model.dropout_probability, cfg.dropout_probability)
        self.assertEqual(model.class_token.shape, (1, 1, cfg.embedding_dim))
        self.assertIsInstance(model.dropout, nn.Dropout)
        self.assertEqual(model.dropout.p, cfg.dropout_probability)
        self.assertIsInstance(model.patch_model, LayerStack)
        self.assertIsInstance(model.patch_model[0].model, Conv2dLayer)
        self.assertIsInstance(model.patch_model[0].model.model, nn.Conv2d)
        self.assertEqual(model.patch_model[0].model.input_dim, cfg.num_input_channels)
        self.assertEqual(model.patch_model[0].model.output_dim, cfg.embedding_dim)
        self.assertEqual(model.patch_model[0].model.kernel_size, cfg.patch_size)

    def test_forward_returns_class_token_plus_projected_patches(self):
        cfg = self.preset(embedding_dim=8, patch_size=4, stride=4, padding=0)
        model = PatchEmbeddingConv(cfg)
        image_height = image_width = 8
        input_batch = torch.randn(2, cfg.num_input_channels, image_height, image_width)

        output = model(input_batch)

        expected_sequence_length = (
            expected_patch_count(image_height, image_width, cfg.patch_size, 4, 0) + 1
        )
        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.shape, (2, expected_sequence_length, cfg.embedding_dim))

    def test_output_matches_conv_patch_sums(self):
        cfg = self.preset(embedding_dim=1, patch_size=2, stride=2, padding=0)
        model = PatchEmbeddingConv(cfg)
        model.eval()

        with torch.no_grad():
            model.class_token.zero_()
            conv = model.patch_model[0].model.model
            conv.weight.fill_(1.0)
            conv.bias.zero_()

        input_batch = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
        output = model(input_batch)

        expected = torch.tensor([[[0.0], [10.0], [18.0], [42.0], [50.0]]])
        torch.testing.assert_close(output, expected)

    def test_gradients_flow_through_patch_embedding(self):
        cfg = self.preset(embedding_dim=6, patch_size=2, stride=2)
        model = PatchEmbeddingConv(cfg)
        input_batch = torch.randn(2, cfg.num_input_channels, 4, 4, requires_grad=True)

        output = model(input_batch)
        output.sum().backward()

        self.assertIsNotNone(model.class_token.grad)
        self.assertEqual(model.class_token.grad.shape, model.class_token.shape)
        self.assertIsNotNone(model.patch_model[0].model.model.weight.grad)
        self.assertIsNotNone(input_batch.grad)

    def test_config_build_returns_conv_patch_embedding(self):
        cfg = self.preset()
        model = cfg.build()

        self.assertIsInstance(model, PatchEmbeddingConv)
        self.assertIsInstance(model, cfg._registry_owner())
        self.assertEqual(model.embedding_dim, cfg.embedding_dim)
        self.assertEqual(model.patch_size, cfg.patch_size)

    def test_config_build_applies_overrides(self):
        cfg = self.preset(embedding_dim=8, patch_size=2)
        overrides = ConvPatchEmbeddingConfig(
            embedding_dim=10,
            num_input_channels=3,
            patch_size=3,
            dropout_probability=0.1,
            conv_stack_config=make_conv_stack_config(
                embedding_dim=10,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        model = cfg.build(overrides)

        self.assertIsInstance(model, PatchEmbeddingConv)
        self.assertEqual(model.embedding_dim, overrides.embedding_dim)
        self.assertEqual(model.num_input_channels, overrides.num_input_channels)
        self.assertEqual(model.patch_size, overrides.patch_size)
        self.assertEqual(model.dropout_probability, overrides.dropout_probability)
        self.assertEqual(model.patch_model[0].model.kernel_size, overrides.patch_size)

    def test_partial_overrides_keep_unset_base_fields(self):
        cfg = self.preset(embedding_dim=8, patch_size=4, dropout_probability=0.0)
        overrides = ConvPatchEmbeddingConfig(dropout_probability=0.2)
        model = PatchEmbeddingConv(cfg, overrides)

        self.assertEqual(model.embedding_dim, cfg.embedding_dim)
        self.assertEqual(model.num_input_channels, cfg.num_input_channels)
        self.assertEqual(model.patch_size, cfg.patch_size)
        self.assertEqual(model.dropout_probability, overrides.dropout_probability)

    def test_init_raises_on_missing_required_fields(self):
        invalid_cases = [
            ("embedding_dim", ConvPatchEmbeddingConfig(
                num_input_channels=1,
                patch_size=4,
                dropout_probability=0.0,
                conv_stack_config=make_conv_stack_config(),
            )),
            ("num_input_channels", ConvPatchEmbeddingConfig(
                embedding_dim=8,
                patch_size=4,
                dropout_probability=0.0,
                conv_stack_config=make_conv_stack_config(),
            )),
            ("patch_size", ConvPatchEmbeddingConfig(
                embedding_dim=8,
                num_input_channels=1,
                dropout_probability=0.0,
                conv_stack_config=make_conv_stack_config(),
            )),
            ("dropout_probability", ConvPatchEmbeddingConfig(
                embedding_dim=8,
                num_input_channels=1,
                patch_size=4,
                conv_stack_config=make_conv_stack_config(),
            )),
            ("conv_stack_config", ConvPatchEmbeddingConfig(
                embedding_dim=8,
                num_input_channels=1,
                patch_size=4,
                dropout_probability=0.0,
            )),
        ]

        for field_name, cfg in invalid_cases:
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError):
                    PatchEmbeddingConv(cfg)

    def test_init_raises_on_invalid_dimensions_or_dropout(self):
        invalid_cases = [
            ("embedding_dim", {"embedding_dim": 0}),
            ("num_input_channels", {"num_input_channels": 0}),
            ("patch_size", {"patch_size": 0}),
            ("dropout_low", {"dropout_probability": -0.1}),
            ("dropout_high", {"dropout_probability": 1.1}),
        ]

        for case, kwargs in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    PatchEmbeddingConv(self.preset(**kwargs))

    def test_forward_raises_on_invalid_input(self):
        model = PatchEmbeddingConv(self.preset(num_input_channels=3))

        invalid_cases = [
            ("non_tensor", [1, 2, 3], TypeError),
            ("rank_3", torch.randn(3, 8, 8), ValueError),
            ("channel_mismatch", torch.randn(2, 1, 8, 8), ValueError),
        ]

        for case, input_batch, error_type in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(error_type):
                    model(input_batch)
