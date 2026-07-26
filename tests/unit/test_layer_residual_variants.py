import math
import unittest

import torch

from emperor.layers import (
    AdditiveResidualConfig,
    AttentionResidualConfig,
    ResidualConfig,
    WeightedBlendResidualConfig,
    WeightedResidualConfig,
)
from emperor.layers._composition.residual.base import (
    ResidualConnectionAbstract,
)
from emperor.layers._composition.residual.pairwise import (
    PairwiseResidualAbstract,
    WeightedPairwiseResidualAbstract,
)
from emperor.layers._composition.residual.variants.additive import AdditiveResidual
from emperor.layers._composition.residual.variants.attention import AttentionResidual
from emperor.layers._composition.residual.variants.weighted import WeightedResidual
from emperor.layers._composition.residual.variants.weighted_blend import (
    WeightedBlendResidual,
)
from emperor.linears import LinearLayerConfig


class TestResidualConfigRegistry(unittest.TestCase):
    def test_each_concrete_config_builds_its_exact_runtime_owner(self):
        cases = (
            (AdditiveResidualConfig(), AdditiveResidual),
            (WeightedResidualConfig(), WeightedResidual),
            (WeightedBlendResidualConfig(), WeightedBlendResidual),
            (
                AttentionResidualConfig(residual_dim=2),
                AttentionResidual,
            ),
        )

        for config, expected_owner in cases:
            with self.subTest(config=type(config).__name__):
                self.assertIs(config.registry_owner(), expected_owner)
                self.assertIs(type(config.build()), expected_owner)

    def test_abstract_config_cannot_be_built(self):
        with self.assertRaisesRegex(
            ValueError,
            "ResidualConfig is abstract.*concrete residual config",
        ):
            ResidualConfig().build()

    def test_config_overrides_are_applied_before_runtime_construction(self):
        residual = WeightedResidualConfig(residual_dim=2).build(
            overrides=WeightedResidualConfig(residual_dim=5)
        )

        self.assertEqual(residual.residual_dim, 5)
        self.assertEqual(residual.cfg.residual_dim, 5)

    def test_runtime_rejects_a_config_owned_by_another_variant(self):
        with self.assertRaisesRegex(
            TypeError,
            "WeightedResidualConfig builds WeightedResidual, not AdditiveResidual",
        ):
            AdditiveResidual(WeightedResidualConfig())


class TestResidualRuntimeHierarchy(unittest.TestCase):
    def test_runtime_hierarchy_exposes_pairwise_monitoring_as_a_capability(self):
        additive = AdditiveResidualConfig().build()
        weighted = WeightedResidualConfig().build()
        blend = WeightedBlendResidualConfig().build()
        attention = AttentionResidualConfig(residual_dim=2).build()

        for residual in (additive, weighted, blend, attention):
            self.assertIsInstance(residual, ResidualConnectionAbstract)
        for residual in (additive, weighted, blend):
            self.assertIsInstance(residual, PairwiseResidualAbstract)
            self.assertTrue(residual.supports_pairwise_diagnostics)
        for residual in (weighted, blend):
            self.assertIsInstance(residual, WeightedPairwiseResidualAbstract)
        self.assertFalse(attention.supports_pairwise_diagnostics)

    def test_pairwise_variants_use_the_default_stateless_lifecycle(self):
        initial_source = torch.ones(1, 2)

        for config in (
            AdditiveResidualConfig(),
            WeightedResidualConfig(),
            WeightedBlendResidualConfig(),
        ):
            with self.subTest(config=type(config).__name__):
                self.assertIsNone(config.build().new_state(initial_source))


class TestPairwiseResidualVariants(unittest.TestCase):
    def test_additive_residual_owns_direct_addition(self):
        current = torch.tensor([[2.0, 3.0]], requires_grad=True)
        previous = torch.tensor([[5.0, 7.0]], requires_grad=True)
        residual = AdditiveResidualConfig().build()

        actual = residual(current, previous)
        actual.sum().backward()

        torch.testing.assert_close(actual, current.detach() + previous.detach())
        torch.testing.assert_close(current.grad, torch.ones_like(current))
        torch.testing.assert_close(previous.grad, torch.ones_like(previous))

    def test_weighted_residual_owns_tanh_composition(self):
        current = torch.tensor([[2.0, 3.0]], requires_grad=True)
        previous = torch.tensor([[5.0, 7.0]], requires_grad=True)
        residual = WeightedResidualConfig().build()
        raw_mix_coefficient = torch.tensor(0.4)
        with torch.no_grad():
            residual.raw_weight.copy_(raw_mix_coefficient)

        actual = residual(current, previous)
        actual.sum().backward()

        torch.testing.assert_close(
            actual,
            previous.detach() + torch.tanh(raw_mix_coefficient) * current.detach(),
        )
        self.assertIsNotNone(residual.raw_weight.grad)
        self.assertGreater(residual.raw_weight.grad.abs().item(), 0.0)

    def test_weighted_blend_owns_sigmoid_convex_composition(self):
        current = torch.tensor([[2.0, 3.0]], requires_grad=True)
        previous = torch.tensor([[5.0, 7.0]], requires_grad=True)
        residual = WeightedBlendResidualConfig().build()
        raw_mix_coefficient = torch.tensor(-0.3)
        with torch.no_grad():
            residual.raw_weight.copy_(raw_mix_coefficient)

        actual = residual(current, previous)
        actual.sum().backward()

        current_coefficient = torch.sigmoid(raw_mix_coefficient)
        expected = (
            current_coefficient * current.detach()
            + (1.0 - current_coefficient) * previous.detach()
        )
        torch.testing.assert_close(actual, expected)
        self.assertIsNotNone(residual.raw_weight.grad)
        self.assertGreater(residual.raw_weight.grad.abs().item(), 0.0)

    def test_scalar_coefficient_initializers_are_preserved(self):
        weighted = WeightedResidualConfig().build()
        blend = WeightedBlendResidualConfig().build()

        torch.testing.assert_close(weighted.raw_weight, torch.tensor(0.0))
        torch.testing.assert_close(
            blend.raw_weight,
            torch.tensor(math.log(0.9 / (1.0 - 0.9))),
        )

    def test_data_dependent_coefficient_model_owns_expected_dimensions_and_init(self):
        residual_dim = 2
        cases = (
            (WeightedResidualConfig, 0.0),
            (
                WeightedBlendResidualConfig,
                math.log(0.9 / (1.0 - 0.9)),
            ),
        )

        for config_type, expected_bias in cases:
            with self.subTest(config=config_type.__name__):
                residual = config_type(
                    residual_dim=residual_dim,
                    model_config=LinearLayerConfig(bias_flag=True),
                ).build()
                model = residual.model

                self.assertIsNone(residual.raw_weight)
                self.assertIsNotNone(model)
                assert model is not None
                self.assertEqual(model.input_dim, residual_dim * 2)
                self.assertEqual(model.output_dim, residual_dim)
                torch.testing.assert_close(
                    model.weight_params,
                    torch.zeros_like(model.weight_params),
                )
                assert model.bias_params is not None
                torch.testing.assert_close(
                    model.bias_params,
                    torch.full_like(model.bias_params, expected_bias),
                )

    def test_data_dependent_coefficient_changes_the_output_and_receives_gradients(self):
        residual = WeightedResidualConfig(
            residual_dim=2,
            model_config=LinearLayerConfig(bias_flag=True),
        ).build()
        assert residual.model is not None
        with torch.no_grad():
            residual.model.weight_params.copy_(
                torch.tensor(
                    [
                        [0.3, 0.0],
                        [0.0, -0.2],
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ]
                )
            )
        current = torch.tensor([[2.0, 3.0]], requires_grad=True)
        previous = torch.tensor([[5.0, 7.0]], requires_grad=True)

        output = residual(current, previous)
        output.sum().backward()

        self.assertIsNotNone(residual.model.weight_params.grad)
        self.assertGreater(
            torch.count_nonzero(residual.model.weight_params.grad).item(),
            0,
        )

    def test_runtime_parameter_names_remain_stable_and_attention_is_direct(self):
        cases = (
            (AdditiveResidualConfig(), ()),
            (WeightedResidualConfig(), ("raw_weight",)),
            (WeightedBlendResidualConfig(), ("raw_weight",)),
            (
                WeightedBlendResidualConfig(
                    residual_dim=2,
                    model_config=LinearLayerConfig(bias_flag=True),
                ),
                ("model.weight_params", "model.bias_params"),
            ),
            (
                AttentionResidualConfig(residual_dim=2),
                ("query", "key_norm.weight"),
            ),
        )

        for config, expected_names in cases:
            with self.subTest(config=type(config).__name__):
                residual = config.build()
                self.assertTupleEqual(tuple(residual.state_dict()), expected_names)
                self.assertTupleEqual(
                    tuple(name for name, _ in residual.named_parameters()),
                    expected_names,
                )

    def test_strict_checkpoint_round_trip_preserves_every_variant_output(self):
        configs = (
            AdditiveResidualConfig(),
            WeightedResidualConfig(),
            WeightedBlendResidualConfig(),
            WeightedBlendResidualConfig(
                residual_dim=2,
                model_config=LinearLayerConfig(bias_flag=True),
            ),
            AttentionResidualConfig(residual_dim=2),
        )
        current = torch.tensor([[2.0, 3.0]])
        previous = torch.tensor([[5.0, 7.0]])

        for config in configs:
            with self.subTest(config=type(config).__name__):
                original = config.build()
                restored = config.build()
                restored.load_state_dict(original.state_dict(), strict=True)
                if isinstance(config, AttentionResidualConfig):
                    expected = original(
                        current,
                        previous,
                        residual_state=original.new_state(previous),
                    )
                    actual = restored(
                        current,
                        previous,
                        residual_state=restored.new_state(previous),
                    )
                else:
                    expected = original(current, previous)
                    actual = restored(current, previous)

                torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
