import math
import unittest

import torch
import torch.nn as nn

from emperor.layers._composition.pairwise_residual import (
    AdditiveResidual,
    WeightedBlendResidual,
    WeightedResidual,
)
from emperor.layers._composition.residual import ResidualConnection
from emperor.layers._config import ResidualConfig
from emperor.layers._options import ResidualConnectionOptions
from emperor.layers._validation import ResidualConnectionValidator
from emperor.linears import LinearLayerConfig


class TestPairwiseResidualVariants(unittest.TestCase):
    @staticmethod
    def build_model(config, **overrides):
        return config.build(overrides=type(config)(**overrides))

    def test_each_pairwise_option_has_its_own_class(self):
        self.assertDictEqual(
            ResidualConnection.PAIRWISE_RESIDUAL_TYPES,
            {
                ResidualConnectionOptions.RESIDUAL: AdditiveResidual,
                ResidualConnectionOptions.WEIGHTED_RESIDUAL: WeightedResidual,
                ResidualConnectionOptions.WEIGHTED_BLEND: WeightedBlendResidual,
            },
        )

    def test_additive_residual_owns_direct_addition(self):
        current = torch.tensor([[2.0, 3.0]])
        previous = torch.tensor([[5.0, 7.0]])
        connection = ResidualConnection(
            ResidualConfig(option=ResidualConnectionOptions.RESIDUAL)
        )

        parameters = AdditiveResidual.build_parameters(
            model_config=None,
            residual_dim=None,
            blend_initial_alpha=0.9,
            build_model=self.build_model,
        )
        actual = AdditiveResidual.forward(connection, current, previous)

        self.assertIsNone(AdditiveResidual.initial_raw_mix_coefficient(0.9))
        self.assertIsNone(parameters.raw_weight)
        self.assertIsNone(parameters.model)
        torch.testing.assert_close(actual, current + previous)

    def test_weighted_residual_owns_initialization_and_tanh_composition(self):
        current = torch.tensor([[2.0, 3.0]])
        previous = torch.tensor([[5.0, 7.0]])
        raw_mix_coefficient = torch.tensor(0.4)
        connection = ResidualConnection(
            ResidualConfig(option=ResidualConnectionOptions.WEIGHTED_RESIDUAL)
        )

        parameters = WeightedResidual.build_parameters(
            model_config=None,
            residual_dim=None,
            blend_initial_alpha=0.9,
            build_model=self.build_model,
        )
        with torch.no_grad():
            connection.raw_weight.copy_(raw_mix_coefficient)
        actual = WeightedResidual.forward(connection, current, previous)

        self.assertIsInstance(parameters.raw_weight, nn.Parameter)
        self.assertIsNone(parameters.model)
        torch.testing.assert_close(parameters.raw_weight, torch.tensor(0.0))
        torch.testing.assert_close(
            actual,
            previous + torch.tanh(raw_mix_coefficient) * current,
        )

    def test_weighted_blend_owns_initialization_and_sigmoid_composition(self):
        current = torch.tensor([[2.0, 3.0]])
        previous = torch.tensor([[5.0, 7.0]])
        raw_mix_coefficient = torch.tensor(-0.3)
        initial_alpha = 0.8
        connection = ResidualConnection(
            ResidualConfig(option=ResidualConnectionOptions.WEIGHTED_BLEND)
        )

        parameters = WeightedBlendResidual.build_parameters(
            model_config=None,
            residual_dim=None,
            blend_initial_alpha=initial_alpha,
            build_model=self.build_model,
        )
        with torch.no_grad():
            connection.raw_weight.copy_(raw_mix_coefficient)
        actual = WeightedBlendResidual.forward(connection, current, previous)

        self.assertIsInstance(parameters.raw_weight, nn.Parameter)
        self.assertIsNone(parameters.model)
        torch.testing.assert_close(
            parameters.raw_weight,
            torch.tensor(math.log(initial_alpha / (1.0 - initial_alpha))),
        )
        current_coefficient = torch.sigmoid(raw_mix_coefficient)
        expected = (
            current_coefficient * current + (1.0 - current_coefficient) * previous
        )
        torch.testing.assert_close(actual, expected)

    def test_weighted_variant_owns_data_dependent_model_generation(self):
        residual_dim = 2
        initial_alpha = 0.8

        parameters = WeightedBlendResidual.build_parameters(
            model_config=LinearLayerConfig(bias_flag=True),
            residual_dim=residual_dim,
            blend_initial_alpha=initial_alpha,
            build_model=self.build_model,
        )

        self.assertIsNone(parameters.raw_weight)
        self.assertIsNotNone(parameters.model)
        model = parameters.model
        assert model is not None
        self.assertEqual(model.input_dim, residual_dim * 2)
        self.assertEqual(model.output_dim, residual_dim)
        torch.testing.assert_close(
            model.weight_params,
            torch.zeros_like(model.weight_params),
        )
        bias_params = model.bias_params
        assert bias_params is not None
        torch.testing.assert_close(
            bias_params,
            torch.full_like(
                bias_params,
                math.log(initial_alpha / (1.0 - initial_alpha)),
            ),
        )

    def test_residual_config_still_builds_the_public_facade(self):
        connection = ResidualConfig(
            option=ResidualConnectionOptions.WEIGHTED_RESIDUAL,
        ).build()

        self.assertIs(type(connection), ResidualConnection)

    def test_facade_forward_only_delegates_to_the_selected_option(self):
        class TrackingResidual(AdditiveResidual):
            @classmethod
            def forward(
                cls,
                connection,
                current,
                previous,
                *,
                residual_state=None,
            ):
                return current - previous

        class TrackingResidualConnection(ResidualConnection):
            RESIDUAL_OPTION_TYPES = {
                **ResidualConnection.RESIDUAL_OPTION_TYPES,
                ResidualConnectionOptions.RESIDUAL: TrackingResidual,
            }

        connection = TrackingResidualConnection(
            ResidualConfig(option=ResidualConnectionOptions.RESIDUAL)
        )
        current = torch.tensor([[7.0, 11.0]])
        previous = torch.tensor([[2.0, 3.0]])

        actual = connection(current, previous)

        torch.testing.assert_close(actual, current - previous)

    def test_variant_extraction_preserves_registered_parameter_names(self):
        cases = (
            (
                ResidualConfig(option=ResidualConnectionOptions.RESIDUAL),
                (),
            ),
            (
                ResidualConfig(option=ResidualConnectionOptions.WEIGHTED_RESIDUAL),
                ("raw_weight",),
            ),
            (
                ResidualConfig(option=ResidualConnectionOptions.WEIGHTED_BLEND),
                ("raw_weight",),
            ),
            (
                ResidualConfig(
                    option=ResidualConnectionOptions.WEIGHTED_BLEND,
                    residual_dim=2,
                    model_config=LinearLayerConfig(bias_flag=True),
                ),
                ("model.weight_params", "model.bias_params"),
            ),
            (
                ResidualConfig(
                    option=ResidualConnectionOptions.ATTENTION_RESIDUAL,
                    residual_dim=2,
                ),
                ("attention_residual.query", "attention_residual.key_norm.weight"),
            ),
        )

        for config, expected_names in cases:
            with self.subTest(option=config.option, data_dependent=config.model_config):
                connection = config.build()

                self.assertTupleEqual(tuple(connection.state_dict()), expected_names)
                self.assertTupleEqual(
                    tuple(name for name, _ in connection.named_parameters()),
                    expected_names,
                )

    def test_stateless_variants_do_not_change_the_torch_module_tree(self):
        connection = ResidualConnection(
            ResidualConfig(option=ResidualConnectionOptions.WEIGHTED_RESIDUAL)
        )

        self.assertTupleEqual(
            tuple(name for name, _ in connection.named_modules()),
            ("",),
        )

    def test_facade_blend_initializer_override_is_preserved(self):
        class CustomBlendResidualConnection(ResidualConnection):
            WEIGHTED_BLEND_INITIAL_ALPHA = 0.75

        connection = CustomBlendResidualConnection(
            ResidualConfig(option=ResidualConnectionOptions.WEIGHTED_BLEND)
        )

        torch.testing.assert_close(
            connection.raw_weight,
            torch.tensor(math.log(0.75 / (1.0 - 0.75))),
        )

    def test_strict_checkpoint_round_trip_preserves_every_option_output(self):
        configs = (
            ResidualConfig(option=ResidualConnectionOptions.RESIDUAL),
            ResidualConfig(option=ResidualConnectionOptions.WEIGHTED_RESIDUAL),
            ResidualConfig(option=ResidualConnectionOptions.WEIGHTED_BLEND),
            ResidualConfig(
                option=ResidualConnectionOptions.WEIGHTED_BLEND,
                residual_dim=2,
                model_config=LinearLayerConfig(bias_flag=True),
            ),
            ResidualConfig(
                option=ResidualConnectionOptions.ATTENTION_RESIDUAL,
                residual_dim=2,
            ),
        )
        current = torch.tensor([[2.0, 3.0]])
        previous = torch.tensor([[5.0, 7.0]])

        for config in configs:
            with self.subTest(option=config.option, data_dependent=config.model_config):
                original = config.build()
                restored = config.build()
                restored.load_state_dict(original.state_dict(), strict=True)

                if config.option == ResidualConnectionOptions.ATTENTION_RESIDUAL:
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

    def test_runtime_option_changes_resolve_the_current_variant(self):
        connection = ResidualConnection(
            ResidualConfig(option=ResidualConnectionOptions.WEIGHTED_RESIDUAL)
        )
        current = torch.tensor([[2.0, 3.0]])
        previous = torch.tensor([[5.0, 7.0]])
        connection.option = ResidualConnectionOptions.WEIGHTED_BLEND

        actual = connection(current, previous)

        torch.testing.assert_close(actual, 0.5 * current + 0.5 * previous)

    def test_unknown_construction_option_keeps_the_coefficient_error_contract(self):
        class PermissiveValidator(ResidualConnectionValidator):
            @classmethod
            def validate(cls, model):
                pass

        class PermissiveResidualConnection(ResidualConnection):
            VALIDATOR = PermissiveValidator

        with self.assertRaisesRegex(
            ValueError,
            "Residual option does not use mixing coefficients: UNSUPPORTED",
        ):
            PermissiveResidualConnection(ResidualConfig(option="UNSUPPORTED"))

    def test_unhashable_runtime_option_keeps_the_dispatch_error_contract(self):
        connection = ResidualConnection(
            ResidualConfig(option=ResidualConnectionOptions.RESIDUAL)
        )
        connection.option = []
        hidden = torch.ones(1, 2)

        with self.assertRaisesRegex(
            ValueError,
            r"Unsupported residual connection option \[\] for ResidualConnection",
        ):
            connection(hidden, hidden)


if __name__ == "__main__":
    unittest.main()
