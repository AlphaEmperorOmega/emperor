import math
import unittest
from dataclasses import dataclass, replace

import torch

from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    AttentionResidualConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStack,
    LayerStackConfig,
    LayerState,
    ResidualConfig,
    WeightedBlendResidualConfig,
    WeightedResidualConfig,
)
from emperor.layers._composition.residual.base import (
    ResidualConnectionAbstract,
)
from emperor.layers._composition.residual.pairwise import (
    WeightedPairwiseResidualAbstract,
)
from emperor.layers._composition.residual.variants.additive import AdditiveResidual
from emperor.layers._composition.residual.variants.attention import AttentionResidual
from emperor.layers._composition.residual.variants.weighted import WeightedResidual
from emperor.layers._composition.residual.variants.weighted_blend import (
    WeightedBlendResidual,
)
from emperor.layers._layer.pipeline.residual import LayerResidualDelegate
from emperor.linears import LinearLayerConfig


def _coefficient_stack_config(
    *,
    bias_flag: bool = True,
    last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT,
) -> LayerStackConfig:
    return LayerStackConfig(
        hidden_dim=3,
        num_layers=2,
        apply_output_postprocessing_flag=False,
        last_layer_bias_option=last_layer_bias_option,
        shared_gate_config=None,
        shared_halting_config=None,
        shared_memory_config=None,
        layer_config=LayerConfig(
            activation=ActivationOptions.GELU,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=bias_flag),
        ),
    )


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
    def test_runtime_hierarchy_shares_the_residual_and_weighted_interfaces(self):
        additive = AdditiveResidualConfig().build()
        weighted = WeightedResidualConfig().build()
        blend = WeightedBlendResidualConfig().build()
        attention = AttentionResidualConfig(
            block_size=1, rms_norm_epsilon=1e-6, residual_dim=2
        ).build()

        for residual in (additive, weighted, blend, attention):
            self.assertIsInstance(residual, ResidualConnectionAbstract)
        for residual in (weighted, blend):
            self.assertIsInstance(residual, WeightedPairwiseResidualAbstract)

    def test_pairwise_variants_use_the_default_stateless_lifecycle(self):
        initial_source = torch.ones(1, 2)

        for config in (
            AdditiveResidualConfig(),
            WeightedResidualConfig(),
            WeightedBlendResidualConfig(),
        ):
            with self.subTest(config=type(config).__name__):
                residual = config.build()

                self.assertIsNone(residual.residual_state_lifecycle)
                self.assertIsNone(residual.new_state(initial_source))

    def test_attention_requires_a_state_lifecycle(self):
        class MissingLifecycleResidual(AttentionResidual):
            def __init__(self, cfg, overrides=None):
                super().__init__(cfg, overrides)
                self.residual_state_lifecycle = None

        @dataclass
        class MissingLifecycleResidualConfig(AttentionResidualConfig):
            def _registry_owner(self) -> type:
                return MissingLifecycleResidual

        with self.assertRaisesRegex(
            RuntimeError,
            "MissingLifecycleResidual requires forward-local residual state",
        ):
            LayerResidualDelegate(
                LayerConfig(
                    output_dim=2,
                    residual_config=MissingLifecycleResidualConfig(
                        block_size=1, rms_norm_epsilon=1e-6
                    ),
                )
            )

    def test_default_state_aware_application_preserves_pairwise_state(self):
        residual = AdditiveResidualConfig(residual_dim=2).build()
        enclosing_residual_state = object()
        current = torch.tensor([[2.0, 3.0]])
        previous = torch.tensor([[5.0, 7.0]])
        layer_state = LayerState(
            hidden=current,
            residual_state=enclosing_residual_state,
        )

        result = residual.apply_to_layer_state(layer_state, previous)

        self.assertIs(result, layer_state)
        self.assertIs(result.residual_state, enclosing_residual_state)
        torch.testing.assert_close(result.hidden, current + previous)


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

    def test_data_dependent_coefficient_preserves_model_initialization(self):
        residual_dim = 2
        model_configs = (
            ("linear", LinearLayerConfig(bias_flag=True)),
            ("biasless_linear", LinearLayerConfig(bias_flag=False)),
            ("stack", _coefficient_stack_config()),
            ("biasless_stack", _coefficient_stack_config(bias_flag=False)),
            (
                "stack_without_final_bias",
                _coefficient_stack_config(
                    last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                ),
            ),
            (
                "stack_with_output_postprocessing",
                replace(
                    _coefficient_stack_config(),
                    apply_output_postprocessing_flag=True,
                ),
            ),
        )

        for config_type in (WeightedResidualConfig, WeightedBlendResidualConfig):
            for model_name, model_config in model_configs:
                with (
                    self.subTest(config=config_type.__name__, model=model_name),
                    torch.random.fork_rng(devices=[]),
                ):
                    torch.manual_seed(17)
                    reference_model = model_config.build(
                        overrides=type(model_config)(
                            input_dim=residual_dim * 2,
                            output_dim=residual_dim,
                        )
                    )
                    expected_rng_state = torch.random.get_rng_state().clone()
                    torch.manual_seed(17)
                    residual = config_type(
                        residual_dim=residual_dim,
                        model_config=model_config,
                    ).build()
                    model = residual.model

                    self.assertIsNone(residual.raw_weight)
                    self.assertEqual(model.input_dim, residual_dim * 2)
                    self.assertEqual(model.output_dim, residual_dim)
                    self.assertEqual(
                        tuple(model.state_dict()), tuple(reference_model.state_dict())
                    )
                    for name, expected in reference_model.state_dict().items():
                        torch.testing.assert_close(
                            model.state_dict()[name], expected, rtol=0, atol=0
                        )
                    torch.testing.assert_close(
                        torch.random.get_rng_state(), expected_rng_state
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

    def test_residual_stack_uses_model_predictions_and_preserves_gradients(self):
        for config_type in (WeightedResidualConfig, WeightedBlendResidualConfig):
            with (
                self.subTest(config=config_type.__name__),
                torch.random.fork_rng(devices=[]),
            ):
                torch.manual_seed(17)
                residual = config_type(
                    residual_dim=2,
                    model_config=_coefficient_stack_config(),
                ).build()
                current = torch.tensor([[0.2, 0.3]], requires_grad=True)
                previous = torch.tensor([[0.5, 0.7]], requires_grad=True)
                coefficient_input = torch.cat((current, previous), dim=-1)
                with torch.no_grad():
                    raw_coefficients = residual.model(
                        LayerState(hidden=coefficient_input)
                    ).hidden
                    if config_type is WeightedResidualConfig:
                        expected = previous + torch.tanh(raw_coefficients) * current
                    else:
                        blend = torch.sigmoid(raw_coefficients)
                        expected = blend * current + (1.0 - blend) * previous

                output = residual(current, previous)
                torch.testing.assert_close(output, expected)
                output.sum().backward()
                for gradient in (
                    current.grad,
                    previous.grad,
                    *(parameter.grad for parameter in residual.model.parameters()),
                ):
                    self.assertIsNotNone(gradient)
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertGreater(torch.count_nonzero(gradient).item(), 0)

    def test_coefficient_model_does_not_use_the_scalar_initializer(self):
        for config_type, residual_type in (
            (WeightedResidualConfig, WeightedResidual),
            (WeightedBlendResidualConfig, WeightedBlendResidual),
        ):

            class ModelInitializedResidual(residual_type):
                @staticmethod
                def _initial_raw_mix_coefficient():
                    raise AssertionError(
                        "model coefficients must use model initialization"
                    )

            with self.subTest(config=config_type.__name__):
                residual = ModelInitializedResidual(
                    config_type(
                        residual_dim=2,
                        model_config=LinearLayerConfig(bias_flag=True),
                    )
                )
                self.assertIsNone(residual.raw_weight)
                self.assertIsNotNone(residual.model)

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


class TestWeightedResidualValidationContracts(unittest.TestCase):
    def test_each_nested_controller_is_rejected_for_both_weighted_variants(self):
        paths = (
            "layer_config.gate_config",
            "layer_config.halting_config",
            "layer_config.memory_config",
            "shared_gate_config",
            "shared_halting_config",
            "shared_memory_config",
        )
        for config_type in (WeightedResidualConfig, WeightedBlendResidualConfig):
            for path in paths:
                with self.subTest(variant=config_type.__name__, path=path):
                    stack = _coefficient_stack_config()
                    if path.startswith("layer_config."):
                        stack.layer_config = replace(
                            stack.layer_config, **{path.split(".")[1]: object()}
                        )
                    else:
                        setattr(stack, path, object())
                    with self.assertRaises(ValueError) as raised:
                        config_type(residual_dim=2, model_config=stack).build()
                    self.assertEqual(
                        str(raised.exception),
                        f"{config_type.__name__}.model_config.{path} must be None "
                        "for a residual coefficient model.",
                    )

    def test_stack_structure_and_error_precedence_are_preserved(self):
        class DerivedLayerConfig(LayerConfig):
            pass

        for config_type in (WeightedResidualConfig, WeightedBlendResidualConfig):
            stack = _coefficient_stack_config()
            cases = (
                (
                    object(),
                    TypeError,
                    "must be a LayerStackConfig or LinearLayerConfig",
                ),
                (
                    replace(stack, layer_config=DerivedLayerConfig()),
                    TypeError,
                    "must be exactly LayerConfig",
                ),
                (
                    replace(
                        stack,
                        layer_config=replace(
                            stack.layer_config, layer_model_config=object()
                        ),
                    ),
                    TypeError,
                    "layer_model_config must be LinearLayerConfig",
                ),
                (
                    replace(
                        stack,
                        last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                        shared_gate_config=object(),
                    ),
                    ValueError,
                    "shared_gate_config must be None",
                ),
                (
                    replace(
                        stack,
                        shared_gate_config=object(),
                        shared_memory_config=object(),
                    ),
                    ValueError,
                    "shared_gate_config must be None",
                ),
                (stack, TypeError, "residual_dim must be int"),
            )
            for model_config, error, message in cases:
                with self.subTest(variant=config_type.__name__, message=message):
                    with self.assertRaisesRegex(error, message):
                        config_type(
                            residual_dim=None, model_config=model_config
                        ).build()

    def test_explicit_final_bias_enables_both_weighted_stack_variants(self):
        for config_type in (WeightedResidualConfig, WeightedBlendResidualConfig):
            with self.subTest(variant=config_type.__name__):
                stack = _coefficient_stack_config(
                    bias_flag=False, last_layer_bias_option=LastLayerBiasOptions.ENABLED
                )
                residual = config_type(residual_dim=2, model_config=stack).build()
                self.assertIsNotNone(residual.model[-1].model.bias_params)
                self.assertFalse(stack.layer_config.layer_model_config.bias_flag)



if __name__ == "__main__":
    unittest.main()
