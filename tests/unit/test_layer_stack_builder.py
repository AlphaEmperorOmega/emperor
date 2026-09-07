import unittest
from copy import deepcopy
from dataclasses import dataclass

from torch import Tensor, nn

from emperor.config import ConfigBase
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    GateConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerGateOptions,
    LayerNormPositionOptions,
    LayerStack,
    LayerStackConfig,
    NormalizationOptions,
)
from emperor.layers._layer.pipeline.normalization_variants import (
    DynamicErf,
    DynamicISRU,
    DynamicTanh,
)
from emperor.layers._stack.builder import LayerStackBuilder
from emperor.linears import LinearLayerConfig
from emperor.nn import Module


class _BiaslessModel(Module):
    def forward(self, values: Tensor) -> Tensor:
        return values


@dataclass
class _BiaslessModelConfig(ConfigBase):
    input_dim: int | None = None
    output_dim: int | None = None

    def build(self, overrides: ConfigBase | None = None) -> Module:
        return _BiaslessModel()


def make_gate_config(dim: int) -> GateConfig:
    return GateConfig(
        option=LayerGateOptions.MULTIPLIER,
        activation=ActivationOptions.SIGMOID,
        model_config=LayerStackConfig(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            num_layers=1,
            apply_output_postprocessing_flag=False,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            shared_gate_config=None,
            shared_halting_config=None,
            shared_memory_config=None,
            layer_config=LayerConfig(
                activation=ActivationOptions.DISABLED,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                residual_config=None,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(bias_flag=True),
            ),
        ),
    )


def make_stack_config(
    *,
    apply_output_postprocessing: bool = True,
    last_layer_bias: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT,
    bias_flag: bool = True,
    gate_config: GateConfig | None = None,
    layer_model_config: ConfigBase | None = None,
) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=4,
        hidden_dim=4,
        output_dim=4,
        num_layers=2,
        apply_output_postprocessing_flag=apply_output_postprocessing,
        last_layer_bias_option=last_layer_bias,
        shared_gate_config=None,
        shared_halting_config=None,
        shared_memory_config=None,
        layer_config=LayerConfig(
            activation=ActivationOptions.RELU,
            dropout_probability=0.25,
            layer_norm_position=LayerNormPositionOptions.AFTER,
            residual_config=AdditiveResidualConfig(),
            gate_config=gate_config,
            halting_config=None,
            memory_config=None,
            layer_model_config=layer_model_config
            if layer_model_config is not None
            else LinearLayerConfig(bias_flag=bias_flag),
        ),
    )


class TestLayerStackBuilder(unittest.TestCase):
    def test_preserves_normalization_selection_through_stack_overrides(self):
        module_types = {
            NormalizationOptions.RMS_NORM: nn.RMSNorm,
            NormalizationOptions.LAYER_NORM: nn.LayerNorm,
            NormalizationOptions.DYNAMIC_TANH: DynamicTanh,
            NormalizationOptions.DERF: DynamicErf,
            NormalizationOptions.DYISRU: DynamicISRU,
        }
        for normalization in NormalizationOptions:
            for output_postprocessing in (True, False):
                with self.subTest(
                    normalization=normalization,
                    output_postprocessing=output_postprocessing,
                ):
                    cfg = make_stack_config(
                        apply_output_postprocessing=output_postprocessing,
                        last_layer_bias=LastLayerBiasOptions.DISABLED,
                    )
                    cfg.layer_config.normalization = normalization
                    original = deepcopy(cfg)
                    stack = LayerStack(cfg)
                    first, final = stack.layers
                    expected_type = module_types[normalization]

                    self.assertIsInstance(first.normalization.module, expected_type)
                    self.assertIs(final.cfg.normalization, normalization)
                    if output_postprocessing:
                        self.assertIsInstance(final.normalization.module, expected_type)
                        self.assertIsNot(
                            first.normalization.module, final.normalization.module
                        )
                    else:
                        self.assertIsNone(final.normalization.module)
                    self.assertEqual(cfg, original)

    def test_builds_layers_in_dimension_order_and_marks_only_the_last(self):
        builder = LayerStackBuilder(
            make_stack_config(),
            supports_rectangular_gate=False,
        )

        layers = builder.build_layer_stack(((4, 4), (4, 6), (6, 2)))

        self.assertEqual(
            [(layer.input_dim, layer.output_dim) for layer in layers],
            [(4, 4), (4, 6), (6, 2)],
        )
        self.assertEqual(
            [layer.halting.is_terminal for layer in layers],
            [False, False, True],
        )
        self.assertEqual(len({id(layer.cfg) for layer in layers}), len(layers))
        self.assertEqual(
            len({id(layer.cfg.layer_model_config) for layer in layers}),
            len(layers),
        )

    def test_disables_only_final_layer_output_postprocessing(self):
        builder = LayerStackBuilder(
            make_stack_config(apply_output_postprocessing=False),
            supports_rectangular_gate=False,
        )

        first_layer, final_layer = builder.build_layer_stack(((4, 4), (4, 4)))

        self.assertEqual(
            first_layer.postprocessing.activation_function,
            ActivationOptions.RELU,
        )
        self.assertIsNotNone(first_layer.postprocessing.dropout)
        self.assertEqual(
            first_layer.normalization.position,
            LayerNormPositionOptions.AFTER,
        )
        self.assertIsNotNone(first_layer.residual.config)
        self.assertEqual(
            final_layer.postprocessing.activation_function,
            ActivationOptions.DISABLED,
        )
        self.assertIsNone(final_layer.postprocessing.dropout)
        self.assertEqual(
            final_layer.normalization.position,
            LayerNormPositionOptions.DISABLED,
        )
        self.assertIsNone(final_layer.residual.config)

    def test_applies_last_layer_bias_policy_without_mutating_base_config(self):
        cases = (
            (LastLayerBiasOptions.DEFAULT, False, False),
            (LastLayerBiasOptions.DEFAULT, True, True),
            (LastLayerBiasOptions.DISABLED, True, False),
            (LastLayerBiasOptions.ENABLED, False, True),
        )

        for bias_option, base_bias_flag, expected_final_bias_flag in cases:
            with self.subTest(
                bias_option=bias_option,
                base_bias_flag=base_bias_flag,
            ):
                stack_config = make_stack_config(
                    last_layer_bias=bias_option,
                    bias_flag=base_bias_flag,
                )
                original_stack_config = deepcopy(stack_config)
                builder = LayerStackBuilder(
                    stack_config,
                    supports_rectangular_gate=False,
                )

                first_layer, final_layer = builder.build_layer_stack(((4, 4), (4, 4)))

                self.assertEqual(first_layer.model.bias_flag, base_bias_flag)
                self.assertEqual(
                    final_layer.model.bias_flag,
                    expected_final_bias_flag,
                )
                self.assertEqual(stack_config, original_stack_config)

    def test_ignores_last_layer_bias_policy_for_biasless_model_config(self):
        builder = LayerStackBuilder(
            make_stack_config(
                last_layer_bias=LastLayerBiasOptions.ENABLED,
                layer_model_config=_BiaslessModelConfig(),
            ),
            supports_rectangular_gate=False,
        )

        layers = builder.build_layer_stack(((4, 4),))

        self.assertIsInstance(layers[0].model, _BiaslessModel)

    def test_disables_residual_for_rectangular_dimensions(self):
        builder = LayerStackBuilder(
            make_stack_config(),
            supports_rectangular_gate=False,
        )

        rectangular_layer = builder.build_layer_stack(((4, 3),))[0]

        self.assertIsNone(rectangular_layer.residual.config)

    def test_rectangular_gate_policy_respects_stack_capability(self):
        for supports_rectangular_gate in (False, True):
            with self.subTest(
                supports_rectangular_gate=supports_rectangular_gate,
            ):
                builder = LayerStackBuilder(
                    make_stack_config(gate_config=make_gate_config(3)),
                    supports_rectangular_gate=supports_rectangular_gate,
                )

                rectangular_layer = builder.build_layer_stack(((4, 3),))[0]

                if supports_rectangular_gate:
                    self.assertIsNotNone(rectangular_layer.postprocessing.gate)
                else:
                    self.assertIsNone(rectangular_layer.postprocessing.gate)

    def test_rejects_unsupported_last_layer_bias_option(self):
        stack_config = make_stack_config()
        stack_config.last_layer_bias_option = object()
        builder = LayerStackBuilder(
            stack_config,
            supports_rectangular_gate=False,
        )

        with self.assertRaisesRegex(
            ValueError,
            r"^Unsupported last layer bias option .* for LayerStack\.$",
        ):
            builder.build_layer_stack(((4, 4),))

    def test_is_stored_without_becoming_a_registered_model_module(self):
        stack = LayerStack(make_stack_config())

        self.assertIsInstance(
            stack.stack_layer_builder,
            LayerStackBuilder,
        )
        self.assertNotIsInstance(stack.stack_layer_builder, Module)
        self.assertIsInstance(stack.layers, nn.Sequential)
        self.assertEqual(
            tuple(name for name, _module in stack.named_children()),
            ("shared_controllers", "layers"),
        )


if __name__ == "__main__":
    unittest.main()
