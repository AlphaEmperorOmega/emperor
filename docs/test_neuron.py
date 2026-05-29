import unittest

import torch
import torch.nn as nn

from dataclasses import dataclass
from torch import Tensor

from emperor.base.layer import LayerConfig, LayerStackConfig
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.base.utils import ConfigBase, Module, optional_field
from emperor.linears.core.config import LinearLayerConfig
from emperor.neuron import (
    Axons,
    AxonsConfig,
    Neuron,
    NeuronCluster,
    NeuronClusterConfig,
    NeuronConfig,
    Nucleus,
    NucleusConfig,
    Terminal,
    TerminalConfig,
    TerminalRangeOptions,
    TerminalZAxisOffsetOptions,
)
from emperor.sampler.core.config import RouterConfig, SamplerConfig


ROUTER_CONFIG_UNSET = object()


@dataclass
class TestProjectionConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    scale: float | None = optional_field("Initial constant weight value.")

    def _registry_owner(self) -> type:
        return TestProjectionModel


class TestProjectionModel(Module):
    def __init__(
        self,
        cfg: TestProjectionConfig,
        overrides: TestProjectionConfig | None = None,
    ):
        super().__init__()
        self.cfg: TestProjectionConfig = self._override_config(cfg, overrides)
        self.weight = nn.Parameter(
            torch.full(
                (self.cfg.input_dim, self.cfg.output_dim),
                float(self.cfg.scale),
            )
        )

    def forward(self, input: Tensor) -> Tensor:
        return input @ self.weight


class NeuronTestCase(unittest.TestCase):
    batch_size = 3
    input_dim = 4

    def projection_config(
        self,
        input_dim: int = 4,
        output_dim: int = 4,
        scale: float = 1.0,
    ) -> TestProjectionConfig:
        return TestProjectionConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            scale=scale,
        )

    def terminal_total_connections(
        self,
        xy_axis_range: TerminalRangeOptions = TerminalRangeOptions.ONE,
        z_axis_range: TerminalRangeOptions = TerminalRangeOptions.ONE,
    ) -> int:
        return (xy_axis_range.value * 2 + 1) ** 2 * (z_axis_range.value + 1)

    def router_config(
        self,
        input_dim: int,
        num_experts: int,
    ) -> RouterConfig:
        return RouterConfig(
            input_dim=input_dim,
            num_experts=num_experts,
            noisy_topk_flag=False,
            model_config=LayerStackConfig(
                input_dim=input_dim,
                hidden_dim=max(input_dim, num_experts),
                output_dim=num_experts,
                num_layers=1,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_pipeline_flag=False,
                layer_config=LayerConfig(
                    activation=ActivationOptions.DISABLED,
                    residual_flag=False,
                    dropout_probability=0.0,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    gate_config=None,
                    halting_config=None,
                    memory_config=None,
                    shared_halting_flag=False,
                    layer_model_config=LinearLayerConfig(bias_flag=True),
                ),
            ),
        )

    def sampler_config(
        self,
        input_dim: int = 4,
        num_experts: int | None = None,
        top_k: int = 2,
        router_config=ROUTER_CONFIG_UNSET,
    ) -> SamplerConfig:
        num_experts = num_experts or self.terminal_total_connections()
        if router_config is ROUTER_CONFIG_UNSET:
            router_config = self.router_config(input_dim, num_experts)
        return SamplerConfig(
            top_k=top_k,
            threshold=0.0,
            filter_above_threshold=False,
            num_topk_samples=0,
            normalize_probabilities_flag=False,
            noisy_topk_flag=False,
            num_experts=num_experts,
            coefficient_of_variation_loss_weight=0.0,
            switch_loss_weight=0.0,
            zero_centred_loss_weight=0.0,
            mutual_information_loss_weight=0.0,
            router_config=router_config,
        )

    def terminal_config(
        self,
        input_dim: int = 4,
        xy_axis_range: TerminalRangeOptions = TerminalRangeOptions.ONE,
        z_axis_range: TerminalRangeOptions = TerminalRangeOptions.ONE,
        z_axis_offset: TerminalZAxisOffsetOptions = TerminalZAxisOffsetOptions.ZERO,
        sampler_config: SamplerConfig | None = None,
    ) -> TerminalConfig:
        num_experts = self.terminal_total_connections(xy_axis_range, z_axis_range)
        return TerminalConfig(
            input_dim=input_dim,
            x_axis_position=1,
            y_axis_position=1,
            z_axis_position=1,
            xy_axis_range=xy_axis_range,
            z_axis_range=z_axis_range,
            z_axis_offset=z_axis_offset,
            sampler_config=sampler_config
            or self.sampler_config(input_dim=input_dim, num_experts=num_experts),
        )

    def neuron_config(self) -> NeuronConfig:
        return NeuronConfig(
            nucleus_config=NucleusConfig(
                model_config=self.projection_config(
                    input_dim=self.input_dim,
                    output_dim=self.input_dim,
                    scale=0.25,
                )
            ),
            axons_config=AxonsConfig(memory_config=None),
            terminal_config=self.terminal_config(input_dim=self.input_dim),
        )


class TestNeuronConfigs(NeuronTestCase):
    def test_config_build_dispatches_to_runtime_classes(self):
        cases = [
            (NucleusConfig(model_config=self.projection_config()), Nucleus),
            (AxonsConfig(memory_config=None), Axons),
            (self.terminal_config(), Terminal),
            (self.neuron_config(), Neuron),
            (
                NeuronClusterConfig(
                    x_axis_total_neurons=1,
                    y_axis_total_neurons=1,
                    z_axis_total_neurons=1,
                    growth_threshold=None,
                    neuron_config=self.neuron_config(),
                ),
                NeuronCluster,
            ),
        ]

        for cfg, expected_type in cases:
            with self.subTest(expected_type=expected_type.__name__):
                self.assertIsInstance(cfg.build(), expected_type)

    def test_overrides_replace_config_fields(self):
        cfg = NucleusConfig(model_config=self.projection_config(scale=1.0))
        overrides = NucleusConfig(model_config=self.projection_config(scale=2.0))

        model = cfg.build(overrides)
        output = model(torch.ones(1, self.input_dim))

        torch.testing.assert_close(
            output,
            torch.full((1, self.input_dim), 8.0),
        )


class TestNucleus(NeuronTestCase):
    def test_delegates_to_config_owned_model(self):
        model = NucleusConfig(
            model_config=self.projection_config(
                input_dim=2,
                output_dim=3,
                scale=0.5,
            )
        ).build()
        input_batch = torch.tensor([[1.0, 3.0]])

        output = model(input_batch)

        torch.testing.assert_close(output, torch.full((1, 3), 2.0))

    def test_gradient_flows_through_nucleus_model(self):
        model = NucleusConfig(
            model_config=self.projection_config(
                input_dim=self.input_dim,
                output_dim=2,
                scale=0.5,
            )
        ).build()
        input_batch = torch.ones(self.batch_size, self.input_dim)

        model(input_batch).sum().backward()

        self.assertIsNotNone(model.model.weight.grad)
        self.assertTrue(torch.any(model.model.weight.grad.abs() > 0.0))


class TestAxons(NeuronTestCase):
    def test_identity_path_preserves_input(self):
        model = AxonsConfig(memory_config=None).build()
        input_batch = torch.randn(self.batch_size, self.input_dim)

        output = model(input_batch)

        self.assertIs(output, input_batch)


class TestTerminal(NeuronTestCase):
    def test_initializes_connection_math(self):
        model = self.terminal_config().build()

        self.assertEqual(model.input_dim, self.input_dim)
        self.assertEqual(model.total_neuron_connections, 18)
        self.assertEqual(model.neuron_connections.shape, (18, 3))
        torch.testing.assert_close(
            model.neuron_connections[:4],
            torch.tensor(
                [
                    [0, 0, 1],
                    [0, 0, 2],
                    [0, 1, 1],
                    [0, 1, 2],
                ]
            ),
        )

    def test_forward_returns_selected_coordinates(self):
        model = self.terminal_config().build()
        input_batch = torch.randn(self.batch_size, self.input_dim)

        output, probabilities, selected_neurons = model(input_batch)

        self.assertIs(output, input_batch)
        self.assertEqual(probabilities.shape, (self.batch_size, 2))
        self.assertEqual(selected_neurons.shape, (self.batch_size, 2, 3))

    def test_logits_only_path_works_without_router(self):
        total_connections = self.terminal_total_connections()
        sampler_config = self.sampler_config(
            input_dim=total_connections,
            num_experts=total_connections,
            router_config=None,
        )
        model = self.terminal_config(
            input_dim=total_connections,
            sampler_config=sampler_config,
        ).build()

        _, probabilities, selected_neurons = model(
            torch.randn(self.batch_size, total_connections)
        )

        self.assertEqual(probabilities.shape, (self.batch_size, 2))
        self.assertEqual(selected_neurons.shape, (self.batch_size, 2, 3))


class TestNeuron(NeuronTestCase):
    def test_composes_nucleus_axons_and_terminal(self):
        model = self.neuron_config().build()
        input_batch = torch.randn(self.batch_size, self.input_dim)

        output, probabilities, selected_neurons = model(input_batch)

        self.assertEqual(output.shape, (self.batch_size, self.input_dim))
        self.assertEqual(probabilities.shape, (self.batch_size, 2))
        self.assertEqual(selected_neurons.shape, (self.batch_size, 2, 3))
        self.assertEqual(model.batch_counter.item(), 1)


class TestNeuronCluster(NeuronTestCase):
    def test_initializes_expected_coordinate_keys(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            growth_threshold=None,
            neuron_config=self.neuron_config(),
        ).build()

        self.assertEqual(
            set(model.cluster.keys()),
            {"neuron_1_1_1", "neuron_2_1_1"},
        )

    def test_forward_returns_neuron_tuple_and_grows_at_threshold(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            growth_threshold=1,
            neuron_config=self.neuron_config(),
        ).build()

        output, probabilities, selected_neurons = model(
            torch.randn(self.batch_size, self.input_dim)
        )

        self.assertEqual(output.shape, (self.batch_size, self.input_dim))
        self.assertEqual(probabilities.shape, (self.batch_size, 2))
        self.assertEqual(selected_neurons.shape, (self.batch_size, 2, 3))
        self.assertEqual(len(model.cluster), 2)
        self.assertIn("neuron_1_1_1", model.cluster)


class TestNeuronValidation(NeuronTestCase):
    def test_missing_child_config_raises(self):
        with self.assertRaises(ValueError):
            NeuronConfig(
                nucleus_config=None,
                axons_config=AxonsConfig(memory_config=None),
                terminal_config=self.terminal_config(),
            ).build()

    def test_invalid_axis_offset_raises(self):
        with self.assertRaises(ValueError):
            self.terminal_config(
                z_axis_range=TerminalRangeOptions.ONE,
                z_axis_offset=TerminalZAxisOffsetOptions.ONE,
            ).build()

    def test_non_positive_cluster_dimensions_raise(self):
        with self.assertRaises(ValueError):
            NeuronClusterConfig(
                x_axis_total_neurons=0,
                y_axis_total_neurons=1,
                z_axis_total_neurons=1,
                growth_threshold=None,
                neuron_config=self.neuron_config(),
            ).build()

    def test_wrong_forward_tensor_rank_raises(self):
        model = self.neuron_config().build()

        with self.assertRaises(ValueError):
            model(torch.randn(2, 3, self.input_dim))

    def test_missing_router_config_raises_when_terminal_requires_routing(self):
        total_connections = self.terminal_total_connections()
        sampler_config = self.sampler_config(
            input_dim=self.input_dim,
            num_experts=total_connections,
            router_config=None,
        )

        with self.assertRaises(ValueError):
            self.terminal_config(
                input_dim=self.input_dim,
                sampler_config=sampler_config,
            ).build()


if __name__ == "__main__":
    unittest.main()
