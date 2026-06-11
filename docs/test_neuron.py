import os
import tempfile
import unittest

from unittest import mock

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
from emperor.halting.config import StickBreakingConfig
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.linears.core.config import LinearLayerConfig
from emperor.neuron import (
    Axons,
    AxonsConfig,
    Neuron,
    NeuronCluster,
    NeuronClusterConfig,
    NeuronClusterTrace,
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


class ScriptedTerminal(nn.Module):
    def __init__(self, routes: list[list[int]]):
        super().__init__()
        self.register_buffer(
            "neuron_connections",
            torch.tensor(routes, dtype=torch.long),
            persistent=False,
        )


class ScriptedNeuron(nn.Module):
    def __init__(
        self,
        routes: list[list[int]],
        probabilities: list[float],
        delta: list[float],
        auxiliary_loss: float = 0.0,
    ):
        super().__init__()
        self.terminal = ScriptedTerminal(routes)
        self.register_buffer(
            "routes",
            torch.tensor(routes, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "probabilities",
            torch.tensor(probabilities, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "delta",
            torch.tensor(delta, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "batch_counter",
            torch.tensor(0, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "atrophy_counter",
            torch.tensor(0, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "route_call_counter",
            torch.tensor(0, dtype=torch.int64),
            persistent=True,
        )
        self.auxiliary_loss = auxiliary_loss

    def process_signal(self, input: Tensor) -> Tensor:
        self.batch_counter += 1
        return input + self.delta.to(device=input.device, dtype=input.dtype)

    def route_signal(self, processed_signal: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        self.route_call_counter += 1
        batch_size = processed_signal.shape[0]
        probabilities = self.probabilities.to(
            device=processed_signal.device,
            dtype=processed_signal.dtype,
        ).expand(batch_size, -1)
        routes = self.routes.to(device=processed_signal.device).expand(
            batch_size, -1, -1
        )
        return (
            probabilities,
            routes,
            processed_signal.new_tensor(self.auxiliary_loss),
        )

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        output = self.process_signal(input)
        probabilities, routes, auxiliary_loss = self.route_signal(output)
        return output, probabilities, routes, auxiliary_loss


class ScriptedSampler(nn.Module):
    def __init__(
        self,
        indices: list[int],
        probabilities: list[float],
        auxiliary_loss: float = 0.0,
    ):
        super().__init__()
        self.register_buffer(
            "indices",
            torch.tensor(indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "probabilities",
            torch.tensor(probabilities, dtype=torch.float32),
            persistent=False,
        )
        self.auxiliary_loss = auxiliary_loss

    def sample_probabilities_and_indices(
        self,
        input: Tensor,
    ) -> tuple[Tensor, Tensor, None, Tensor]:
        batch_size = input.shape[0]
        return (
            self.probabilities.to(device=input.device, dtype=input.dtype).expand(
                batch_size,
                -1,
            ),
            self.indices.to(device=input.device).expand(batch_size, -1),
            None,
            input.new_tensor(self.auxiliary_loss),
        )


@dataclass
class RecordingHaltingState:
    halt_mask: Tensor
    hidden: Tensor


class RecordingHaltingModel(nn.Module):
    def __init__(
        self,
        halt_after_updates: int | None = None,
        finalize_offset: float = 0.0,
        ponder_loss: float = 0.0,
    ):
        super().__init__()
        self.halt_after_updates = halt_after_updates
        self.finalize_offset = finalize_offset
        self.ponder_loss = ponder_loss
        self.inputs: list[Tensor] = []
        self.update_count = 0
        self.finalize_count = 0

    def update_halting_state(
        self,
        previous_state: RecordingHaltingState | None,
        model_hidden_state: Tensor,
    ) -> tuple[RecordingHaltingState, Tensor]:
        self.update_count += 1
        self.inputs.append(model_hidden_state)
        halt_mask = torch.zeros(
            model_hidden_state.shape[0],
            dtype=torch.bool,
            device=model_hidden_state.device,
        )
        if (
            self.halt_after_updates is not None
            and self.update_count >= self.halt_after_updates
        ):
            halt_mask = torch.ones_like(halt_mask)
        if previous_state is not None:
            halt_mask = halt_mask | previous_state.halt_mask
        return RecordingHaltingState(halt_mask, model_hidden_state), model_hidden_state

    def finalize_weighted_accumulation(
        self,
        state: RecordingHaltingState,
        current_hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self.finalize_count += 1
        return (
            state.hidden + self.finalize_offset,
            current_hidden.new_full((current_hidden.shape[0],), self.ponder_loss),
        )


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

    def halting_config(
        self,
        input_dim: int = 4,
        threshold: float = 0.99,
    ) -> StickBreakingConfig:
        return StickBreakingConfig(
            input_dim=input_dim,
            threshold=threshold,
            halting_dropout=0.0,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            halting_gate_config=LayerStackConfig(
                input_dim=input_dim,
                hidden_dim=input_dim,
                output_dim=2,
                num_layers=1,
                last_layer_bias_option=LastLayerBiasOptions.DISABLED,
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

    def full_sampler_neuron_config(self) -> NeuronConfig:
        total_connections = self.terminal_total_connections()
        return NeuronConfig(
            nucleus_config=NucleusConfig(
                model_config=self.projection_config(
                    input_dim=self.input_dim,
                    output_dim=self.input_dim,
                    scale=0.25,
                )
            ),
            axons_config=AxonsConfig(memory_config=None),
            terminal_config=self.terminal_config(
                input_dim=self.input_dim,
                sampler_config=self.sampler_config(
                    input_dim=self.input_dim,
                    num_experts=total_connections,
                    top_k=total_connections,
                ),
            ),
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
                    max_steps=1,
                    growth_threshold=None,
                    neuron_config=self.neuron_config(),
                ),
                NeuronCluster,
            ),
        ]

        for cfg, expected_type in cases:
            with self.subTest(expected_type=expected_type.__name__):
                self.assertIsInstance(cfg.build(), expected_type)

    def test_cluster_config_builds_with_halting(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            halting_config=self.halting_config(input_dim=self.input_dim),
            neuron_config=self.neuron_config(),
        ).build()

        self.assertIsInstance(model, NeuronCluster)
        self.assertIsNotNone(model.halting_model)

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

        output, probabilities, selected_neurons, auxiliary_loss = model(input_batch)

        self.assertIs(output, input_batch)
        self.assertEqual(probabilities.shape, (self.batch_size, 2))
        self.assertEqual(selected_neurons.shape, (self.batch_size, 2, 3))
        self.assertIsInstance(auxiliary_loss, Tensor)
        self.assertEqual(auxiliary_loss.shape, ())

    def test_sparse_forward_returns_matrix_shapes(self):
        total_connections = self.terminal_total_connections()
        model = self.terminal_config(
            sampler_config=self.sampler_config(
                input_dim=self.input_dim,
                num_experts=total_connections,
                top_k=1,
            )
        ).build()

        _, probabilities, selected_neurons, _ = model(
            torch.randn(self.batch_size, self.input_dim)
        )

        self.assertEqual(probabilities.shape, (self.batch_size, 1))
        self.assertEqual(selected_neurons.shape, (self.batch_size, 1, 3))

    def test_full_forward_returns_all_coordinate_shapes(self):
        total_connections = self.terminal_total_connections()
        model = self.terminal_config(
            sampler_config=self.sampler_config(
                input_dim=self.input_dim,
                num_experts=total_connections,
                top_k=total_connections,
            )
        ).build()

        _, probabilities, selected_neurons, _ = model(
            torch.randn(self.batch_size, self.input_dim)
        )

        self.assertEqual(probabilities.shape, (self.batch_size, total_connections))
        self.assertEqual(selected_neurons.shape, (self.batch_size, total_connections, 3))

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

        _, probabilities, selected_neurons, auxiliary_loss = model(
            torch.randn(self.batch_size, total_connections)
        )

        self.assertEqual(probabilities.shape, (self.batch_size, 2))
        self.assertEqual(selected_neurons.shape, (self.batch_size, 2, 3))
        self.assertIsInstance(auxiliary_loss, Tensor)


class TestNeuron(NeuronTestCase):
    def test_composes_nucleus_axons_and_terminal(self):
        model = self.neuron_config().build()
        input_batch = torch.randn(self.batch_size, self.input_dim)

        output, probabilities, selected_neurons, auxiliary_loss = model(input_batch)

        self.assertEqual(output.shape, (self.batch_size, self.input_dim))
        self.assertEqual(probabilities.shape, (self.batch_size, 2))
        self.assertEqual(selected_neurons.shape, (self.batch_size, 2, 3))
        self.assertIsInstance(auxiliary_loss, Tensor)
        self.assertEqual(model.batch_counter.item(), 1)


class TestNeuronCluster(NeuronTestCase):
    def scripted_cluster(
        self,
        max_steps: int = 1,
        halting_model: nn.Module | None = None,
        input_dim: int = 1,
        x_axis_total_neurons: int = 4,
        y_axis_total_neurons: int = 1,
        z_axis_total_neurons: int = 1,
        initial_x_axis_total_neurons: int | None = None,
        initial_y_axis_total_neurons: int | None = None,
        initial_z_axis_total_neurons: int | None = None,
    ) -> NeuronCluster:
        neuron_config = NeuronConfig(
            nucleus_config=NucleusConfig(
                model_config=self.projection_config(
                    input_dim=input_dim,
                    output_dim=input_dim,
                    scale=0.25,
                )
            ),
            axons_config=AxonsConfig(memory_config=None),
            terminal_config=self.terminal_config(input_dim=input_dim),
        )
        model = NeuronClusterConfig(
            x_axis_total_neurons=x_axis_total_neurons,
            y_axis_total_neurons=y_axis_total_neurons,
            z_axis_total_neurons=z_axis_total_neurons,
            initial_x_axis_total_neurons=initial_x_axis_total_neurons,
            initial_y_axis_total_neurons=initial_y_axis_total_neurons,
            initial_z_axis_total_neurons=initial_z_axis_total_neurons,
            max_steps=max_steps,
            growth_threshold=None,
            neuron_config=neuron_config,
        ).build()
        model.entry_sampler = ScriptedSampler(indices=[0], probabilities=[1.0])
        model.halting_model = halting_model
        return model

    def test_initializes_expected_coordinate_keys(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.neuron_config(),
        ).build()

        self.assertEqual(
            set(model.cluster.keys()),
            {"neuron_1_1_1", "neuron_2_1_1"},
        )

    def test_sparse_initial_grid_centers_within_xy_capacity(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=10,
            y_axis_total_neurons=10,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=2,
            initial_y_axis_total_neurons=2,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.neuron_config(),
        ).build()

        expected_keys = {
            f"neuron_{x}_{y}_1" for x in range(5, 7) for y in range(5, 7)
        }
        self.assertEqual(
            set(model.cluster.keys()),
            expected_keys,
        )
        torch.testing.assert_close(
            model.entry_coordinates,
            torch.tensor(
                [
                    [5, 5, 1],
                    [5, 6, 1],
                    [6, 5, 1],
                    [6, 6, 1],
                ]
            ),
        )

    def test_sparse_initial_grid_centers_within_xyz_capacity(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=5,
            y_axis_total_neurons=4,
            z_axis_total_neurons=3,
            initial_x_axis_total_neurons=3,
            initial_y_axis_total_neurons=2,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.neuron_config(),
        ).build()

        expected_keys = {
            f"neuron_{x}_{y}_2" for x in range(2, 5) for y in range(2, 4)
        }
        self.assertEqual(set(model.cluster.keys()), expected_keys)
        torch.testing.assert_close(
            model.entry_coordinates,
            torch.tensor(
                [
                    [2, 2, 2],
                    [2, 3, 2],
                    [3, 2, 2],
                    [3, 3, 2],
                    [4, 2, 2],
                    [4, 3, 2],
                ]
            ),
        )

    def test_feature_last_inputs_restore_original_non_feature_shape(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()

        for shape in (
            (self.batch_size, self.input_dim),
            (2, 3, self.input_dim),
            (2, 2, 3, self.input_dim),
        ):
            with self.subTest(shape=shape):
                output, auxiliary_loss = model(torch.randn(*shape))

                self.assertEqual(output.shape, shape)
                self.assertEqual(auxiliary_loss.shape, ())

    def test_forward_returns_output_and_grows_at_threshold(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()

        output, auxiliary_loss = model(torch.randn(self.batch_size, self.input_dim))

        self.assertEqual(output.shape, (self.batch_size, self.input_dim))
        self.assertIsInstance(auxiliary_loss, Tensor)
        self.assertEqual(auxiliary_loss.shape, ())
        self.assertEqual(len(model.cluster), 2)
        self.assertIn("neuron_1_1_1", model.cluster)

    def test_growth_expands_from_centered_seed(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=10,
            y_axis_total_neurons=10,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=2,
            initial_y_axis_total_neurons=2,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()

        seed_keys = {f"neuron_{x}_{y}_1" for x in range(5, 7) for y in range(5, 7)}
        expected_growth_keys = {
            "neuron_4_5_1",
            "neuron_4_6_1",
            "neuron_5_4_1",
            "neuron_5_7_1",
            "neuron_6_4_1",
            "neuron_6_7_1",
            "neuron_7_5_1",
            "neuron_7_6_1",
        }
        self.assertEqual(set(model.cluster.keys()), seed_keys)

        model(torch.randn(self.batch_size, self.input_dim))

        cluster_keys = set(model.cluster.keys())
        added_keys = cluster_keys - seed_keys
        self.assertTrue(seed_keys.issubset(cluster_keys))
        self.assertEqual(len(cluster_keys), 5)
        self.assertEqual(len(added_keys), 1)
        self.assertTrue(added_keys.issubset(expected_growth_keys))

    def test_propagation_fires_deeper_layer_branch(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=2,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()

        output, _ = model(torch.randn(self.batch_size, self.input_dim))

        self.assertEqual(output.shape, (self.batch_size, self.input_dim))
        self.assertEqual(int(model.cluster["neuron_1_1_1"].batch_counter.item()), 2)
        self.assertEqual(int(model.cluster["neuron_1_1_2"].batch_counter.item()), 1)

    def test_growth_prefers_neuron_with_highest_counter(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=4,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=2,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=100,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        self.assertEqual(
            set(model.cluster.keys()),
            {"neuron_2_1_1", "neuron_3_1_1"},
        )
        model.cluster["neuron_2_1_1"].batch_counter.fill_(100)
        model.cluster["neuron_3_1_1"].batch_counter.fill_(200)

        model(torch.randn(self.batch_size, self.input_dim))

        self.assertIn("neuron_4_1_1", model.cluster)
        self.assertNotIn("neuron_1_1_1", model.cluster)
        self.assertEqual(
            int(model.cluster["neuron_3_1_1"].batch_counter.item()),
            0,
        )

    def escape_growth_cluster(self) -> NeuronCluster:
        return NeuronClusterConfig(
            x_axis_total_neurons=5,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            escape_driven_growth_flag=True,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()

    def test_escape_driven_growth_targets_most_escaped_coordinate(self):
        model = self.escape_growth_cluster()
        self.assertEqual(set(model.cluster.keys()), {"neuron_3_1_1"})
        model.escape_counts[3, 0, 0] = 10_000

        model(torch.randn(self.batch_size, self.input_dim))

        self.assertIn("neuron_4_1_1", model.cluster)
        self.assertNotIn("neuron_2_1_1", model.cluster)
        self.assertEqual(int(model.escape_counts[3, 0, 0].item()), 0)

    def test_escape_driven_growth_without_planted_signal_keeps_closest(self):
        model = self.escape_growth_cluster()

        model(torch.randn(self.batch_size, self.input_dim))

        self.assertIn("neuron_2_1_1", model.cluster)
        self.assertNotIn("neuron_4_1_1", model.cluster)

    def test_mitosis_initialization_copies_parent_parameters(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            mitosis_initialization_flag=True,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        parent = model.cluster["neuron_1_1_1"]
        parent.nucleus.model.weight.data.fill_(0.7)

        model(torch.randn(self.batch_size, self.input_dim))

        child = model.cluster["neuron_2_1_1"]
        torch.testing.assert_close(
            child.nucleus.model.weight,
            torch.full_like(parent.nucleus.model.weight, 0.7),
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(int(child.batch_counter.item()), 0)
        for child_param, parent_param in zip(
            child.parameters(),
            parent.parameters(),
            strict=True,
        ):
            with self.subTest(shape=tuple(child_param.shape)):
                parent_std = float(
                    parent_param.detach().float().std(correction=0)
                )
                max_difference = float((child_param - parent_param).abs().max())
                if parent_std > 1e-6:
                    self.assertGreater(max_difference, 0.0)
                    self.assertLess(max_difference, 0.1 * parent_std)
                else:
                    self.assertEqual(max_difference, 0.0)

    def test_mitosis_flag_disabled_uses_fresh_initialization(self):
        model = self.growth_cluster_config(growth_threshold=1).build()
        parent = model.cluster["neuron_1_1_1"]
        parent.nucleus.model.weight.data.fill_(0.7)

        model(torch.randn(self.batch_size, self.input_dim))

        child = model.cluster["neuron_2_1_1"]
        torch.testing.assert_close(
            child.nucleus.model.weight,
            torch.full_like(parent.nucleus.model.weight, 0.25),
            rtol=0.0,
            atol=0.0,
        )

    def test_growth_only_targets_positive_coordinates(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()

        for _ in range(3):
            model(torch.randn(self.batch_size, self.input_dim))

        for neuron_name in model.cluster:
            _, x, y, z = neuron_name.split("_")
            self.assertTrue(1 <= int(x) <= 2)
            self.assertTrue(1 <= int(y) <= 1)
            self.assertTrue(1 <= int(z) <= 1)

    def test_eval_forward_does_not_grow_at_threshold(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        model.eval()

        with torch.no_grad():
            output, auxiliary_loss = model(torch.randn(self.batch_size, self.input_dim))

        self.assertEqual(output.shape, (self.batch_size, self.input_dim))
        self.assertEqual(auxiliary_loss.shape, ())
        self.assertEqual(len(model.cluster), 1)

    def test_eval_forward_does_not_increment_batch_counters(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        model.eval()

        with torch.no_grad():
            model(torch.randn(self.batch_size, self.input_dim))

        for neuron_name, neuron in model.cluster.items():
            with self.subTest(neuron_name=neuron_name):
                self.assertEqual(int(neuron.batch_counter.item()), 0)

    def growth_cluster_config(self, growth_threshold: int | None) -> NeuronClusterConfig:
        return NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=growth_threshold,
            neuron_config=self.full_sampler_neuron_config(),
        )

    @unittest.skipUnless(
        torch.distributed.is_available() and torch.distributed.is_gloo_available(),
        "gloo process group support is required",
    )
    def test_growth_proceeds_under_initialized_process_group(self):
        model = self.growth_cluster_config(growth_threshold=1).build()

        with tempfile.TemporaryDirectory() as temp_dir:
            init_file = os.path.join(temp_dir, "process_group_init")
            torch.distributed.init_process_group(
                backend="gloo",
                init_method=f"file://{init_file}",
                rank=0,
                world_size=1,
            )
            try:
                output, _ = model(torch.randn(self.batch_size, self.input_dim))
            finally:
                torch.distributed.destroy_process_group()

        self.assertEqual(output.shape, (self.batch_size, self.input_dim))
        self.assertIn("neuron_2_1_1", model.cluster)

    def test_distributed_training_without_growth_does_not_raise(self):
        eval_model = self.growth_cluster_config(growth_threshold=1).build()
        eval_model.eval()
        growthless_model = self.growth_cluster_config(growth_threshold=None).build()

        with mock.patch.object(
            torch.distributed,
            "is_initialized",
            return_value=True,
        ):
            with self.subTest(case="eval_forward"), torch.no_grad():
                output, _ = eval_model(
                    torch.randn(self.batch_size, self.input_dim)
                )
                self.assertEqual(output.shape, (self.batch_size, self.input_dim))
            with self.subTest(case="growth_disabled"):
                output, _ = growthless_model(
                    torch.randn(self.batch_size, self.input_dim)
                )
                self.assertEqual(output.shape, (self.batch_size, self.input_dim))

    def test_load_state_dict_rebuilds_grown_neurons(self):
        source_model = self.growth_cluster_config(growth_threshold=1).build()
        source_model(torch.randn(self.batch_size, self.input_dim))
        self.assertEqual(len(source_model.cluster), 2)

        target_model = self.growth_cluster_config(growth_threshold=1).build()
        self.assertEqual(len(target_model.cluster), 1)

        target_model.load_state_dict(source_model.state_dict(), strict=True)

        self.assertEqual(
            set(target_model.cluster.keys()),
            set(source_model.cluster.keys()),
        )
        for neuron_name, neuron in source_model.cluster.items():
            with self.subTest(neuron_name=neuron_name):
                self.assertEqual(
                    int(target_model.cluster[neuron_name].batch_counter.item()),
                    int(neuron.batch_counter.item()),
                )

        source_model.eval()
        target_model.eval()
        input_batch = torch.randn(self.batch_size, self.input_dim)
        with torch.no_grad():
            source_output, _ = source_model(input_batch)
            target_output, _ = target_model(input_batch)
        torch.testing.assert_close(target_output, source_output)

    def test_load_state_dict_without_grown_neurons_is_unchanged(self):
        source_model = self.growth_cluster_config(growth_threshold=None).build()
        target_model = self.growth_cluster_config(growth_threshold=None).build()

        target_model.load_state_dict(source_model.state_dict(), strict=True)

        self.assertEqual(
            set(target_model.cluster.keys()),
            set(source_model.cluster.keys()),
        )

    def pruning_cluster_model(
        self,
        pruning_threshold: int | None,
        growth_threshold: int | None = None,
    ) -> "NeuronCluster":
        return NeuronClusterConfig(
            x_axis_total_neurons=5,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=growth_threshold,
            pruning_threshold=pruning_threshold,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()

    def plant_isolated_neuron(self, model, x_coordinate: int) -> str:
        """Adds a real neuron outside the entry neuron's terminal range
        (entry neuron_3_1_1 only reaches x in {2, 3, 4}), so routing never
        process-calls it and its atrophy counter accrues every forward."""
        name = f"neuron_{x_coordinate}_1_1"
        model.cluster[name] = model._initialize_neuron(x_coordinate, 1, 1)
        return name

    def test_forward_prunes_idle_neuron_at_threshold(self):
        model = self.pruning_cluster_model(pruning_threshold=2)
        planted_name = self.plant_isolated_neuron(model, 5)
        input_batch = torch.randn(self.batch_size, self.input_dim)

        model(input_batch)
        self.assertIn(planted_name, model.cluster)
        self.assertEqual(
            int(model.cluster[planted_name].atrophy_counter.item()),
            1,
        )

        model(input_batch)
        self.assertNotIn(planted_name, model.cluster)
        self.assertIn("neuron_3_1_1", model.cluster)

    def test_used_neuron_atrophy_counter_resets_to_zero(self):
        model = self.pruning_cluster_model(pruning_threshold=10)
        planted_name = self.plant_isolated_neuron(model, 5)
        model.cluster["neuron_3_1_1"].atrophy_counter.fill_(5)

        model(torch.randn(self.batch_size, self.input_dim))

        self.assertEqual(
            int(model.cluster["neuron_3_1_1"].atrophy_counter.item()),
            0,
        )
        self.assertEqual(
            int(model.cluster[planted_name].atrophy_counter.item()),
            1,
        )

    def test_prune_prefers_highest_atrophy_counter(self):
        model = self.pruning_cluster_model(pruning_threshold=2)
        lower_name = self.plant_isolated_neuron(model, 1)
        higher_name = self.plant_isolated_neuron(model, 5)
        model.cluster[lower_name].atrophy_counter.fill_(3)
        model.cluster[higher_name].atrophy_counter.fill_(10)

        model(torch.randn(self.batch_size, self.input_dim))

        self.assertNotIn(higher_name, model.cluster)
        self.assertIn(lower_name, model.cluster)

    def test_prune_tie_break_is_sorted_name(self):
        model = self.pruning_cluster_model(pruning_threshold=2)
        first_name = self.plant_isolated_neuron(model, 1)
        second_name = self.plant_isolated_neuron(model, 5)
        model.cluster[first_name].atrophy_counter.fill_(5)
        model.cluster[second_name].atrophy_counter.fill_(5)

        model(torch.randn(self.batch_size, self.input_dim))

        self.assertNotIn(first_name, model.cluster)
        self.assertIn(second_name, model.cluster)

    def test_at_most_one_prune_per_forward(self):
        model = self.pruning_cluster_model(pruning_threshold=2)
        self.plant_isolated_neuron(model, 1)
        self.plant_isolated_neuron(model, 5)
        model.cluster["neuron_1_1_1"].atrophy_counter.fill_(10)
        model.cluster["neuron_5_1_1"].atrophy_counter.fill_(10)
        input_batch = torch.randn(self.batch_size, self.input_dim)

        model(input_batch)
        self.assertEqual(len(model.cluster), 2)

        model(input_batch)
        self.assertEqual(set(model.cluster.keys()), {"neuron_3_1_1"})

    def test_entry_plane_neurons_are_never_pruned(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=5,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=2,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            pruning_threshold=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        model.entry_sampler = ScriptedSampler(indices=[0], probabilities=[1.0])
        model.cluster = nn.ModuleDict(
            {
                "neuron_2_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1]],
                    probabilities=[1.0],
                    delta=[0.0, 0.0, 0.0, 0.0],
                ),
                "neuron_3_1_1": ScriptedNeuron(
                    routes=[[3, 1, 1]],
                    probabilities=[1.0],
                    delta=[0.0, 0.0, 0.0, 0.0],
                ),
            }
        )
        input_batch = torch.randn(self.batch_size, self.input_dim)

        for _ in range(3):
            model(input_batch)

        self.assertEqual(
            set(model.cluster.keys()),
            {"neuron_2_1_1", "neuron_3_1_1"},
        )
        self.assertGreaterEqual(
            int(model.cluster["neuron_3_1_1"].atrophy_counter.item()),
            1,
        )

    def test_eval_forward_does_not_prune(self):
        model = self.pruning_cluster_model(pruning_threshold=1)
        planted_name = self.plant_isolated_neuron(model, 5)
        model.cluster[planted_name].atrophy_counter.fill_(99)
        model.eval()

        with torch.no_grad():
            model(torch.randn(self.batch_size, self.input_dim))

        self.assertIn(planted_name, model.cluster)
        self.assertEqual(
            int(model.cluster[planted_name].atrophy_counter.item()),
            99,
        )

    def test_pruning_disabled_when_threshold_none(self):
        model = self.pruning_cluster_model(pruning_threshold=None)
        planted_name = self.plant_isolated_neuron(model, 5)
        input_batch = torch.randn(self.batch_size, self.input_dim)

        for _ in range(3):
            model(input_batch)

        self.assertIn(planted_name, model.cluster)
        self.assertEqual(
            int(model.cluster[planted_name].atrophy_counter.item()),
            0,
        )

    def test_grown_neuron_counts_as_used_on_its_birth_forward(self):
        model = self.growth_cluster_config(growth_threshold=1).build()
        model.pruning_threshold = 1

        model(torch.randn(self.batch_size, self.input_dim))

        self.assertIn("neuron_2_1_1", model.cluster)
        self.assertEqual(
            int(model.cluster["neuron_2_1_1"].atrophy_counter.item()),
            0,
        )

    def test_grow_and_prune_same_forward(self):
        model = self.pruning_cluster_model(
            pruning_threshold=2,
            growth_threshold=50,
        )
        planted_name = self.plant_isolated_neuron(model, 5)
        model.cluster[planted_name].atrophy_counter.fill_(10)
        model.cluster["neuron_3_1_1"].batch_counter.fill_(100)

        model(torch.randn(self.batch_size, self.input_dim))

        self.assertIn("neuron_2_1_1", model.cluster)
        self.assertNotIn(planted_name, model.cluster)

    def test_pruned_coordinate_becomes_growth_candidate_again(self):
        model = self.pruning_cluster_model(
            pruning_threshold=1,
            growth_threshold=50,
        )
        planted_name = self.plant_isolated_neuron(model, 5)
        input_batch = torch.randn(self.batch_size, self.input_dim)

        model(input_batch)
        self.assertNotIn(planted_name, model.cluster)

        bridge_name = self.plant_isolated_neuron(model, 4)
        model.cluster[bridge_name].batch_counter.fill_(1000)
        model(input_batch)

        self.assertIn(planted_name, model.cluster)

    def test_load_state_dict_reconciles_pruned_neurons(self):
        source_model = self.pruning_cluster_model(pruning_threshold=None)
        target_model = self.pruning_cluster_model(pruning_threshold=None)
        self.plant_isolated_neuron(target_model, 5)

        target_model.load_state_dict(source_model.state_dict(), strict=True)

        self.assertEqual(
            set(target_model.cluster.keys()),
            set(source_model.cluster.keys()),
        )

    def test_load_legacy_checkpoint_without_atrophy_counter(self):
        source_model = self.pruning_cluster_model(pruning_threshold=None)
        legacy_state_dict = {
            key: value
            for key, value in source_model.state_dict().items()
            if not key.endswith(".atrophy_counter")
        }
        target_model = self.pruning_cluster_model(pruning_threshold=None)
        target_model.cluster["neuron_3_1_1"].atrophy_counter.fill_(7)

        target_model.load_state_dict(legacy_state_dict, strict=True)

        self.assertEqual(
            int(target_model.cluster["neuron_3_1_1"].atrophy_counter.item()),
            0,
        )

    def test_state_dict_roundtrip_preserves_atrophy_counter(self):
        source_model = self.pruning_cluster_model(pruning_threshold=None)
        source_model.cluster["neuron_3_1_1"].atrophy_counter.fill_(5)
        target_model = self.pruning_cluster_model(pruning_threshold=None)

        target_model.load_state_dict(source_model.state_dict(), strict=True)

        self.assertEqual(
            int(target_model.cluster["neuron_3_1_1"].atrophy_counter.item()),
            5,
        )

    def test_grown_neuron_inherits_cluster_dtype_and_device(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        model.entry_sampler = ScriptedSampler(indices=[0], probabilities=[1.0])
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1]],
                    probabilities=[1.0],
                    delta=[0.0, 0.0, 0.0, 0.0],
                )
            }
        )
        model = model.to(dtype=torch.float64)
        initial_keys = set(model.cluster.keys())

        model(torch.randn(self.batch_size, self.input_dim, dtype=torch.float64))

        added_keys = set(model.cluster.keys()) - initial_keys
        self.assertEqual(len(added_keys), 1)
        new_neuron = model.cluster[added_keys.pop()]
        expected_device = next(model.parameters()).device
        self.assertEqual(
            {parameter.device for parameter in new_neuron.parameters()},
            {expected_device},
        )
        self.assertEqual(
            {
                parameter.dtype
                for parameter in new_neuron.parameters()
                if parameter.is_floating_point()
            },
            {torch.float64},
        )

    def test_entry_fanout_processes_selected_neurons_and_highest_continues(self):
        model = self.scripted_cluster(
            max_steps=1,
            x_axis_total_neurons=2,
            initial_x_axis_total_neurons=2,
        )
        model.entry_sampler = ScriptedSampler(
            indices=[0, 1],
            probabilities=[0.25, 0.75],
        )
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[1, 1, 1]],
                    probabilities=[1.0],
                    delta=[1.0],
                ),
                "neuron_2_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1]],
                    probabilities=[1.0],
                    delta=[10.0],
                ),
            }
        )

        output, _ = model(torch.zeros(1, 1))

        torch.testing.assert_close(output, torch.tensor([[20.0]]))
        self.assertEqual(int(model.cluster["neuron_1_1_1"].batch_counter.item()), 1)
        self.assertEqual(int(model.cluster["neuron_2_1_1"].batch_counter.item()), 2)
        self.assertEqual(
            int(model.cluster["neuron_1_1_1"].route_call_counter.item()),
            0,
        )
        self.assertEqual(
            int(model.cluster["neuron_2_1_1"].route_call_counter.item()),
            1,
        )

    def test_entry_weighted_candidate_updates_cluster_halting(self):
        halting_model = RecordingHaltingModel(halt_after_updates=1)
        model = self.scripted_cluster(
            max_steps=5,
            halting_model=halting_model,
            x_axis_total_neurons=2,
            initial_x_axis_total_neurons=2,
        )
        model.entry_sampler = ScriptedSampler(
            indices=[0, 1],
            probabilities=[0.25, 0.75],
        )
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[1, 1, 1]],
                    probabilities=[1.0],
                    delta=[1.0],
                ),
                "neuron_2_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1]],
                    probabilities=[1.0],
                    delta=[10.0],
                ),
            }
        )

        output, _ = model(torch.zeros(1, 1))

        expected_entry_candidate = torch.tensor([[7.75]])
        torch.testing.assert_close(halting_model.inputs[0], expected_entry_candidate)
        torch.testing.assert_close(output, expected_entry_candidate)
        self.assertEqual(halting_model.update_count, 1)
        self.assertEqual(int(model.cluster["neuron_1_1_1"].batch_counter.item()), 1)
        self.assertEqual(int(model.cluster["neuron_2_1_1"].batch_counter.item()), 1)

    def test_saturated_neuron_keeps_counter(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=2,
            z_axis_total_neurons=2,
            max_steps=1,
            growth_threshold=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()

        model(torch.randn(self.batch_size, self.input_dim))

        self.assertTrue(int(model.cluster["neuron_1_1_1"].batch_counter.item()) >= 1)

    def test_topk_branches_run_and_highest_probability_continues(self):
        model = self.scripted_cluster(max_steps=1, input_dim=2)
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1], [3, 1, 1], [4, 1, 1]],
                    probabilities=[0.2, 0.7, 0.1],
                    delta=[1.0, 0.0],
                ),
                "neuron_2_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1]],
                    probabilities=[1.0],
                    delta=[0.0, 2.0],
                ),
                "neuron_3_1_1": ScriptedNeuron(
                    routes=[[3, 1, 1]],
                    probabilities=[1.0],
                    delta=[0.0, 3.0],
                ),
                "neuron_4_1_1": ScriptedNeuron(
                    routes=[[4, 1, 1]],
                    probabilities=[1.0],
                    delta=[0.0, 4.0],
                ),
            }
        )

        output, auxiliary_loss = model(torch.zeros(2, 2))

        torch.testing.assert_close(output, torch.tensor([[1.0, 3.0], [1.0, 3.0]]))
        self.assertEqual(auxiliary_loss.shape, ())
        self.assertEqual(int(model.cluster["neuron_1_1_1"].batch_counter.item()), 1)
        self.assertEqual(
            int(model.cluster["neuron_1_1_1"].route_call_counter.item()),
            1,
        )
        self.assertEqual(int(model.cluster["neuron_2_1_1"].batch_counter.item()), 1)
        self.assertEqual(
            int(model.cluster["neuron_2_1_1"].route_call_counter.item()),
            0,
        )
        self.assertEqual(int(model.cluster["neuron_3_1_1"].batch_counter.item()), 1)
        self.assertEqual(
            int(model.cluster["neuron_3_1_1"].route_call_counter.item()),
            0,
        )
        self.assertEqual(int(model.cluster["neuron_4_1_1"].batch_counter.item()), 1)
        self.assertEqual(
            int(model.cluster["neuron_4_1_1"].route_call_counter.item()),
            0,
        )

    def test_weighted_topk_candidate_updates_halting_without_renormalizing(self):
        halting_model = RecordingHaltingModel(ponder_loss=0.5)
        model = self.scripted_cluster(
            max_steps=1,
            halting_model=halting_model,
            input_dim=2,
        )
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1], [99, 1, 1]],
                    probabilities=[0.25, 0.75],
                    delta=[1.0, 0.0],
                ),
                "neuron_2_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1]],
                    probabilities=[1.0],
                    delta=[0.0, 4.0],
                ),
            }
        )

        output, auxiliary_loss = model(torch.zeros(1, 2))

        torch.testing.assert_close(halting_model.inputs[0], torch.tensor([[1.0, 0.0]]))
        expected_candidate = torch.tensor([[1.0, 1.0]])
        torch.testing.assert_close(halting_model.inputs[1], expected_candidate)
        torch.testing.assert_close(output, expected_candidate)
        torch.testing.assert_close(auxiliary_loss, torch.tensor(0.5))
        self.assertEqual(halting_model.finalize_count, 1)

    def test_self_route_stops_at_max_steps(self):
        model = self.scripted_cluster(max_steps=3)
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[1, 1, 1]],
                    probabilities=[1.0],
                    delta=[1.0],
                )
            }
        )

        output, _ = model(torch.zeros(1, 1))

        torch.testing.assert_close(output, torch.tensor([[4.0]]))
        self.assertEqual(int(model.cluster["neuron_1_1_1"].batch_counter.item()), 4)
        self.assertEqual(
            int(model.cluster["neuron_1_1_1"].route_call_counter.item()),
            3,
        )

    def test_ping_pong_route_stops_at_max_steps(self):
        model = self.scripted_cluster(max_steps=3)
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1]],
                    probabilities=[1.0],
                    delta=[1.0],
                ),
                "neuron_2_1_1": ScriptedNeuron(
                    routes=[[1, 1, 1]],
                    probabilities=[1.0],
                    delta=[10.0],
                ),
            }
        )

        model(torch.zeros(1, 1))

        self.assertEqual(int(model.cluster["neuron_1_1_1"].batch_counter.item()), 2)
        self.assertEqual(int(model.cluster["neuron_2_1_1"].batch_counter.item()), 2)
        self.assertEqual(
            int(model.cluster["neuron_1_1_1"].route_call_counter.item()),
            2,
        )
        self.assertEqual(
            int(model.cluster["neuron_2_1_1"].route_call_counter.item()),
            1,
        )

    def test_backward_route_is_allowed_when_coordinate_exists(self):
        model = self.scripted_cluster(max_steps=2, z_axis_total_neurons=2)
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[1, 1, 2]],
                    probabilities=[1.0],
                    delta=[1.0],
                ),
                "neuron_1_1_2": ScriptedNeuron(
                    routes=[[1, 1, 1]],
                    probabilities=[1.0],
                    delta=[2.0],
                ),
            }
        )

        model(torch.zeros(1, 1))

        self.assertEqual(int(model.cluster["neuron_1_1_1"].batch_counter.item()), 2)
        self.assertEqual(int(model.cluster["neuron_1_1_2"].batch_counter.item()), 1)
        self.assertEqual(
            int(model.cluster["neuron_1_1_1"].route_call_counter.item()),
            1,
        )
        self.assertEqual(
            int(model.cluster["neuron_1_1_2"].route_call_counter.item()),
            1,
        )

    def test_missing_target_marks_sample_inactive(self):
        model = self.scripted_cluster(max_steps=5)
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[99, 1, 1]],
                    probabilities=[1.0],
                    delta=[1.0],
                )
            }
        )

        output, _ = model(torch.zeros(1, 1))

        torch.testing.assert_close(output, torch.tensor([[1.0]]))
        self.assertEqual(int(model.cluster["neuron_1_1_1"].batch_counter.item()), 1)
        self.assertEqual(
            int(model.cluster["neuron_1_1_1"].route_call_counter.item()),
            1,
        )

    def test_highest_probability_escape_exits_with_weighted_candidate(self):
        model = self.scripted_cluster(
            max_steps=3,
            x_axis_total_neurons=2,
        )
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[99, 1, 1], [2, 1, 1]],
                    probabilities=[0.8, 0.2],
                    delta=[1.0],
                ),
                "neuron_2_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1]],
                    probabilities=[1.0],
                    delta=[10.0],
                ),
            }
        )

        output, _ = model(torch.zeros(1, 1))

        torch.testing.assert_close(output, torch.tensor([[3.0]]))
        self.assertEqual(int(model.cluster["neuron_1_1_1"].batch_counter.item()), 1)
        self.assertEqual(int(model.cluster["neuron_2_1_1"].batch_counter.item()), 1)
        self.assertEqual(
            int(model.cluster["neuron_2_1_1"].route_call_counter.item()),
            0,
        )

    def test_halting_stops_routing_before_max_steps(self):
        halting_model = RecordingHaltingModel(halt_after_updates=1)
        model = self.scripted_cluster(max_steps=5, halting_model=halting_model)
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[1, 1, 1]],
                    probabilities=[1.0],
                    delta=[1.0],
                )
            }
        )

        model(torch.zeros(1, 1))

        self.assertEqual(halting_model.update_count, 1)
        self.assertEqual(halting_model.finalize_count, 1)
        self.assertEqual(int(model.cluster["neuron_1_1_1"].batch_counter.item()), 1)
        self.assertEqual(
            int(model.cluster["neuron_1_1_1"].route_call_counter.item()),
            0,
        )

    def test_return_trace_records_detached_entry_and_route_steps(self):
        model = self.scripted_cluster(
            max_steps=1,
            x_axis_total_neurons=2,
        )
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1], [99, 1, 1]],
                    probabilities=[0.4, 0.6],
                    delta=[1.0],
                ),
                "neuron_2_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1]],
                    probabilities=[1.0],
                    delta=[10.0],
                ),
            }
        )

        output, auxiliary_loss, trace = model(
            torch.zeros(1, 1, requires_grad=True),
            return_trace=True,
        )

        self.assertEqual(output.shape, (1, 1))
        self.assertEqual(auxiliary_loss.shape, ())
        self.assertIsInstance(trace, NeuronClusterTrace)
        self.assertEqual(trace.input_shape, (1, 1))
        self.assertEqual(len(trace.steps), 1)
        self.assertFalse(trace.entry_probabilities.requires_grad)
        self.assertFalse(trace.entry_selected_coordinates.requires_grad)
        self.assertFalse(trace.steps[0].probabilities.requires_grad)
        self.assertFalse(trace.steps[0].selected_coordinates.requires_grad)
        torch.testing.assert_close(
            trace.entry_selected_coordinates,
            torch.tensor([[[1, 1, 1]]]),
        )
        torch.testing.assert_close(
            trace.steps[0].escape_mask,
            torch.tensor([[False, True]]),
        )
        torch.testing.assert_close(
            trace.steps[0].chosen_branch_indices,
            torch.tensor([1]),
        )

    def test_gradient_flows_through_cluster_branching_and_halting(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            halting_config=self.halting_config(input_dim=self.input_dim),
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        model.eval()
        input_batch = torch.randn(
            self.batch_size,
            self.input_dim,
            requires_grad=True,
        )

        output, auxiliary_loss = model(input_batch)
        (output.sum() + auxiliary_loss).backward()

        self.assertIsNotNone(input_batch.grad)
        self.assertTrue(torch.any(input_batch.grad.abs() > 0.0))
        nonzero_grads = [
            param.grad
            for param in model.parameters()
            if param.requires_grad
            and param.grad is not None
            and torch.any(param.grad.abs() > 0.0)
        ]
        self.assertTrue(len(nonzero_grads) > 0)


def _init_distributed_growth_process_group(
    rank: int,
    world_size: int,
    init_file: str,
) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )


def _assert_equal_across_ranks(world_size: int, value, label: str) -> None:
    gathered_values = [None] * world_size
    torch.distributed.all_gather_object(gathered_values, value)
    assert all(
        gathered_value == gathered_values[0] for gathered_value in gathered_values
    ), f"{label} diverged across ranks: {gathered_values}"


def _assert_tensors_bitwise_equal_across_ranks(
    world_size: int,
    tensors: dict[str, Tensor],
    label: str,
) -> None:
    gathered_tensors = [None] * world_size
    torch.distributed.all_gather_object(gathered_tensors, tensors)
    reference_tensors = gathered_tensors[0]
    for rank_index, rank_tensors in enumerate(gathered_tensors[1:], start=1):
        assert rank_tensors.keys() == reference_tensors.keys(), (
            f"{label} keys diverged on rank {rank_index}"
        )
        for key in reference_tensors:
            assert torch.equal(rank_tensors[key], reference_tensors[key]), (
                f"{label} diverged across ranks for {key} on rank {rank_index}"
            )


def _distributed_growth_worker_assert_identical_growth(
    rank: int,
    world_size: int,
    init_file: str,
    config: NeuronClusterConfig,
    expected_grown_neuron_name: str,
) -> None:
    _init_distributed_growth_process_group(rank, world_size, init_file)
    try:
        input_dim = config.neuron_config.terminal_config.input_dim
        torch.manual_seed(0)
        model = config.build()
        torch.manual_seed(100 + rank)

        model(torch.randn(3, input_dim))

        cluster_keys = sorted(model.cluster.keys())
        _assert_equal_across_ranks(world_size, cluster_keys, "cluster keys")
        assert expected_grown_neuron_name in model.cluster, (
            f"rank {rank} expected {expected_grown_neuron_name} to be grown, "
            f"found {cluster_keys}"
        )
        grown_neuron_state = {
            key: value.detach().cpu().clone()
            for key, value in model.cluster[
                expected_grown_neuron_name
            ].state_dict().items()
        }
        _assert_tensors_bitwise_equal_across_ranks(
            world_size,
            grown_neuron_state,
            "grown neuron state",
        )
    finally:
        torch.distributed.destroy_process_group()


def _distributed_growth_worker_assert_counter_all_reduce(
    rank: int,
    world_size: int,
    init_file: str,
    config: NeuronClusterConfig,
) -> None:
    _init_distributed_growth_process_group(rank, world_size, init_file)
    try:
        input_dim = config.neuron_config.terminal_config.input_dim
        torch.manual_seed(0)
        below_global_threshold_model = config.build()
        saturated_by_global_sum_model = config.build()
        torch.manual_seed(100 + rank)

        below_global_threshold_model.cluster["neuron_1_1_1"].batch_counter.fill_(10)
        below_global_threshold_model(torch.randn(3, input_dim))
        assert len(below_global_threshold_model.cluster) == 1, (
            f"rank {rank} grew a neuron although the all-reduced counter is "
            "below growth_threshold"
        )

        saturated_by_global_sum_model.cluster["neuron_1_1_1"].batch_counter.fill_(49)
        saturated_by_global_sum_model(torch.randn(3, input_dim))
        assert "neuron_2_1_1" in saturated_by_global_sum_model.cluster, (
            f"rank {rank} did not grow although the all-reduced counter "
            "crosses growth_threshold"
        )
    finally:
        torch.distributed.destroy_process_group()


def _distributed_growth_worker_assert_escape_count_alignment(
    rank: int,
    world_size: int,
    init_file: str,
    config: NeuronClusterConfig,
) -> None:
    _init_distributed_growth_process_group(rank, world_size, init_file)
    try:
        input_dim = config.neuron_config.terminal_config.input_dim
        torch.manual_seed(0)
        model = config.build()
        torch.manual_seed(100 + rank)
        if rank == 0:
            model.escape_counts[3, 0, 0] = 10_000

        model(torch.randn(3, input_dim))

        assert "neuron_4_1_1" in model.cluster, (
            f"rank {rank} ignored the all-reduced escape counts, "
            f"found {sorted(model.cluster.keys())}"
        )
        assert "neuron_2_1_1" not in model.cluster, (
            f"rank {rank} grew the Manhattan-closest neuron instead of the "
            "most escaped coordinate"
        )
        assert int(model.escape_counts[3, 0, 0].item()) == 0, (
            f"rank {rank} did not reset the grown coordinate's escape count"
        )
    finally:
        torch.distributed.destroy_process_group()


def _distributed_pruning_worker_assert_identical_pruning(
    rank: int,
    world_size: int,
    init_file: str,
    config: NeuronClusterConfig,
) -> None:
    _init_distributed_growth_process_group(rank, world_size, init_file)
    try:
        input_dim = config.neuron_config.terminal_config.input_dim
        torch.manual_seed(0)
        model = config.build()
        model.cluster["neuron_5_1_1"] = model._initialize_neuron(5, 1, 1)
        torch.manual_seed(100 + rank)

        model(torch.randn(3, input_dim))
        model(torch.randn(3, input_dim))

        cluster_keys = sorted(model.cluster.keys())
        _assert_equal_across_ranks(world_size, cluster_keys, "cluster keys")
        assert "neuron_5_1_1" not in model.cluster, (
            f"rank {rank} kept the idle neuron, found {cluster_keys}"
        )
    finally:
        torch.distributed.destroy_process_group()


def _distributed_pruning_worker_assert_min_reduce_protects_neuron(
    rank: int,
    world_size: int,
    init_file: str,
    config: NeuronClusterConfig,
) -> None:
    _init_distributed_growth_process_group(rank, world_size, init_file)
    try:
        input_dim = config.neuron_config.terminal_config.input_dim
        torch.manual_seed(0)
        model = config.build()
        model.cluster["neuron_5_1_1"] = model._initialize_neuron(5, 1, 1)
        torch.manual_seed(100 + rank)
        if rank == 0:
            model.cluster["neuron_5_1_1"].atrophy_counter.fill_(1000)

        model(torch.randn(3, input_dim))

        assert "neuron_5_1_1" in model.cluster, (
            f"rank {rank} pruned a neuron whose cross-rank minimum atrophy "
            "count is below pruning_threshold"
        )
    finally:
        torch.distributed.destroy_process_group()


@unittest.skipUnless(
    torch.distributed.is_available() and torch.distributed.is_gloo_available(),
    "gloo process group support is required",
)
class TestNeuronClusterDistributedGrowth(NeuronTestCase):
    world_size = 2

    def distributed_growth_cluster_config(
        self,
        growth_threshold: int,
        mitosis_initialization_flag: bool = False,
    ) -> NeuronClusterConfig:
        return NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=growth_threshold,
            mitosis_initialization_flag=mitosis_initialization_flag,
            neuron_config=self.full_sampler_neuron_config(),
        )

    def distributed_escape_growth_cluster_config(self) -> NeuronClusterConfig:
        return NeuronClusterConfig(
            x_axis_total_neurons=5,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            escape_driven_growth_flag=True,
            neuron_config=self.full_sampler_neuron_config(),
        )

    def spawn_distributed_growth_workers(self, worker, *worker_args) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            init_file = os.path.join(temp_dir, "process_group_init")
            torch.multiprocessing.spawn(
                worker,
                args=(self.world_size, init_file, *worker_args),
                nprocs=self.world_size,
                join=True,
            )

    def test_growth_is_identical_across_ranks(self):
        self.spawn_distributed_growth_workers(
            _distributed_growth_worker_assert_identical_growth,
            self.distributed_growth_cluster_config(growth_threshold=1),
            "neuron_2_1_1",
        )

    def test_mitosis_growth_is_bitwise_identical_across_ranks(self):
        self.spawn_distributed_growth_workers(
            _distributed_growth_worker_assert_identical_growth,
            self.distributed_growth_cluster_config(
                growth_threshold=1,
                mitosis_initialization_flag=True,
            ),
            "neuron_2_1_1",
        )

    def test_growth_triggers_on_all_reduced_batch_counters(self):
        self.spawn_distributed_growth_workers(
            _distributed_growth_worker_assert_counter_all_reduce,
            self.distributed_growth_cluster_config(growth_threshold=100),
        )

    def test_escape_driven_placement_uses_all_reduced_escape_counts(self):
        self.spawn_distributed_growth_workers(
            _distributed_growth_worker_assert_escape_count_alignment,
            self.distributed_escape_growth_cluster_config(),
        )

    def distributed_pruning_cluster_config(
        self,
        pruning_threshold: int,
    ) -> NeuronClusterConfig:
        return NeuronClusterConfig(
            x_axis_total_neurons=5,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            pruning_threshold=pruning_threshold,
            neuron_config=self.full_sampler_neuron_config(),
        )

    def test_pruning_is_identical_across_ranks(self):
        self.spawn_distributed_growth_workers(
            _distributed_pruning_worker_assert_identical_pruning,
            self.distributed_pruning_cluster_config(pruning_threshold=2),
        )

    def test_pruning_uses_cross_rank_minimum_atrophy(self):
        self.spawn_distributed_growth_workers(
            _distributed_pruning_worker_assert_min_reduce_protects_neuron,
            self.distributed_pruning_cluster_config(pruning_threshold=50),
        )


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
                max_steps=1,
                growth_threshold=None,
                neuron_config=self.neuron_config(),
            ).build()

    def test_non_positive_max_steps_raises(self):
        with self.assertRaises(ValueError):
            NeuronClusterConfig(
                x_axis_total_neurons=1,
                y_axis_total_neurons=1,
                z_axis_total_neurons=1,
                max_steps=0,
                growth_threshold=None,
                neuron_config=self.neuron_config(),
            ).build()

    def test_initial_grid_larger_than_max_raises(self):
        with self.assertRaises(ValueError):
            NeuronClusterConfig(
                x_axis_total_neurons=1,
                y_axis_total_neurons=1,
                z_axis_total_neurons=1,
                initial_x_axis_total_neurons=2,
                max_steps=1,
                growth_threshold=None,
                neuron_config=self.neuron_config(),
            ).build()

    def test_entry_sampler_num_experts_mismatch_raises(self):
        with self.assertRaises(ValueError):
            NeuronClusterConfig(
                x_axis_total_neurons=1,
                y_axis_total_neurons=1,
                z_axis_total_neurons=1,
                max_steps=1,
                growth_threshold=None,
                entry_sampler_config=self.sampler_config(
                    input_dim=self.input_dim,
                    num_experts=2,
                    top_k=1,
                ),
                neuron_config=self.neuron_config(),
            ).build()

    def test_wrong_cluster_halting_config_type_raises(self):
        with self.assertRaises(TypeError):
            NeuronClusterConfig(
                x_axis_total_neurons=1,
                y_axis_total_neurons=1,
                z_axis_total_neurons=1,
                max_steps=1,
                growth_threshold=None,
                halting_config=self.projection_config(),
                neuron_config=self.neuron_config(),
            ).build()

    def test_cluster_halting_input_dim_mismatch_raises(self):
        with self.assertRaises(ValueError):
            NeuronClusterConfig(
                x_axis_total_neurons=1,
                y_axis_total_neurons=1,
                z_axis_total_neurons=1,
                max_steps=1,
                growth_threshold=None,
                halting_config=self.halting_config(input_dim=self.input_dim + 1),
                neuron_config=self.neuron_config(),
            ).build()

    def test_wrong_forward_tensor_rank_raises(self):
        model = self.neuron_config().build()

        with self.assertRaises(ValueError):
            model(torch.randn(2, 3, self.input_dim))

    def test_cluster_feature_dim_mismatch_raises(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.neuron_config(),
        ).build()

        with self.assertRaises(ValueError):
            model(torch.randn(2, self.input_dim + 1))

    def test_cluster_rank_one_input_raises(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.neuron_config(),
        ).build()

        with self.assertRaises(ValueError):
            model(torch.randn(self.input_dim))

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

    def logits_only_neuron_config(self) -> NeuronConfig:
        total_connections = self.terminal_total_connections()
        return NeuronConfig(
            nucleus_config=NucleusConfig(
                model_config=self.projection_config(
                    input_dim=total_connections,
                    output_dim=total_connections,
                    scale=0.25,
                )
            ),
            axons_config=AxonsConfig(memory_config=None),
            terminal_config=self.terminal_config(
                input_dim=total_connections,
                sampler_config=self.sampler_config(
                    input_dim=total_connections,
                    num_experts=total_connections,
                    router_config=None,
                ),
            ),
        )

    def test_growth_placement_flags_require_growth_threshold(self):
        for flag_name in (
            "escape_driven_growth_flag",
            "mitosis_initialization_flag",
        ):
            with self.subTest(flag=flag_name):
                with self.assertRaises(ValueError):
                    NeuronClusterConfig(
                        x_axis_total_neurons=1,
                        y_axis_total_neurons=1,
                        z_axis_total_neurons=1,
                        max_steps=1,
                        growth_threshold=None,
                        neuron_config=self.neuron_config(),
                        **{flag_name: True},
                    ).build()

    def test_nucleus_model_dimension_mismatch_raises(self):
        neuron_config = NeuronConfig(
            nucleus_config=NucleusConfig(
                model_config=self.projection_config(
                    input_dim=self.input_dim,
                    output_dim=self.input_dim + 1,
                    scale=0.25,
                )
            ),
            axons_config=AxonsConfig(memory_config=None),
            terminal_config=self.terminal_config(input_dim=self.input_dim),
        )

        with self.assertRaises(ValueError):
            NeuronClusterConfig(
                x_axis_total_neurons=1,
                y_axis_total_neurons=1,
                z_axis_total_neurons=1,
                max_steps=1,
                growth_threshold=None,
                neuron_config=neuron_config,
            ).build()

    def test_derived_logits_only_entry_sampler_dim_mismatch_raises(self):
        with self.assertRaises(ValueError):
            NeuronClusterConfig(
                x_axis_total_neurons=2,
                y_axis_total_neurons=1,
                z_axis_total_neurons=1,
                max_steps=1,
                growth_threshold=None,
                neuron_config=self.logits_only_neuron_config(),
            ).build()

    def test_derived_logits_only_entry_sampler_matching_dim_builds(self):
        total_connections = self.terminal_total_connections()
        model = NeuronClusterConfig(
            x_axis_total_neurons=total_connections,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.logits_only_neuron_config(),
        ).build()

        self.assertEqual(len(model.cluster), total_connections)


if __name__ == "__main__":
    unittest.main()
