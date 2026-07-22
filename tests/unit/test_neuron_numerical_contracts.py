import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.func import functional_call

from emperor.halting import HaltingHiddenStateModeOptions, HaltingUsageTrackerManager
from emperor.neuron import (
    AxonsConfig,
    NeuronClusterConfig,
    NeuronClusterTrace,
    NeuronClusterTraceStep,
    NeuronConfig,
    NucleusConfig,
    TerminalConnectionShapeOptions,
)
from emperor.neuron._cluster.beam_routes import _NeuronClusterBeamRoutesMixin
from emperor.neuron._cluster.recurrent_routes import (
    _NeuronClusterRecurrentRoutesMixin,
)
from emperor.neuron._cluster.state import (
    NeuronClusterRouteState,
    _NeuronClusterStateMixin,
)
from emperor.neuron._monitoring.diagnostics import _NeuronDiagnostics
from unit import test_neuron as neuron_test_fixtures
from unit.test_neuron import (
    NeuronTestCase,
    ScriptedNeuron,
    ScriptedSampler,
    ScriptedTerminal,
)


class _DifferentiableSelfRouteNeuron(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float64))
        self.terminal = ScriptedTerminal([[1, 1, 1]])

    def process_signal(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor * self.scale

    def route_signal(
        self,
        processed_signal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = processed_signal.shape[0]
        return (
            processed_signal.new_ones((batch_size, 1)),
            self.terminal.neuron_connections.to(processed_signal.device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1),
            processed_signal.new_zeros(()),
        )


class TestNeuronRoutingLifecycleContracts(unittest.TestCase):
    def test_beam_slot_padding_is_inactive_and_has_zero_probability(self) -> None:
        beam_routes = _NeuronClusterBeamRoutesMixin()
        beam_routes.beam_width = 3

        path_probabilities, branch_indices = (
            beam_routes._NeuronClusterBeamRoutesMixin__top_beam_slots(
                torch.tensor([[0.2, -0.4]], dtype=torch.float64)
            )
        )

        torch.testing.assert_close(
            path_probabilities,
            torch.tensor([[0.2, 0.0, 0.0]], dtype=torch.float64),
        )
        torch.testing.assert_close(
            branch_indices,
            torch.tensor([[0, 1, 0]], dtype=torch.long),
        )

    def test_beam_missing_current_route_finalizes_only_the_missing_slot(self) -> None:
        class BeamHarness(
            _NeuronClusterBeamRoutesMixin,
            _NeuronClusterStateMixin,
        ):
            @staticmethod
            def _coordinate_from_row(row):
                return tuple(int(value) for value in row)

            @staticmethod
            def _neuron_name(x, y, z):
                return f"neuron_{x}_{y}_{z}"

        hidden = torch.tensor(
            [[1.0], [2.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        route_state = NeuronClusterRouteState(
            hidden=hidden,
            positions=torch.tensor([[9, 9, 9], [1, 1, 1]]),
            active_mask=torch.tensor([True, True]),
            escaped_mask=torch.tensor([False, False]),
            final_mask=torch.tensor([False, False]),
            halting_state=SimpleNamespace(marker="halting"),
            loss=torch.tensor(0.25, dtype=torch.float64),
            beam_path_probabilities=torch.tensor([0.75, 0.25], dtype=torch.float64),
        )
        harness = BeamHarness()
        harness.beam_width = 2
        harness.cluster = {}

        finalized = harness._NeuronClusterBeamRoutesMixin__run_beam_route_step(
            route_state,
            torch.tensor([True, False]),
        )

        self.assertIs(finalized.hidden, route_state.hidden)
        self.assertIs(finalized.positions, route_state.positions)
        self.assertIs(finalized.halting_state, route_state.halting_state)
        self.assertIs(
            finalized.beam_path_probabilities,
            route_state.beam_path_probabilities,
        )
        torch.testing.assert_close(finalized.active_mask, torch.tensor([False, True]))
        torch.testing.assert_close(finalized.final_mask, torch.tensor([True, False]))
        torch.testing.assert_close(finalized.escaped_mask, route_state.escaped_mask)
        torch.testing.assert_close(finalized.loss, route_state.loss)
        self.assertIsNone(finalized.trace)

        finalized.hidden.sum().backward()
        torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))

    def test_recurrent_missing_current_route_finalizes_without_graph_break(
        self,
    ) -> None:
        class RecurrentHarness(
            _NeuronClusterRecurrentRoutesMixin,
            _NeuronClusterStateMixin,
        ):
            @staticmethod
            def _coordinate_from_row(row):
                return tuple(int(value) for value in row)

            @staticmethod
            def _neuron_name(x, y, z):
                return f"neuron_{x}_{y}_{z}"

        hidden = torch.tensor(
            [[1.0], [2.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        route_state = NeuronClusterRouteState(
            hidden=hidden,
            positions=torch.tensor([[9, 9, 9], [1, 1, 1]]),
            active_mask=torch.tensor([True, True]),
            escaped_mask=torch.tensor([False, False]),
            final_mask=torch.tensor([False, False]),
            halting_state=SimpleNamespace(marker="halting"),
            loss=torch.tensor(0.25, dtype=torch.float64),
            trace=SimpleNamespace(marker="trace"),
        )
        harness = RecurrentHarness()
        harness.cluster = {}

        finalized = (
            harness._NeuronClusterRecurrentRoutesMixin__run_recurrent_route_step(
                route_state,
                torch.tensor([True, False]),
            )
        )

        torch.testing.assert_close(finalized.hidden, hidden)
        torch.testing.assert_close(finalized.positions, route_state.positions)
        torch.testing.assert_close(finalized.active_mask, torch.tensor([False, True]))
        torch.testing.assert_close(finalized.final_mask, torch.tensor([True, False]))
        torch.testing.assert_close(finalized.escaped_mask, route_state.escaped_mask)
        torch.testing.assert_close(finalized.loss, route_state.loss)
        self.assertIs(finalized.halting_state, route_state.halting_state)
        self.assertIs(finalized.trace, route_state.trace)

        finalized.hidden.sum().backward()
        torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))

    def test_halting_masks_and_state_rows_preserve_batch_ownership(self) -> None:
        state = _NeuronClusterStateMixin()
        halting_state = SimpleNamespace(
            halt_mask=torch.tensor([[True, True], [True, False]]),
            scores=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            unrelated_rows=torch.tensor([5.0, 6.0, 7.0]),
            scalar=torch.tensor(8.0),
            label="state",
        )

        reduced_mask = state._get_halt_mask(halting_state)
        mask_tensor = state._halt_mask_tensor(
            halting_state,
            batch_size=2,
            device=torch.device("cpu"),
        )
        gathered = state._gather_halting_state_rows(
            halting_state,
            torch.tensor([1, 0]),
        )

        torch.testing.assert_close(reduced_mask, torch.tensor([True, False]))
        torch.testing.assert_close(mask_tensor, reduced_mask)
        self.assertIsNot(gathered, halting_state)
        torch.testing.assert_close(
            gathered.halt_mask,
            torch.tensor([[True, False], [True, True]]),
        )
        torch.testing.assert_close(
            gathered.scores,
            torch.tensor([[3.0, 4.0], [1.0, 2.0]]),
        )
        self.assertIs(gathered.unrelated_rows, halting_state.unrelated_rows)
        self.assertIs(gathered.scalar, halting_state.scalar)
        self.assertEqual(gathered.label, "state")

    def test_state_wrapper_places_absent_halt_mask_on_requested_device(self) -> None:
        state = _NeuronClusterStateMixin()

        halt_mask = state._halt_mask_tensor(
            None,
            batch_size=3,
            device=torch.device("meta"),
        )

        self.assertEqual(halt_mask.shape, (3,))
        self.assertEqual(halt_mask.dtype, torch.bool)
        self.assertEqual(halt_mask.device, torch.device("meta"))

    def test_ponder_loss_ignores_padded_beam_slots(self) -> None:
        class FixedHaltingModel:
            def finalize_weighted_accumulation(self, state, current_hidden):
                return current_hidden, current_hidden.new_tensor([0.75, 0.0])

        class StateHarness(_NeuronClusterStateMixin):
            @staticmethod
            def _accumulate_auxiliary_loss(loss, auxiliary_loss):
                return loss + auxiliary_loss

        state_mixin = StateHarness()
        state_mixin.halting_model = FixedHaltingModel()
        halting_state = SimpleNamespace(
            valid_mask=torch.tensor([True, False]),
            advanced_mask=torch.tensor([True, False]),
        )
        route_state = NeuronClusterRouteState(
            hidden=torch.tensor([[1.0], [0.0]]),
            positions=torch.zeros(2, 3, dtype=torch.long),
            active_mask=torch.tensor([True, False]),
            escaped_mask=torch.tensor([False, False]),
            final_mask=torch.tensor([False, True]),
            halting_state=halting_state,
            loss=torch.zeros(()),
            beam_path_probabilities=torch.tensor([1.0, 0.0]),
        )

        finalized = state_mixin._maybe_finalize_cluster_halting(route_state)

        torch.testing.assert_close(finalized.loss, torch.tensor(0.75))

    def test_padded_ponder_loss_gradient_updates_only_the_live_beam(self) -> None:
        ponder_loss = torch.tensor([0.75, 9.0], dtype=torch.float64, requires_grad=True)

        class FixedHaltingModel:
            def finalize_weighted_accumulation(self, state, current_hidden):
                return current_hidden, ponder_loss

        class StateHarness(_NeuronClusterStateMixin):
            @staticmethod
            def _accumulate_auxiliary_loss(loss, auxiliary_loss):
                return loss + auxiliary_loss

        halting_state = SimpleNamespace(
            valid_mask=torch.tensor([True, False]),
            advanced_mask=torch.tensor([True, False]),
        )
        route_state = NeuronClusterRouteState(
            hidden=torch.tensor([[1.0], [0.0]], dtype=torch.float64),
            positions=torch.zeros(2, 3, dtype=torch.long),
            active_mask=torch.tensor([True, False]),
            escaped_mask=torch.tensor([False, False]),
            final_mask=torch.tensor([False, True]),
            halting_state=halting_state,
            loss=torch.zeros((), dtype=torch.float64),
            beam_path_probabilities=torch.tensor(
                [1.0, 0.0],
                dtype=torch.float64,
            ),
        )
        state = StateHarness()
        state.halting_model = FixedHaltingModel()

        finalized = state._maybe_finalize_cluster_halting(route_state)
        finalized.loss.backward()

        torch.testing.assert_close(
            ponder_loss.grad,
            torch.tensor([1.0, 0.0], dtype=torch.float64),
        )


class TestNeuronRecurrentGradientContract(NeuronTestCase):
    def _assert_cluster_stick_breaking_gradcheck(self, cluster) -> None:
        parameters_by_name = dict(cluster.named_parameters())
        parameter_names = tuple(
            name for name in parameters_by_name if name.startswith("halting_model.")
        )
        parameter_values = tuple(
            parameters_by_name[name].detach().clone().requires_grad_()
            for name in parameter_names
        )
        input_tensor = torch.tensor(
            [[0.31]],
            dtype=torch.float64,
            requires_grad=True,
        )

        def smooth_fixed_routes(input_value, *parameters):
            replacements = dict(zip(parameter_names, parameters, strict=True))
            output, ponder_loss = functional_call(
                cluster,
                replacements,
                (input_value,),
            )
            return torch.cat((output.reshape(-1), ponder_loss.reshape(-1)))

        self.assertTrue(
            torch.autograd.gradcheck(
                smooth_fixed_routes,
                (input_tensor, *parameter_values),
                eps=1e-6,
                atol=3e-6,
                rtol=2e-4,
            )
        )

    def test_recurrent_stick_breaking_output_and_ponder_pass_gradcheck(
        self,
    ) -> None:
        halting_model = self.halting_config(input_dim=1, threshold=0.999).build()
        cluster = neuron_test_fixtures.TestNeuronCluster.scripted_cluster(
            self,
            max_steps=2,
            halting_model=halting_model,
            input_dim=1,
            x_axis_total_neurons=1,
        )
        cluster.entry_sampler = ScriptedSampler(indices=[0], probabilities=[1.0])
        cluster.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[1, 1, 1]],
                    probabilities=[1.0],
                    delta=[0.2],
                )
            }
        )

        self._assert_cluster_stick_breaking_gradcheck(cluster.double().eval())

    def test_early_halt_stick_breaking_weighted_output_and_parameters_pass_gradcheck(
        self,
    ) -> None:
        halting_config = self.halting_config(input_dim=1, threshold=0.4)
        halting_config.hidden_state_mode = HaltingHiddenStateModeOptions.ACCUMULATED
        halting_model = halting_config.build()
        cluster = neuron_test_fixtures.TestNeuronCluster.scripted_cluster(
            self,
            max_steps=5,
            halting_model=halting_model,
            input_dim=1,
            x_axis_total_neurons=1,
        )
        cluster.entry_sampler = ScriptedSampler(indices=[0], probabilities=[1.0])
        neuron = ScriptedNeuron(
            routes=[[1, 1, 1]],
            probabilities=[1.0],
            delta=[0.2],
        )
        cluster.cluster = nn.ModuleDict({"neuron_1_1_1": neuron})
        cluster = cluster.double().eval()

        input_tensor = torch.tensor(
            [[0.31]],
            dtype=torch.float64,
            requires_grad=True,
        )
        output, ponder_loss = cluster(input_tensor)
        processed_hidden = input_tensor.detach() + neuron.delta.to(
            dtype=input_tensor.dtype,
        )
        expected_output = 0.5 * processed_hidden
        torch.testing.assert_close(output, expected_output, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(ponder_loss, torch.zeros_like(ponder_loss))

        gate_weight = cluster.halting_model.halting_gate_model[-1].model.weight_params
        input_gradient, gate_gradient = torch.autograd.grad(
            output.sum() + ponder_loss,
            (input_tensor, gate_weight),
        )
        expected_gate_gradient = torch.cat(
            (
                -0.25 * processed_hidden.square(),
                0.25 * processed_hidden.square(),
            ),
            dim=-1,
        )
        torch.testing.assert_close(
            input_gradient,
            torch.full_like(input_gradient, 0.5),
            rtol=1e-13,
            atol=1e-13,
        )
        torch.testing.assert_close(
            gate_gradient,
            expected_gate_gradient,
            rtol=1e-13,
            atol=1e-13,
        )

        self._assert_cluster_stick_breaking_gradcheck(cluster)
        self.assertEqual(int(neuron.route_call_counter), 0)

    def test_beam_stick_breaking_output_and_ponder_pass_gradcheck(self) -> None:
        halting_model = self.halting_config(input_dim=1, threshold=0.999).build()
        cluster = neuron_test_fixtures.TestNeuronCluster.scripted_cluster(
            self,
            max_steps=1,
            halting_model=halting_model,
            input_dim=1,
            x_axis_total_neurons=2,
            initial_x_axis_total_neurons=2,
            beam_width=2,
        )
        cluster.entry_sampler = ScriptedSampler(
            indices=[0, 1],
            probabilities=[0.25, 0.75],
        )
        cluster.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[99, 1, 1]],
                    probabilities=[1.0],
                    delta=[0.2],
                ),
                "neuron_2_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1]],
                    probabilities=[1.0],
                    delta=[0.7],
                ),
            }
        )

        self._assert_cluster_stick_breaking_gradcheck(cluster.double().eval())

    def test_sparse_halting_monitor_records_only_advanced_rows(self) -> None:
        class StateHarness(_NeuronClusterStateMixin):
            @staticmethod
            def _accumulate_auxiliary_loss(loss, auxiliary_loss):
                return loss + auxiliary_loss

        halting_model = (
            self.halting_config(input_dim=1, threshold=0.7).build().double().eval()
        )
        with torch.no_grad():
            halting_model.halting_gate_model[-1].model.weight_params.copy_(
                torch.tensor([[0.0, 1.0]], dtype=torch.float64)
            )
        tracker_manager = HaltingUsageTrackerManager()
        tracker = tracker_manager.attach(halting_model)
        state_harness = StateHarness()
        state_harness.halting_model = halting_model
        hidden = torch.tensor([[0.0], [2.0]], dtype=torch.float64)

        try:
            halting_state = state_harness._maybe_update_halting_state(
                None,
                hidden,
                hidden,
                torch.tensor([True, False]),
            )
            route_state = NeuronClusterRouteState(
                hidden=hidden,
                positions=torch.zeros(2, 3, dtype=torch.long),
                active_mask=torch.tensor([True, False]),
                escaped_mask=torch.tensor([False, True]),
                final_mask=torch.tensor([False, True]),
                halting_state=halting_state,
                loss=torch.zeros((), dtype=torch.float64),
            )

            finalized_state = state_harness._maybe_finalize_cluster_halting(route_state)

            torch.testing.assert_close(
                tracker.last_survival,
                torch.tensor([1.0]),
            )
            torch.testing.assert_close(finalized_state.loss, hidden.new_tensor(0.5))
            torch.testing.assert_close(
                tracker.last_ponder_loss,
                finalized_state.loss.float(),
            )
        finally:
            tracker_manager.detach(halting_model)

    def test_finished_beam_stick_breaking_output_and_ponder_match_route_oracle(
        self,
    ) -> None:
        cluster_halting_model = (
            self.halting_config(input_dim=1, threshold=0.999).build().double().eval()
        )
        cluster = neuron_test_fixtures.TestNeuronCluster.scripted_cluster(
            self,
            max_steps=1,
            halting_model=cluster_halting_model,
            input_dim=1,
            x_axis_total_neurons=2,
            initial_x_axis_total_neurons=2,
            beam_width=2,
        ).double()
        cluster.entry_sampler = ScriptedSampler(
            indices=[0, 1],
            probabilities=[0.25, 0.75],
        )
        cluster.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[99, 1, 1]],
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
        cluster.eval()
        reference_halting_model = (
            self.halting_config(input_dim=1, threshold=0.999).build().double().eval()
        )
        cluster_input = torch.zeros((1, 1), dtype=torch.float64, requires_grad=True)
        reference_input = cluster_input.detach().clone().requires_grad_(True)

        cluster_output, cluster_ponder_loss = cluster(cluster_input)
        finished_hidden = reference_input + 1.0
        continuing_hidden = reference_input + 10.0
        finished_state, _ = reference_halting_model.update_halting_state(
            None,
            finished_hidden,
        )
        continuing_state, _ = reference_halting_model.update_halting_state(
            None,
            continuing_hidden,
        )
        continued_hidden = reference_input + 20.0
        continuing_state, _ = reference_halting_model.update_halting_state(
            continuing_state,
            continued_hidden,
        )
        _, finished_ponder_loss = (
            reference_halting_model.finalize_weighted_accumulation(
                finished_state,
                finished_hidden,
            )
        )
        finalized_continuing_hidden, continuing_ponder_loss = (
            reference_halting_model.finalize_weighted_accumulation(
                continuing_state,
                continued_hidden,
            )
        )
        reference_ponder_loss = torch.cat(
            (finished_ponder_loss, continuing_ponder_loss)
        ).mean()
        reference_output = 0.25 * finished_hidden + 0.75 * finalized_continuing_hidden

        torch.testing.assert_close(cluster_output, reference_output)
        torch.testing.assert_close(cluster_ponder_loss, reference_ponder_loss)
        cluster_parameters = tuple(cluster_halting_model.parameters())
        reference_parameters = tuple(reference_halting_model.parameters())
        cluster_objective = 0.37 * cluster_output.sum() + 0.3 * cluster_ponder_loss
        reference_objective = (
            0.37 * reference_output.sum() + 0.3 * reference_ponder_loss
        )
        cluster_gradients = torch.autograd.grad(
            cluster_objective,
            (*cluster_parameters, cluster_input),
        )
        reference_gradients = torch.autograd.grad(
            reference_objective,
            (*reference_parameters, reference_input),
        )
        for cluster_gradient, reference_gradient in zip(
            cluster_gradients,
            reference_gradients,
            strict=True,
        ):
            torch.testing.assert_close(cluster_gradient, reference_gradient)

    def test_missing_recurrent_target_does_not_advance_stick_breaking(
        self,
    ) -> None:
        cluster_halting_model = (
            self.halting_config(input_dim=1, threshold=0.999).build().double().eval()
        )
        cluster = neuron_test_fixtures.TestNeuronCluster.scripted_cluster(
            self,
            max_steps=1,
            halting_model=cluster_halting_model,
            input_dim=1,
            x_axis_total_neurons=1,
        ).double()
        cluster.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[99, 1, 1]],
                    probabilities=[1.0],
                    delta=[1.0],
                )
            }
        )
        cluster.eval()
        reference_halting_model = (
            self.halting_config(input_dim=1, threshold=0.999).build().double().eval()
        )
        cluster_input = torch.tensor(
            [[0.2]],
            dtype=torch.float64,
            requires_grad=True,
        )
        reference_input = cluster_input.detach().clone().requires_grad_(True)

        cluster_output, cluster_ponder_loss = cluster(cluster_input)
        reference_processed_hidden = reference_input + 1.0
        reference_state, _ = reference_halting_model.update_halting_state(
            None,
            reference_processed_hidden,
        )
        _, reference_ponder_loss = (
            reference_halting_model.finalize_weighted_accumulation(
                reference_state,
                reference_processed_hidden,
            )
        )
        reference_ponder_loss = reference_ponder_loss.mean()

        torch.testing.assert_close(cluster_output, reference_processed_hidden.detach())
        torch.testing.assert_close(cluster_ponder_loss, reference_ponder_loss)
        cluster_parameters = tuple(cluster_halting_model.parameters())
        reference_parameters = tuple(reference_halting_model.parameters())
        cluster_gradients = torch.autograd.grad(
            cluster_ponder_loss,
            (*cluster_parameters, cluster_input),
        )
        reference_gradients = torch.autograd.grad(
            reference_ponder_loss,
            (*reference_parameters, reference_input),
        )
        for cluster_gradient, reference_gradient in zip(
            cluster_gradients,
            reference_gradients,
            strict=True,
        ):
            torch.testing.assert_close(cluster_gradient, reference_gradient)

    def test_missing_entry_row_has_no_stick_breaking_loss_or_gradient(
        self,
    ) -> None:
        class PerRowEntrySampler(nn.Module):
            @staticmethod
            def sample_probabilities_and_indices(input_tensor):
                probabilities = input_tensor.new_ones((2, 1))
                indices = torch.tensor(
                    [[0], [1]],
                    dtype=torch.long,
                    device=input_tensor.device,
                )
                return (
                    probabilities,
                    indices,
                    None,
                    input_tensor.new_zeros(()),
                )

        mixed_halting_model = (
            self.halting_config(input_dim=1, threshold=0.999).build().double().eval()
        )
        mixed_cluster = neuron_test_fixtures.TestNeuronCluster.scripted_cluster(
            self,
            max_steps=1,
            halting_model=mixed_halting_model,
            input_dim=1,
            x_axis_total_neurons=2,
            initial_x_axis_total_neurons=2,
        ).double()
        mixed_cluster.entry_sampler = PerRowEntrySampler()
        mixed_cluster.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[1, 1, 1]],
                    probabilities=[1.0],
                    delta=[1.0],
                )
            }
        )
        mixed_cluster.eval()

        reference_halting_model = (
            self.halting_config(input_dim=1, threshold=0.999).build().double().eval()
        )
        reference_cluster = neuron_test_fixtures.TestNeuronCluster.scripted_cluster(
            self,
            max_steps=1,
            halting_model=reference_halting_model,
            input_dim=1,
            x_axis_total_neurons=1,
        ).double()
        reference_cluster.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[1, 1, 1]],
                    probabilities=[1.0],
                    delta=[1.0],
                )
            }
        )
        reference_cluster.eval()
        mixed_input = torch.tensor(
            [[0.2], [9.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        reference_input = mixed_input[:1].detach().clone().requires_grad_(True)

        mixed_output, mixed_ponder_loss = mixed_cluster(mixed_input)
        reference_output, reference_ponder_loss = reference_cluster(reference_input)

        torch.testing.assert_close(mixed_output[:1], reference_output)
        torch.testing.assert_close(mixed_output[1], mixed_input.detach()[1])
        torch.testing.assert_close(mixed_ponder_loss, reference_ponder_loss)
        mixed_parameters = tuple(mixed_halting_model.parameters())
        reference_parameters = tuple(reference_halting_model.parameters())
        mixed_gradients = torch.autograd.grad(
            mixed_ponder_loss,
            (*mixed_parameters, mixed_input),
        )
        reference_gradients = torch.autograd.grad(
            reference_ponder_loss,
            (*reference_parameters, reference_input),
        )
        for mixed_gradient, reference_gradient in zip(
            mixed_gradients[:-1],
            reference_gradients[:-1],
            strict=True,
        ):
            torch.testing.assert_close(mixed_gradient, reference_gradient)
        torch.testing.assert_close(
            mixed_gradients[-1][0],
            reference_gradients[-1][0],
        )
        torch.testing.assert_close(
            mixed_gradients[-1][1],
            torch.zeros_like(mixed_gradients[-1][1]),
        )

    def test_sparse_halting_update_freezes_inactive_stick_breaking_row(
        self,
    ) -> None:
        class StateHarness(_NeuronClusterStateMixin):
            @staticmethod
            def _accumulate_auxiliary_loss(loss, auxiliary_loss):
                return loss + auxiliary_loss

        halting_model = (
            self.halting_config(input_dim=2, threshold=0.999).build().double().eval()
        )
        state_harness = StateHarness()
        state_harness.halting_model = halting_model
        initial_hidden = torch.tensor(
            [[0.2, -0.4], [0.5, 0.3]],
            dtype=torch.float64,
            requires_grad=True,
        )
        initial_state = state_harness._maybe_update_halting_state(
            None,
            initial_hidden,
            initial_hidden,
            torch.tensor([True, True]),
        )
        current_hidden = torch.tensor(
            [[7.0, -5.0], [0.1, 0.8]],
            dtype=torch.float64,
            requires_grad=True,
        )
        weighted_candidate = torch.tensor(
            [[11.0, 13.0], [0.7, -0.2]],
            dtype=torch.float64,
            requires_grad=True,
        )

        updated_state = state_harness._maybe_update_halting_state(
            initial_state,
            current_hidden,
            weighted_candidate,
            torch.tensor([False, True]),
        )

        for attribute_name in (
            "halt_mask",
            "log_continuation",
            "accumulated_hidden",
            "output_hidden",
            "accumulated_halt_probabilities",
        ):
            with self.subTest(attribute=attribute_name):
                torch.testing.assert_close(
                    getattr(updated_state, attribute_name)[0],
                    getattr(initial_state, attribute_name)[0],
                )
        torch.testing.assert_close(
            updated_state.step_count,
            torch.tensor([0, 1]),
        )

        initial_inactive_value = (
            initial_state.accumulated_halt_probabilities[0]
            + initial_state.accumulated_hidden[0].sum()
        )
        updated_inactive_value = (
            updated_state.accumulated_halt_probabilities[0]
            + updated_state.accumulated_hidden[0].sum()
        )
        gate_parameters = tuple(halting_model.halting_gate_model.parameters())
        initial_gate_gradients = torch.autograd.grad(
            initial_inactive_value,
            gate_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        updated_gate_gradients = torch.autograd.grad(
            updated_inactive_value,
            gate_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        for initial_gradient, updated_gradient in zip(
            initial_gate_gradients,
            updated_gate_gradients,
            strict=True,
        ):
            self.assertEqual(initial_gradient is None, updated_gradient is None)
            if initial_gradient is not None:
                torch.testing.assert_close(updated_gradient, initial_gradient)
        inactive_current_gradient = torch.autograd.grad(
            updated_inactive_value,
            current_hidden,
            allow_unused=True,
        )[0]
        torch.testing.assert_close(
            inactive_current_gradient,
            torch.zeros_like(current_hidden),
        )

    def test_zero_probability_beam_expansions_are_excluded_without_redistribution(
        self,
    ) -> None:
        cluster = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=2,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            beam_width=2,
            growth_threshold=None,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        cluster.entry_sampler = ScriptedSampler(
            indices=[0, 1],
            probabilities=[0.25, 0.75],
        )
        cluster.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[1, 1, 1], [99, 1, 1]],
                    probabilities=[0.0, 1.0],
                    delta=[1.0] * self.input_dim,
                ),
                "neuron_2_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1], [99, 1, 1]],
                    probabilities=[0.0, 1.0],
                    delta=[10.0] * self.input_dim,
                ),
            }
        )

        output, auxiliary_loss = cluster(torch.zeros(1, self.input_dim))

        torch.testing.assert_close(output, torch.full_like(output, 7.75))
        torch.testing.assert_close(auxiliary_loss, torch.zeros_like(auxiliary_loss))

    def test_padded_beam_is_invalid_for_ponder_value_and_gradient(self) -> None:
        class BeamPonderHalting(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.final_hidden = None
                self.final_state = None

            def update_halting_state(
                self,
                previous_state,
                model_hidden_state,
            ):
                state = SimpleNamespace(
                    halt_mask=torch.zeros(
                        model_hidden_state.shape[0],
                        dtype=torch.bool,
                        device=model_hidden_state.device,
                    )
                )
                return state, model_hidden_state

            def finalize_weighted_accumulation(self, state, current_hidden):
                self.final_state = state
                self.final_hidden = current_hidden
                current_hidden.retain_grad()
                return current_hidden, current_hidden[:, 0]

        halting_model = BeamPonderHalting()
        cluster = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=2,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            beam_width=3,
            growth_threshold=None,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        cluster.entry_sampler = ScriptedSampler(
            indices=[0, 1],
            probabilities=[0.75, 0.25],
        )
        cluster.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[1, 1, 1]],
                    probabilities=[1.0],
                    delta=[10.0] * self.input_dim,
                ),
                "neuron_2_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1]],
                    probabilities=[1.0],
                    delta=[1.0] * self.input_dim,
                ),
            }
        )
        cluster.halting_model = halting_model

        output, ponder_loss = cluster(
            torch.zeros(1, self.input_dim, dtype=torch.float64, requires_grad=True)
        )
        ponder_loss.backward()

        torch.testing.assert_close(
            output[:, 0], torch.tensor([15.5], dtype=torch.float64)
        )
        torch.testing.assert_close(ponder_loss, torch.tensor(11.0, dtype=torch.float64))
        torch.testing.assert_close(
            halting_model.final_hidden.grad[:, 0],
            torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64),
        )

    def test_forward_only_neuron_compatibility_routes_and_processes_signal(
        self,
    ) -> None:
        class ForwardOnlyNeuron(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.forward_call_count = 0

            def forward(self, input_tensor):
                self.forward_call_count += 1
                batch_size = input_tensor.shape[0]
                coordinates = torch.tensor(
                    [1, 1, 1],
                    dtype=torch.long,
                    device=input_tensor.device,
                ).expand(batch_size, 1, 3)
                return (
                    input_tensor + 1,
                    input_tensor.new_ones((batch_size, 1)),
                    coordinates,
                    input_tensor.new_tensor(0.5),
                )

        cluster = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        cluster.entry_sampler = ScriptedSampler(indices=[0], probabilities=[1.0])
        neuron = ForwardOnlyNeuron()
        cluster.cluster = nn.ModuleDict({"neuron_1_1_1": neuron})

        output, auxiliary_loss = cluster(torch.zeros(2, self.input_dim))

        torch.testing.assert_close(output, torch.full_like(output, 2.0))
        torch.testing.assert_close(auxiliary_loss, torch.tensor(0.5))
        self.assertEqual(neuron.forward_call_count, 3)

    def test_completed_warmup_uses_full_processed_output(self) -> None:
        class WarmNeuron(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "warmup_remaining_steps",
                    torch.zeros((), dtype=torch.int64),
                )

            @staticmethod
            def process_signal(input_tensor):
                return input_tensor + 2

        cluster = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=100,
            growth_warmup_steps=2,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        neuron = WarmNeuron()
        input_tensor = torch.zeros(2, self.input_dim)

        output = cluster._process_neuron(neuron, input_tensor)

        torch.testing.assert_close(output, torch.full_like(output, 2.0))
        self.assertEqual(int(neuron.warmup_remaining_steps), 0)

    def test_real_neuron_input_nucleus_and_terminal_router_pass_gradcheck(
        self,
    ) -> None:
        neuron = (
            self.full_sampler_neuron_config(coordinate_embedding_flag=True)
            .build()
            .double()
            .eval()
        )
        parameter_names = tuple(name for name, _ in neuron.named_parameters())
        parameter_values = tuple(
            parameter.detach().clone().requires_grad_()
            for parameter in neuron.parameters()
        )
        input_tensor = torch.tensor(
            [[0.2, -0.4, 0.7, 1.1]],
            dtype=torch.float64,
            requires_grad=True,
        )

        def smooth_full_route_neuron(input_value, *parameters):
            replacements = dict(zip(parameter_names, parameters, strict=True))
            output, probabilities, _coordinates, _auxiliary_loss = functional_call(
                neuron,
                replacements,
                (input_value,),
            )
            return output, probabilities

        self.assertTrue(
            torch.autograd.gradcheck(
                smooth_full_route_neuron,
                (input_tensor, *parameter_values),
                eps=1e-6,
                atol=2e-6,
                rtol=1e-4,
            )
        )

    def test_axons_configured_memory_input_and_parameters_pass_gradcheck(
        self,
    ) -> None:
        memory_config = neuron_test_fixtures.make_memory_config(
            input_dim=2,
            output_dim=3,
        )
        axons = AxonsConfig(memory_config=memory_config).build().double().eval()
        parameter_names = tuple(name for name, _ in axons.named_parameters())
        parameter_values = tuple(
            parameter.detach().clone().requires_grad_()
            for parameter in axons.parameters()
        )
        input_tensor = torch.tensor(
            [[0.2, -0.4]],
            dtype=torch.float64,
            requires_grad=True,
        )

        def smooth_axons(input_value, *parameters):
            return functional_call(
                axons,
                dict(zip(parameter_names, parameters, strict=True)),
                (input_value,),
            )

        self.assertEqual(len(parameter_values), 4)
        self.assertTrue(
            torch.autograd.gradcheck(
                smooth_axons,
                (input_tensor, *parameter_values),
                eps=1e-6,
                atol=3e-6,
                rtol=2e-4,
            )
        )

        output = axons(input_tensor)
        direction = torch.tensor([[0.7, -0.3]], dtype=torch.float64)
        gradients = torch.autograd.grad(
            (output * direction).sum(),
            (input_tensor, *tuple(axons.parameters())),
        )
        for gradient_name, gradient in zip(
            ("input", *parameter_names),
            gradients,
            strict=True,
        ):
            with self.subTest(gradient=gradient_name):
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(torch.count_nonzero(gradient).item(), 0)

    def test_real_cluster_fixed_selection_gradient_matrix_and_gradcheck(self) -> None:
        with torch.random.fork_rng():
            torch.manual_seed(9)
            cluster = (
                NeuronClusterConfig(
                    x_axis_total_neurons=2,
                    y_axis_total_neurons=1,
                    z_axis_total_neurons=1,
                    max_steps=1,
                    growth_threshold=None,
                    neuron_config=self.full_sampler_neuron_config(),
                )
                .build()
                .double()
                .eval()
            )
        with torch.no_grad():
            for neuron_name, scale in (
                ("neuron_1_1_1", 0.2),
                ("neuron_2_1_1", 0.7),
            ):
                weight = cluster.cluster[neuron_name].nucleus.model.weight
                weight.copy_(torch.eye(self.input_dim, dtype=torch.float64) * scale)
            entry_layer = cluster.entry_sampler.router.model.layers[0].model
            entry_layer.weight_params.zero_()
            entry_layer.bias_params.copy_(
                torch.tensor([0.4, -0.2], dtype=torch.float64)
            )
        active_parameter_names = (
            "cluster.neuron_1_1_1.nucleus.model.weight",
            "cluster.neuron_2_1_1.nucleus.model.weight",
            "cluster.neuron_1_1_1.terminal.sampler.router.model.layers.0.model.weight_params",
            "cluster.neuron_1_1_1.terminal.sampler.router.model.layers.0.model.bias_params",
            "entry_sampler.router.model.layers.0.model.weight_params",
            "entry_sampler.router.model.layers.0.model.bias_params",
        )
        parameters_by_name = dict(cluster.named_parameters())
        parameter_values = tuple(
            parameters_by_name[name].detach().clone().requires_grad_()
            for name in active_parameter_names
        )
        input_tensor = torch.tensor(
            [[0.2, -0.4, 0.7, 1.1]],
            dtype=torch.float64,
            requires_grad=True,
        )

        entry_probabilities, entry_coordinates, _ = cluster._route_entry_input(
            input_tensor.detach()
        )
        chosen_coordinate = entry_coordinates[0, entry_probabilities.argmax(dim=1)[0]]
        torch.testing.assert_close(
            chosen_coordinate,
            torch.tensor([1, 1, 1]),
        )

        def fixed_selection_cluster(input_value, *parameters):
            replacements = dict(zip(active_parameter_names, parameters, strict=True))
            output, _auxiliary_loss = functional_call(
                cluster,
                replacements,
                (input_value,),
            )
            return output

        self.assertTrue(
            torch.autograd.gradcheck(
                fixed_selection_cluster,
                (input_tensor, *parameter_values),
                eps=1e-6,
                atol=3e-6,
                rtol=2e-4,
            )
        )

        output, auxiliary_loss = cluster(input_tensor)
        (output.square().sum() + auxiliary_loss).backward()
        gradients = {
            name: parameter.grad for name, parameter in cluster.named_parameters()
        }
        for name in active_parameter_names:
            with self.subTest(active_parameter=name):
                gradient = gradients[name]
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(torch.count_nonzero(gradient).item(), 0)
        inactive_prefix = "cluster.neuron_2_1_1.terminal.sampler.router"
        for name, gradient in gradients.items():
            if name.startswith(inactive_prefix):
                self.assertIsNone(gradient, name)

    def test_real_terminal_zero_centred_auxiliary_loss_gradients_match_oracles(
        self,
    ) -> None:
        auxiliary_weight = 0.7
        terminal_sampler_config = self.sampler_config(
            input_dim=self.input_dim,
            num_experts=3,
            top_k=2,
        )
        terminal_sampler_config.zero_centred_loss_weight = auxiliary_weight
        neuron_config = NeuronConfig(
            nucleus_config=NucleusConfig(
                model_config=self.projection_config(
                    input_dim=self.input_dim,
                    output_dim=self.input_dim,
                    scale=1.0,
                )
            ),
            axons_config=AxonsConfig(memory_config=None),
            terminal_config=self.terminal_config(
                input_dim=self.input_dim,
                sampler_config=terminal_sampler_config,
                connection_shape=TerminalConnectionShapeOptions.LINE_LEFT_RIGHT,
            ),
        )
        entry_sampler_config = self.sampler_config(
            input_dim=self.input_dim,
            num_experts=3,
            top_k=1,
        )
        cluster = (
            NeuronClusterConfig(
                x_axis_total_neurons=3,
                y_axis_total_neurons=1,
                z_axis_total_neurons=1,
                initial_x_axis_total_neurons=3,
                initial_y_axis_total_neurons=1,
                initial_z_axis_total_neurons=1,
                entry_sampler_config=entry_sampler_config,
                max_steps=1,
                growth_threshold=None,
                neuron_config=neuron_config,
            )
            .build()
            .double()
            .eval()
        )
        cluster.entry_sampler = ScriptedSampler(indices=[1], probabilities=[1.0])
        routed_neuron = cluster.cluster["neuron_2_1_1"]
        router_layer = routed_neuron.terminal.sampler.router.model.layers[0].model
        router_bias = torch.tensor([2.0, 0.0, -2.0], dtype=torch.float64)
        with torch.no_grad():
            routed_neuron.nucleus.model.weight.copy_(
                torch.eye(self.input_dim, dtype=torch.float64)
            )
            router_layer.weight_params.zero_()
            router_layer.bias_params.copy_(router_bias)

        input_tensor = torch.tensor(
            [
                [0.2, -0.4, 0.7, 1.1],
                [-0.3, 0.8, -0.1, 0.5],
            ],
            dtype=torch.float64,
        )
        _, auxiliary_loss = cluster(input_tensor)
        weight_gradient, bias_gradient = torch.autograd.grad(
            auxiliary_loss,
            (router_layer.weight_params, router_layer.bias_params),
        )

        log_normalizer = torch.logsumexp(router_bias, dim=0)
        probabilities = torch.softmax(router_bias, dim=0)
        score_gradient_per_row = (
            2.0
            * auxiliary_weight
            / input_tensor.shape[0]
            * log_normalizer
            * probabilities
        )
        expected_weight_gradient = input_tensor.transpose(0, 1) @ (
            score_gradient_per_row.expand(input_tensor.shape[0], -1)
        )
        expected_bias_gradient = score_gradient_per_row * input_tensor.shape[0]
        torch.testing.assert_close(
            auxiliary_loss,
            auxiliary_weight * log_normalizer.square(),
            rtol=2e-13,
            atol=2e-13,
        )
        torch.testing.assert_close(
            weight_gradient,
            expected_weight_gradient,
            rtol=2e-13,
            atol=2e-13,
        )
        torch.testing.assert_close(
            bias_gradient,
            expected_bias_gradient,
            rtol=2e-13,
            atol=2e-13,
        )

        weight_direction = torch.tensor(
            [
                [0.3, -0.2, 0.7],
                [-0.5, 0.4, 0.1],
                [0.6, -0.8, 0.2],
                [0.9, 0.25, -0.35],
            ],
            dtype=torch.float64,
        )
        bias_direction = torch.tensor(
            [0.45, -0.65, 0.15],
            dtype=torch.float64,
        )
        direction_norm = torch.sqrt(
            weight_direction.square().sum() + bias_direction.square().sum()
        )
        weight_direction = weight_direction / direction_norm
        bias_direction = bias_direction / direction_norm
        epsilon = 1e-6
        with torch.no_grad():
            router_layer.weight_params.copy_(epsilon * weight_direction)
            router_layer.bias_params.copy_(router_bias + epsilon * bias_direction)
            positive_loss = cluster(input_tensor)[1].clone()
            router_layer.weight_params.copy_(-epsilon * weight_direction)
            router_layer.bias_params.copy_(router_bias - epsilon * bias_direction)
            negative_loss = cluster(input_tensor)[1].clone()
            router_layer.weight_params.zero_()
            router_layer.bias_params.copy_(router_bias)

        finite_difference = (positive_loss - negative_loss) / (2.0 * epsilon)
        analytical_directional_derivative = (
            weight_gradient * weight_direction
        ).sum() + (bias_gradient * bias_direction).sum()
        torch.testing.assert_close(
            analytical_directional_derivative,
            finite_difference,
            rtol=1e-8,
            atol=1e-10,
        )
        for gradient in (weight_gradient, bias_gradient):
            self.assertEqual(gradient.dtype, torch.float64)
            self.assertEqual(gradient.device.type, "cpu")
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertEqual(torch.count_nonzero(gradient), gradient.numel())

    def test_shared_self_route_accumulates_input_and_parameter_gradients_each_step(
        self,
    ) -> None:
        cluster = (
            NeuronClusterConfig(
                x_axis_total_neurons=1,
                y_axis_total_neurons=1,
                z_axis_total_neurons=1,
                max_steps=2,
                growth_threshold=None,
                neuron_config=self.full_sampler_neuron_config(),
            )
            .build()
            .double()
            .eval()
        )
        cluster.entry_sampler = ScriptedSampler(indices=[0], probabilities=[1.0])
        neuron = _DifferentiableSelfRouteNeuron(scale=1.2)
        cluster.cluster = nn.ModuleDict({"neuron_1_1_1": neuron})
        input_tensor = torch.tensor(
            [[0.2, -0.4, 0.7, 1.1]],
            dtype=torch.float64,
            requires_grad=True,
        )
        direction = torch.tensor([[0.3, -0.2, 0.5, 0.4]], dtype=torch.float64)

        output, auxiliary_loss = cluster(input_tensor)
        objective = (output * direction).sum() + auxiliary_loss
        objective.backward()

        scale = neuron.scale.detach()
        expected_input_gradient = scale.pow(3) * direction
        expected_scale_gradient = (
            3 * scale.pow(2) * (input_tensor.detach() * direction).sum()
        )
        torch.testing.assert_close(input_tensor.grad, expected_input_gradient)
        torch.testing.assert_close(neuron.scale.grad, expected_scale_gradient)
        torch.testing.assert_close(output, input_tensor.detach() * scale.pow(3))

        epsilon = 1e-6
        original_scale = float(neuron.scale.detach())
        objectives = []
        with torch.no_grad():
            for perturbed_scale in (
                original_scale + epsilon,
                original_scale - epsilon,
            ):
                neuron.scale.fill_(perturbed_scale)
                perturbed_output, perturbed_loss = cluster(input_tensor.detach())
                objectives.append(
                    float((perturbed_output * direction).sum() + perturbed_loss)
                )
            neuron.scale.fill_(original_scale)
        central_difference = (objectives[0] - objectives[1]) / (2 * epsilon)
        self.assertAlmostEqual(
            float(neuron.scale.grad),
            central_difference,
            delta=1e-7 * max(1.0, abs(central_difference)),
        )


class TestNeuronMonitoringNumerics(unittest.TestCase):
    def test_empty_probabilities_and_all_invalid_routes_have_zero_diagnostics(
        self,
    ) -> None:
        step = NeuronClusterTraceStep(
            probabilities=torch.ones(2, 1),
            selected_coordinates=torch.zeros(2, 1, 3, dtype=torch.long),
            valid_mask=torch.zeros(2, 1, dtype=torch.bool),
            escape_mask=torch.zeros(2, 1, dtype=torch.bool),
            chosen_branch_indices=torch.zeros(2, dtype=torch.long),
            halt_mask=torch.zeros(2, dtype=torch.bool),
            active_mask=torch.zeros(2, dtype=torch.bool),
        )
        trace = NeuronClusterTrace(
            input_shape=(2, 4),
            entry_coordinates=torch.empty(0, 3, dtype=torch.long),
            entry_probabilities=torch.empty(2, 0),
            entry_selected_coordinates=torch.zeros(2, 1, 3, dtype=torch.long),
            entry_valid_mask=torch.zeros(2, 1, dtype=torch.bool),
            entry_escape_mask=torch.zeros(2, 1, dtype=torch.bool),
            entry_chosen_branch_indices=torch.zeros(2, dtype=torch.long),
            entry_halt_mask=torch.zeros(2, dtype=torch.bool),
            entry_active_mask=torch.zeros(2, dtype=torch.bool),
            steps=[step],
        )

        route_metrics = _NeuronDiagnostics.calculate_route(trace)

        self.assertIsNone(_NeuronDiagnostics.calculate_entry_routing(trace))
        torch.testing.assert_close(route_metrics.route_depth, torch.zeros(2))
        torch.testing.assert_close(route_metrics.escape_fraction, torch.zeros(()))
        torch.testing.assert_close(route_metrics.valid_fraction, torch.zeros(()))
        torch.testing.assert_close(route_metrics.halted_fraction, torch.zeros(()))
        torch.testing.assert_close(route_metrics.active_neuron_count, torch.zeros(()))
        torch.testing.assert_close(route_metrics.survival, torch.zeros(2))
        for metric in (
            route_metrics.route_depth,
            route_metrics.escape_fraction,
            route_metrics.valid_fraction,
            route_metrics.halted_fraction,
            route_metrics.active_neuron_count,
            route_metrics.survival,
        ):
            self.assertTrue(torch.isfinite(metric).all())

    def test_single_entry_routing_metrics_are_finite_and_have_zero_variation(
        self,
    ) -> None:
        trace = NeuronClusterTrace(
            input_shape=(2, 4),
            entry_coordinates=torch.tensor([[1, 1, 1]]),
            entry_probabilities=torch.ones(2, 1),
            entry_selected_coordinates=torch.tensor([[[1, 1, 1]], [[1, 1, 1]]]),
            entry_valid_mask=torch.ones(2, 1, dtype=torch.bool),
            entry_escape_mask=torch.zeros(2, 1, dtype=torch.bool),
            entry_chosen_branch_indices=torch.zeros(2, dtype=torch.long),
            entry_halt_mask=torch.zeros(2, dtype=torch.bool),
            entry_active_mask=torch.ones(2, dtype=torch.bool),
        )

        metrics = _NeuronDiagnostics.calculate_entry_routing(trace)

        self.assertIsNotNone(metrics)
        self.assertTrue(torch.isfinite(metrics.mean_entropy))
        self.assertTrue(torch.isfinite(metrics.marginal_entropy))
        self.assertTrue(torch.isfinite(metrics.coefficient_of_variation))
        torch.testing.assert_close(metrics.mean_entropy, torch.zeros(()))
        torch.testing.assert_close(metrics.marginal_entropy, torch.zeros(()))
        torch.testing.assert_close(metrics.coefficient_of_variation, torch.zeros(()))


if __name__ == "__main__":
    unittest.main()
