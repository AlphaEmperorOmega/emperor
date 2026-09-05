import copy

import torch
from torch import nn

from emperor.halting import HaltingHiddenStateModeOptions, SoftHaltingConfig
from unit import test_neuron as neuron_fixtures
from unit.test_neuron import NeuronTestCase, ScriptedSampler, ScriptedTerminal


class ScalingNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
        self.terminal = ScriptedTerminal([[1, 1, 1]])
        self.inputs = []

    def process_signal(self, hidden):
        self.inputs.append(hidden.detach().clone())
        return hidden * self.scale

    def route_signal(self, hidden):
        return (
            hidden.new_ones((hidden.shape[0], 1)),
            self.terminal.neuron_connections.unsqueeze(0).expand(
                hidden.shape[0], -1, -1
            ),
            hidden.new_zeros(()),
        )


class TestNeuronAccumulatedHalting(NeuronTestCase):
    @staticmethod
    def soft_config(config):
        return SoftHaltingConfig(
            **{name: getattr(config, name) for name in config.__dataclass_fields__}
        )

    def halting_cluster(self, config, beam_width=1, max_steps=2):
        cluster = (
            neuron_fixtures.TestNeuronCluster.scripted_cluster(
                self,
                max_steps=max_steps,
                halting_model=config.build(),
                input_dim=1,
                x_axis_total_neurons=1,
                beam_width=beam_width,
            )
            .double()
            .eval()
        )
        cluster.entry_sampler = ScriptedSampler(indices=[0], probabilities=[1.0])
        cluster.cluster = nn.ModuleDict({"neuron_1_1_1": ScalingNeuron()})
        return cluster

    def test_soft_raw_continuation_preserves_existing_raw_inputs(self):
        config = self.soft_config(self.halting_config(input_dim=1))
        cluster = self.halting_cluster(config)
        source = torch.ones(1, 1, dtype=torch.float64, requires_grad=True)
        output, _ = cluster(source)
        torch.testing.assert_close(
            torch.cat(cluster.cluster["neuron_1_1_1"].inputs),
            source.new_tensor([[1.0], [2.0], [4.0]]),
        )
        torch.testing.assert_close(output, source.new_tensor([[4.0]]))

    def test_accumulated_continuation_changes_multistep_result(self):
        config = self.halting_config(input_dim=1, threshold=0.999)
        config.hidden_state_mode = HaltingHiddenStateModeOptions.ACCUMULATED
        cluster = (
            neuron_fixtures.TestNeuronCluster.scripted_cluster(
                self,
                max_steps=2,
                halting_model=config.build(),
                input_dim=1,
                x_axis_total_neurons=1,
            )
            .double()
            .eval()
        )
        cluster.entry_sampler = ScriptedSampler(indices=[0], probabilities=[1.0])
        neuron = ScalingNeuron()
        cluster.cluster = nn.ModuleDict({"neuron_1_1_1": neuron})
        source = torch.ones(1, 1, dtype=torch.float64, requires_grad=True)
        output, loss = cluster(source)
        torch.testing.assert_close(
            torch.cat(neuron.inputs), source.new_tensor([[1.0], [1.0], [1.5]])
        )
        torch.testing.assert_close(output, source.new_tensor([[2.25]]))
        output.sum().backward()
        torch.testing.assert_close(source.grad, source.new_tensor([[2.25]]))
        self.assertTrue(torch.isfinite(loss))

    def test_both_strategies_match_direct_owner_steps_and_gradients(self):
        for soft in (False, True):
            for mode in HaltingHiddenStateModeOptions:
                for beam_width in (1, 3):
                    for min_steps, threshold in ((1, 0.999), (2, 0.999), (1, 0.6)):
                        with self.subTest(
                            soft=soft,
                            mode=mode,
                            beam=beam_width,
                            minimum=min_steps,
                            threshold=threshold,
                        ):
                            config = self.halting_config(
                                input_dim=1, threshold=threshold
                            )
                            config.min_steps = min_steps
                            config.hidden_state_mode = mode
                            if soft:
                                config = self.soft_config(config)
                            cluster = self.halting_cluster(config, beam_width)
                            reference_halting = copy.deepcopy(cluster.halting_model)
                            neuron = cluster.cluster["neuron_1_1_1"]
                            scale = neuron.scale.detach().clone().requires_grad_()
                            source = torch.tensor(
                                [[1.0], [0.25]], dtype=torch.float64, requires_grad=True
                            )
                            reference_source = source.detach().clone().requires_grad_()
                            hidden = reference_source
                            state = None
                            reference_inputs = []
                            for _ in range(3):
                                reference_inputs.append(hidden.detach().clone())
                                candidate = hidden * scale
                                state, returned = (
                                    reference_halting.update_halting_state(
                                        state, candidate
                                    )
                                )
                                hidden = (
                                    candidate
                                    if mode == HaltingHiddenStateModeOptions.RAW
                                    else returned
                                )
                                if state.halt_mask.all():
                                    break
                            expected, expected_loss = (
                                reference_halting.finalize_weighted_accumulation(
                                    state, candidate
                                )
                            )
                            expected_loss = expected_loss.mean()
                            actual, actual_loss = cluster(source)
                            torch.testing.assert_close(
                                actual, expected, rtol=1e-12, atol=1e-12
                            )
                            torch.testing.assert_close(
                                actual_loss, expected_loss, rtol=1e-12, atol=1e-12
                            )
                            torch.testing.assert_close(
                                torch.cat(neuron.inputs), torch.cat(reference_inputs)
                            )
                            (actual.sum() + actual_loss).backward()
                            (expected.sum() + expected_loss).backward()
                            torch.testing.assert_close(
                                source.grad, reference_source.grad
                            )
                            torch.testing.assert_close(neuron.scale.grad, scale.grad)
                            reference_parameters = dict(
                                reference_halting.named_parameters()
                            )
                            for (
                                name,
                                parameter,
                            ) in cluster.halting_model.named_parameters():
                                reference_gradient = reference_parameters[name].grad
                                if reference_gradient is None:
                                    self.assertIsNone(parameter.grad)
                                else:
                                    torch.testing.assert_close(
                                        parameter.grad, reference_gradient
                                    )

    def test_soft_accumulated_has_distinct_closed_form_continuation(self):
        config = self.soft_config(self.halting_config(input_dim=1))
        config.hidden_state_mode = HaltingHiddenStateModeOptions.ACCUMULATED
        for beam_width in (1, 2):
            cluster = self.halting_cluster(config, beam_width)
            source = torch.ones(1, 1, dtype=torch.float64, requires_grad=True)
            output, _ = cluster(source)
            torch.testing.assert_close(
                torch.cat(cluster.cluster["neuron_1_1_1"].inputs),
                source.new_tensor([[1.0], [2.0], [3.0]]),
            )
            torch.testing.assert_close(output, source.new_tensor([[3.5]]))
            output.sum().backward()
            torch.testing.assert_close(source.grad, source.new_tensor([[3.5]]))

    def test_partial_halting_matches_independent_per_row_trajectories(self):
        for soft in (False, True):
            for beam_width in (1, 3):
                with self.subTest(soft=soft, beam=beam_width):
                    config = self.halting_config(input_dim=1, threshold=0.9)
                    config.hidden_state_mode = HaltingHiddenStateModeOptions.ACCUMULATED
                    if soft:
                        config = self.soft_config(config)
                    cluster = self.halting_cluster(config, beam_width)
                    gate = (
                        cluster.halting_model._gate
                        if soft
                        else cluster.halting_model.halting_gate_model
                    )
                    with torch.no_grad():
                        gate[-1].model.weight_params.copy_(
                            torch.tensor([[-1.0, 1.0]], dtype=torch.float64)
                        )
                    reference = copy.deepcopy(cluster.halting_model)
                    source = torch.tensor(
                        [[2.0], [-0.1]], dtype=torch.float64, requires_grad=True
                    )
                    reference_source = source.detach().clone().requires_grad_()
                    neuron = cluster.cluster["neuron_1_1_1"]
                    reference_scale = neuron.scale.detach().clone().requires_grad_()
                    outputs, losses, call_counts = [], [], []
                    for row in reference_source.split(1):
                        hidden, state, calls = row, None, 0
                        for _ in range(3):
                            candidate = hidden * reference_scale
                            state, hidden = reference.update_halting_state(
                                state, candidate
                            )
                            calls += 1
                            if state.halt_mask.all():
                                break
                        output, loss = reference.finalize_weighted_accumulation(
                            state, candidate
                        )
                        outputs.append(output)
                        losses.append(loss.mean())
                        call_counts.append(calls)
                    self.assertLess(call_counts[0], call_counts[1])
                    expected, expected_loss = (
                        torch.cat(outputs),
                        torch.stack(losses).mean(),
                    )
                    actual, loss = cluster(source)
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(loss, expected_loss)
                    self.assertEqual(
                        sum(batch.shape[0] for batch in neuron.inputs), sum(call_counts)
                    )
                    (actual.sum() + loss).backward()
                    (expected.sum() + expected_loss).backward()
                    torch.testing.assert_close(source.grad, reference_source.grad)
                    torch.testing.assert_close(neuron.scale.grad, reference_scale.grad)
                    for name, parameter in cluster.halting_model.named_parameters():
                        torch.testing.assert_close(
                            parameter.grad,
                            dict(reference.named_parameters())[name].grad,
                        )

    def test_escaped_final_output_is_not_replaced_by_halting_accumulation(self):
        for beam_width in (1, 2):
            config = self.halting_config(input_dim=1, threshold=0.999)
            config.hidden_state_mode = HaltingHiddenStateModeOptions.ACCUMULATED
            cluster = self.halting_cluster(config, beam_width)
            cluster.cluster["neuron_1_1_1"].terminal.neuron_connections.fill_(99)
            source = torch.ones(1, 1, dtype=torch.float64, requires_grad=True)
            output, _ = cluster(source)
            torch.testing.assert_close(output, source)
            output.sum().backward()
            torch.testing.assert_close(source.grad, torch.ones_like(source))
