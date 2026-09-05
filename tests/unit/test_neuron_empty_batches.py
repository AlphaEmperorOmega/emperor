from contextlib import nullcontext
from unittest.mock import patch

import torch

from emperor.neuron import NeuronClusterConfig
from unit.test_neuron import NeuronTestCase


class TestNeuronEmptyBatches(NeuronTestCase):
    def test_empty_terminal_zero_centred_loss_is_finite_and_connected(self):
        for top_k in (1, 2):
            with self.subTest(top_k=top_k):
                sampler_config = self.sampler_config(top_k=top_k)
                sampler_config.zero_centred_loss_weight = 0.5
                terminal = self.terminal_config(sampler_config=sampler_config).build()
                source = torch.empty(0, self.input_dim, requires_grad=True)
                _, _, _, loss = terminal(source)
                self.assertTrue(torch.isfinite(loss))
                torch.testing.assert_close(loss, torch.zeros(()))
                loss.backward()
                torch.testing.assert_close(source.grad, torch.zeros_like(source))

    def test_empty_beam_batch_preserves_shape_and_backward(self):
        cluster = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=2,
            beam_width=2,
            neuron_config=self.neuron_config(),
        ).build()
        source = torch.empty(0, self.input_dim, requires_grad=True)
        output, loss = cluster(source)
        self.assertEqual(output.shape, source.shape)
        self.assertTrue(torch.isfinite(loss))
        (output.sum() + loss).backward()
        torch.testing.assert_close(source.grad, torch.zeros_like(source))

    def test_empty_leading_dimensions_modes_and_lifecycle(self):
        for beam_width in (1, 2, 4):
            for halting in (False, True):
                for precision in ("float32", "float64", "autocast"):
                    for shape in ((0, 4), (2, 0, 4), (0, 3, 4)):
                        with self.subTest(
                            beam=beam_width,
                            halting=halting,
                            precision=precision,
                            shape=shape,
                        ):
                            cluster = NeuronClusterConfig(
                                x_axis_total_neurons=1,
                                y_axis_total_neurons=1,
                                z_axis_total_neurons=1,
                                max_steps=2,
                                beam_width=beam_width,
                                neuron_config=self.neuron_config(),
                                halting_config=self.halting_config()
                                if halting
                                else None,
                                growth_threshold=100,
                                growth_cooldown_steps=10,
                                pruning_threshold=100,
                            ).build()
                            dtype = (
                                torch.float64
                                if precision == "float64"
                                else torch.float32
                            )
                            cluster.to(dtype=dtype)
                            source = torch.empty(shape, dtype=dtype, requires_grad=True)
                            neuron = cluster.cluster["neuron_1_1_1"]
                            with patch.object(
                                neuron,
                                "process_signal",
                                side_effect=AssertionError("empty neuron call"),
                            ):
                                with (
                                    torch.autocast("cpu", dtype=torch.bfloat16)
                                    if precision == "autocast"
                                    else nullcontext()
                                ):
                                    output, loss = cluster(source)
                            self.assertEqual(output.shape, source.shape)
                            self.assertEqual(output.dtype, dtype)
                            self.assertEqual(output.device, source.device)
                            self.assertTrue(torch.isfinite(loss))
                            self.assertEqual(neuron.batch_counter.item(), 0)
                            self.assertEqual(neuron.atrophy_counter.item(), 1)
                            self.assertEqual(
                                cluster.forwards_since_last_growth.item(), 1
                            )
                            (output.sum() + loss).backward()
                            torch.testing.assert_close(
                                source.grad, torch.zeros_like(source)
                            )
                            self.assertTrue(
                                all(
                                    parameter.grad is None
                                    for parameter in neuron.parameters()
                                )
                            )

    def test_empty_beams_still_reject_single_route_trace(self):
        cluster = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            beam_width=2,
            neuron_config=self.neuron_config(),
        ).build()
        with self.assertRaisesRegex(NotImplementedError, "return_trace"):
            cluster(torch.empty(0, self.input_dim), return_trace=True)
