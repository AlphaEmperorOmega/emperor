import torch
from torch import Tensor

from unit.test_neuron import FourFieldOnlySampler, NeuronTestCase


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

    def test_forward_uses_only_four_field_sampler_interface(self):
        model = self.neuron_config().build()
        model.terminal.sampler = FourFieldOnlySampler(
            indices=[0, 1],
            probabilities=[0.25, 0.75],
        )

        _, probabilities, selected_neurons, auxiliary_loss = model(
            torch.randn(self.batch_size, self.input_dim)
        )

        torch.testing.assert_close(
            probabilities,
            torch.tensor([[0.25, 0.75]]).expand(self.batch_size, -1),
        )
        self.assertEqual(selected_neurons.shape, (self.batch_size, 2, 3))
        self.assertEqual(auxiliary_loss.shape, ())

    def test_coordinate_embedding_disabled_by_default(self):
        model = self.neuron_config().build()

        self.assertIsNone(model.coordinate_embedding)

    def test_coordinate_embedding_matches_sinusoidal_encoding(self):
        model = self.neuron_config(coordinate_embedding_flag=True).build()

        torch.testing.assert_close(
            model.coordinate_embedding,
            self.expected_coordinate_embedding(1, 1, 1),
        )

    def test_coordinate_embedding_differs_across_coordinates(self):
        base_model = self.neuron_config(coordinate_embedding_flag=True).build()
        shifted_config = self.neuron_config(coordinate_embedding_flag=True)
        shifted_config.terminal_config.x_axis_position = 2
        shifted_model = shifted_config.build()

        self.assertFalse(
            torch.allclose(
                base_model.coordinate_embedding,
                shifted_model.coordinate_embedding,
            )
        )

    def test_process_signal_injects_coordinate_embedding_into_nucleus(self):
        model = self.neuron_config(coordinate_embedding_flag=True).build()
        input_batch = torch.randn(self.batch_size, self.input_dim)

        output = model.process_signal(input_batch)

        torch.testing.assert_close(
            output,
            model.nucleus(input_batch + model.coordinate_embedding),
        )

    def test_route_signal_injects_coordinate_embedding_into_terminal(self):
        model = self.neuron_config(coordinate_embedding_flag=True).build()
        model.eval()
        processed_signal = torch.randn(self.batch_size, self.input_dim)

        probabilities, selected_neurons, _ = model.route_signal(processed_signal)

        _, expected_probabilities, expected_selected_neurons, _ = model.terminal(
            processed_signal + model.coordinate_embedding
        )
        torch.testing.assert_close(probabilities, expected_probabilities)
        torch.testing.assert_close(selected_neurons, expected_selected_neurons)

    def test_route_signal_runs_terminal_forward_hook_once(self):
        model = self.neuron_config().build()
        processed_signal = torch.randn(self.batch_size, self.input_dim)
        hook_outputs: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
        hook_handle = model.terminal.register_forward_hook(
            lambda _module, _inputs, output: hook_outputs.append(output)
        )
        try:
            probabilities, selected_neurons, auxiliary_loss = model.route_signal(
                processed_signal
            )
        finally:
            hook_handle.remove()

        self.assertEqual(len(hook_outputs), 1)
        self.assertEqual(len(hook_outputs[0]), 4)
        self.assertEqual(probabilities.shape[0], processed_signal.shape[0])
        self.assertEqual(selected_neurons.shape[0], processed_signal.shape[0])
        self.assertEqual(auxiliary_loss.shape, ())

    def test_route_signal_runs_terminal_backward_hook_once(self):
        model = self.neuron_config().build()
        processed_signal = torch.randn(
            self.batch_size,
            self.input_dim,
            requires_grad=True,
        )
        hook_calls: list[
            tuple[tuple[Tensor | None, ...], tuple[Tensor | None, ...]]
        ] = []
        hook_handle = model.terminal.register_full_backward_hook(
            lambda _module, grad_input, grad_output: hook_calls.append(
                (grad_input, grad_output)
            )
        )
        try:
            probabilities, _, _ = model.route_signal(processed_signal)
            probabilities.sum().backward()
        finally:
            hook_handle.remove()

        self.assertEqual(len(hook_calls), 1)
        self.assertIsNotNone(processed_signal.grad)
        self.assertTrue(torch.isfinite(processed_signal.grad).all().item())

    def test_coordinate_embedding_excluded_from_state_dict(self):
        model = self.neuron_config(coordinate_embedding_flag=True).build()

        self.assertNotIn("coordinate_embedding", model.state_dict())

    def test_coordinate_embedding_requires_minimum_input_dim(self):
        config = self.neuron_config(coordinate_embedding_flag=True)
        config.terminal_config.input_dim = 2

        with self.assertRaisesRegex(
            ValueError,
            "coordinate_embedding_flag requires terminal_config.input_dim",
        ):
            config.build()

    def test_coordinate_embedding_flag_rejects_non_bool(self):
        config = self.neuron_config()
        config.coordinate_embedding_flag = 1

        with self.assertRaisesRegex(
            TypeError,
            "coordinate_embedding_flag must be a bool",
        ):
            config.build()
