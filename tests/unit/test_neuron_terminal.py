import torch
from torch import Tensor

from unit.test_neuron import FourFieldOnlySampler, NeuronTestCase


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

    def test_connection_shape_is_required(self):
        config = self.terminal_config()
        config.connection_shape = None
        torch.manual_seed(20260830)
        rng_before = torch.random.get_rng_state().clone()

        with self.assertRaisesRegex(
            ValueError,
            "connection_shape is required for TerminalConfig, received None",
        ):
            config.build()

        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)

    def test_forward_returns_selected_coordinates(self):
        model = self.terminal_config().build()
        input_batch = torch.randn(self.batch_size, self.input_dim)

        result = model(input_batch)
        self.assertEqual(len(result), 4)
        output, probabilities, selected_neurons, auxiliary_loss = result

        self.assertIs(output, input_batch)
        self.assertEqual(probabilities.shape, (self.batch_size, 2))
        self.assertEqual(selected_neurons.shape, (self.batch_size, 2, 3))
        self.assertIsInstance(auxiliary_loss, Tensor)
        self.assertEqual(auxiliary_loss.shape, ())

        with self.assertRaises(TypeError):
            model(input_batch, return_log_probabilities=True)

    def test_forward_uses_only_four_field_sampler_interface(self):
        model = self.terminal_config().build()
        model.sampler = FourFieldOnlySampler(
            indices=[0, 1],
            probabilities=[0.25, 0.75],
        )
        input_batch = torch.randn(self.batch_size, self.input_dim)

        routed_input, probabilities, selected_neurons, auxiliary_loss = model(
            input_batch
        )

        self.assertIs(routed_input, input_batch)
        torch.testing.assert_close(
            probabilities,
            torch.tensor([[0.25, 0.75]]).expand(self.batch_size, -1),
        )
        self.assertEqual(selected_neurons.shape, (self.batch_size, 2, 3))
        self.assertEqual(auxiliary_loss.shape, ())

    def test_routerless_full_selection_follows_input_device(self) -> None:
        total_connections = self.terminal_total_connections()
        model = self.terminal_config(
            input_dim=total_connections,
            sampler_config=self.sampler_config(
                input_dim=total_connections,
                num_experts=total_connections,
                top_k=total_connections,
                router_config=None,
            ),
        ).build()
        meta_input = torch.empty(2, total_connections, device="meta")

        output, probabilities, selected_neurons, _ = model(meta_input)

        self.assertEqual(output.device, meta_input.device)
        self.assertEqual(probabilities.device, meta_input.device)
        self.assertEqual(selected_neurons.device, meta_input.device)
        self.assertEqual(selected_neurons.shape, (2, total_connections, 3))

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
        self.assertEqual(
            selected_neurons.shape, (self.batch_size, total_connections, 3)
        )

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
