import math
import unittest

import torch

from unit.test_neuron import NeuronTestCase


class TestNeuronCoordinateEmbeddingContracts(NeuronTestCase):
    input_dim = 7

    def test_embedding_matches_independent_odd_width_sinusoidal_oracle(self) -> None:
        config = self.neuron_config(coordinate_embedding_flag=True)
        config.terminal_config.x_axis_position = 2
        config.terminal_config.y_axis_position = 3
        config.terminal_config.z_axis_position = 4

        neuron = config.build()

        x_frequency = neuron.COORDINATE_EMBEDDING_FREQUENCY_BASE ** (2.0 / 3.0)
        expected = torch.tensor(
            [
                math.sin(2.0),
                math.cos(2.0),
                math.sin(2.0 / x_frequency),
                math.sin(3.0),
                math.cos(3.0),
                math.sin(4.0),
                math.cos(4.0),
            ],
            dtype=torch.float32,
        )
        self.assertEqual(neuron.coordinate_embedding.shape, (7,))
        self.assertEqual(neuron.coordinate_embedding.dtype, torch.float32)
        torch.testing.assert_close(
            neuron.coordinate_embedding,
            expected,
            rtol=2e-7,
            atol=2e-7,
        )

    def test_embedding_injection_preserves_mixed_precision_input_contract(
        self,
    ) -> None:
        neuron = self.neuron_config(coordinate_embedding_flag=True).build()
        input_tensor = torch.zeros(
            2,
            self.input_dim,
            dtype=torch.float16,
            requires_grad=True,
        )

        output = neuron._Neuron__inject_coordinate_embedding(input_tensor)

        self.assertEqual(output.shape, input_tensor.shape)
        self.assertEqual(output.dtype, input_tensor.dtype)
        self.assertEqual(output.device, input_tensor.device)
        torch.testing.assert_close(
            output,
            neuron.coordinate_embedding.to(dtype=input_tensor.dtype).expand_as(output),
            rtol=0.0,
            atol=0.0,
        )
        output.sum().backward()
        torch.testing.assert_close(
            input_tensor.grad,
            torch.ones_like(input_tensor),
            rtol=0.0,
            atol=0.0,
        )

    def test_embedding_injection_follows_meta_input_device(self) -> None:
        neuron = self.neuron_config(coordinate_embedding_flag=True).build()
        input_tensor = torch.empty(
            2,
            self.input_dim,
            dtype=torch.float64,
            device="meta",
            requires_grad=True,
        )

        output = neuron._Neuron__inject_coordinate_embedding(input_tensor)

        self.assertEqual(output.shape, input_tensor.shape)
        self.assertEqual(output.dtype, input_tensor.dtype)
        self.assertEqual(output.device, input_tensor.device)
        output.sum().backward()
        self.assertIsNotNone(input_tensor.grad)
        self.assertEqual(input_tensor.grad.shape, input_tensor.shape)
        self.assertEqual(input_tensor.grad.dtype, input_tensor.dtype)
        self.assertEqual(input_tensor.grad.device, input_tensor.device)


if __name__ == "__main__":
    unittest.main()
