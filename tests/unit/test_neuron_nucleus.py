import torch

from emperor.neuron import NucleusConfig
from unit.test_neuron import NeuronTestCase


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
