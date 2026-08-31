import torch

from emperor.memory import GatedResidualDynamicMemoryConfig
from emperor.memory._variants.gated_residual import GatedResidualDynamicMemory
from emperor.neuron import AxonsConfig
from unit.test_memory import make_memory_config
from unit.test_neuron import NeuronTestCase


class TestAxons(NeuronTestCase):
    def test_identity_path_preserves_input(self):
        model = AxonsConfig(memory_config=None).build()
        input_batch = torch.randn(self.batch_size, self.input_dim)

        output = model(input_batch)

        self.assertIs(output, input_batch)

    def test_builds_and_applies_dynamic_memory_config(self):
        memory_config = make_memory_config(
            config_cls=GatedResidualDynamicMemoryConfig,
            input_dim=self.input_dim,
            output_dim=self.input_dim + 2,
        )
        model = AxonsConfig(memory_config=memory_config).build()
        input_batch = torch.randn(self.batch_size, self.input_dim)

        output = model(input_batch)

        self.assertIsInstance(model.memory_model, GatedResidualDynamicMemory)
        self.assertEqual(model.memory_model.input_dim, self.input_dim)
        self.assertEqual(model.memory_model.output_dim, self.input_dim)
        self.assertEqual(output.shape, input_batch.shape)
        self.assertFalse(torch.allclose(output, input_batch))

    def test_rejects_non_memory_config_base(self):
        with self.assertRaises(TypeError):
            AxonsConfig(memory_config=self.projection_config()).build()
