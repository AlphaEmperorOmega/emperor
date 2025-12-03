import unittest
from torch.nn import Parameter
from Emperor.generators.utils.mixtures.base import MixtureBase, MixtureConfig
from Emperor.generators.utils.mixtures.vector import (
    VectorBiasMixture,
    VectorMixtureBase,
    VectorWeightsMixture,
)


class TestVectorMixture(unittest.TestCase):
    def setUp(self):
        self.cfg = MixtureConfig(
            input_dim=4,
            output_dim=5,
            depth_dim=6,
            top_k=2,
            num_experts=6,
            weighted_parameters_flag=False,
        )

    def test_init(self):
        model_types = [VectorWeightsMixture, VectorBiasMixture]
        c = self.cfg

        for model_type in model_types:
            message = f"Testing model type: {model_type.__name__}"
            with self.subTest(msg=message):
                m = model_type(self.cfg)

                bank_shape = (c.output_dim, c.depth_dim)
                if model_type is VectorWeightsMixture:
                    bank_shape = (c.input_dim, c.depth_dim, c.output_dim)

                self.assertIsInstance(m, MixtureBase)
                self.assertIsInstance(m, VectorMixtureBase)
                self.assertEqual(m.input_dim, c.input_dim)
                self.assertEqual(m.output_dim, c.output_dim)
                self.assertEqual(m.depth_dim, c.depth_dim)
                self.assertEqual(m.top_k, c.top_k)
                self.assertEqual(m.num_experts, c.num_experts)
                self.assertEqual(m.weighted_parameters_flag, c.weighted_parameters_flag)
                self.assertIsInstance(m.parameter_bank, Parameter)
                self.assertEqual(m.parameter_bank.shape, bank_shape)
