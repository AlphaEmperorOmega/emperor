import torch
import unittest
import torch.nn.functional as F

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

    def test_init_parameter_select_range(self):
        model_types = [VectorWeightsMixture, VectorBiasMixture]
        top_k_values = [1, 3, 6]
        for model_type in model_types:
            for top_k in top_k_values:
                message = (
                    f"Testing model type: {model_type.__name__} with top_k={top_k}"
                )
                with self.subTest(msg=message):
                    overrides = MixtureConfig(
                        top_k=top_k,
                        num_experts=6,
                        weighted_parameters_flag=True,
                    )
                    m = model_type(self.cfg, overrides)
                    range_indices = m._init_parameter_select_range()

                    expected_range_shape = (1, m.range_dim)
                    if 1 < top_k < m.depth_dim:
                        expected_range_shape = (1, m.range_dim, 1)

                    self.assertEqual(range_indices.shape, expected_range_shape)

    def test_select_parameters(self):
        model_types = [VectorWeightsMixture, VectorBiasMixture]
        top_k_values = [1, 3, 6]
        for model_type in model_types:
            for top_k in top_k_values:
                message = (
                    f"Testing model type: {model_type.__name__} with top_k={top_k}"
                )
                with self.subTest(msg=message):
                    overrides = MixtureConfig(
                        top_k=top_k,
                        depth_dim=6,
                        num_experts=6,
                        weighted_parameters_flag=True,
                    )
                    m = model_type(self.cfg, overrides)

                    batch_size = 5

                    if top_k == 1:
                        indexes_shape = (m.input_dim, batch_size)
                        if model_type is VectorBiasMixture:
                            indexes_shape = (m.output_dim, batch_size)
                        indexes = torch.randint(0, m.depth_dim, indexes_shape)
                    elif 1 < top_k < m.depth_dim:
                        indexes_shape = (m.input_dim, batch_size, top_k)
                        if model_type is VectorBiasMixture:
                            indexes_shape = (m.output_dim, batch_size, top_k)
                        indexes = torch.randint(0, m.depth_dim, indexes_shape)
                    else:
                        indexes = None

                    selected_params = m._select_parameters(indexes)

                    if top_k == 1:
                        expected_shape = (batch_size, m.input_dim, m.output_dim)
                        if model_type is VectorBiasMixture:
                            expected_shape = (batch_size, m.output_dim)
                    elif 1 < top_k < m.depth_dim:
                        expected_shape = (batch_size, m.input_dim, top_k, m.output_dim)
                        if model_type is VectorBiasMixture:
                            expected_shape = (batch_size, m.output_dim, m.top_k)
                    else:
                        expected_shape = (m.input_dim, m.depth_dim, m.output_dim)
                        if model_type is VectorBiasMixture:
                            expected_shape = (m.output_dim, m.depth_dim)

                    self.assertEqual(selected_params.shape, expected_shape)

    def test__compute_parameter_mixture(self):
        model_types = [VectorWeightsMixture, VectorBiasMixture]
        top_k_values = [1, 3, 6]
        for model_type in model_types:
            for top_k in top_k_values:
                message = (
                    f"Testing model type: {model_type.__name__} with top_k={top_k}"
                )
                with self.subTest(msg=message):
                    overrides = MixtureConfig(
                        top_k=top_k,
                        depth_dim=6,
                        num_experts=6,
                        weighted_parameters_flag=True,
                    )
                    m = model_type(self.cfg, overrides)

                    batch_size = 5

                    if top_k == 1:
                        selected_shape = (batch_size, m.input_dim, m.output_dim)
                        if model_type is VectorBiasMixture:
                            selected_shape = (batch_size, m.output_dim)
                        probs_shape = (m.input_dim, batch_size, m.depth_dim)
                    elif 1 < top_k < m.depth_dim:
                        selected_shape = (batch_size, m.input_dim, top_k, m.output_dim)
                        if model_type is VectorBiasMixture:
                            selected_shape = (batch_size, m.output_dim, top_k)
                        probs_shape = (m.input_dim, batch_size, m.depth_dim)
                    else:
                        selected_shape = (m.input_dim, m.depth_dim, m.output_dim)
                        if model_type is VectorBiasMixture:
                            selected_shape = (batch_size, m.output_dim, top_k)
                        probs_shape = (m.input_dim, batch_size, m.depth_dim)

                    selected_parameters = torch.randn(selected_shape)
                    probs = F.sigmoid(torch.randn(probs_shape))
                    selected_params = m._VectorMixtureBase__compute_parameter_mixture(
                        selected_parameters, probs
                    )

                    # if top_k == 1:
                    #     expected_shape = (batch_size, m.input_dim, m.output_dim)
                    #     if model_type is VectorBiasMixture:
                    #         expected_shape = (batch_size, m.output_dim)
                    # elif 1 < top_k < m.depth_dim:
                    #     expected_shape = (batch_size, m.input_dim, top_k, m.output_dim)
                    #     if model_type is VectorBiasMixture:
                    #         expected_shape = (batch_size, m.output_dim, m.top_k)
                    # else:
                    #     expected_shape = (m.input_dim, m.depth_dim, m.output_dim)
                    #     if model_type is VectorBiasMixture:
                    #         expected_shape = (m.output_dim, m.depth_dim)
                    #
                    # self.assertEqual(selected_params.shape, expected_shape)
