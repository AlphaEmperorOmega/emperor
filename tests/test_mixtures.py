import copy
import unittest
import torch
import torch.nn.functional as F
from math import prod
from torch.nn import Parameter
from Emperor.components.parameter_generators.utils import mixture
from Emperor.components.parameter_generators.utils.mixture import (
    MixtureConfig,
    MixtureBase,
    ParameterBank,
    VectorChoiceMixture,
    MatrixChoiceMixture,
    GeneratorChoiceMixture,
)
from Emperor.config import ModelConfig


class TestMixtureBase(unittest.TestCase):
    def setUp(self):
        self.cfg = ModelConfig()
        self.mixture_cfg = self.cfg.mixture_model_config

    def test__init_with_custom_mixture_config_only(self):
        cfg = MixtureConfig(
            top_k=3,
            weighted_parameters_flag=True,
            bias_parameters_flag=True,
            router_output_dim=10,
        )
        m = MixtureBase(cfg)
        self.assertEqual(m.cfg, cfg)
        self.assertEqual(m.top_k, cfg.top_k)
        self.assertEqual(m.bias_parameters_flag, cfg.bias_parameters_flag)
        self.assertEqual(m.weighted_parameters_flag, cfg.weighted_parameters_flag)
        self.assertEqual(m.router_output_dim, cfg.router_output_dim)

    def test__init_with_main_config(self):
        m = MixtureBase(self.cfg)
        self.assertEqual(m.top_k, self.mixture_cfg.top_k)
        self.assertEqual(m.bias_parameters_flag, self.mixture_cfg.bias_parameters_flag)
        self.assertEqual(
            m.weighted_parameters_flag, self.mixture_cfg.weighted_parameters_flag
        )
        self.assertEqual(m.router_output_dim, self.mixture_cfg.router_output_dim)

    def test__init_with_overrides(self):
        overrides = MixtureConfig(
            input_dim=2,
            output_dim=5,
            depth_dim=7,
            top_k=5,
            weighted_parameters_flag=False,
            bias_parameters_flag=False,
            router_output_dim=20,
            cross_diagonal_flag=False,
        )
        m = MixtureBase(self.cfg, overrides)
        self.assertEqual(m.cfg, overrides)
        self.assertEqual(m.top_k, overrides.top_k)
        self.assertEqual(m.bias_parameters_flag, overrides.bias_parameters_flag)
        self.assertEqual(m.router_output_dim, overrides.router_output_dim)


class TestParameterBank(unittest.TestCase):
    def setUp(self):
        self.shape = (3, 4)
        self.initializer = lambda x: torch.nn.init.uniform_(x, -1, 1)

    def test__init(self):
        m = ParameterBank(self.shape, self.initializer)
        self.assertEqual(m.shape, self.shape)
        self.assertEqual(m.initializer, self.initializer)
        self.assertIsInstance(m.parameter_bank, Parameter)
        self.assertEqual(m.parameter_bank.shape, self.shape)

    def test__create_bank(self):
        m = ParameterBank(self.shape, self.initializer)
        self.assertTrue(torch.all(m.parameter_bank >= -1))
        self.assertTrue(torch.all(m.parameter_bank <= 1))

    def test__get(self):
        m = ParameterBank(self.shape, self.initializer)
        param = m.get()
        self.assertIs(param, m.parameter_bank)
        self.assertIsInstance(param, Parameter)
        self.assertEqual(param.shape, self.shape)


class TestVectorChoiceMixture(unittest.TestCase):
    def setUp(self):
        self.cfg = ModelConfig()
        self.mixture_cfg = self.cfg.mixture_model_config

    def test__init_with_default_config(self):
        m = VectorChoiceMixture(self.cfg)
        input = self.mixture_cfg.input_dim
        depth = self.mixture_cfg.depth_dim
        output = self.mixture_cfg.output_dim

        self.assertIsInstance(m.weight_bank, Parameter)
        self.assertEqual(list(m.weight_bank.shape), [input, depth, output])
        self.assertIsInstance(m.bias_bank, Parameter)
        self.assertEqual(list(m.bias_bank.shape), [output, depth])

        self.assertTrue(hasattr(m, "range_weights"))
        self.assertTrue(hasattr(m, "range_biases"))

    def test__init_parameter_banks(self):
        m = VectorChoiceMixture(self.cfg)
        weight_bank, bias_bank = m._VectorChoiceMixture__init_parameter_banks()
        input = self.mixture_cfg.input_dim
        depth = self.mixture_cfg.depth_dim
        output = self.mixture_cfg.output_dim

        self.assertIsInstance(weight_bank, Parameter)
        self.assertEqual(list(weight_bank.shape), [input, depth, output])

        self.assertIsInstance(bias_bank, Parameter)
        self.assertEqual(list(bias_bank.shape), [output, depth])

        m.bias_parameters_flag = False
        weight_bank, bias_bank = m._VectorChoiceMixture__init_parameter_banks()
        self.assertIsInstance(weight_bank, Parameter)
        self.assertIsNone(bias_bank)

    def test__init_parameter_choice_ranges(self):
        m = VectorChoiceMixture(self.cfg)
        input = self.mixture_cfg.input_dim
        output = self.mixture_cfg.output_dim

        m.top_k = 3
        range_weights, range_biases = (
            m._VectorChoiceMixture__init_parameter_choice_ranges()
        )
        self.assertEqual(range_weights.shape, torch.Size([1, input, 1]))
        self.assertEqual(range_biases.shape, torch.Size([1, output, 1]))

        m.top_k = 1
        range_weights, range_biases = (
            m._VectorChoiceMixture__init_parameter_choice_ranges()
        )

        self.assertEqual(range_weights.shape, torch.Size([1, input]))
        self.assertEqual(range_biases.shape, torch.Size([1, output]))

        m.top_k = 10
        m.depth_dim = 10
        range_weights, range_biases = (
            m._VectorChoiceMixture__init_parameter_choice_ranges()
        )

        self.assertEqual(range_weights.shape, torch.Size([1, input]))
        self.assertEqual(range_biases.shape, torch.Size([1, output]))

    def test__select_parameters_top_1(self):
        sz = lambda x: torch.Size(x)

        input = self.mixture_cfg.input_dim
        depth = self.mixture_cfg.depth_dim
        output = self.mixture_cfg.output_dim
        batch_size = 5

        overrides = MixtureConfig(top_k=1)
        m = VectorChoiceMixture(self.cfg, overrides)

        weight_indexes = torch.randint(0, depth, (input, batch_size))
        bias_indexes = torch.randint(0, depth, (output, batch_size))

        selected_weights, selected_biases = m._select_parameters(
            weight_indexes, bias_indexes
        )

        self.assertEqual(selected_weights.shape, sz([batch_size, input, output]))
        self.assertEqual(selected_biases.shape, sz([batch_size, output]))

        selected_weights, selected_biases = m._select_parameters(weight_indexes, None)

        self.assertEqual(selected_weights.shape, sz([batch_size, input, output]))
        self.assertIsNone(selected_biases)

    def test__select_parameters_top_k(self):
        sz = lambda x: torch.Size(x)

        input = self.mixture_cfg.input_dim
        depth = self.mixture_cfg.depth_dim
        output = self.mixture_cfg.output_dim
        top_k = self.mixture_cfg.top_k
        batch_size = 5

        m = VectorChoiceMixture(self.cfg)

        weight_indexes = torch.randint(0, depth, (input, batch_size, top_k))
        bias_indexes = torch.randint(0, depth, (output, batch_size, top_k))

        selected_weights, selected_biases = m._select_parameters(
            weight_indexes, bias_indexes
        )

        self.assertEqual(selected_weights.shape, sz([batch_size, input, top_k, output]))
        self.assertEqual(selected_biases.shape, sz([batch_size, output, top_k]))

        selected_weights, selected_biases = m._select_parameters(weight_indexes, None)

        self.assertEqual(selected_weights.shape, sz([batch_size, input, top_k, output]))
        self.assertIsNone(selected_biases)

    def test__compute_parameter_mixture_top_1(self):
        input = self.mixture_cfg.input_dim
        depth = self.mixture_cfg.depth_dim
        output = self.mixture_cfg.output_dim
        batch_size = 5

        overrides = MixtureConfig(top_k=1, weighted_parameters_flag=True)
        m = VectorChoiceMixture(self.cfg, overrides)

        weight_indexes = torch.randint(0, depth, (input, batch_size))
        bias_indexes = torch.randint(0, depth, (output, batch_size))
        weight_probs = F.sigmoid(torch.randn(input, batch_size))
        bias_probs = F.sigmoid(torch.randn(output, batch_size))

        selected_weights, selected_biases = m._select_parameters(
            weight_indexes, bias_indexes
        )

        weight_mixture, bias_mixture = m._compute_parameter_mixture(
            selected_weights, weight_probs, selected_biases, bias_probs
        )

        self.assertEqual(weight_mixture.shape, torch.Size([batch_size, input, output]))
        self.assertEqual(bias_mixture.shape, torch.Size([batch_size, output]))

        weight_probs_test = weight_probs.transpose(1, 0).unsqueeze(-1)
        bias_probs_test = bias_probs.transpose(1, 0)

        expected_weight_mixture = selected_weights * weight_probs_test
        expected_bias_mixture = selected_biases * bias_probs_test

        self.assertTrue(torch.allclose(weight_mixture, expected_weight_mixture))
        self.assertTrue(torch.allclose(bias_mixture, expected_bias_mixture))

        weight_mixture, bias_mixture = m._compute_parameter_mixture(
            selected_weights, weight_probs, None, None
        )
        self.assertEqual(weight_mixture.shape, torch.Size([batch_size, input, output]))
        self.assertIsNone(bias_mixture)

    def test__compute_parameter_mixture_top_1_weighted_parameters_false(self):
        input = self.mixture_cfg.input_dim
        depth = self.mixture_cfg.depth_dim
        output = self.mixture_cfg.output_dim
        batch_size = 5

        overrides = MixtureConfig(top_k=1, weighted_parameters_flag=False)
        m = VectorChoiceMixture(self.cfg, overrides)

        weight_indexes = torch.randint(0, depth, (input, batch_size))
        bias_indexes = torch.randint(0, depth, (output, batch_size))
        weight_probs = F.softmax(torch.randn(input, batch_size), dim=-1)
        bias_probs = F.softmax(torch.randn(output, batch_size), dim=-1)

        selected_weights, selected_biases = m._select_parameters(
            weight_indexes, bias_indexes
        )

        weight_mixture, bias_mixture = m._compute_parameter_mixture(
            selected_weights, weight_probs, selected_biases, bias_probs
        )

        self.assertEqual(weight_mixture.shape, torch.Size([batch_size, input, output]))
        self.assertEqual(bias_mixture.shape, torch.Size([batch_size, output]))

        self.assertTrue(torch.allclose(weight_mixture, selected_weights))
        self.assertTrue(torch.allclose(bias_mixture, selected_biases))

        weight_mixture, bias_mixture = m._compute_parameter_mixture(
            selected_weights, weight_probs, None, None
        )
        self.assertEqual(weight_mixture.shape, torch.Size([batch_size, input, output]))
        self.assertIsNone(bias_mixture)

    def test__compute_parameter_mixture_top_k(self):
        input = self.mixture_cfg.input_dim
        depth = self.mixture_cfg.depth_dim
        output = self.mixture_cfg.output_dim
        top_k = self.mixture_cfg.top_k
        batch_size = 5

        m = VectorChoiceMixture(self.cfg)

        weight_indexes = torch.randint(0, depth, (input, batch_size, top_k))
        bias_indexes = torch.randint(0, depth, (output, batch_size, top_k))
        weight_probs = F.softmax(torch.randn(input, batch_size, top_k), dim=-1)
        bias_probs = F.softmax(torch.randn(output, batch_size, top_k), dim=-1)

        selected_weights, selected_biases = m._select_parameters(
            weight_indexes, bias_indexes
        )
        weight_mixture, bias_mixture = m._compute_parameter_mixture(
            selected_weights, weight_probs, selected_biases, bias_probs
        )

        self.assertEqual(weight_mixture.shape, torch.Size([batch_size, input, output]))
        self.assertEqual(bias_mixture.shape, torch.Size([batch_size, output]))

        weight_probs_test = weight_probs.transpose(1, 0).unsqueeze(-1)
        bias_probs_test = bias_probs.transpose(1, 0)

        weighted_weights = selected_weights * weight_probs_test
        weighted_biases = selected_biases * bias_probs_test

        expected_weight_mixture = weighted_weights.sum(dim=-2)
        expected_bias_mixture = weighted_biases.sum(dim=-1)

        self.assertTrue(torch.allclose(weight_mixture, expected_weight_mixture))
        self.assertTrue(torch.allclose(bias_mixture, expected_bias_mixture))

        weight_mixture, bias_mixture = m._compute_parameter_mixture(
            selected_weights, weight_probs, None, None
        )
        self.assertEqual(weight_mixture.shape, torch.Size([batch_size, input, output]))
        self.assertIsNone(bias_mixture)

    def test__compute_parameter_mixture_top_k_weighted_parameters_false(self):
        input = self.mixture_cfg.input_dim
        depth = self.mixture_cfg.depth_dim
        output = self.mixture_cfg.output_dim
        top_k = self.mixture_cfg.top_k
        batch_size = 5

        overrides = MixtureConfig(weighted_parameters_flag=False)
        m = VectorChoiceMixture(self.cfg, overrides)

        weight_indexes = torch.randint(0, depth, (input, batch_size, top_k))
        bias_indexes = torch.randint(0, depth, (output, batch_size, top_k))
        weight_probs = F.softmax(torch.randn(input, batch_size, top_k), dim=-1)
        bias_probs = F.softmax(torch.randn(output, batch_size, top_k), dim=-1)

        selected_weights, selected_biases = m._select_parameters(
            weight_indexes, bias_indexes
        )

        weight_mixture, bias_mixture = m._compute_parameter_mixture(
            selected_weights, weight_probs, selected_biases, bias_probs
        )

        self.assertEqual(weight_mixture.shape, torch.Size([batch_size, input, output]))
        self.assertEqual(bias_mixture.shape, torch.Size([batch_size, output]))

        expected_weight_mixture = selected_weights.sum(dim=-2)
        expected_bias_mixture = selected_biases.sum(dim=-1)

        self.assertTrue(torch.allclose(weight_mixture, expected_weight_mixture))
        self.assertTrue(torch.allclose(bias_mixture, expected_bias_mixture))

        weight_mixture, bias_mixture = m._compute_parameter_mixture(
            selected_weights, weight_probs, None, None
        )
        self.assertEqual(weight_mixture.shape, torch.Size([batch_size, input, output]))
        self.assertIsNone(bias_mixture)

    def test__compute_parameter_mixture_full(self):
        input = self.mixture_cfg.input_dim
        depth = self.mixture_cfg.depth_dim
        output = self.mixture_cfg.output_dim
        top_k = self.mixture_cfg.top_k
        router_output_dim = self.mixture_cfg.router_output_dim
        batch_size = 5

        overrides = MixtureConfig(
            top_k=router_output_dim,
            depth_dim=router_output_dim,
            weighted_parameters_flag=True,
        )
        m = VectorChoiceMixture(self.cfg, overrides)

        weight_probs = F.softmax(
            torch.randn(input, batch_size, router_output_dim), dim=-1
        )
        bias_probs = F.softmax(
            torch.randn(output, batch_size, router_output_dim), dim=-1
        )

        weight_mixture, bias_mixture = m._compute_parameter_mixture(
            m.weight_bank.unsqueeze(0),
            weight_probs,
            m.bias_bank.unsqueeze(0),
            bias_probs,
        )
        self.assertEqual(weight_mixture.shape, torch.Size([batch_size, input, output]))
        self.assertEqual(bias_mixture.shape, torch.Size([batch_size, output]))

        weight_probs_test = weight_probs.transpose(1, 0).unsqueeze(-1)
        bias_probs_test = bias_probs.transpose(1, 0)

        weighted_weights = m.weight_bank.unsqueeze(0) * weight_probs_test
        weighted_biases = m.bias_bank.unsqueeze(0) * bias_probs_test

        expected_weight_mixture = weighted_weights.sum(dim=-2)
        expected_bias_mixture = weighted_biases.sum(dim=-1)

        self.assertTrue(torch.allclose(weight_mixture, expected_weight_mixture))
        self.assertTrue(torch.allclose(bias_mixture, expected_bias_mixture))

        weight_mixture, bias_mixture = m._compute_parameter_mixture(
            m.weight_bank.unsqueeze(0), weight_probs, None, None
        )
        self.assertEqual(weight_mixture.shape, torch.Size([batch_size, input, output]))
        self.assertIsNone(bias_mixture)

    # def test__compute_parameter_mixture_full_weighted_parameters_false(self):
    #     input = self.mixture_cfg.input_dim
    #     depth = self.mixture_cfg.depth_dim
    #     output = self.mixture_cfg.output_dim
    #     top_k = self.mixture_cfg.top_k
    #     router_output_dim = self.mixture_cfg.router_output_dim
    #     threshold = 0.1
    #     batch_size = 5
    #
    #     overrides = MixtureConfig(
    #         top_k=router_output_dim,
    #         depth_dim=router_output_dim,
    #         weighted_parameters_flag=False,
    #     )
    #     m = VectorChoiceMixture(self.cfg, overrides)
    #
    #     weight_probs_shape = (input, batch_size, router_output_dim)
    #     weight_logits = torch.randn(weight_probs_shape)
    #     weight_mask = weight_logits > 0
    #     weight_logits = weight_logits.masked_fill(weight_mask, float("-inf"))
    #     weight_probs = F.softmax(weight_logits, dim=-1)
    #
    #     bias_probs_shape = (output, batch_size, router_output_dim)
    #     bias_logits = torch.randn(bias_probs_shape)
    #     bias_mask = bias_logits > 0
    #     bias_logits = bias_logits.masked_fill(bias_mask, float("-inf"))
    #     bias_probs = F.softmax(bias_logits, dim=-1)
    #
    #     weight_mixture, bias_mixture = m._compute_parameter_mixture(
    #         m.weight_bank.unsqueeze(0),
    #         weight_probs,
    #         m.bias_bank.unsqueeze(0),
    #         bias_probs,
    #     )
    #
    #     self.assertEqual(weight_mixture.shape, torch.Size([batch_size, input, output]))
    #     self.assertEqual(bias_mixture.shape, torch.Size([batch_size, output]))
    #
    #     weight_probs_test = (weight_probs > 0).int().transpose(1, 0).unsqueeze(-1)
    #     bias_probs_test = (bias_probs > 0).int().transpose(1, 0)
    #
    #     weighted_weights = m.weight_bank.unsqueeze(0) * weight_probs_test
    #     weighted_biases = m.bias_bank.unsqueeze(0) * bias_probs_test
    #
    #     expected_weight_mixture = weighted_weights.sum(dim=-2)
    #     expected_bias_mixture = weighted_biases.sum(dim=-1)
    #
    #     self.assertTrue(torch.allclose(weight_mixture, expected_weight_mixture))
    #     self.assertTrue(torch.allclose(bias_mixture, expected_bias_mixture))
    #
    #     weight_mixture, bias_mixture = m._compute_parameter_mixture(
    #         m.weight_bank, weight_probs, None, None
    #     )
    #     self.assertEqual(weight_mixture.shape, torch.Size([batch_size, input, output]))
    #     self.assertIsNone(bias_mixture)


class TestMatrixChoiceMixture(unittest.TestCase):
    def setUp(self):
        self.cfg = MixtureConfig(
            input_dim=4,
            output_dim=5,
            depth_dim=6,
            top_k=2,
            router_output_dim=6,
            cross_diagonal_flag=False,
            weighted_parameters_flag=False,
            bias_parameters_flag=False,
        )

    def test__class_initialization_with_custom_config(self):
        c = copy.deepcopy(self.cfg)
        m = MatrixChoiceMixture(c)

        self.assertIsInstance(m, MatrixChoiceMixture)
        self.assertEqual(m.input_dim, c.input_dim)
        self.assertEqual(m.output_dim, c.output_dim)
        self.assertEqual(m.depth_dim, c.depth_dim)
        self.assertEqual(m.top_k, c.top_k)
        self.assertEqual(m.router_output_dim, c.router_output_dim)
        self.assertEqual(m.cross_diagonal_flag, c.cross_diagonal_flag)
        self.assertEqual(m.weighted_parameters_flag, c.weighted_parameters_flag)
        self.assertEqual(m.bias_parameters_flag, c.bias_parameters_flag)

    def test__class_initialization_with_overwride(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            input_dim=40,
            output_dim=50,
            depth_dim=60,
            top_k=20,
            router_output_dim=60,
            cross_diagonal_flag=True,
            weighted_parameters_flag=True,
            bias_parameters_flag=True,
        )
        m = MatrixChoiceMixture(c, overrides)

        self.assertIsInstance(m, MatrixChoiceMixture)
        self.assertEqual(m.input_dim, overrides.input_dim)
        self.assertEqual(m.output_dim, overrides.output_dim)
        self.assertEqual(m.depth_dim, overrides.depth_dim)
        self.assertEqual(m.top_k, overrides.top_k)
        self.assertEqual(m.router_output_dim, overrides.router_output_dim)
        self.assertEqual(m.cross_diagonal_flag, overrides.cross_diagonal_flag)
        self.assertEqual(m.weighted_parameters_flag, overrides.weighted_parameters_flag)
        self.assertEqual(m.bias_parameters_flag, overrides.bias_parameters_flag)

    def test__init_parameter_banks(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            bias_parameters_flag=True,
        )
        m = MatrixChoiceMixture(c, overrides)
        input_weight_bank, bias_bank = m._MatrixChoiceMixture__init_parameter_banks()

        s = lambda x: torch.Size(x)
        self.assertIsInstance(input_weight_bank, Parameter)
        self.assertEqual(
            input_weight_bank.shape, s([c.depth_dim, c.input_dim, c.output_dim])
        )
        self.assertIsInstance(bias_bank, Parameter)
        self.assertEqual(bias_bank.shape, s([c.depth_dim, c.output_dim]))

    def test__generate_probability_shapes__top_k__1(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=1,
        )
        m = MatrixChoiceMixture(c, overrides)
        weight_probs_shape, bias_probs_shape = (
            m._MatrixChoiceMixture__generate_probability_shapes()
        )

        self.assertEqual(weight_probs_shape, (-1, 1, 1))
        self.assertEqual(bias_probs_shape, (-1, 1))

    def test__generate_probability_shapes__top_k__greater_than_1(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=3,
        )
        m = MatrixChoiceMixture(c, overrides)
        weight_probs_shape, bias_probs_shape = (
            m._MatrixChoiceMixture__generate_probability_shapes()
        )

        self.assertEqual(weight_probs_shape, (-1, 3, 1, 1))
        self.assertEqual(bias_probs_shape, (-1, 3, 1))

    def test__select_parameters__bias_parameters_flag__False(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=2,
        )
        m = MatrixChoiceMixture(c, overrides)

        batch_size = 2
        weight_indexes = torch.randint(0, c.depth_dim, (batch_size, m.top_k))
        bias_indexes = torch.randint(0, c.depth_dim, (batch_size, m.top_k))

        (
            selected_weights,
            selected_biases,
        ) = m._select_parameters(weight_indexes, bias_indexes)

        s = lambda x: torch.Size(x)
        self.assertEqual(
            selected_weights.shape,
            s([batch_size, m.top_k, c.input_dim, c.output_dim]),
        )
        self.assertIsNone(selected_biases)

    def test__select_parameters__bias_parameters_flag__True(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=2,
            bias_parameters_flag=True,
        )
        m = MatrixChoiceMixture(c, overrides)

        batch_size = 2
        weight_indexes = torch.randint(0, c.depth_dim, (batch_size, m.top_k))
        bias_indexes = torch.randint(0, c.depth_dim, (batch_size, m.top_k))

        selected_weights, selected_biases = m._select_parameters(
            weight_indexes, bias_indexes
        )

        s = lambda x: torch.Size(x)
        self.assertEqual(
            selected_weights.shape,
            s([batch_size, m.top_k, c.input_dim, c.output_dim]),
        )
        self.assertEqual(
            selected_biases.shape,
            s([batch_size, m.top_k, c.output_dim]),
        )

    def test__select_parameters__top_k__1(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=1,
            bias_parameters_flag=True,
        )
        m = MatrixChoiceMixture(c, overrides)

        batch_size = 2
        weight_indexes = torch.randint(0, c.depth_dim, (batch_size,))
        bias_indexes = torch.randint(0, c.depth_dim, (batch_size,))

        selected_weights, selected_biases = m._select_parameters(
            weight_indexes, bias_indexes
        )

        s = lambda x: torch.Size(x)
        self.assertEqual(
            selected_weights.shape, s([batch_size, c.input_dim, c.output_dim])
        )
        self.assertEqual(selected_biases.shape, s([batch_size, c.output_dim]))

    def test__select_parameters__top_k__k(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=3,
            bias_parameters_flag=True,
        )
        m = MatrixChoiceMixture(c, overrides)

        batch_size = 2
        weight_indexes = torch.randint(0, c.depth_dim, (batch_size, m.top_k))
        bias_indexes = torch.randint(0, c.depth_dim, (batch_size, m.top_k))

        (
            selected_weights,
            selected_biases,
        ) = m._select_parameters(weight_indexes, bias_indexes)

        s = lambda x: torch.Size(x)
        self.assertEqual(
            selected_weights.shape,
            s([batch_size, c.top_k, c.input_dim, c.output_dim]),
        )
        self.assertEqual(selected_biases.shape, s([batch_size, c.top_k, c.output_dim]))

    def test__compute_weighted_parameters__weighted_parameters_flag__False(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=2,
        )
        m = MatrixChoiceMixture(c, overrides)

        batch_size = 2
        selected_weight_shape = (batch_size, c.top_k, c.input_dim, c.output_dim)
        selected_weight_params = torch.arange(prod(selected_weight_shape)).reshape(
            selected_weight_shape
        )
        probability = torch.randint(0, c.depth_dim, (batch_size, m.top_k))

        weighted_parameters = m._MatrixChoiceMixture__compute_weighted_parameters(
            selected_weight_params, probability
        )

        s = lambda x: torch.Size(x)
        self.assertEqual(
            weighted_parameters.shape,
            s([batch_size, c.top_k, c.input_dim, c.output_dim]),
        )
        self.assertTrue(torch.equal(weighted_parameters, selected_weight_params))

    def test__compute_weighted_parameters__weighted_parameters_flag__True(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=2,
            weighted_parameters_flag=True,
        )
        m = MatrixChoiceMixture(c, overrides)

        batch_size = 2
        selected_weight_shape = (batch_size, c.top_k, c.input_dim, c.output_dim)
        selected_weight_params = torch.arange(prod(selected_weight_shape)).reshape(
            selected_weight_shape
        )
        probability = torch.randint(0, c.depth_dim, (batch_size, m.top_k))

        weighted_parameters = m._MatrixChoiceMixture__compute_weighted_parameters(
            selected_weight_params, probability
        )

        s = lambda x: torch.Size(x)
        self.assertEqual(
            weighted_parameters.shape,
            s([batch_size, c.top_k, c.input_dim, c.output_dim]),
        )

    def test__compute_weighted_parameters__top_k__1(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=1,
            weighted_parameters_flag=True,
        )
        m = MatrixChoiceMixture(c, overrides)

        batch_size = 2
        selected_weight_shape = (batch_size, c.input_dim, c.output_dim)
        selected_weight_params = torch.arange(prod(selected_weight_shape)).reshape(
            selected_weight_shape
        )
        probability = F.sigmoid(torch.randn((batch_size, c.top_k)))

        weighted_parameters = m._MatrixChoiceMixture__compute_weighted_parameters(
            selected_weight_params, probability
        )

        s = lambda x: torch.Size(x)
        self.assertEqual(
            weighted_parameters.shape,
            s([batch_size, c.input_dim, c.output_dim]),
        )

    def test__compute_weighted_parameters__weighted_parameters_flag__False__top_k__1(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=1,
            weighted_parameters_flag=False,
        )
        m = MatrixChoiceMixture(c, overrides)

        batch_size = 2
        selected_weight_shape = (batch_size, c.input_dim, c.output_dim)
        selected_weight_params = torch.arange(prod(selected_weight_shape)).reshape(
            selected_weight_shape
        )
        probability = F.sigmoid(torch.randn((batch_size, c.top_k)))

        weighted_parameters = m._MatrixChoiceMixture__compute_weighted_parameters(
            selected_weight_params, probability
        )

        s = lambda x: torch.Size(x)
        self.assertEqual(
            weighted_parameters.shape,
            s([batch_size, c.input_dim, c.output_dim]),
        )
        self.assertTrue(torch.equal(weighted_parameters, selected_weight_params))

    def test__compute_weighted_parameters__top_k__k(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=3,
            weighted_parameters_flag=True,
        )
        m = MatrixChoiceMixture(c, overrides)

        batch_size = 2
        selected_weight_shape = (batch_size, m.top_k, c.input_dim, c.output_dim)
        selected_weight_params = torch.arange(prod(selected_weight_shape)).reshape(
            selected_weight_shape
        )
        probability = F.sigmoid(torch.randn((batch_size, c.top_k)))

        weighted_parameters = m._MatrixChoiceMixture__compute_weighted_parameters(
            selected_weight_params, probability
        )

        s = lambda x: torch.Size(x)
        self.assertEqual(
            weighted_parameters.shape,
            s([batch_size, m.top_k, c.input_dim, c.output_dim]),
        )

    def test__compute_weighted_parameters__weighted_parameters_flag__False__top_k__k(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=3,
            weighted_parameters_flag=False,
        )
        m = MatrixChoiceMixture(c, overrides)

        batch_size = 2
        selected_weight_shape = (batch_size, m.top_k, c.input_dim, c.output_dim)
        selected_weight_params = torch.arange(prod(selected_weight_shape)).reshape(
            selected_weight_shape
        )
        probability = F.sigmoid(torch.randn((batch_size, c.top_k)))

        weighted_parameters = m._MatrixChoiceMixture__compute_weighted_parameters(
            selected_weight_params, probability
        )

        s = lambda x: torch.Size(x)
        self.assertEqual(
            weighted_parameters.shape,
            s([batch_size, m.top_k, c.input_dim, c.output_dim]),
        )
        self.assertTrue(torch.equal(weighted_parameters, selected_weight_params))

    def test__compute_weighted_parameters__full_mixture(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            depth_dim=10,
            top_k=10,
            weighted_parameters_flag=True,
        )
        m = MatrixChoiceMixture(c, overrides)

        batch_size = 2
        selected_weight_shape = (c.depth_dim, c.input_dim, c.output_dim)
        selected_weight_params = torch.arange(prod(selected_weight_shape)).reshape(
            selected_weight_shape
        )
        probability = F.sigmoid(torch.randn((batch_size, c.depth_dim)))

        weighted_parameters = m._MatrixChoiceMixture__compute_weighted_parameters(
            selected_weight_params, probability
        )

        s = lambda x: torch.Size(x)
        self.assertEqual(
            weighted_parameters.shape,
            s([batch_size, m.depth_dim, c.input_dim, c.output_dim]),
        )

    def test__compute_mixture__top_k__1(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=1,
            weighted_parameters_flag=True,
        )
        m = MatrixChoiceMixture(c, overrides)

        batch_size = 2
        selected_weight_shape = (batch_size, c.input_dim, c.output_dim)
        selected_weight_params = torch.arange(prod(selected_weight_shape)).reshape(
            selected_weight_shape
        )
        probability = F.sigmoid(torch.randn((batch_size, c.top_k)))

        weighted_parameters = m._compute_mixture(selected_weight_params, probability)

        s = lambda x: torch.Size(x)
        self.assertEqual(
            weighted_parameters.shape,
            s([batch_size, c.input_dim, c.output_dim]),
        )

    def test__compute_mixture__top_k__k(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=2,
            weighted_parameters_flag=True,
        )
        m = MatrixChoiceMixture(c, overrides)

        batch_size = 2
        selected_weight_shape = (batch_size, m.top_k, c.input_dim, c.output_dim)
        selected_weight_params = torch.arange(prod(selected_weight_shape)).reshape(
            selected_weight_shape
        )
        probability = F.sigmoid(torch.randn((batch_size, c.top_k)))

        weighted_parameters = m._compute_mixture(selected_weight_params, probability)

        s = lambda x: torch.Size(x)
        self.assertEqual(
            weighted_parameters.shape,
            s([batch_size, c.input_dim, c.output_dim]),
        )

    def test__compute_mixture__full_mixture(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=6,
            weighted_parameters_flag=True,
        )
        m = MatrixChoiceMixture(c, overrides)

        batch_size = 2
        selected_weight_shape = (c.depth_dim, c.input_dim, c.output_dim)
        selected_weight_params = torch.arange(prod(selected_weight_shape)).reshape(
            selected_weight_shape
        )
        probability = F.sigmoid(torch.randn((batch_size, c.depth_dim)))

        weighted_parameters = m._compute_mixture(selected_weight_params, probability)

        s = lambda x: torch.Size(x)
        self.assertEqual(
            weighted_parameters.shape,
            s([batch_size, c.input_dim, c.output_dim]),
        )

    def test__compute_mixture__bias_parameters_flag__True__top_k__1(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=1,
            weighted_parameters_flag=True,
            bias_parameters_flag=True,
        )
        m = MatrixChoiceMixture(c, overrides)

        batch_size = 2
        selected_weight_shape = (batch_size, c.output_dim)
        selected_weight_params = torch.arange(prod(selected_weight_shape)).reshape(
            selected_weight_shape
        )
        probability = F.sigmoid(torch.randn((batch_size,)))
        bias_parameters = m._compute_mixture(
            selected_weight_params, probability, is_weight=False
        )

        s = lambda x: torch.Size(x)
        self.assertEqual(bias_parameters.shape, s([batch_size, c.output_dim]))

    def test__compute_mixture__bias_parameters_flag__True__top_k__greater_than_1(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=3,
            weighted_parameters_flag=True,
            bias_parameters_flag=True,
        )
        m = MatrixChoiceMixture(c, overrides)

        batch_size = 2
        selected_weight_shape = (batch_size, c.top_k, c.output_dim)
        selected_weight_params = torch.arange(prod(selected_weight_shape)).reshape(
            selected_weight_shape
        )
        probability = F.sigmoid(torch.randn((batch_size, c.top_k)))
        bias_parameters = m._compute_mixture(
            selected_weight_params, probability, is_weight=False
        )

        s = lambda x: torch.Size(x)
        self.assertEqual(bias_parameters.shape, s([batch_size, c.output_dim]))

    def test__compute_mixture__bias_parameters_flag__True__full_mixture(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            depth_dim=10,
            top_k=10,
            weighted_parameters_flag=True,
            bias_parameters_flag=True,
        )
        m = MatrixChoiceMixture(c, overrides)

        batch_size = 2
        selected_weight_shape = (c.depth_dim, c.output_dim)
        selected_weight_params = torch.arange(prod(selected_weight_shape)).reshape(
            selected_weight_shape
        )
        probability = F.sigmoid(torch.randn((batch_size, c.depth_dim)))
        bias_parameters = m._compute_mixture(
            selected_weight_params, probability, is_weight=False
        )

        s = lambda x: torch.Size(x)
        self.assertEqual(bias_parameters.shape, s([batch_size, c.output_dim]))

    def test_gradient_flow(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=2,
            weighted_parameters_flag=True,
        )
        m = MatrixChoiceMixture(c, overrides)

        batch_size = 2
        selected_weight_shape = (batch_size, m.top_k, c.input_dim, c.output_dim)
        selected_weight_params = torch.randn(*selected_weight_shape, requires_grad=True)
        probability = torch.tensor([[0.3, 0.7]], requires_grad=True)

        result = m._compute_mixture(selected_weight_params, probability)
        loss = result.sum()
        loss.backward()

        self.assertIsNotNone(selected_weight_params.grad)
        self.assertIsNotNone(probability.grad)


class TestGeneratorChoiceMixture(unittest.TestCase):
    def setUp(self):
        self.cfg = MixtureConfig(
            input_dim=4,
            output_dim=5,
            depth_dim=6,
            top_k=2,
            router_output_dim=6,
            cross_diagonal_flag=False,
            weighted_parameters_flag=False,
            bias_parameters_flag=False,
        )

    def test__init_with_default_config(self):
        main_c = ModelConfig()
        c = main_c.mixture_model_config

        overrides = MixtureConfig(
            cross_diagonal_flag=True,
        )

        m = GeneratorChoiceMixture(main_c, overrides)

        self.assertIsInstance(m.input_weight_bank, Parameter)
        self.assertEqual(
            list(m.input_weight_bank.shape),
            [c.depth_dim, c.input_dim, c.input_dim],
        )
        self.assertIsInstance(m.output_weight_bank, Parameter)
        self.assertEqual(
            list(m.output_weight_bank.shape),
            [c.depth_dim, c.input_dim, c.output_dim],
        )
        self.assertIsInstance(m.diagonal_weight_bank, Parameter)
        self.assertEqual(
            list(m.diagonal_weight_bank.shape),
            [c.depth_dim, c.input_dim, m.diagonal_dim],
        )
        self.assertIsInstance(m.anti_diagonal_weight_bank, Parameter)
        self.assertEqual(
            list(m.anti_diagonal_weight_bank.shape),
            [c.depth_dim, c.input_dim, m.diagonal_dim],
        )
        self.assertIsInstance(m.bias_bank, Parameter)
        self.assertEqual(
            list(m.bias_bank.shape), [c.depth_dim, c.input_dim, m.output_dim]
        )

    def test__class_initialization_with_custom_config(self):
        c = copy.deepcopy(self.cfg)
        m = GeneratorChoiceMixture(c)

        self.assertIsInstance(m, GeneratorChoiceMixture)
        self.assertEqual(m.input_dim, c.input_dim)
        self.assertEqual(m.output_dim, c.output_dim)
        self.assertEqual(m.depth_dim, c.depth_dim)
        self.assertEqual(m.top_k, c.top_k)
        self.assertEqual(m.router_output_dim, c.router_output_dim)
        self.assertEqual(m.cross_diagonal_flag, c.cross_diagonal_flag)
        self.assertEqual(m.weighted_parameters_flag, c.weighted_parameters_flag)
        self.assertEqual(m.bias_parameters_flag, c.bias_parameters_flag)

    def test__class_initialization_with_overwride(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            input_dim=40,
            output_dim=50,
            depth_dim=60,
            top_k=20,
            router_output_dim=60,
            cross_diagonal_flag=True,
            weighted_parameters_flag=True,
            bias_parameters_flag=True,
        )
        m = GeneratorChoiceMixture(c, overrides)

        self.assertIsInstance(m, GeneratorChoiceMixture)
        self.assertEqual(m.input_dim, overrides.input_dim)
        self.assertEqual(m.output_dim, overrides.output_dim)
        self.assertEqual(m.depth_dim, overrides.depth_dim)
        self.assertEqual(m.top_k, overrides.top_k)
        self.assertEqual(m.router_output_dim, overrides.router_output_dim)
        self.assertEqual(m.cross_diagonal_flag, overrides.cross_diagonal_flag)
        self.assertEqual(m.weighted_parameters_flag, overrides.weighted_parameters_flag)
        self.assertEqual(m.bias_parameters_flag, overrides.bias_parameters_flag)

    def test__generate_probability_shapes(self):
        c = copy.deepcopy(self.cfg)
        m = GeneratorChoiceMixture(c)
        weight_probs_shape, bias_probs_shape = (
            m._GeneratorChoiceMixture__generate_probability_shapes()
        )

        self.assertEqual(weight_probs_shape, (-1, c.top_k, 1, 1))
        self.assertEqual(bias_probs_shape, (-1, c.top_k, 1))

    def test__init_parameter_banks(self):
        c = copy.deepcopy(self.cfg)
        m = GeneratorChoiceMixture(c)
        (
            input_weight_bank,
            output_weight_bank,
            diagonal_weight_bank,
            anti_diagonal_weight_bank,
            bias_bank,
        ) = m._GeneratorChoiceMixture__init_parameter_banks()

        s = lambda x: torch.Size(x)
        self.assertIsInstance(input_weight_bank, Parameter)
        self.assertEqual(
            input_weight_bank.shape, s([c.depth_dim, c.input_dim, c.input_dim])
        )
        self.assertIsInstance(output_weight_bank, Parameter)
        self.assertEqual(
            output_weight_bank.shape, s([c.depth_dim, c.input_dim, c.output_dim])
        )
        self.assertIsInstance(diagonal_weight_bank, Parameter)
        self.assertEqual(
            diagonal_weight_bank.shape, s([c.depth_dim, c.input_dim, c.input_dim])
        )
        self.assertIsNone(anti_diagonal_weight_bank)
        self.assertIsNone(bias_bank)

    def test__init_parameter_banks__cross_diagonal_flag__true(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            cross_diagonal_flag=True,
        )
        m = GeneratorChoiceMixture(c, overrides)
        (
            input_weight_bank,
            output_weight_bank,
            diagonal_weight_bank,
            anti_diagonal_weight_bank,
            bias_bank,
        ) = m._GeneratorChoiceMixture__init_parameter_banks()

        s = lambda x: torch.Size(x)
        diagonal = min(c.input_dim, c.output_dim)
        self.assertIsInstance(input_weight_bank, Parameter)
        self.assertEqual(
            input_weight_bank.shape, s([c.depth_dim, c.input_dim, c.input_dim])
        )
        self.assertIsInstance(output_weight_bank, Parameter)
        self.assertEqual(
            output_weight_bank.shape, s([c.depth_dim, c.input_dim, c.output_dim])
        )
        self.assertIsInstance(diagonal_weight_bank, Parameter)
        self.assertEqual(
            diagonal_weight_bank.shape, s([c.depth_dim, c.input_dim, diagonal])
        )
        self.assertIsInstance(anti_diagonal_weight_bank, Parameter)
        self.assertEqual(
            anti_diagonal_weight_bank.shape, s([c.depth_dim, c.input_dim, diagonal])
        )
        self.assertIsNone(bias_bank)

    def test__init_parameter_banks__bias_parameters_flag__true(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            bias_parameters_flag=True,
        )

        m = GeneratorChoiceMixture(c, overrides)
        (
            input_weight_bank,
            output_weight_bank,
            diagonal_weight_bank,
            anti_diagonal_weight_bank,
            bias_bank,
        ) = m._GeneratorChoiceMixture__init_parameter_banks()

        s = lambda x: torch.Size(x)
        diagonal = min(c.input_dim, c.output_dim)
        self.assertIsInstance(input_weight_bank, Parameter)
        self.assertEqual(
            input_weight_bank.shape, s([c.depth_dim, c.input_dim, c.input_dim])
        )
        self.assertIsInstance(output_weight_bank, Parameter)
        self.assertEqual(
            output_weight_bank.shape, s([c.depth_dim, c.input_dim, c.output_dim])
        )
        self.assertIsInstance(diagonal_weight_bank, Parameter)
        self.assertEqual(
            diagonal_weight_bank.shape, s([c.depth_dim, c.input_dim, diagonal])
        )
        self.assertIsNone(anti_diagonal_weight_bank)
        self.assertIsInstance(bias_bank, Parameter)
        self.assertEqual(bias_bank.shape, s([c.depth_dim, c.input_dim, c.output_dim]))

    def test__select_parameters_top_1(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=1,
        )
        m = GeneratorChoiceMixture(c, overrides)

        batch_size = 2
        diagonal = min(c.input_dim, c.output_dim)
        weight_indexes = torch.randint(0, c.depth_dim, (batch_size,))
        bias_indexes = torch.randint(0, c.depth_dim, (batch_size,))

        (
            selected_input_params,
            selected_output_params,
            selected_diagonal_params,
            selected_anti_diagonal_params,
            selected_bias_params,
        ) = m._select_parameters(weight_indexes, bias_indexes)

        s = lambda x: torch.Size(x)
        self.assertEqual(
            selected_input_params.shape,
            s([batch_size, c.top_k, c.input_dim, c.input_dim]),
        )
        self.assertEqual(
            selected_output_params.shape,
            s([batch_size, c.top_k, c.input_dim, c.output_dim]),
        )
        self.assertEqual(
            selected_diagonal_params.shape,
            s([batch_size, c.top_k, c.input_dim, diagonal]),
        )
        self.assertIsNone(selected_anti_diagonal_params)
        self.assertIsNone(selected_bias_params)

    def test__select_parameters_top_1__cross_diagonal_flag__true(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=1,
            cross_diagonal_flag=True,
        )
        m = GeneratorChoiceMixture(c, overrides)

        batch_size = 2
        diagonal = min(c.input_dim, c.output_dim)
        weight_indexes = torch.randint(0, c.depth_dim, (batch_size,))
        bias_indexes = torch.randint(0, c.depth_dim, (batch_size,))

        (
            selected_input_params,
            selected_output_params,
            selected_diagonal_params,
            selected_anti_diagonal_params,
            selected_bias_params,
        ) = m._select_parameters(weight_indexes, bias_indexes)

        s = lambda x: torch.Size(x)
        self.assertEqual(
            selected_input_params.shape,
            s([batch_size, c.top_k, c.input_dim, c.input_dim]),
        )
        self.assertEqual(
            selected_output_params.shape,
            s([batch_size, c.top_k, c.input_dim, c.output_dim]),
        )
        self.assertEqual(
            selected_diagonal_params.shape,
            s([batch_size, c.top_k, c.input_dim, diagonal]),
        )
        self.assertEqual(
            selected_anti_diagonal_params.shape,
            s([batch_size, c.top_k, c.input_dim, diagonal]),
        )
        self.assertIsNone(selected_bias_params)

    def test__select_parameters_top_1__bias_parameters_flag__true(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=1,
            bias_parameters_flag=True,
        )
        m = GeneratorChoiceMixture(c, overrides)

        batch_size = 2
        diagonal = min(c.input_dim, c.output_dim)
        weight_indexes = torch.randint(0, c.depth_dim, (batch_size,))
        bias_indexes = torch.randint(0, c.depth_dim, (batch_size,))

        (
            selected_input_params,
            selected_output_params,
            selected_diagonal_params,
            selected_anti_diagonal_params,
            selected_bias_params,
        ) = m._select_parameters(weight_indexes, bias_indexes)

        s = lambda x: torch.Size(x)
        self.assertEqual(
            selected_input_params.shape,
            s([batch_size, c.top_k, c.input_dim, c.input_dim]),
        )
        self.assertEqual(
            selected_output_params.shape,
            s([batch_size, c.top_k, c.input_dim, c.output_dim]),
        )
        self.assertEqual(
            selected_diagonal_params.shape,
            s([batch_size, c.top_k, c.input_dim, diagonal]),
        )
        self.assertIsNone(selected_anti_diagonal_params)
        self.assertEqual(
            selected_bias_params.shape,
            s([batch_size, c.top_k, c.input_dim, c.output_dim]),
        )

    def test__select_parameters_top_k(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=5,
        )
        m = GeneratorChoiceMixture(c, overrides)

        batch_size = 2
        diagonal = min(c.input_dim, c.output_dim)
        weight_indexes = torch.randint(0, c.depth_dim, (batch_size, c.top_k))
        bias_indexes = torch.randint(0, c.depth_dim, (batch_size, c.top_k))

        (
            selected_input_params,
            selected_output_params,
            selected_diagonal_params,
            selected_anti_diagonal_params,
            selected_bias_params,
        ) = m._select_parameters(weight_indexes, bias_indexes)

        s = lambda x: torch.Size(x)
        self.assertEqual(
            selected_input_params.shape,
            s([batch_size, c.top_k, c.input_dim, c.input_dim]),
        )
        self.assertEqual(
            selected_output_params.shape,
            s([batch_size, c.top_k, c.input_dim, c.output_dim]),
        )
        self.assertEqual(
            selected_diagonal_params.shape,
            s([batch_size, c.top_k, c.input_dim, diagonal]),
        )
        self.assertIsNone(selected_anti_diagonal_params)
        self.assertIsNone(selected_bias_params)

    def test__select_parameters_top_k__cross_diagonal_flag__true(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=5,
            cross_diagonal_flag=True,
        )
        m = GeneratorChoiceMixture(c, overrides)

        batch_size = 2
        diagonal = min(c.input_dim, c.output_dim)
        weight_indexes = torch.randint(0, c.depth_dim, (batch_size, c.top_k))
        bias_indexes = torch.randint(0, c.depth_dim, (batch_size, c.top_k))

        (
            selected_input_params,
            selected_output_params,
            selected_diagonal_params,
            selected_anti_diagonal_params,
            selected_bias_params,
        ) = m._select_parameters(weight_indexes, bias_indexes)

        s = lambda x: torch.Size(x)
        self.assertEqual(
            selected_input_params.shape,
            s([batch_size, c.top_k, c.input_dim, c.input_dim]),
        )
        self.assertEqual(
            selected_output_params.shape,
            s([batch_size, c.top_k, c.input_dim, c.output_dim]),
        )
        self.assertEqual(
            selected_diagonal_params.shape,
            s([batch_size, c.top_k, c.input_dim, diagonal]),
        )
        self.assertEqual(
            selected_anti_diagonal_params.shape,
            s([batch_size, c.top_k, c.input_dim, diagonal]),
        )
        self.assertIsNone(selected_bias_params)

    def test__select_parameters_top_k__bias_parameters_flag__true(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=5,
            bias_parameters_flag=True,
        )
        m = GeneratorChoiceMixture(c, overrides)

        batch_size = 2
        diagonal = min(c.input_dim, c.output_dim)
        weight_indexes = torch.randint(0, c.depth_dim, (batch_size, c.top_k))
        bias_indexes = torch.randint(0, c.depth_dim, (batch_size, c.top_k))

        (
            selected_input_params,
            selected_output_params,
            selected_diagonal_params,
            selected_anti_diagonal_params,
            selected_bias_params,
        ) = m._select_parameters(weight_indexes, bias_indexes)

        s = lambda x: torch.Size(x)
        self.assertEqual(
            selected_input_params.shape,
            s([batch_size, c.top_k, c.input_dim, c.input_dim]),
        )
        self.assertEqual(
            selected_output_params.shape,
            s([batch_size, c.top_k, c.input_dim, c.output_dim]),
        )
        self.assertEqual(
            selected_diagonal_params.shape,
            s([batch_size, c.top_k, c.input_dim, diagonal]),
        )
        self.assertIsNone(selected_anti_diagonal_params)
        self.assertEqual(
            selected_bias_params.shape,
            s([batch_size, c.top_k, c.input_dim, c.output_dim]),
        )

    def test__compute_einsum_top_1(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=1,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2

        input_shape = (batch_size, c.input_dim)
        input_batch = torch.arange(prod(input_shape)).reshape(input_shape)

        selected_weight_shape = (batch_size, c.top_k, c.input_dim, c.input_dim)
        selected_weight_params = torch.arange(prod(selected_weight_shape)).reshape(
            selected_weight_shape
        )

        output = m._GeneratorChoiceMixture__compute_einsum(
            input_batch, selected_weight_params
        )

        self.assertEqual(output.shape, torch.Size([batch_size, c.top_k, c.input_dim]))
        for batch in range(batch_size):
            input_sample = input_batch[batch]
            selected_parameters = selected_weight_params[batch][0]
            expected_result = torch.matmul(input_sample, selected_parameters)
            actual_output = output[batch][0]
            # print()
            # print(f"Input sample {batch} : \n", input_sample)
            # print(f"Selected weights: \n", selected_parameters)
            # print("Expected result: \n", expected_result)
            # print("Actual result: \n", actual_output)
            # print()
            self.assertTrue(torch.equal(actual_output, expected_result))

    def test__compute_einsum_top_k(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=2,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2

        input_shape = (batch_size, c.input_dim)
        input_batch = torch.arange(prod(input_shape)).reshape(input_shape)

        selected_weight_shape = (batch_size, c.top_k, c.input_dim, c.input_dim)
        selected_weight_params = torch.arange(prod(selected_weight_shape)).reshape(
            selected_weight_shape
        )

        output = m._GeneratorChoiceMixture__compute_einsum(
            input_batch, selected_weight_params
        )

        self.assertEqual(output.shape, torch.Size([batch_size, c.top_k, c.input_dim]))
        for batch in range(batch_size):
            for vector in range(c.top_k):
                input_sample = input_batch[batch]
                selected_parameters = selected_weight_params[batch][vector]
                expected_result = torch.matmul(input_sample, selected_parameters)
                actual_output = output[batch][vector]
                # print()
                # print(f"Input sample {batch} : \n", input_sample)
                # print(
                #     f"Selected weights, sample: {batch}, matrix {vector}: \n",
                #     selected_parameters,
                # )
                # print("Expected result: \n", expected_result)
                # print("Actual result: \n", actual_output)
                # print()
                self.assertTrue(torch.equal(actual_output, expected_result))

    def test__compute_einsum_full(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            depth_dim=6,
            top_k=6,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2

        input_shape = (batch_size, c.input_dim)
        input_batch = torch.arange(prod(input_shape)).reshape(input_shape)

        full_weight_bank_shape = (c.depth_dim, c.input_dim, c.input_dim)
        full_weight_bank = torch.arange(prod(full_weight_bank_shape)).reshape(
            full_weight_bank_shape
        )

        output = m._GeneratorChoiceMixture__compute_einsum(
            input_batch, full_weight_bank
        )

        self.assertEqual(
            output.shape, torch.Size([batch_size, c.depth_dim, c.input_dim])
        )
        for batch in range(batch_size):
            for vector in range(c.depth_dim):
                input_sample = input_batch[batch]
                bank_parameters = full_weight_bank[vector]
                expected_result = torch.matmul(input_sample, bank_parameters)
                actual_output = output[batch][vector]
                # print()
                # print(f"Input sample {batch} : \n", input_sample)
                # print(f"Weight bank {vector}: \n", bank_parameters)
                # print("Expected result: \n", expected_result)
                # print("Actual result: \n", actual_output)
                # print()
                self.assertTrue(torch.equal(actual_output, expected_result))

    def test__compute_parameter_vectors(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=2,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2

        input_batch_shape = (batch_size, c.input_dim)
        input_batch = torch.arange(prod(input_batch_shape)).reshape(input_batch_shape)

        input_params_shape = (batch_size, c.top_k, c.input_dim, c.input_dim)
        selected_input_weight_params = torch.arange(prod(input_params_shape)).reshape(
            input_params_shape
        )

        output_params_shape = (batch_size, c.top_k, c.input_dim, c.output_dim)
        selected_output_weight_params = torch.arange(prod(output_params_shape)).reshape(
            output_params_shape
        )

        diagonal = min(c.input_dim, c.output_dim)
        diagonal_params_shape = (batch_size, c.top_k, c.input_dim, diagonal)
        selected_diagonal_weight_params = torch.arange(
            prod(diagonal_params_shape)
        ).reshape(diagonal_params_shape)

        (
            input_vectors,
            output_vectors,
            diagonal_vectors,
            anti_diagonal_vectors,
            bias_output,
        ) = m._GeneratorChoiceMixture__compute_parameter_vectors(
            input_batch,
            selected_input_weight_params,
            selected_output_weight_params,
            selected_diagonal_weight_params,
        )

        s = lambda x: torch.Size(x)
        expected_input_vectors_shape = s([batch_size, c.top_k, c.input_dim])
        self.assertEqual(input_vectors.shape, expected_input_vectors_shape)

        expected_output_vector_shape = s([batch_size, c.top_k, c.output_dim])
        self.assertEqual(output_vectors.shape, expected_output_vector_shape)

        expected_diagonal_vector_shape = s([batch_size, c.top_k, diagonal])
        self.assertEqual(diagonal_vectors.shape, expected_diagonal_vector_shape)

        self.assertIsNone(anti_diagonal_vectors)
        self.assertIsNone(bias_output)

    def test__compute_parameter_vectors__cross_diagonal_flag__true(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=2,
            cross_diagonal_flag=True,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2

        input_batch_shape = (batch_size, c.input_dim)
        input_batch = torch.arange(prod(input_batch_shape)).reshape(input_batch_shape)

        input_params_shape = (batch_size, c.top_k, c.input_dim, c.input_dim)
        selected_input_weight_params = torch.arange(prod(input_params_shape)).reshape(
            input_params_shape
        )

        output_params_shape = (batch_size, c.top_k, c.input_dim, c.output_dim)
        selected_output_weight_params = torch.arange(prod(output_params_shape)).reshape(
            output_params_shape
        )

        diagonal = min(c.input_dim, c.output_dim)
        diagonal_params_shape = (batch_size, c.top_k, c.input_dim, diagonal)
        selected_diagonal_weight_params = torch.arange(
            prod(diagonal_params_shape)
        ).reshape(diagonal_params_shape)

        anti_diagonal_params_shape = (batch_size, c.top_k, c.input_dim, diagonal)
        selected_anti_diagonal_weight_params = torch.arange(
            prod(anti_diagonal_params_shape)
        ).reshape(anti_diagonal_params_shape)

        (
            input_vectors,
            output_vectors,
            diagonal_vectors,
            anti_diagonal_vectors,
            bias_output,
        ) = m._GeneratorChoiceMixture__compute_parameter_vectors(
            input_batch,
            selected_input_weight_params,
            selected_output_weight_params,
            selected_diagonal_weight_params,
            selected_anti_diagonal_weight_params,
        )

        s = lambda x: torch.Size(x)
        expected_input_vectors_shape = s([batch_size, c.top_k, c.input_dim])
        self.assertEqual(input_vectors.shape, expected_input_vectors_shape)

        expected_output_vector_shape = s([batch_size, c.top_k, c.output_dim])
        self.assertEqual(output_vectors.shape, expected_output_vector_shape)

        expected_diagonal_vector_shape = s([batch_size, c.top_k, diagonal])
        self.assertEqual(diagonal_vectors.shape, expected_diagonal_vector_shape)

        expected_anti_diagonal_vector_shape = s([batch_size, c.top_k, diagonal])
        self.assertEqual(
            anti_diagonal_vectors.shape, expected_anti_diagonal_vector_shape
        )
        self.assertIsNone(bias_output)

    def test__compute_parameter_vectors__bias_parameters_flag__true(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=2,
            bias_parameters_flag=True,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2

        input_batch_shape = (batch_size, c.input_dim)
        input_batch = torch.arange(prod(input_batch_shape)).reshape(input_batch_shape)

        input_params_shape = (batch_size, c.top_k, c.input_dim, c.input_dim)
        selected_input_weight_params = torch.arange(prod(input_params_shape)).reshape(
            input_params_shape
        )

        output_params_shape = (batch_size, c.top_k, c.input_dim, c.output_dim)
        selected_output_weight_params = torch.arange(prod(output_params_shape)).reshape(
            output_params_shape
        )

        diagonal = min(c.input_dim, c.output_dim)
        diagonal_params_shape = (batch_size, c.top_k, c.input_dim, diagonal)
        selected_diagonal_weight_params = torch.arange(
            prod(diagonal_params_shape)
        ).reshape(diagonal_params_shape)

        bias_params_shape = (batch_size, c.top_k, c.input_dim, c.output_dim)
        selected_bias_params = torch.arange(prod(bias_params_shape)).reshape(
            bias_params_shape
        )

        (
            input_vectors,
            output_vectors,
            diagonal_vectors,
            anti_diagonal_vectors,
            bias_output,
        ) = m._GeneratorChoiceMixture__compute_parameter_vectors(
            input_batch,
            selected_input_weight_params,
            selected_output_weight_params,
            selected_diagonal_weight_params,
            bias_params=selected_bias_params,
        )

        s = lambda x: torch.Size(x)
        expected_input_vectors_shape = s([batch_size, c.top_k, c.input_dim])
        self.assertEqual(input_vectors.shape, expected_input_vectors_shape)

        expected_output_vector_shape = s([batch_size, c.top_k, c.output_dim])
        self.assertEqual(output_vectors.shape, expected_output_vector_shape)

        expected_diagonal_vector_shape = s([batch_size, c.top_k, diagonal])
        self.assertEqual(diagonal_vectors.shape, expected_diagonal_vector_shape)

        self.assertIsNone(anti_diagonal_vectors)

        expected_bias_vector_shape = s([batch_size, c.top_k, c.output_dim])
        self.assertEqual(bias_output.shape, expected_bias_vector_shape)

    def test__maybe_compute_einsum(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=2,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2
        input_batch_shape = (batch_size, c.input_dim)
        input_batch = torch.arange(prod(input_batch_shape)).reshape(input_batch_shape)

        weight_tensor_shape = (batch_size, c.top_k, c.input_dim, c.input_dim)
        weight_tensor = torch.arange(prod(weight_tensor_shape)).reshape(
            weight_tensor_shape
        )

        output = m._GeneratorChoiceMixture__maybe_compute_einsum(
            input_batch, weight_tensor
        )

        self.assertIsNone(output)

    def test__maybe_compute_einsum__einsum_flag__true(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=2,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2

        input_batch_shape = (batch_size, c.input_dim)
        input_batch = torch.arange(prod(input_batch_shape)).reshape(input_batch_shape)

        weight_tensor_shape = (batch_size, c.top_k, c.input_dim, c.input_dim)
        weight_tensor = torch.arange(prod(weight_tensor_shape)).reshape(
            weight_tensor_shape
        )

        output = m._GeneratorChoiceMixture__maybe_compute_einsum(
            input_batch, weight_tensor, True
        )

        expected_output = m._GeneratorChoiceMixture__compute_einsum(
            input_batch, weight_tensor
        )
        self.assertTrue(torch.equal(output, expected_output))

    def test__compute_outer_product__output_bigger(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            input_dim=4,
            output_dim=5,
            top_k=2,
        )
        m = GeneratorChoiceMixture(c, overrides)

        batch_size = 2
        diagonal_dim = min(c.input_dim, c.output_dim)

        input_vectors_shape = (batch_size, c.top_k, c.input_dim)
        input_vectors = torch.arange(prod(input_vectors_shape)).reshape(
            input_vectors_shape
        )
        output_vectors_shape = (batch_size, c.top_k, c.output_dim)
        output_vectors = torch.arange(prod(output_vectors_shape)).reshape(
            output_vectors_shape
        )

        output = m._GeneratorChoiceMixture__compute_outer_product(
            input_vectors, output_vectors
        )

        scaled_input_vectors = input_vectors * diagonal_dim**-0.5

        self.assertEqual(
            output.shape, torch.Size([batch_size, c.top_k, c.input_dim, c.output_dim])
        )
        for batch in range(batch_size):
            for k in range(c.top_k):
                input_vector = scaled_input_vectors[batch][k]
                output_vector = output_vectors[batch][k]
                expected_output = torch.outer(input_vector, output_vector)
                actual_output = output[batch][k]
                # print()
                # print(f"Input vector {batch}, {k}: \n", input_vector)
                # print(f"Output vector {batch}, {k}: \n", output_vector)
                # print("Expected result: \n", expected_output)
                # print("Actual result: \n", actual_output)
                # print()
                self.assertTrue(torch.equal(actual_output, expected_output))

    def test__compute_outer_product__input_bigger(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            input_dim=6,
            output_dim=5,
            top_k=2,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2
        diagonal_dim = min(c.input_dim, c.output_dim)

        input_vectors_shape = (batch_size, c.top_k, c.input_dim)
        input_vectors = torch.arange(prod(input_vectors_shape)).reshape(
            input_vectors_shape
        )
        output_vectors_shape = (batch_size, c.top_k, c.output_dim)
        output_vectors = torch.arange(prod(output_vectors_shape)).reshape(
            output_vectors_shape
        )

        output = m._GeneratorChoiceMixture__compute_outer_product(
            input_vectors, output_vectors
        )

        scaled_input_vectors = input_vectors * diagonal_dim**-0.5

        self.assertEqual(
            output.shape, torch.Size([batch_size, c.top_k, c.input_dim, c.output_dim])
        )
        for batch in range(batch_size):
            for k in range(c.top_k):
                input_vector = scaled_input_vectors[batch][k]
                output_vector = output_vectors[batch][k]
                expected_output = torch.outer(input_vector, output_vector)
                actual_output = output[batch][k]
                # print()
                # print(f"Input vector {batch}, {k}: \n", input_vector)
                # print(f"Output vector {batch}, {k}: \n", output_vector)
                # print("Expected result: \n", expected_output)
                # print("Actual result: \n", actual_output)
                # print()
                self.assertTrue(torch.equal(actual_output, expected_output))

    def test__compute_outer_product__top_k__1(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            input_dim=6,
            output_dim=5,
            top_k=1,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2
        diagonal_dim = min(c.input_dim, c.output_dim)

        input_vectors_shape = (batch_size, c.top_k, c.input_dim)
        input_vectors = torch.arange(prod(input_vectors_shape)).reshape(
            input_vectors_shape
        )
        output_vectors_shape = (batch_size, c.top_k, c.output_dim)
        output_vectors = torch.arange(prod(output_vectors_shape)).reshape(
            output_vectors_shape
        )

        output = m._GeneratorChoiceMixture__compute_outer_product(
            input_vectors, output_vectors
        )

        scaled_input_vectors = input_vectors * diagonal_dim**-0.5

        self.assertEqual(
            output.shape, torch.Size([batch_size, c.top_k, c.input_dim, c.output_dim])
        )
        for batch in range(batch_size):
            for k in range(c.top_k):
                input_vector = scaled_input_vectors[batch][k]
                output_vector = output_vectors[batch][k]
                expected_output = torch.outer(input_vector, output_vector)
                actual_output = output[batch][k]
                # print()
                # print(f"Input vector {batch}, {k}: \n", input_vector)
                # print(f"Output vector {batch}, {k}: \n", output_vector)
                # print("Expected result: \n", expected_output)
                # print("Actual result: \n", actual_output)
                # print()
                self.assertTrue(torch.equal(actual_output, expected_output))

    def test__compute_outer_product__top_k__k(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            input_dim=6,
            output_dim=5,
            top_k=2,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2
        diagonal_dim = min(c.input_dim, c.output_dim)

        input_vectors_shape = (batch_size, c.top_k, c.input_dim)
        input_vectors = torch.arange(prod(input_vectors_shape)).reshape(
            input_vectors_shape
        )
        output_vectors_shape = (batch_size, c.top_k, c.output_dim)
        output_vectors = torch.arange(prod(output_vectors_shape)).reshape(
            output_vectors_shape
        )

        output = m._GeneratorChoiceMixture__compute_outer_product(
            input_vectors, output_vectors
        )

        scaled_input_vectors = input_vectors * diagonal_dim**-0.5

        self.assertEqual(
            output.shape, torch.Size([batch_size, c.top_k, c.input_dim, c.output_dim])
        )
        for batch in range(batch_size):
            for k in range(c.top_k):
                input_vector = scaled_input_vectors[batch][k]
                output_vector = output_vectors[batch][k]
                expected_output = torch.outer(input_vector, output_vector)
                actual_output = output[batch][k]
                # print()
                # print(f"Input vector {batch}, {k}: \n", input_vector)
                # print(f"Output vector {batch}, {k}: \n", output_vector)
                # print("Expected result: \n", expected_output)
                # print("Actual result: \n", actual_output)
                # print()
                self.assertTrue(torch.equal(actual_output, expected_output))

    def test__compute_outer_product__full_mixture(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            input_dim=6,
            output_dim=5,
            top_k=6,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2
        diagonal_dim = min(c.input_dim, c.output_dim)

        input_vectors_shape = (batch_size, c.top_k, c.input_dim)
        input_vectors = torch.arange(prod(input_vectors_shape)).reshape(
            input_vectors_shape
        )
        output_vectors_shape = (batch_size, c.top_k, c.output_dim)
        output_vectors = torch.arange(prod(output_vectors_shape)).reshape(
            output_vectors_shape
        )

        output = m._GeneratorChoiceMixture__compute_outer_product(
            input_vectors, output_vectors
        )

        scaled_input_vectors = input_vectors * diagonal_dim**-0.5

        self.assertEqual(
            output.shape, torch.Size([batch_size, c.top_k, c.input_dim, c.output_dim])
        )
        for batch in range(batch_size):
            for k in range(c.top_k):
                input_vector = scaled_input_vectors[batch][k]
                output_vector = output_vectors[batch][k]
                expected_output = torch.outer(input_vector, output_vector)
                actual_output = output[batch][k]
                # print()
                # print(f"Input vector {batch}, {k}: \n", input_vector)
                # print(f"Output vector {batch}, {k}: \n", output_vector)
                # print("Expected result: \n", expected_output)
                # print("Actual result: \n", actual_output)
                # print()
                self.assertTrue(torch.equal(actual_output, expected_output))

    def test__compute_diagonal__bigger_input(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            input_dim=6,
            output_dim=5,
            top_k=2,
        )
        m = GeneratorChoiceMixture(c, overrides)

        batch_size = 2
        diagonal_dim = min(c.input_dim, c.output_dim)

        diagonal_vectors_shape = (batch_size, c.top_k, diagonal_dim)
        diagonal_vectors = torch.arange(prod(diagonal_vectors_shape)).reshape(
            diagonal_vectors_shape
        )

        output = m._GeneratorChoiceMixture__compute_diagonal_matrix(diagonal_vectors)

        self.assertEqual(
            output.shape,
            torch.Size([batch_size, c.top_k, c.input_dim, c.output_dim]),
        )

        for batch in range(batch_size):
            for k in range(c.top_k):
                diagonal_vector = diagonal_vectors[batch][k]
                expected_output = torch.diag_embed(diagonal_vector)
                expected_output = F.pad(expected_output, m.diagonal_padding_shape)
                actual_output = output[batch][k]
                # print()
                # print(f"Diagonal vector {batch}, {k}: \n", diagonal_vector)
                # print("Expected result: \n", expected_output)
                # print("Actual result: \n", actual_output)
                # print()
                self.assertTrue(torch.equal(actual_output, expected_output))

    def test__compute_diagonal__bigger_output(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            input_dim=4,
            output_dim=5,
            top_k=2,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2
        diagonal_dim = min(c.input_dim, c.output_dim)

        diagonal_vectors_shape = (batch_size, c.top_k, diagonal_dim)
        diagonal_vectors = torch.arange(prod(diagonal_vectors_shape)).reshape(
            diagonal_vectors_shape
        )

        output = m._GeneratorChoiceMixture__compute_diagonal_matrix(diagonal_vectors)

        self.assertEqual(
            output.shape,
            torch.Size([batch_size, c.top_k, c.input_dim, c.output_dim]),
        )

        for batch in range(batch_size):
            for k in range(c.top_k):
                diagonal_vector = diagonal_vectors[batch][k]
                expected_output = torch.diag_embed(diagonal_vector)
                expected_output = F.pad(expected_output, m.diagonal_padding_shape)
                actual_output = output[batch][k]
                # print()
                # print(f"Diagonal vector {batch}, {k}: \n", diagonal_vector)
                # print("Expected result: \n", expected_output)
                # print("Actual result: \n", actual_output)
                # print()
                self.assertTrue(torch.equal(actual_output, expected_output))

    def test__compute_diagonal__equal_input_output_dims(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            input_dim=4,
            output_dim=4,
            top_k=2,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2
        diagonal_dim = min(c.input_dim, c.output_dim)

        diagonal_vectors_shape = (batch_size, c.top_k, diagonal_dim)
        diagonal_vectors = torch.arange(prod(diagonal_vectors_shape)).reshape(
            diagonal_vectors_shape
        )

        output = m._GeneratorChoiceMixture__compute_diagonal_matrix(diagonal_vectors)

        self.assertEqual(
            output.shape, torch.Size([batch_size, c.top_k, c.input_dim, diagonal_dim])
        )

        for batch in range(batch_size):
            for k in range(c.top_k):
                diagonal_vector = diagonal_vectors[batch][k]
                expected_output = torch.diag_embed(diagonal_vector)
                actual_output = output[batch][k]
                # print()
                # print(f"Diagonal vector {batch}, {k}: \n", diagonal_vector)
                # print("Expected result: \n", expected_output)
                # print("Actual result: \n", actual_output)
                # print()
                self.assertTrue(torch.equal(actual_output, expected_output))

    def test__anti_compute_diagonal(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=2,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2
        diagonal_dim = min(c.input_dim, c.output_dim)

        anti_diagonal_vectors_shape = (batch_size, c.top_k, diagonal_dim)
        anti_diagonal_vectors = torch.arange(prod(anti_diagonal_vectors_shape)).reshape(
            anti_diagonal_vectors_shape
        )

        output = m._GeneratorChoiceMixture__compute_anti_diagonal_matrix(
            anti_diagonal_vectors
        )

        self.assertIsNone(output)

    def test__anti_compute_diagonal__cross_diagonal_flag__true(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=2,
            cross_diagonal_flag=True,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2
        diagonal_dim = min(c.input_dim, c.output_dim)

        anti_diagonal_vectors_shape = (batch_size, c.top_k, diagonal_dim)
        anti_diagonal_vectors = torch.arange(prod(anti_diagonal_vectors_shape)).reshape(
            anti_diagonal_vectors_shape
        )

        output = m._GeneratorChoiceMixture__compute_anti_diagonal_matrix(
            anti_diagonal_vectors
        )

        self.assertEqual(
            output.shape, torch.Size([batch_size, c.top_k, c.input_dim, c.output_dim])
        )

        for batch in range(batch_size):
            for k in range(c.top_k):
                anti_diagonal_vector = anti_diagonal_vectors[batch][k]
                expected_output = m._GeneratorChoiceMixture__compute_diagonal_matrix(
                    anti_diagonal_vector
                )
                expected_output = expected_output.flip(dims=[0])
                actual_output = output[batch][k]
                # print()
                # print(f"Diagonal vector {batch}, {k}: \n", anti_diagonal_vector)
                # print("Expected result: \n", expected_output)
                # print("Actual result: \n", actual_output)
                # print()
                self.assertTrue(torch.equal(actual_output, expected_output))

    def test__assemble_parameters_matrix(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=2,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2
        diagonal_dim = min(c.input_dim, c.output_dim) + abs(c.input_dim - c.output_dim)

        outer_product_shape = (batch_size, c.top_k, c.input_dim, c.output_dim)
        outer_product = torch.arange(prod(outer_product_shape)).reshape(
            outer_product_shape
        )

        diagonal_matrix_shape = (batch_size, c.top_k, c.input_dim, diagonal_dim)
        diagonal_matrix = torch.arange(prod(diagonal_matrix_shape)).reshape(
            diagonal_matrix_shape
        )

        output = m._GeneratorChoiceMixture__assemble_parameters_matrix(
            outer_product,
            diagonal_matrix,
        )

        self.assertEqual(
            output.shape, torch.Size([batch_size, c.top_k, c.input_dim, c.output_dim])
        )
        for batch in range(batch_size):
            for k in range(c.top_k):
                expected_output = outer_product[batch][k] + diagonal_matrix[batch][k]
                actual_output = output[batch][k]
                # print()
                # print("Expected result: \n", expected_output)
                # print("Actual result: \n", actual_output)
                # print()
                self.assertTrue(torch.equal(actual_output, expected_output))

    def test__assemble_parameters_matrix__cross_diagonal_flag__True(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=2,
            cross_diagonal_flag=True,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2
        diagonal_dim = min(c.input_dim, c.output_dim) + abs(c.input_dim - c.output_dim)

        outer_product_shape = (batch_size, c.top_k, c.input_dim, c.output_dim)
        outer_product = torch.arange(prod(outer_product_shape)).reshape(
            outer_product_shape
        )

        diagonal_matrix_shape = (batch_size, c.top_k, c.input_dim, diagonal_dim)
        diagonal_matrix = torch.arange(prod(diagonal_matrix_shape)).reshape(
            diagonal_matrix_shape
        )

        anti_diagonal_matrix_shape = (batch_size, c.top_k, c.input_dim, diagonal_dim)
        anti_diagonal_matrix = torch.arange(prod(anti_diagonal_matrix_shape)).reshape(
            anti_diagonal_matrix_shape
        )

        output = m._GeneratorChoiceMixture__assemble_parameters_matrix(
            outer_product,
            diagonal_matrix,
            anti_diagonal_matrix,
        )

        self.assertEqual(
            output.shape, torch.Size([batch_size, c.top_k, c.input_dim, c.output_dim])
        )
        for batch in range(batch_size):
            for k in range(c.top_k):
                expected_output = outer_product[batch][k] + diagonal_matrix[batch][k]
                expected_output = expected_output + anti_diagonal_matrix[batch][k]
                actual_output = output[batch][k]
                # print()
                # print("Expected result: \n", expected_output)
                # print("Actual result: \n", actual_output)
                # print()
                self.assertTrue(torch.equal(actual_output, expected_output))

    def test__apply_parameter_weighting(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=2,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2

        generated_parameters_shape = (batch_size, c.top_k, c.input_dim, c.output_dim)
        generated_parameters = torch.arange(prod(generated_parameters_shape)).reshape(
            generated_parameters_shape
        )

        output = m._GeneratorChoiceMixture__apply_parameter_weighting(
            generated_parameters, m.weight_probs_shape
        )

        self.assertEqual(
            output.shape, torch.Size([batch_size, c.top_k, c.input_dim, c.output_dim])
        )
        for batch in range(batch_size):
            for k in range(c.top_k):
                expected_output = generated_parameters[batch][k]
                actual_output = output[batch][k]
                # print()
                # print("Expected result: \n", expected_output)
                # print("Actual result: \n", actual_output)
                # print()
                self.assertTrue(torch.equal(actual_output, expected_output))

    def test__apply_parameter_weighting__top_k__1(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=1,
            weighted_parameters_flag=True,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2

        generated_parameters_shape = (batch_size, c.top_k, c.input_dim, c.output_dim)
        generated_parameters = torch.arange(prod(generated_parameters_shape)).reshape(
            generated_parameters_shape,
        )

        weight_probs = F.sigmoid(torch.randn((batch_size,)))

        output = m._GeneratorChoiceMixture__apply_parameter_weighting(
            generated_parameters,
            m.weight_probs_shape,
            weight_probs,
        )

        self.assertEqual(
            output.shape, torch.Size([batch_size, c.top_k, c.input_dim, c.output_dim])
        )
        for batch in range(batch_size):
            for k in range(c.top_k):
                weight_prob = weight_probs[batch]
                expected_output = generated_parameters[batch][k] * weight_prob
                actual_output = output[batch][k]
                # print()
                # print("Expected result: \n", expected_output)
                # print("Actual result: \n", actual_output)
                # print()
                self.assertTrue(torch.equal(actual_output, expected_output))

    def test__apply_parameter_weighting__top_k__k(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=2,
            weighted_parameters_flag=True,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2

        generated_parameters_shape = (batch_size, c.top_k, c.input_dim, c.output_dim)
        generated_parameters = torch.arange(prod(generated_parameters_shape)).reshape(
            generated_parameters_shape,
        )

        weight_probs = F.sigmoid(torch.randn((batch_size, c.top_k)))

        output = m._GeneratorChoiceMixture__apply_parameter_weighting(
            generated_parameters,
            m.weight_probs_shape,
            weight_probs,
        )

        self.assertEqual(
            output.shape, torch.Size([batch_size, c.top_k, c.input_dim, c.output_dim])
        )
        for batch in range(batch_size):
            for k in range(c.top_k):
                weight_prob = weight_probs[batch][k]
                expected_output = generated_parameters[batch][k] * weight_prob
                actual_output = output[batch][k]
                # print()
                # print("Expected result: \n", expected_output)
                # print("Actual result: \n", actual_output)
                # print()
                self.assertTrue(torch.equal(actual_output, expected_output))

    def test__apply_parameter_weighting__full_mixture(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=6,
            weighted_parameters_flag=True,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2

        generated_parameters_shape = (batch_size, c.top_k, c.input_dim, c.output_dim)
        generated_parameters = torch.arange(prod(generated_parameters_shape)).reshape(
            generated_parameters_shape,
        )

        weight_probs = F.sigmoid(torch.randn((batch_size, c.top_k)))

        output = m._GeneratorChoiceMixture__apply_parameter_weighting(
            generated_parameters,
            m.weight_probs_shape,
            weight_probs,
        )

        self.assertEqual(
            output.shape, torch.Size([batch_size, c.top_k, c.input_dim, c.output_dim])
        )
        for batch in range(batch_size):
            for k in range(c.top_k):
                weight_prob = weight_probs[batch][k]
                expected_output = generated_parameters[batch][k] * weight_prob
                actual_output = output[batch][k]
                # print()
                # print("Expected result: \n", expected_output)
                # print("Actual result: \n", actual_output)
                # print()
                self.assertTrue(torch.equal(actual_output, expected_output))

    def test__generate_weight_parameters__top_k__1(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=1,
            cross_diagonal_flag=True,
            weighted_parameters_flag=True,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2

        input_vectors_shape = (batch_size, c.top_k, c.input_dim)
        input_vectors = torch.arange(prod(input_vectors_shape)).reshape(
            input_vectors_shape
        )

        output_vectors_shape = (batch_size, c.top_k, c.output_dim)
        output_vectors = torch.arange(prod(output_vectors_shape)).reshape(
            output_vectors_shape
        )

        diagonal_vectors_shape = (batch_size, c.top_k, m.diagonal_dim)
        diagonal_vectors = torch.arange(prod(diagonal_vectors_shape)).reshape(
            diagonal_vectors_shape
        )

        anti_diagonal_vectors_shape = (batch_size, c.top_k, m.diagonal_dim)
        anti_diagonal_vectors = torch.arange(prod(anti_diagonal_vectors_shape)).reshape(
            anti_diagonal_vectors_shape
        )

        weight_probs = F.sigmoid(torch.randn((batch_size, c.top_k)))

        output = m._GeneratorChoiceMixture__generate_weight_parameters(
            input_vectors,
            output_vectors,
            diagonal_vectors,
            anti_diagonal_vectors,
            weight_probs,
        )

        self.assertEqual(
            output.shape, torch.Size([batch_size, c.input_dim, c.output_dim])
        )

    def test__generate_weight_parameters__top_k__k(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=3,
            cross_diagonal_flag=True,
            weighted_parameters_flag=True,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2

        input_vectors_shape = (batch_size, c.top_k, c.input_dim)
        input_vectors = torch.arange(prod(input_vectors_shape)).reshape(
            input_vectors_shape
        )

        output_vectors_shape = (batch_size, c.top_k, c.output_dim)
        output_vectors = torch.arange(prod(output_vectors_shape)).reshape(
            output_vectors_shape
        )

        diagonal_vectors_shape = (batch_size, c.top_k, m.diagonal_dim)
        diagonal_vectors = torch.arange(prod(diagonal_vectors_shape)).reshape(
            diagonal_vectors_shape
        )

        anti_diagonal_vectors_shape = (batch_size, c.top_k, m.diagonal_dim)
        anti_diagonal_vectors = torch.arange(prod(anti_diagonal_vectors_shape)).reshape(
            anti_diagonal_vectors_shape
        )

        weight_probs = F.sigmoid(torch.randn((batch_size, c.top_k)))

        output = m._GeneratorChoiceMixture__generate_weight_parameters(
            input_vectors,
            output_vectors,
            diagonal_vectors,
            anti_diagonal_vectors,
            weight_probs,
        )

        self.assertEqual(
            output.shape, torch.Size([batch_size, c.input_dim, c.output_dim])
        )

    def test__generate_weight_parameters__full_mixture(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=6,
            cross_diagonal_flag=True,
            weighted_parameters_flag=True,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2

        input_vectors_shape = (batch_size, c.top_k, c.input_dim)
        input_vectors = torch.arange(prod(input_vectors_shape)).reshape(
            input_vectors_shape
        )

        output_vectors_shape = (batch_size, c.top_k, c.output_dim)
        output_vectors = torch.arange(prod(output_vectors_shape)).reshape(
            output_vectors_shape
        )

        diagonal_vectors_shape = (batch_size, c.top_k, m.diagonal_dim)
        diagonal_vectors = torch.arange(prod(diagonal_vectors_shape)).reshape(
            diagonal_vectors_shape
        )

        anti_diagonal_vectors_shape = (batch_size, c.top_k, m.diagonal_dim)
        anti_diagonal_vectors = torch.arange(prod(anti_diagonal_vectors_shape)).reshape(
            anti_diagonal_vectors_shape
        )

        weight_probs = F.sigmoid(torch.randn((batch_size, c.top_k)))

        output = m._GeneratorChoiceMixture__generate_weight_parameters(
            input_vectors,
            output_vectors,
            diagonal_vectors,
            anti_diagonal_vectors,
            weight_probs,
        )

        self.assertEqual(
            output.shape, torch.Size([batch_size, c.input_dim, c.output_dim])
        )

    def test__generate_bias_parameters(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=2,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2

        generated_biases_shape = (batch_size, c.top_k, c.input_dim)
        generated_biases = torch.arange(prod(generated_biases_shape)).reshape(
            generated_biases_shape
        )
        bias_probs = F.sigmoid(torch.randn((batch_size, c.top_k)))

        output = m._GeneratorChoiceMixture__generate_bias_parameters(
            generated_biases,
            bias_probs,
        )

        self.assertIsNone(output)

    def test__generate_bias_parameters__top_k__1(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=1,
            bias_parameters_flag=True,
            weighted_parameters_flag=True,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2

        generated_biases_shape = (batch_size, c.top_k, c.output_dim)
        generated_biases = torch.arange(prod(generated_biases_shape)).reshape(
            generated_biases_shape
        )
        bias_probs = F.sigmoid(torch.randn((batch_size, c.top_k)))

        output = m._GeneratorChoiceMixture__generate_bias_parameters(
            generated_biases,
            bias_probs,
        )

        self.assertEqual(output.shape, torch.Size([batch_size, c.output_dim]))

    def test__generate_bias_parameters__top_k__k(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=4,
            bias_parameters_flag=True,
            weighted_parameters_flag=True,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2

        generated_biases_shape = (batch_size, c.top_k, c.output_dim)
        generated_biases = torch.arange(prod(generated_biases_shape)).reshape(
            generated_biases_shape
        )
        bias_probs = F.sigmoid(torch.randn((batch_size, c.top_k)))

        output = m._GeneratorChoiceMixture__generate_bias_parameters(
            generated_biases,
            bias_probs,
        )

        self.assertEqual(output.shape, torch.Size([batch_size, c.output_dim]))

    def test__generate_bias_parameters__full_mixture(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=6,
            bias_parameters_flag=True,
            weighted_parameters_flag=True,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2

        generated_biases_shape = (batch_size, c.top_k, c.output_dim)
        generated_biases = torch.arange(prod(generated_biases_shape)).reshape(
            generated_biases_shape
        )
        bias_probs = F.sigmoid(torch.randn((batch_size, c.top_k)))

        output = m._GeneratorChoiceMixture__generate_bias_parameters(
            generated_biases,
            bias_probs,
        )

        self.assertEqual(output.shape, torch.Size([batch_size, c.output_dim]))

    def test__compute_parameter_mixture__top_k__1(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=1,
            bias_parameters_flag=True,
            weighted_parameters_flag=True,
            cross_diagonal_flag=True,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2

        input_batch_shape = (batch_size, c.input_dim)
        input_batch = torch.arange(prod(input_batch_shape)).reshape(input_batch_shape)

        input_params_shape = (batch_size, c.top_k, c.input_dim, c.input_dim)
        selected_input_weight_params = torch.arange(prod(input_params_shape)).reshape(
            input_params_shape
        )

        output_params_shape = (batch_size, c.top_k, c.input_dim, c.output_dim)
        selected_output_weight_params = torch.arange(prod(output_params_shape)).reshape(
            output_params_shape
        )

        diagonal = min(c.input_dim, c.output_dim)
        diagonal_params_shape = (batch_size, c.top_k, c.input_dim, diagonal)
        selected_diagonal_weight_params = torch.arange(
            prod(diagonal_params_shape)
        ).reshape(diagonal_params_shape)

        anti_diagonal_params_shape = (batch_size, c.top_k, c.input_dim, diagonal)
        selected_anti_diagonal_weight_params = torch.arange(
            prod(anti_diagonal_params_shape)
        ).reshape(anti_diagonal_params_shape)

        selected_biases_shape = (batch_size, c.top_k, c.input_dim, c.output_dim)
        selected_bias_params = torch.arange(prod(selected_biases_shape)).reshape(
            selected_biases_shape
        )

        weight_probs = F.sigmoid(torch.randn((batch_size,)))
        bias_probs = F.sigmoid(torch.randn((batch_size,)))

        output_weights, output_biases = m._compute_parameter_mixture(
            input_batch,
            selected_input_weight_params,
            selected_output_weight_params,
            selected_diagonal_weight_params,
            selected_anti_diagonal_weight_params,
            selected_bias_params,
            weight_probs,
            bias_probs,
        )

        self.assertEqual(
            output_weights.shape, torch.Size([batch_size, c.input_dim, c.output_dim])
        )
        self.assertEqual(output_biases.shape, torch.Size([batch_size, c.output_dim]))

    def test__compute_parameter_mixture__top_k__k(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=3,
            bias_parameters_flag=True,
            weighted_parameters_flag=True,
            cross_diagonal_flag=True,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2

        input_batch_shape = (batch_size, c.input_dim)
        input_batch = torch.arange(prod(input_batch_shape)).reshape(input_batch_shape)

        input_params_shape = (batch_size, c.top_k, c.input_dim, c.input_dim)
        selected_input_weight_params = torch.arange(prod(input_params_shape)).reshape(
            input_params_shape
        )

        output_params_shape = (batch_size, c.top_k, c.input_dim, c.output_dim)
        selected_output_weight_params = torch.arange(prod(output_params_shape)).reshape(
            output_params_shape
        )

        diagonal = min(c.input_dim, c.output_dim)
        diagonal_params_shape = (batch_size, c.top_k, c.input_dim, diagonal)
        selected_diagonal_weight_params = torch.arange(
            prod(diagonal_params_shape)
        ).reshape(diagonal_params_shape)

        anti_diagonal_params_shape = (batch_size, c.top_k, c.input_dim, diagonal)
        selected_anti_diagonal_weight_params = torch.arange(
            prod(anti_diagonal_params_shape)
        ).reshape(anti_diagonal_params_shape)

        selected_biases_shape = (batch_size, c.top_k, c.input_dim, c.output_dim)
        selected_bias_params = torch.arange(prod(selected_biases_shape)).reshape(
            selected_biases_shape
        )

        weight_probs = F.sigmoid(torch.randn((batch_size, c.top_k)))
        bias_probs = F.sigmoid(torch.randn((batch_size, c.top_k)))

        output_weights, output_biases = m._compute_parameter_mixture(
            input_batch,
            selected_input_weight_params,
            selected_output_weight_params,
            selected_diagonal_weight_params,
            selected_anti_diagonal_weight_params,
            selected_bias_params,
            weight_probs,
            bias_probs,
        )

        self.assertEqual(
            output_weights.shape, torch.Size([batch_size, c.input_dim, c.output_dim])
        )
        self.assertEqual(output_biases.shape, torch.Size([batch_size, c.output_dim]))

    def test__compute_parameter_mixture__full_mixture(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureConfig(
            top_k=6,
            bias_parameters_flag=True,
            weighted_parameters_flag=True,
            cross_diagonal_flag=True,
        )
        m = GeneratorChoiceMixture(c, overrides)
        batch_size = 2

        input_batch_shape = (batch_size, c.input_dim)
        input_batch = torch.arange(prod(input_batch_shape)).reshape(input_batch_shape)

        input_params_shape = (c.depth_dim, c.input_dim, c.input_dim)
        input_weight_bank_params = torch.arange(prod(input_params_shape)).reshape(
            input_params_shape
        )

        output_params_shape = (c.depth_dim, c.input_dim, c.output_dim)
        output_weight_bank_params = torch.arange(prod(output_params_shape)).reshape(
            output_params_shape
        )

        diagonal = min(c.input_dim, c.output_dim)
        diagonal_params_shape = (c.depth_dim, c.input_dim, diagonal)
        diagonal_weight_bank_params = torch.arange(prod(diagonal_params_shape)).reshape(
            diagonal_params_shape
        )

        anti_diagonal_params_shape = (c.depth_dim, c.input_dim, diagonal)
        anti_diagonal_weight_bank_params = torch.arange(
            prod(anti_diagonal_params_shape)
        ).reshape(anti_diagonal_params_shape)

        selected_biases_shape = (c.depth_dim, c.input_dim, c.output_dim)
        selected_bias_params = torch.arange(prod(selected_biases_shape)).reshape(
            selected_biases_shape
        )

        weight_probs = F.sigmoid(torch.randn((batch_size, c.top_k)))
        bias_probs = F.sigmoid(torch.randn((batch_size, c.top_k)))

        output_weights, output_biases = m._compute_parameter_mixture(
            input_batch,
            input_weight_bank_params,
            output_weight_bank_params,
            diagonal_weight_bank_params,
            anti_diagonal_weight_bank_params,
            selected_bias_params,
            weight_probs,
            bias_probs,
        )

        self.assertEqual(
            output_weights.shape, torch.Size([batch_size, c.input_dim, c.output_dim])
        )
        self.assertEqual(output_biases.shape, torch.Size([batch_size, c.output_dim]))
