import copy
import torch.nn as nn
import unittest
from Emperor.base.utils import randn
from Emperor.layers.utils.base import LayerBlock
from Emperor.layers.utils.linears import (
    DynamicDiagonalLinearLayer,
    LinearLayer,
)
from Emperor.layers.utils.routers import (
    RouterModel,
    RouterConfig,
    VectorRouterModel,
)
from Emperor.config import ModelConfig


class TestRouterModel(unittest.TestCase):
    def setUp(self):
        self.cfg = RouterConfig(
            input_dim=5,
            hidden_dim=6,
            output_dim=7,
            noisy_topk_flag=False,
            residual_flag=False,
            activation=nn.Sigmoid(),
            num_layers=3,
            diagonal_linear_model_flag=False,
        )

    def test__init_with_invalid_num_layers(self):
        with self.assertRaises(AssertionError):
            overrides = RouterConfig(num_layers=0)
            RouterModel(self.cfg, overrides)

    def test__init__noisy_topk_flag__False(self):
        c = copy.deepcopy(self.cfg)
        overrides = RouterConfig(
            noisy_topk_flag=False,
        )
        model = RouterModel(self.cfg, overrides)
        self.assertEqual(model.num_experts, c.output_dim)

    def test__init__noisy_topk_flag__True(self):
        c = copy.deepcopy(self.cfg)
        overrides = RouterConfig(
            noisy_topk_flag=True,
        )
        model = RouterModel(self.cfg, overrides)
        self.assertEqual(model.num_experts, c.output_dim * 2)

    def test__main_config_override(self):
        c = ModelConfig()
        overrides = RouterConfig(
            input_dim=256,
            noisy_topk_flag=True,
            output_dim=512,
            num_layers=40,
            diagonal_linear_model_flag=True,
        )

        m = RouterModel(c, overrides)
        self.assertEqual(m.input_dim, overrides.input_dim)
        self.assertEqual(m.output_dim, overrides.output_dim)
        self.assertEqual(m.num_layers, overrides.num_layers)
        self.assertEqual(m.noisy_topk_flag, overrides.noisy_topk_flag)
        self.assertEqual(
            m.diagonal_linear_model_flag, overrides.diagonal_linear_model_flag
        )

    def test__init_with_main_config(self):
        m = RouterModel(self.cfg)

        self.assertEqual(m.input_dim, self.cfg.input_dim)
        self.assertEqual(m.hidden_dim, self.cfg.hidden_dim)
        self.assertEqual(m.output_dim, self.cfg.output_dim)

    def test__init_with_router_config_only(self):
        config = RouterConfig(
            input_dim=5,
            hidden_dim=6,
            output_dim=7,
            noisy_topk_flag=False,
            residual_flag=True,
            activation=nn.Sigmoid(),
            num_layers=3,
            diagonal_linear_model_flag=True,
        )
        m = RouterModel(config)

        self.assertEqual(m.input_dim, 5)
        self.assertEqual(m.hidden_dim, 6)
        self.assertEqual(m.output_dim, 7)
        self.assertEqual(m.noisy_topk_flag, False)
        self.assertEqual(m.residual_flag, True)
        self.assertEqual(m.num_layers, 3)

    def test__build_model__num_layers__1(self):
        c = copy.deepcopy(self.cfg)
        overrides = RouterConfig(
            num_layers=1,
        )
        m = RouterModel(c, overrides)
        model = m._RouterModel__build_model()

        self.assertIsInstance(model, nn.Sequential)

    def test__build_model__num_layers__3(self):
        c = copy.deepcopy(self.cfg)
        overrides = RouterConfig(
            num_layers=3,
        )
        m = RouterModel(c, overrides)
        model = m._RouterModel__build_model()

        self.assertIsInstance(model, nn.Sequential)
        self.assertIsInstance(model[0], LayerBlock)
        for layer in model[:-1]:
            self.assertIsInstance(layer, LayerBlock)
        self.assertIsInstance(model[-1], nn.Linear)

    def test__compute_logit_scores__noisy_topk__False(self):
        c = copy.deepcopy(self.cfg)
        overrides = RouterConfig(
            noisy_topk_flag=False,
            diagonal_linear_model_flag=True,
        )
        m = RouterModel(c, overrides)

        batch_size = 2
        input_batch = randn(batch_size, m.input_dim)
        output = m.compute_logit_scores(input_batch)

        self.assertEqual(list(output.shape), [batch_size, m.output_dim])

    def test__compute_logit_scores__noisy_topk__True(self):
        c = copy.deepcopy(self.cfg)
        overrides = RouterConfig(
            noisy_topk_flag=True,
            diagonal_linear_model_flag=True,
        )
        m = RouterModel(c, overrides)

        batch_size = 2
        input_batch = randn(batch_size, m.input_dim)
        output = m.compute_logit_scores(input_batch)

        self.assertEqual(list(output.shape), [batch_size, m.output_dim * 2])

    def test__create_router_layer_model__diagonal_linear_model_flag__False(self):
        c = copy.deepcopy(self.cfg)
        overrides = RouterConfig(noisy_topk_flag=True, diagonal_linear_model_flag=False)
        m = RouterModel(c, overrides)

        model = m._RouterModel__create_router_layer_model(c.input_dim, c.output_dim)

        self.assertIsInstance(model, LinearLayer)

    def test__create_router_layer_model__diagonal_linear_model_flag__True(self):
        c = copy.deepcopy(self.cfg)
        overrides = RouterConfig(
            noisy_topk_flag=True,
            diagonal_linear_model_flag=True,
        )
        m = RouterModel(c, overrides)

        model = m._RouterModel__create_router_layer_model(c.input_dim, c.output_dim)

        self.assertIsInstance(model, DynamicDiagonalLinearLayer)


class TestVectorRouterModel(unittest.TestCase):
    def setUp(self):
        self.cfg = RouterConfig(
            input_dim=5,
            hidden_dim=6,
            output_dim=7,
            noisy_topk_flag=False,
            residual_flag=False,
            activation=nn.Sigmoid(),
            num_layers=3,
            diagonal_linear_model_flag=False,
        )

    def test__generate_parameter_bank__bias_parameters_flag__False(self):
        c = copy.deepcopy(self.cfg)
        m = VectorRouterModel(c)

        parameters = m._VectorRouterModel__generate_parameter_bank()

        expected_shape = [c.input_dim, c.input_dim, c.output_dim]
        self.assertEqual(list(parameters.shape), expected_shape)

    def test__generate_parameter_bank__bias_parameters_flag__True(self):
        c = copy.deepcopy(self.cfg)
        m = VectorRouterModel(c)

        parameters = m._VectorRouterModel__generate_parameter_bank()

        expected_shape = [c.input_dim, c.input_dim, c.output_dim]
        self.assertEqual(list(parameters.shape), expected_shape)

    def test__compute_logit_scores__noisy_topk__False(self):
        c = copy.deepcopy(self.cfg)
        overrides = RouterConfig(
            noisy_topk_flag=False,
            diagonal_linear_model_flag=True,
        )
        m = VectorRouterModel(c)

        batch_size = 2
        input_batch = randn(batch_size, m.input_dim)
        output = m.compute_logit_scores(input_batch)

        self.assertEqual(list(output.shape), [m.input_dim, batch_size, m.output_dim])

    def test__compute_logit_scores__noisy_topk__True(self):
        c = copy.deepcopy(self.cfg)
        overrides = RouterConfig(
            noisy_topk_flag=True,
            diagonal_linear_model_flag=True,
        )
        m = VectorRouterModel(c, overrides)

        batch_size = 2
        input_batch = randn(batch_size, m.input_dim)
        output = m.compute_logit_scores(input_batch)

        self.assertEqual(
            list(output.shape), [m.input_dim, batch_size, m.output_dim * 2]
        )
