import copy
import torch
import torch.nn as nn
import unittest
from Emperor.base.utils import randn
from Emperor.components.parameter_generators.utils.routers import (
    RouterModel,
    RouterLayer,
    VectorRouterModel,
    RouterConfig,
)
from Emperor.config import ROUTER_NUM_LAYERNUM_LAYERSS, ModelConfig


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
        self.assertEqual(model.router_output_dim, c.output_dim)

    def test__init__noisy_topk_flag__True(self):
        c = copy.deepcopy(self.cfg)
        overrides = RouterConfig(
            noisy_topk_flag=True,
        )
        model = RouterModel(self.cfg, overrides)
        self.assertEqual(model.router_output_dim, c.output_dim * 2)

    def test__main_config_override(self):
        c = ModelConfig()
        overrides = RouterConfig(
            input_dim=256,
            noisy_topk_flag=True,
            output_dim=512,
            num_layers=40,
        )

        m = RouterModel(c, overrides)
        self.assertEqual(m.input_dim, overrides.input_dim)
        self.assertEqual(m.output_dim, overrides.output_dim)
        self.assertEqual(m.num_layers, overrides.num_layers)
        self.assertEqual(m.noisy_topk_flag, overrides.noisy_topk_flag)

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

        self.assertIsInstance(model, nn.Linear)

    def test__build_model__num_layers__3(self):
        c = copy.deepcopy(self.cfg)
        overrides = RouterConfig(
            num_layers=3,
        )
        m = RouterModel(c, overrides)
        model = m._RouterModel__build_model()

        self.assertIsInstance(model, nn.Sequential)

    def test__build_multilayer_router(self):
        c = copy.deepcopy(self.cfg)
        overrides = RouterConfig(
            num_layers=5,
        )
        m = RouterModel(c, overrides)
        model = m._RouterModel__build_multilayer_router()

        self.assertIsInstance(model[0], nn.Linear)
        for idx in range(2, m.num_layers):
            self.assertIsInstance(model[idx], RouterLayer)
        self.assertIsInstance(model[-1], nn.Linear)

    def test__compute_logit_scores__noisy_topk__False(self):
        c = copy.deepcopy(self.cfg)
        overrides = RouterConfig(
            noisy_topk_flag=False,
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
        )
        m = RouterModel(c, overrides)

        batch_size = 2
        input_batch = randn(batch_size, m.input_dim)
        output = m.compute_logit_scores(input_batch)

        self.assertEqual(list(output.shape), [batch_size, m.output_dim * 2])
