import torch
import unittest
from Emperor.base.utils import randn
from Emperor.components.parameter_generators.utils.routers import (
    RouterModel,
    RouterLayer,
    VectorChoiceRouterModel,
)
from Emperor.config import ModelConfig


class TestRouterModel(unittest.TestCase):
    def setUp(self):
        self.cfg = ModelConfig()
        self.cfg.bias_flag = True
        self.router_cfg = self.cfg.router_model_config
        self.model = RouterModel(cfg=self.cfg)

    def test_forward_pass_weights(self):
        self.cfg.bias_flag = False
        model = RouterModel(cfg=self.cfg)
        input_batch = randn(self.cfg.batch_size, self.router_cfg.input_dim)
        output = model.compute_logit_scores(input_batch)

        batch_size, output_dim = output.size()
        self.assertEqual(batch_size, self.cfg.batch_size)
        self.assertEqual(output_dim, self.router_cfg.output_dim)

    def test_forward_pass_bias(self):
        self.cfg.bias_flag = True
        model = RouterModel(cfg=self.cfg)
        input_batch = randn(self.cfg.batch_size, self.router_cfg.input_dim)
        output = model.compute_logit_scores(input_batch, False)

        batch_size, output_dim = output.size()
        self.assertEqual(batch_size, self.cfg.batch_size)
        self.assertEqual(output_dim, self.router_cfg.output_dim)

    def test_weight_router_model(self):
        input_batch = randn(self.cfg.batch_size, self.router_cfg.input_dim)
        output = self.model.weight_router_model(input_batch)
        batch_size, output_dim = output.size()
        self.assertEqual(batch_size, self.cfg.batch_size)
        self.assertEqual(output_dim, self.router_cfg.output_dim)

    def test_bias_router_model(self):
        if self.cfg.bias_flag:
            input_batch = randn(self.cfg.batch_size, self.router_cfg.input_dim)
            output = self.model.bias_router_model(input_batch)
            batch_size, output_dim = output.size()
            self.assertEqual(batch_size, self.cfg.batch_size)
            self.assertEqual(output_dim, self.router_cfg.output_dim)

    def test_noisy_topk_flag_weights(self):
        bias_flag = True
        noisy_topk_flag = True
        model = RouterModel(
            cfg=self.cfg, bias_flag=bias_flag, noisy_topk_flag=noisy_topk_flag
        )
        input_batch = randn(self.cfg.batch_size, self.router_cfg.input_dim)
        output = model.compute_logit_scores(input_batch)

        batch_size, output_dim = output.size()
        self.assertEqual(batch_size, self.cfg.batch_size)
        self.assertEqual(output_dim, model.router_output_dim)

    def test_noisy_topk_flag_bias(self):
        bias_flag = True
        noisy_topk_flag = True
        model = RouterModel(
            cfg=self.cfg, bias_flag=bias_flag, noisy_topk_flag=noisy_topk_flag
        )
        input_batch = randn(self.cfg.batch_size, self.router_cfg.input_dim)
        output = model.compute_logit_scores(input_batch, False)

        batch_size, output_dim = output.size()
        self.assertEqual(batch_size, self.cfg.batch_size)
        self.assertEqual(output_dim, model.router_output_dim)

    def test_residual_connection(self):
        residual_flag = True
        input_batch = randn(self.cfg.batch_size, self.router_cfg.hidden_dim)
        router_layer = RouterLayer(
            self.router_cfg.hidden_dim, self.router_cfg.activation, residual_flag
        )
        output = router_layer(input_batch)
        simulated_output = input_batch + router_layer.activation(
            router_layer.layer(input_batch)
        )

        self.assertTrue(torch.allclose(output, simulated_output))

    def test_no_residual_connection(self):
        residual_flag = False
        input_batch = randn(self.cfg.batch_size, self.router_cfg.hidden_dim)
        router_layer = RouterLayer(
            self.router_cfg.hidden_dim,
            self.router_cfg.activation,
            residual_flag,
        )
        output = router_layer(input_batch)
        simulated_output = input_batch + router_layer.activation(
            router_layer.layer(input_batch)
        )
        self.assertFalse(torch.allclose(output, simulated_output))

    def test_single_layer_mlp_router_model(self):
        self.router_cfg.num_layers = 1
        input_batch = randn(self.cfg.batch_size, self.router_cfg.input_dim)
        model = RouterModel(cfg=self.cfg)
        weights_output = model.compute_logit_scores(input_batch)
        bias_output = model.compute_logit_scores(input_batch, False)

        batch_size, output_dim = weights_output.size()
        self.assertEqual(batch_size, self.cfg.batch_size)
        self.assertEqual(output_dim, model.router_output_dim)

        batch_size, output_dim = bias_output.size()
        self.assertEqual(batch_size, self.cfg.batch_size)
        self.assertEqual(output_dim, model.router_output_dim)

    def test_two_layer_mlp_router_model(self):
        self.router_cfg.num_layers = 2
        input_batch = randn(self.cfg.batch_size, self.router_cfg.input_dim)
        model = RouterModel(cfg=self.cfg)
        output = model.compute_logit_scores(input_batch)

        batch_size, output_dim = output.size()
        self.assertEqual(batch_size, self.cfg.batch_size)
        self.assertEqual(output_dim, model.router_output_dim)

    def test_multiple_layers_mlp_router_model(self):
        self.router_cfg.num_layers = 5
        input_batch = randn(self.cfg.batch_size, self.router_cfg.input_dim)
        model = RouterModel(cfg=self.cfg)
        output = model.compute_logit_scores(input_batch)

        batch_size, output_dim = output.size()
        self.assertEqual(batch_size, self.cfg.batch_size)
        self.assertEqual(output_dim, model.router_output_dim)

    def test_forward_with_different_input_shapes(self):
        input_batch = randn(4, self.router_cfg.input_dim)
        output = self.model.compute_logit_scores(input_batch)
        batch_size, _ = output.size()
        self.assertEqual(batch_size, 4)

        input_batch = randn(1, self.router_cfg.input_dim)
        output = self.model.compute_logit_scores(input_batch)
        batch_size, _ = output.size()
        self.assertEqual(batch_size, 1)


class TestVectorChoiceRouterModel(unittest.TestCase):
    def setUp(self):
        self.cfg = ModelConfig()
        self.router_cfg = self.cfg.router_model_config
        self.bias_flag = True
        self.model = VectorChoiceRouterModel(cfg=self.cfg)

    def test_forward_weight_pass_no_bias(self):
        input_batch = randn(self.cfg.batch_size, self.router_cfg.input_dim)
        output = self.model.compute_logit_scores(input_batch)

        input_dim, batch_size, router_output_dim = output.size()
        self.assertEqual(input_dim, self.router_cfg.input_dim)
        self.assertEqual(batch_size, self.cfg.batch_size)
        self.assertEqual(router_output_dim, self.model.router_output_dim)

    def test_forward_weight_pass_bias(self):
        input_batch = randn(self.cfg.batch_size, self.router_cfg.input_dim)
        output = self.model.compute_logit_scores(input_batch, True)

        output_dim, batch_size, router_output_dim = output.size()
        self.assertEqual(output_dim, self.router_cfg.output_dim)
        self.assertEqual(batch_size, self.cfg.batch_size)
        self.assertEqual(router_output_dim, self.model.router_output_dim)
