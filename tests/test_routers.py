import torch
import torch.nn as nn
import unittest
from Emperor.base.utils import randn
from Emperor.components.parameter_generators.utils.routers import (
    RouterModel,
    RouterLayer,
    VectorChoiceRouterModel,
    RouterConfig,
)
from Emperor.config import ModelConfig


class TestRouterModel(unittest.TestCase):
    def setUp(self):
        self.cfg = ModelConfig()

        self.router_cfg = self.cfg.router_model_config

        self.input_dim = self.router_cfg.input_dim
        self.hidden_dim = self.router_cfg.hidden_dim
        self.output_dim = self.router_cfg.output_dim
        self.batch_size = self.cfg.batch_size
        self.activation = self.router_cfg.activation
        self.residual_flag = self.router_cfg.residual_flag
        self.compute_bias_logits_flag = self.router_cfg.compute_bias_logits_flag
        self.noisy_topk_flag = self.router_cfg.noisy_topk_flag
        self.num_layers = self.router_cfg.num_layers

        self.model = RouterModel(cfg=self.cfg)
        self.test_input = randn(self.batch_size, self.input_dim)

    def test__main_config_override(self):
        overrides = RouterConfig(
            input_dim=256,
            noisy_topk_flag=True,
            compute_bias_logits_flag=False,
            output_dim=512,
            num_layers=40,
        )

        cfg = ModelConfig()

        model = RouterModel(cfg=cfg, overrides=overrides)
        self.assertEqual(model.input_dim, overrides.input_dim)
        self.assertEqual(model.output_dim, overrides.output_dim)
        self.assertEqual(model.num_layers, overrides.num_layers)
        self.assertEqual(model.noisy_topk_flag, overrides.noisy_topk_flag)
        self.assertEqual(
            model.compute_bias_logits_flag, overrides.compute_bias_logits_flag
        )

    def test__init_with_config_only(self):
        model = RouterModel(cfg=self.cfg)

        self.assertEqual(model.input_dim, self.input_dim)
        self.assertEqual(model.hidden_dim, self.hidden_dim)
        self.assertEqual(model.output_dim, self.output_dim)
        self.assertEqual(model.compute_bias_logits_flag, self.compute_bias_logits_flag)
        self.assertIsNotNone(model.weight_router_model)
        self.assertIsNotNone(model.bias_router_model)

    def test__init_with_router_config_only(self):
        config = RouterConfig(
            input_dim=5,
            hidden_dim=6,
            output_dim=7,
            compute_bias_logits_flag=True,
            noisy_topk_flag=False,
            residual_flag=True,
            activation=nn.Sigmoid(),
            num_layers=3,
        )
        model = RouterModel(config)

        self.assertEqual(model.input_dim, 5)
        self.assertEqual(model.hidden_dim, 6)
        self.assertEqual(model.output_dim, 7)
        self.assertEqual(model.compute_bias_logits_flag, True)
        self.assertEqual(model.noisy_topk_flag, False)
        self.assertEqual(model.residual_flag, True)
        self.assertEqual(model.num_layers, 3)

    def test__compute_logit_scores_noisy_topk_false(self):
        overrides = RouterConfig(noisy_topk_flag=False)
        model = RouterModel(self.cfg, overrides)

        input_batch = randn(self.batch_size, self.input_dim)
        output = model.compute_logit_scores(input_batch)

        batch_size, output_dim = output.size()
        self.assertEqual(batch_size, self.batch_size)
        self.assertEqual(output_dim, self.output_dim)

    def test__compute_logit_scores_noisy_topk_true(self):
        overrides = RouterConfig(noisy_topk_flag=True)
        model = RouterModel(self.cfg, overrides)

        input_test = randn(self.batch_size, self.input_dim)
        output = model.compute_logit_scores(input_test)

        batch_size, output_dim = output.size()
        self.assertEqual(batch_size, self.batch_size)
        self.assertEqual(output_dim, self.output_dim * 2)

    def test__init_with_invalid_num_layers(self):
        with self.assertRaises(AssertionError):
            overrides = RouterConfig(num_layers=0)
            RouterModel(self.cfg, overrides)

    def test__noisy_topk_affects_output_dim(self):
        overrides1 = RouterConfig(noisy_topk_flag=False)
        overrides2 = RouterConfig(noisy_topk_flag=True)

        model1 = RouterModel(self.cfg, overrides1)
        model2 = RouterModel(self.cfg, overrides2)

        self.assertEqual(model1.router_output_dim, self.output_dim)
        self.assertEqual(model2.router_output_dim, 2 * self.output_dim)

    def test__same_output_single_vs_multilayer(self):
        single_layer_overrides = RouterConfig(num_layers=1, noisy_topk_flag=False)
        multi_layer_overrides = RouterConfig(num_layers=7, noisy_topk_flag=False)
        single_layer = RouterModel(self.cfg, single_layer_overrides)
        multi_layer = RouterModel(self.cfg, multi_layer_overrides)

        test_input = randn(self.batch_size, self.input_dim)
        output1 = single_layer.compute_logit_scores(test_input)
        output2 = multi_layer.compute_logit_scores(test_input)

        self.assertEqual(output1.shape, (self.batch_size, self.output_dim))
        self.assertEqual(output2.shape, (self.batch_size, self.output_dim))

    def test__forward_pass_compute_weights_logits(self):
        overrides = RouterConfig(noisy_topk_flag=False, compute_bias_logits_flag=True)
        model = RouterModel(self.cfg, overrides)
        test_input = randn(self.batch_size, self.input_dim)
        model.set_compute_weight_flag(True)
        weight_logits_output = model.compute_logit_scores(test_input)
        model.set_compute_weight_flag(False)
        bias_logits_output = model.compute_logit_scores(test_input)

        batch_size, output_dim = weight_logits_output.size()
        self.assertEqual(batch_size, self.batch_size)
        self.assertEqual(output_dim, self.output_dim)

        batch_size, output_dim = bias_logits_output.size()
        self.assertEqual(batch_size, self.batch_size)
        self.assertEqual(output_dim, self.output_dim)

    def test_noisy_topk_flag_weights(self):
        overrides = RouterConfig(
            compute_bias_logits_flag=True,
            noisy_topk_flag=True,
        )
        model = RouterModel(self.cfg, overrides)
        model.set_compute_weight_flag(True)
        test_input = randn(self.batch_size, self.input_dim)
        output = model.compute_logit_scores(test_input)

        batch_size, output_dim = output.size()
        self.assertEqual(batch_size, self.batch_size)
        self.assertEqual(output_dim, model.router_output_dim)

    def test_noisy_topk_flag_bias(self):
        overrides = RouterConfig(
            compute_bias_logits_flag=True,
            noisy_topk_flag=True,
        )
        model = RouterModel(self.cfg, overrides)
        test_input = randn(self.batch_size, self.input_dim)
        model.set_compute_weight_flag(False)
        output = model.compute_logit_scores(test_input)

        batch_size, output_dim = output.size()
        self.assertEqual(batch_size, self.batch_size)
        self.assertEqual(output_dim, model.router_output_dim)

    def test_residual_connection(self):
        residual_flag = True
        router_layer = RouterLayer(self.hidden_dim, self.activation, residual_flag)
        test_input = randn(self.batch_size, self.hidden_dim)
        output = router_layer(test_input)
        simulated_output = test_input + router_layer.activation(
            router_layer.layer(test_input)
        )

        self.assertTrue(torch.allclose(output, simulated_output))

    def test_no_residual_connection(self):
        residual_flag = False
        router_layer = RouterLayer(self.hidden_dim, self.activation, residual_flag)
        test_input = randn(self.batch_size, self.hidden_dim)
        output = router_layer(test_input)
        simulated_output = test_input + router_layer.activation(
            router_layer.layer(test_input)
        )
        self.assertFalse(torch.allclose(output, simulated_output))

    def test_single_layer_mlp_router_model(self):
        overrides = RouterConfig(num_layers=1)
        model = RouterModel(self.cfg, overrides)

        test_input = randn(self.batch_size, self.input_dim)
        model.set_compute_weight_flag(True)
        weights_output = model.compute_logit_scores(test_input)
        model.set_compute_weight_flag(False)
        bias_output = model.compute_logit_scores(test_input)

        batch_size, output_dim = weights_output.size()
        self.assertEqual(batch_size, self.batch_size)
        self.assertEqual(output_dim, model.router_output_dim)

        batch_size, output_dim = bias_output.size()
        self.assertEqual(batch_size, self.batch_size)
        self.assertEqual(output_dim, model.router_output_dim)

    def test_two_layer_mlp_router_model(self):
        overrides = RouterConfig(num_layers=2)
        model = RouterModel(self.cfg, overrides)

        test_input = randn(self.batch_size, self.input_dim)
        model.set_compute_weight_flag(True)
        weights_output = model.compute_logit_scores(test_input)
        model.set_compute_weight_flag(False)
        bias_output = model.compute_logit_scores(test_input)

        batch_size, output_dim = weights_output.size()
        self.assertEqual(batch_size, self.batch_size)
        self.assertEqual(output_dim, model.router_output_dim)

        batch_size, output_dim = bias_output.size()
        self.assertEqual(batch_size, self.batch_size)
        self.assertEqual(output_dim, model.router_output_dim)

    def test_multiple_layers_mlp_router_model(self):
        overrides = RouterConfig(num_layers=5)
        model = RouterModel(self.cfg, overrides)

        test_input = randn(self.batch_size, self.input_dim)
        model.set_compute_weight_flag(True)
        weights_output = model.compute_logit_scores(test_input)
        model.set_compute_weight_flag(False)
        bias_output = model.compute_logit_scores(test_input)

        batch_size, output_dim = weights_output.size()
        self.assertEqual(batch_size, self.batch_size)
        self.assertEqual(output_dim, model.router_output_dim)

        batch_size, output_dim = bias_output.size()
        self.assertEqual(batch_size, self.batch_size)
        self.assertEqual(output_dim, model.router_output_dim)

    def test_forward_with_different_batch_sizes(self):
        input_batch = randn(4, self.input_dim)
        output = self.model.compute_logit_scores(input_batch)
        batch_size, _ = output.size()
        self.assertEqual(batch_size, 4)

        input_batch = randn(1, self.input_dim)
        output = self.model.compute_logit_scores(input_batch)
        batch_size, _ = output.size()
        self.assertEqual(batch_size, 1)

        input_batch = randn(32, self.input_dim)
        output = self.model.compute_logit_scores(input_batch)
        batch_size, _ = output.size()
        self.assertEqual(batch_size, 32)


class TestVectorChoiceRouterModel(unittest.TestCase):
    def setUp(self):
        self.cfg = ModelConfig()

        self.router_cfg = self.cfg.router_model_config

        self.input_dim = self.router_cfg.input_dim
        self.hidden_dim = self.router_cfg.hidden_dim
        self.output_dim = self.router_cfg.output_dim
        self.batch_size = self.cfg.batch_size
        self.activation = self.router_cfg.activation
        self.residual_flag = self.router_cfg.residual_flag
        self.compute_bias_logits_flag = self.router_cfg.compute_bias_logits_flag
        self.noisy_topk_flag = self.router_cfg.noisy_topk_flag
        self.num_layers = self.router_cfg.num_layers

        self.test_input = randn(self.batch_size, self.input_dim)

    def test_forward_weight_logits(self):
        test_input = randn(self.batch_size, self.input_dim)
        model = VectorChoiceRouterModel(self.cfg)
        model.set_compute_weight_flag(True)
        output = model.compute_logit_scores(test_input)

        input_dim, batch_size, router_output_dim = output.size()
        self.assertEqual(input_dim, self.input_dim)
        self.assertEqual(batch_size, self.batch_size)
        self.assertEqual(router_output_dim, model.router_output_dim)

    def test_forward_bias_logits(self):
        test_input = randn(self.batch_size, self.input_dim)
        model = VectorChoiceRouterModel(self.cfg)
        model.set_compute_weight_flag(False)
        output = model.compute_logit_scores(test_input)

        output_dim, batch_size, router_output_dim = output.size()
        self.assertEqual(output_dim, self.output_dim)
        self.assertEqual(batch_size, self.batch_size)
        self.assertEqual(router_output_dim, model.router_output_dim)
