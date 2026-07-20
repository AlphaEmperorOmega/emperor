import unittest
from dataclasses import dataclass

import torch

from emperor.config import ConfigBase, optional_field
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    LayerState,
)
from emperor.linears import LinearLayerConfig
from emperor.nn import Module
from emperor.sampler import RouterConfig, RouterModel


@dataclass
class ConstantRouterConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")

    def _registry_owner(self) -> type:
        return ConstantRouterModel


class ConstantRouterModel(Module):
    def __init__(
        self,
        cfg: ConstantRouterConfig,
        overrides: ConstantRouterConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg: ConstantRouterConfig = self._override_config(cfg, overrides)
        self.output_dim = self.cfg.output_dim

    def forward(self, state: LayerState) -> LayerState:
        return LayerState(
            hidden=torch.zeros(
                *state.hidden.shape[:-1],
                self.output_dim,
                dtype=state.hidden.dtype,
                device=state.hidden.device,
            ),
            loss=state.loss,
            halting_state=state.halting_state,
        )


class TestRouterModel(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 8,
        hidden_dim: int = 12,
        num_experts: int = 4,
        noisy_topk_flag: bool = False,
        num_layers: int = 2,
        model_config: ConfigBase | None = None,
    ) -> RouterConfig:
        if model_config is None:
            model_config = self.layer_stack_config(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=num_experts,
                num_layers=num_layers,
            )
        return RouterConfig(
            input_dim=input_dim,
            num_experts=num_experts,
            noisy_topk_flag=noisy_topk_flag,
            model_config=model_config,
        )

    def layer_stack_config(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
    ) -> LayerStackConfig:
        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=True,
            layer_config=LayerConfig(
                activation=ActivationOptions.RELU,
                residual_config=None,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(bias_flag=True),
            ),
        )

    def test_init_stores_config_attributes(self):
        cfg = self.preset(input_dim=8, num_experts=4, noisy_topk_flag=False)
        model = RouterModel(cfg)

        self.assertEqual(model.input_dim, 8)
        self.assertEqual(model.num_experts, 4)
        self.assertFalse(model.noisy_topk_flag)
        self.assertEqual(model.model_config, cfg.model_config)

    def test_noisy_topk_doubles_router_output_dim(self):
        cfg = self.preset(input_dim=8, num_experts=4, noisy_topk_flag=True)
        model = RouterModel(cfg)

        self.assertEqual(model.num_experts, 4)
        self.assertEqual(model.router_output_dim, 8)

    def test_invalid_config_values_raise(self):
        cases = [
            ("input_dim", 0, ValueError),
            ("input_dim", -1, ValueError),
            ("input_dim", True, TypeError),
            ("input_dim", 1.5, TypeError),
            ("num_experts", 0, ValueError),
            ("num_experts", -1, ValueError),
            ("num_experts", True, TypeError),
            ("num_experts", 1.5, TypeError),
            ("noisy_topk_flag", None, ValueError),
            ("noisy_topk_flag", 1, TypeError),
            ("model_config", None, ValueError),
            ("model_config", object(), TypeError),
        ]
        for field_name, value, error_type in cases:
            with self.subTest(field_name=field_name, value=value):
                cfg = self.preset()
                setattr(cfg, field_name, value)
                with self.assertRaises(error_type):
                    RouterModel(cfg)

    def test_compute_logit_scores_returns_expected_shape(self):
        cases = [
            (1, 4, False, 4),
            (2, 4, False, 4),
            (3, 4, True, 8),
        ]
        for num_layers, num_experts, noisy_topk_flag, expected_dim in cases:
            with self.subTest(
                num_layers=num_layers,
                num_experts=num_experts,
                noisy_topk_flag=noisy_topk_flag,
            ):
                cfg = self.preset(
                    input_dim=8,
                    num_experts=num_experts,
                    noisy_topk_flag=noisy_topk_flag,
                    num_layers=num_layers,
                )
                model = RouterModel(cfg)

                input_batch = torch.randn(3, cfg.input_dim)
                output = model.compute_logit_scores(input_batch)

                self.assertIsInstance(output, torch.Tensor)
                self.assertEqual(output.shape, (3, expected_dim))

    def test_compute_logit_scores_accepts_config_base_model_config(self):
        cfg = self.preset(
            input_dim=8,
            num_experts=4,
            model_config=ConstantRouterConfig(input_dim=8, output_dim=4),
        )
        model = RouterModel(cfg)

        output = model.compute_logit_scores(torch.randn(3, 8))

        self.assertEqual(output.shape, (3, 4))

    def test_compute_logit_scores_rejects_invalid_input_shape(self):
        model = RouterModel(self.preset(input_dim=8))
        invalid_inputs = [
            torch.randn(8),
            torch.randn(2, 3, 8),
            torch.randn(2, 7),
        ]

        for input_batch in invalid_inputs:
            with self.subTest(shape=tuple(input_batch.shape)):
                with self.assertRaises(ValueError):
                    model.compute_logit_scores(input_batch)

    def test_compute_logit_scores_rejects_non_tensor_input(self):
        model = RouterModel(self.preset(input_dim=8))

        with self.assertRaises(TypeError):
            model.compute_logit_scores([[1.0] * 8])
