import math
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
)
from emperor.linears import LinearLayerConfig
from emperor.sampler import RouterConfig, RouterModel, SamplerConfig
from emperor.sampler._monitoring import _SamplerDiagnostics
from emperor.sampler._selection.base import SamplerBase
from emperor.sampler._selection.sparse import SamplerSparse


def sampler_config(**overrides: object) -> SamplerConfig:
    values = {
        "top_k": 2,
        "threshold": 0.0,
        "filter_above_threshold": False,
        "num_topk_samples": 0,
        "normalize_probabilities_flag": False,
        "noisy_topk_flag": False,
        "num_experts": 4,
        "coefficient_of_variation_loss_weight": 0.0,
        "switch_loss_weight": 0.0,
        "zero_centred_loss_weight": 0.0,
        "mutual_information_loss_weight": 0.0,
        "router_config": None,
    }
    values.update(overrides)
    return SamplerConfig(**values)


def router_layer_stack_config() -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=2,
        hidden_dim=3,
        output_dim=2,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            activation=ActivationOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=True),
        ),
    )


@dataclass
class WrappedRouterConfig(ConfigBase):
    input_dim: int | None = optional_field("Outer model input dimension.")
    router_model_config: RouterConfig | None = optional_field(
        "Nested router configuration."
    )


class SamplerRegressionTests(unittest.TestCase):
    def test_sparse_sampler_rejects_non_sparse_top_k(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "^top_k must be 1 when using SamplerSparse, received 2\\.$",
        ):
            SamplerSparse(sampler_config(top_k=2))

    def test_non_finite_auxiliary_loss_weights_are_rejected(self) -> None:
        field_names = (
            "coefficient_of_variation_loss_weight",
            "switch_loss_weight",
            "zero_centred_loss_weight",
            "mutual_information_loss_weight",
        )
        for field_name in field_names:
            for value in (math.inf, -math.inf, math.nan):
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaisesRegex(
                        ValueError,
                        f"^{field_name} must be finite and >= 0\\.0, received ",
                    ):
                        SamplerBase(sampler_config(**{field_name: value}))

    def test_wrapped_router_does_not_hide_invalid_nested_input_dimension(self) -> None:
        wrapped = WrappedRouterConfig(
            input_dim=3,
            router_model_config=RouterConfig(
                input_dim=0,
                num_experts=2,
                noisy_topk_flag=False,
                model_config=router_layer_stack_config(),
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "^input_dim must be a positive integer, received 0\\.$",
        ):
            RouterModel(wrapped)

    def test_single_expert_diagnostics_are_finite(self) -> None:
        metrics = _SamplerDiagnostics.calculate_usage(
            torch.tensor([3.0]),
            torch.tensor([2.5]),
        )

        self.assertTrue(torch.isfinite(metrics.coefficient_of_variation))
        torch.testing.assert_close(
            metrics.coefficient_of_variation,
            torch.zeros(()),
        )


if __name__ == "__main__":
    unittest.main()
