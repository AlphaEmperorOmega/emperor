import math
import unittest
from dataclasses import dataclass

import torch
from lightning import LightningModule

from emperor.config import ConfigBase, optional_field
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.sampler import (
    RouterConfig,
    RouterModel,
    SamplerConfig,
    SamplerModel,
    SamplerMonitorCallback,
)
from emperor.sampler._monitoring import _SamplerDiagnostics
from emperor.sampler._selection.base import SamplerBase
from emperor.sampler._selection.full import SamplerFull
from emperor.sampler._selection.losses import (
    AuxiliaryLossBase,
    CoefficientOfVariationLoss,
    MutualInformationLoss,
    SamplerAuxiliaryLosses,
    SwitchLoss,
    ZeroCentredLoss,
)
from emperor.sampler._selection.sparse import SamplerSparse
from emperor.sampler._selection.top_k import SamplerTopk
from emperor.sampler._usage import SamplerUsageTrackerManager


def sampler_config(**overrides) -> SamplerConfig:
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


def router_network_config(input_dim: int, output_dim: int) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=input_dim,
        output_dim=output_dim,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.ENABLED,
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


def router_config(
    *,
    input_dim: int | None = 2,
    num_experts: int = 3,
    noisy_topk_flag: bool = False,
) -> RouterConfig:
    resolved_input_dim = 2 if input_dim is None else input_dim
    output_dim = num_experts * (2 if noisy_topk_flag else 1)
    return RouterConfig(
        input_dim=input_dim,
        num_experts=num_experts,
        noisy_topk_flag=noisy_topk_flag,
        model_config=router_network_config(resolved_input_dim, output_dim),
    )


@dataclass
class WrappedSamplerConfig(ConfigBase):
    sampler_model_config: SamplerConfig | None = optional_field(
        "Nested sampler configuration."
    )


@dataclass
class WrappedRouterConfig(ConfigBase):
    router_model_config: RouterConfig | None = optional_field(
        "Nested router configuration."
    )
    input_dim: int | None = optional_field("Outer input dimension.")


@dataclass
class WrappedRouterConfigWithoutInputDimension(ConfigBase):
    router_model_config: RouterConfig | None = optional_field(
        "Nested router configuration."
    )


class _RecordingLightningModule(LightningModule):
    def __init__(self, **samplers: SamplerModel) -> None:
        super().__init__()
        for name, sampler in samplers.items():
            self.add_module(name, sampler)
        self.logged_scalars: list[tuple[str, torch.Tensor]] = []
        self._recorded_global_step = 0

    @property
    def logger(self):
        return None

    @property
    def global_step(self) -> int:
        return self._recorded_global_step

    def log(self, name, value, *args, **kwargs) -> None:
        self.logged_scalars.append((name, value))


class ExactExceptionMixin:
    def assert_exact_exception(
        self,
        exception_type: type[BaseException],
        message: str,
        operation,
    ) -> None:
        with self.assertRaises(exception_type) as raised:
            operation()
        self.assertEqual(str(raised.exception), message)


class SamplerValidationContractTests(ExactExceptionMixin, unittest.TestCase):
    def test_base_boundaries_and_exact_configuration_errors(self) -> None:
        SamplerBase(sampler_config(threshold=1.0, num_topk_samples=2))

        cases = (
            (
                {"num_experts": 0},
                "num_experts must be a positive integer, received 0.",
            ),
            (
                {"num_topk_samples": -1},
                "num_topk_samples must be a non-negative integer, received -1.",
            ),
            (
                {"threshold": 1.1},
                "threshold must be between 0.0 and 1.0 inclusive, received 1.1.",
            ),
            (
                {"num_topk_samples": 3},
                "num_topk_samples cannot exceed top_k, "
                "received num_topk_samples=3, top_k=2.",
            ),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                self.assert_exact_exception(
                    ValueError,
                    message,
                    lambda overrides=overrides: SamplerBase(
                        sampler_config(**overrides)
                    ),
                )

    def test_specialized_sampler_boundaries_and_exact_errors(self) -> None:
        self.assertEqual(SamplerTopk(sampler_config(top_k=1)).top_k, 1)

        cases = (
            (
                SamplerSparse,
                {"top_k": 2},
                "top_k must be 1 when using SamplerSparse, received 2.",
            ),
            (
                SamplerSparse,
                {"top_k": 1, "normalize_probabilities_flag": True},
                "normalize_probabilities_flag must be False when using "
                "SamplerSparse, received True.",
            ),
            (
                SamplerSparse,
                {"top_k": 1, "num_topk_samples": 1},
                "num_topk_samples must be 0 when using SamplerSparse, received 1.",
            ),
            (
                SamplerSparse,
                {"top_k": 1, "mutual_information_loss_weight": 0.25},
                "mutual_information_loss_weight must be 0.0 when using "
                "SamplerSparse, received 0.25.",
            ),
            (
                SamplerTopk,
                {"top_k": 4},
                "top_k must be greater than 0 and less than num_experts when "
                "using SamplerTopk, received top_k=4, num_experts=4.",
            ),
            (
                SamplerFull,
                {"top_k": 4, "num_topk_samples": 1},
                "num_topk_samples must be 0 when using SamplerFull, received 1.",
            ),
            (
                SamplerFull,
                {"top_k": 4, "coefficient_of_variation_loss_weight": 0.25},
                "coefficient_of_variation_loss_weight must be 0.0 when using "
                "SamplerFull, received 0.25.",
            ),
            (
                SamplerFull,
                {"top_k": 4, "switch_loss_weight": 0.25},
                "switch_loss_weight must be 0.0 when using SamplerFull, received 0.25.",
            ),
            (
                SamplerFull,
                {"top_k": 4, "zero_centred_loss_weight": 0.25},
                "zero_centred_loss_weight must be 0.0 when using SamplerFull, "
                "received 0.25.",
            ),
            (
                SamplerFull,
                {"top_k": 4, "mutual_information_loss_weight": 0.25},
                "mutual_information_loss_weight must be 0.0 when using "
                "SamplerFull, received 0.25.",
            ),
            (
                SamplerFull,
                {"top_k": 3},
                "top_k must be equal to num_experts when using SamplerFull, "
                "received top_k=3, num_experts=4.",
            ),
        )
        for sampler_type, overrides, message in cases:
            with self.subTest(sampler_type=sampler_type.__name__, overrides=overrides):
                self.assert_exact_exception(
                    ValueError,
                    message,
                    lambda sampler_type=sampler_type, overrides=overrides: sampler_type(
                        sampler_config(**overrides)
                    ),
                )

    def test_selection_runtime_errors_include_exact_types_ranks_and_shapes(
        self,
    ) -> None:
        sampler = SamplerTopk(sampler_config())

        cases = (
            (
                TypeError,
                "router_logit_scores must be a Tensor, received list.",
                lambda: sampler.get_probabilities_and_indices([[1.0] * 4]),
            ),
            (
                ValueError,
                "router_logit_scores must be a 2D tensor "
                "(batch_size, num_experts), received a 1D tensor with shape (4,).",
                lambda: sampler.get_probabilities_and_indices(torch.ones(4)),
            ),
            (
                ValueError,
                "router_logit_scores feature dimension is invalid, expected 4, "
                "received shape (2, 3).",
                lambda: sampler.get_probabilities_and_indices(torch.ones(2, 3)),
            ),
            (
                TypeError,
                "skip_mask must be a Tensor when provided, received list.",
                lambda: sampler.get_probabilities_and_indices(torch.ones(2, 4), [1, 1]),
            ),
            (
                ValueError,
                "skip_mask must be a 2D tensor with shape (batch_size, 1), "
                "received a 1D tensor with shape (2,).",
                lambda: sampler.get_probabilities_and_indices(
                    torch.ones(2, 4), torch.ones(2)
                ),
            ),
            (
                ValueError,
                "skip_mask batch dimension must match router_logit_scores, "
                "received skip_mask shape (3, 1) and "
                "router_logit_scores shape (2, 4).",
                lambda: sampler.get_probabilities_and_indices(
                    torch.ones(2, 4), torch.ones(3, 1)
                ),
            ),
            (
                ValueError,
                "skip_mask feature dimension must be 1 so it broadcasts across "
                "experts, received skip_mask shape (2, 2) for num_experts=4.",
                lambda: sampler.get_probabilities_and_indices(
                    torch.ones(2, 4), torch.ones(2, 2)
                ),
            ),
        )
        for exception_type, message, operation in cases:
            with self.subTest(message=message):
                self.assert_exact_exception(exception_type, message, operation)

        noisy_sampler = SamplerTopk(sampler_config(noisy_topk_flag=True))
        self.assert_exact_exception(
            ValueError,
            "router_logit_scores feature dimension is invalid, expected 8, "
            "received shape (2, 4).",
            lambda: noisy_sampler.get_probabilities_and_indices(torch.ones(2, 4)),
        )

    def test_model_and_router_validation_errors_are_exact(self) -> None:
        self.assert_exact_exception(
            ValueError,
            "top_k must be a positive integer, received 0.",
            lambda: SamplerModel(sampler_config(top_k=0)),
        )
        self.assert_exact_exception(
            ValueError,
            "num_experts must be a positive integer, received 0.",
            lambda: SamplerModel(sampler_config(top_k=1, num_experts=0)),
        )

        invalid_router = object()
        self.assert_exact_exception(
            TypeError,
            "router_config must be a RouterConfig for SamplerModel, got object.",
            lambda: SamplerModel(sampler_config(router_config=invalid_router)),
        )

        mismatched_experts = router_config(num_experts=3)
        self.assert_exact_exception(
            ValueError,
            "router_config.num_experts must match sampler_config.num_experts, "
            "received router_config.num_experts=3 and "
            "sampler_config.num_experts=4.",
            lambda: SamplerModel(
                sampler_config(num_experts=4, router_config=mismatched_experts)
            ),
        )

        mismatched_noise = router_config(num_experts=4, noisy_topk_flag=True)
        self.assert_exact_exception(
            ValueError,
            "router_config.noisy_topk_flag must match "
            "sampler_config.noisy_topk_flag, received "
            "router_config.noisy_topk_flag=True and "
            "sampler_config.noisy_topk_flag=False.",
            lambda: SamplerModel(
                sampler_config(
                    num_experts=4,
                    noisy_topk_flag=False,
                    router_config=mismatched_noise,
                )
            ),
        )

        sampler = SamplerModel(sampler_config())
        self.assert_exact_exception(
            TypeError,
            "input_matrix must be a Tensor, received list.",
            lambda: sampler.sample_probabilities_and_indices([[1.0] * 4]),
        )
        self.assert_exact_exception(
            ValueError,
            "SamplerModel expects a 2D input tensor (batch_size, features), "
            "received a 1D tensor with shape (4,).",
            lambda: sampler.sample_probabilities_and_indices(torch.ones(4)),
        )

        bad_model_config = router_config()
        bad_model_config.model_config = object()
        self.assert_exact_exception(
            TypeError,
            "model_config must be a ConfigBase for RouterConfig, got object.",
            lambda: RouterModel(bad_model_config),
        )
        self.assert_exact_exception(
            ValueError,
            "num_experts must be a positive integer, received 0.",
            lambda: RouterModel(router_config(num_experts=0)),
        )

        router = RouterModel(router_config(input_dim=2))
        self.assert_exact_exception(
            TypeError,
            "RouterModel input_batch must be a Tensor, received list.",
            lambda: router.compute_logit_scores([[1.0, 2.0]]),
        )
        self.assert_exact_exception(
            ValueError,
            "RouterModel expects a 2D input tensor (batch_size, input_dim), "
            "received a 1D tensor with shape (2,).",
            lambda: router.compute_logit_scores(torch.ones(2)),
        )
        self.assert_exact_exception(
            ValueError,
            "RouterModel input feature dimension must match input_dim, "
            "received input_dim=2 and input shape (3, 1).",
            lambda: router.compute_logit_scores(torch.ones(3, 1)),
        )

    def test_router_accepts_minimum_positive_dimensions(self) -> None:
        model = RouterModel(router_config(input_dim=1, num_experts=1))
        output = model.compute_logit_scores(torch.tensor([[2.0], [3.0]]))

        self.assertEqual(output.shape, (2, 1))

    def test_router_requires_an_input_dimension_from_nested_or_outer_config(
        self,
    ) -> None:
        self.assert_exact_exception(
            ValueError,
            "input_dim is required for RouterConfig, received None",
            lambda: RouterModel(router_config(input_dim=None)),
        )
        self.assert_exact_exception(
            ValueError,
            "input_dim is required for RouterConfig, received None",
            lambda: RouterModel(
                WrappedRouterConfigWithoutInputDimension(
                    router_model_config=router_config(input_dim=None)
                )
            ),
        )

    def test_sampler_base_abstract_strategy_has_exact_error(self) -> None:
        sampler = SamplerBase(sampler_config())
        self.assert_exact_exception(
            NotImplementedError,
            "`_sample_probabilities_and_indices` has to be implemented in "
            "classes that inherit `SamplerBase`.",
            lambda: sampler._sample_probabilities_and_indices(torch.ones(2, 4)),
        )


class SamplerConstructionAndRouterTests(unittest.TestCase):
    def test_wrapped_configs_and_partial_overrides_preserve_precedence(self) -> None:
        nested = sampler_config(threshold=0.35, switch_loss_weight=0.2)
        wrapped = WrappedSamplerConfig(sampler_model_config=nested)

        base = SamplerBase(wrapped)
        model = SamplerModel(wrapped)
        losses = SamplerAuxiliaryLosses(wrapped)

        self.assertIs(base.cfg, nested)
        self.assertIs(model.sampler_config, nested)
        self.assertIs(losses.cfg, nested)
        self.assertEqual(base.threshold, 0.35)
        self.assertEqual(model.sampler_model.threshold, 0.35)
        self.assertEqual(losses.switch_loss_weight, 0.2)

        for sampler_type, top_k in (
            (SamplerBase, 2),
            (SamplerSparse, 1),
            (SamplerTopk, 2),
            (SamplerFull, 4),
        ):
            with self.subTest(sampler_type=sampler_type.__name__):
                direct = sampler_type(
                    sampler_config(top_k=top_k),
                    SamplerConfig(threshold=0.6),
                )
                self.assertEqual(direct.threshold, 0.6)

        overridden_losses = SamplerAuxiliaryLosses(
            sampler_config(switch_loss_weight=0.1),
            SamplerConfig(switch_loss_weight=0.75),
        )
        self.assertEqual(overridden_losses.switch_loss_weight, 0.75)
        self.assertEqual(overridden_losses.switch_loss.loss_weight, 0.75)

    def test_sampler_model_passes_overrides_to_every_dispatched_variant(self) -> None:
        cases = (
            (2, 1, SamplerSparse),
            (1, 2, SamplerTopk),
            (2, 4, SamplerFull),
        )
        for base_top_k, override_top_k, expected_type in cases:
            with self.subTest(override_top_k=override_top_k):
                overrides = SamplerConfig(top_k=override_top_k, threshold=0.4)
                model = SamplerModel(
                    sampler_config(top_k=base_top_k),
                    overrides,
                )

                self.assertIs(model.overrides, overrides)
                self.assertIsInstance(model.sampler_model, expected_type)
                self.assertEqual(model.sampler_model.top_k, override_top_k)
                self.assertEqual(model.sampler_model.threshold, 0.4)

    def test_router_wrappers_fallback_and_overrides_are_applied(self) -> None:
        nested = router_config(input_dim=None, num_experts=3)
        wrapped = WrappedRouterConfig(
            router_model_config=nested,
            input_dim=2,
        )
        wrapped_model = RouterModel(wrapped)

        self.assertEqual(wrapped_model.input_dim, 2)
        self.assertEqual(wrapped_model.num_experts, 3)
        self.assertEqual(
            wrapped_model.compute_logit_scores(torch.ones(4, 2)).shape,
            (4, 3),
        )

        overridden = RouterModel(
            router_config(input_dim=2, num_experts=3),
            RouterConfig(input_dim=3, num_experts=2, noisy_topk_flag=True),
        )
        self.assertEqual(overridden.input_dim, 3)
        self.assertEqual(overridden.num_experts, 2)
        self.assertTrue(overridden.noisy_topk_flag)
        self.assertEqual(overridden.router_output_dim, 4)
        self.assertEqual(
            overridden.compute_logit_scores(torch.ones(5, 3)).shape,
            (5, 4),
        )

    def test_router_computes_exact_affine_values_and_backpropagates(self) -> None:
        model = RouterModel(router_config(input_dim=2, num_experts=3)).double()
        linear = model.model.layers[0].model
        weight = torch.tensor(
            [[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]],
            dtype=torch.float64,
        )
        bias = torch.tensor([0.25, -0.5, 1.0], dtype=torch.float64)
        with torch.no_grad():
            linear.weight_params.copy_(weight)
            linear.bias_params.copy_(bias)

        input_matrix = (
            torch.tensor(
                [[2.0, -1.0], [0.0, 3.0]],
                dtype=torch.float64,
            )
            .t()
            .detach()
            .requires_grad_()
        )
        self.assertFalse(input_matrix.is_contiguous())

        output = model.compute_logit_scores(input_matrix)
        expected = input_matrix.detach() @ weight + bias

        self.assertEqual(output.dtype, torch.float64)
        self.assertEqual(output.device.type, "cpu")
        self.assertEqual(output.shape, (2, 3))
        torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            model.compute_logit_scores(input_matrix[:1]),
            output[:1],
            rtol=0.0,
            atol=0.0,
        )

        loss = output.square().sum()
        loss.backward()
        for gradient in (
            input_matrix.grad,
            linear.weight_params.grad,
            linear.bias_params.grad,
        ):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)

        before = linear.weight_params.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        optimizer.step()
        self.assertFalse(torch.equal(before, linear.weight_params.detach()))


class SamplerSelectionMathTests(unittest.TestCase):
    def test_noisy_logits_match_seeded_equation_and_gradients(self) -> None:
        sampler = SamplerBase(
            sampler_config(
                noisy_topk_flag=True,
                num_experts=3,
            )
        )
        sampler.train()
        self.assertEqual(sampler.noise_epsilon, 0.01)

        logits = torch.tensor(
            [[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]],
            dtype=torch.float64,
        )
        raw_noise_scale = torch.tensor(
            [[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]],
            dtype=torch.float64,
        )
        combined = torch.hstack((logits, raw_noise_scale)).requires_grad_()

        with torch.random.fork_rng():
            torch.manual_seed(117)
            expected_noise = torch.randn_like(logits)
            torch.manual_seed(117)
            output = sampler._SamplerBase__add_noise_to_logits(combined)

        expected_scale = torch.sigmoid(raw_noise_scale) + 0.01
        expected = logits + expected_noise * expected_scale
        torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)

        output.sum().backward()
        torch.testing.assert_close(
            combined.grad[:, :3],
            torch.ones_like(logits),
            rtol=0.0,
            atol=0.0,
        )
        expected_scale_gradient = (
            expected_noise
            * torch.sigmoid(raw_noise_scale)
            * (1.0 - torch.sigmoid(raw_noise_scale))
        )
        torch.testing.assert_close(
            combined.grad[:, 3:],
            expected_scale_gradient,
            rtol=1e-14,
            atol=1e-14,
        )

    def test_normalization_uses_exact_stabilizer_and_detached_denominator(self) -> None:
        sampler = SamplerBase(sampler_config(normalize_probabilities_flag=True))
        probabilities = torch.tensor(
            [[0.2, 0.3], [1.0, 3.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        weights = torch.tensor(
            [[1.0, 2.0], [3.0, 5.0]],
            dtype=torch.float64,
        )

        output = sampler._normalize_probabilities(probabilities)
        denominator = probabilities.detach().sum(dim=1, keepdim=True) + 1e-6
        expected = probabilities.detach() / denominator

        torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)
        (output * weights).sum().backward()
        torch.testing.assert_close(
            probabilities.grad,
            weights / denominator,
            rtol=0.0,
            atol=0.0,
        )

    def test_threshold_equality_obeys_both_filter_modes(self) -> None:
        logits = torch.zeros(2, 2, dtype=torch.float64)
        skip_mask = torch.ones(2, 1, dtype=torch.float64)

        for filter_above_threshold, expected_mask_value in ((False, 1.0), (True, 0.0)):
            with self.subTest(filter_above_threshold=filter_above_threshold):
                sampler = SamplerFull(
                    sampler_config(
                        top_k=2,
                        num_experts=2,
                        threshold=0.5,
                        filter_above_threshold=filter_above_threshold,
                    )
                )
                probabilities, indices, updated_mask, loss = (
                    sampler.get_probabilities_and_indices(logits, skip_mask.clone())
                )

                torch.testing.assert_close(
                    probabilities,
                    torch.full((2, 2), 0.5, dtype=torch.float64),
                    rtol=0.0,
                    atol=0.0,
                )
                self.assertIsNone(indices)
                torch.testing.assert_close(
                    updated_mask,
                    torch.full((2, 1), expected_mask_value, dtype=torch.float64),
                    rtol=0.0,
                    atol=0.0,
                )
                torch.testing.assert_close(loss, loss.new_zeros(()))

    def test_full_threshold_zero_is_a_true_noop(self) -> None:
        sampler = SamplerFull(
            sampler_config(
                top_k=2,
                num_experts=2,
                threshold=0.0,
                normalize_probabilities_flag=True,
            )
        )
        probabilities = torch.tensor([[0.2, 0.3]], dtype=torch.float64)

        output = sampler._SamplerFull__apply_dynamic_topk_threshold_mask(probabilities)

        self.assertIs(output, probabilities)

    def test_skip_mask_contract_across_sparse_topk_and_full_samplers(self) -> None:
        logits = torch.tensor(
            [
                [4.0, 3.0, 2.0, 1.0],
                [1.0, 2.0, 3.0, 4.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        sampler_cases = (
            (SamplerSparse, 1),
            (SamplerTopk, 2),
            (SamplerFull, 4),
        )

        for sampler_type, top_k in sampler_cases:
            with self.subTest(sampler=sampler_type.__name__, mask=None):
                sampler = sampler_type(
                    sampler_config(top_k=top_k, num_experts=4, threshold=0.0)
                ).double()
                _, _, returned_mask, loss = sampler.get_probabilities_and_indices(
                    logits
                )
                self.assertIsNone(returned_mask)
                self.assertTrue(torch.isfinite(loss))

            for dtype in (torch.bool, torch.float64):
                for mask_values in ((1, 1, 1), (1, 0, 1)):
                    with self.subTest(
                        sampler=sampler_type.__name__,
                        dtype=dtype,
                        mask_values=mask_values,
                    ):
                        sampler = sampler_type(
                            sampler_config(
                                top_k=top_k,
                                num_experts=4,
                                threshold=0.0,
                            )
                        ).double()
                        skip_mask = torch.tensor(
                            mask_values,
                            dtype=dtype,
                            device=logits.device,
                        ).unsqueeze(1)
                        baseline, baseline_indices, _, baseline_loss = (
                            sampler.get_probabilities_and_indices(logits)
                        )
                        probabilities, indices, returned_mask, loss = (
                            sampler.get_probabilities_and_indices(logits, skip_mask)
                        )

                        torch.testing.assert_close(probabilities, baseline)
                        if indices is None:
                            self.assertIsNone(baseline_indices)
                        else:
                            torch.testing.assert_close(indices, baseline_indices)
                        torch.testing.assert_close(loss, baseline_loss)
                        self.assertIs(returned_mask, skip_mask)
                        self.assertEqual(returned_mask.dtype, dtype)
                        self.assertEqual(returned_mask.device, logits.device)
                        torch.testing.assert_close(returned_mask, skip_mask)

    def test_positive_threshold_updates_masks_in_both_filter_modes(self) -> None:
        logits = torch.tensor(
            [
                [4.0, 3.0, 2.0, 1.0],
                [1.0, 2.0, 3.0, 4.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        sampler_cases = (
            (SamplerSparse, 1),
            (SamplerTopk, 2),
            (SamplerFull, 4),
        )
        expected_by_filter_mode = {
            False: torch.tensor([[1], [0], [1]], dtype=torch.bool),
            True: torch.zeros(3, 1, dtype=torch.bool),
        }

        for sampler_type, top_k in sampler_cases:
            for (
                filter_above_threshold,
                expected_mask,
            ) in expected_by_filter_mode.items():
                for dtype in (torch.bool, torch.float64):
                    with self.subTest(
                        sampler=sampler_type.__name__,
                        filter_above_threshold=filter_above_threshold,
                        dtype=dtype,
                    ):
                        sampler = sampler_type(
                            sampler_config(
                                top_k=top_k,
                                num_experts=4,
                                threshold=0.2,
                                filter_above_threshold=filter_above_threshold,
                            )
                        ).double()
                        skip_mask = torch.tensor(
                            [[1], [0], [1]],
                            dtype=dtype,
                            device=logits.device,
                        )
                        probabilities, _, returned_mask, loss = (
                            sampler.get_probabilities_and_indices(logits, skip_mask)
                        )

                        self.assertTrue(torch.isfinite(probabilities).all())
                        self.assertTrue(torch.isfinite(loss))
                        self.assertEqual(returned_mask.dtype, dtype)
                        self.assertEqual(returned_mask.device, logits.device)
                        torch.testing.assert_close(
                            returned_mask,
                            expected_mask.to(dtype=dtype, device=logits.device),
                        )
                        if top_k == 1:
                            torch.testing.assert_close(
                                probabilities[1], probabilities.new_zeros(())
                            )
                        else:
                            torch.testing.assert_close(
                                probabilities[1],
                                probabilities.new_zeros(probabilities.shape[1]),
                            )

    def test_masked_topk_auxiliary_loss_is_finite_and_ignores_zero_rows(
        self,
    ) -> None:
        config = sampler_config(
            top_k=2,
            num_experts=4,
            threshold=0.1,
            mutual_information_loss_weight=1.0,
        )
        logits = torch.tensor(
            [
                [1.0, 0.5, -0.5, -1.0],
                [-0.5, 1.0, 0.25, -1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        skip_mask = torch.tensor([[1.0], [0.5], [0.0]], dtype=torch.float64)
        sampler = SamplerTopk(config).double()
        _, _, returned_mask, masked_loss = sampler.get_probabilities_and_indices(
            logits, skip_mask
        )
        active_sampler = SamplerTopk(config).double()
        _, _, _, active_loss = active_sampler.get_probabilities_and_indices(
            logits[:2], skip_mask[:2]
        )

        self.assertTrue(torch.isfinite(masked_loss))
        torch.testing.assert_close(masked_loss, active_loss)
        self.assertEqual(returned_mask.dtype, skip_mask.dtype)
        masked_loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad[:2].abs().sum().item(), 0.0)

    def test_sparse_auxiliary_loss_matches_hand_calculated_equations(self) -> None:
        sampler = SamplerSparse(
            sampler_config(
                top_k=1,
                num_experts=3,
                coefficient_of_variation_loss_weight=0.3,
                switch_loss_weight=0.5,
                zero_centred_loss_weight=0.7,
            )
        ).double()
        logits = torch.tensor(
            [[0.0, math.log(2.0), math.log(3.0)], [math.log(4.0), 0.0, math.log(2.0)]],
            dtype=torch.float64,
            requires_grad=True,
        )
        full_probabilities = torch.tensor(
            [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]],
            dtype=torch.float64,
            requires_grad=True,
        )
        sampled_probabilities = torch.tensor(
            [0.6, 0.5],
            dtype=torch.float64,
            requires_grad=True,
        )
        indices = torch.tensor([0, 1])

        output = sampler._compute_loss(
            logits,
            full_probabilities,
            sampled_probabilities,
            indices,
            torch.ones(2, 1, dtype=torch.float64),
        )

        gates = torch.tensor(
            [[0.6, 0.0, 0.0], [0.0, 0.5, 0.0]],
            dtype=torch.float64,
        )
        gate_mass = gates.sum(dim=0)
        normalized_gate_mass = gate_mass / gate_mass.abs().sum()
        cv = normalized_gate_mass.float().var() / (
            normalized_gate_mass.float().mean().square() + 1e-10
        )
        probability_mass = full_probabilities.detach().sum(dim=0)
        frequency = (gates > 0).float().sum(dim=0)
        switch = (
            3
            * (
                probability_mass
                / probability_mass.abs().sum()
                * (frequency / frequency.abs().sum())
            ).sum()
        )
        zero_centred = torch.logsumexp(logits.detach(), dim=1).square().sum() / 2
        expected = 0.3 * cv + 0.5 * switch + 0.7 * zero_centred

        torch.testing.assert_close(output, expected, rtol=1e-7, atol=1e-7)
        output.backward()
        for gradient in (
            logits.grad,
            full_probabilities.grad,
            sampled_probabilities.grad,
        ):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_random_topk_preserves_deterministic_choice_and_weighted_sampling(
        self,
    ) -> None:
        sampler = SamplerTopk(
            sampler_config(
                top_k=2,
                num_experts=4,
                num_topk_samples=1,
            )
        )
        sampler.train()
        row = torch.tensor([0.9, 0.099, 0.001, 0.0])
        probabilities = row.repeat(2048, 1)

        with torch.random.fork_rng():
            torch.manual_seed(19)
            selected_probabilities, selected_indices = (
                sampler._sample_probabilities_and_indices(probabilities)
            )

        self.assertTrue(torch.all(selected_indices[:, 0] == 0))
        fraction_second_expert = (selected_indices[:, 1] == 1).float().mean()
        self.assertGreater(float(fraction_second_expert), 0.97)
        torch.testing.assert_close(
            selected_probabilities,
            torch.gather(probabilities, 1, selected_indices),
        )

    def test_random_topk_allows_every_selected_slot_to_be_sampled(self) -> None:
        sampler = SamplerTopk(
            sampler_config(
                top_k=2,
                num_experts=3,
                num_topk_samples=2,
            )
        )
        sampler.train()
        probabilities = torch.tensor(
            [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]],
            dtype=torch.float64,
        )

        with torch.random.fork_rng():
            torch.manual_seed(5)
            selected_probabilities, selected_indices = (
                sampler._sample_probabilities_and_indices(probabilities)
            )

        self.assertEqual(selected_indices.shape, (2, 2))
        self.assertTrue(torch.all(selected_indices[:, 0] != selected_indices[:, 1]))
        torch.testing.assert_close(
            selected_probabilities,
            torch.gather(probabilities, 1, selected_indices),
        )

    def test_empty_batch_preserves_shapes_dtype_and_scalar_loss(self) -> None:
        sampler = SamplerTopk(sampler_config()).double()

        probabilities, indices, skip_mask, loss = sampler.get_probabilities_and_indices(
            torch.empty(0, 4, dtype=torch.float64)
        )

        self.assertEqual(probabilities.shape, (0, 2))
        self.assertEqual(indices.shape, (0, 2))
        self.assertEqual(probabilities.dtype, torch.float64)
        self.assertIsNone(skip_mask)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(loss))


class SamplerAuxiliaryLossContractTests(ExactExceptionMixin, unittest.TestCase):
    def test_all_loss_defaults_are_disabled(self) -> None:
        losses = (
            AuxiliaryLossBase(),
            CoefficientOfVariationLoss(),
            SwitchLoss(num_experts=3),
            ZeroCentredLoss(),
            MutualInformationLoss(),
        )

        for loss in losses:
            with self.subTest(loss_type=type(loss).__name__):
                self.assertEqual(loss.loss_weight, 0.0)

    def test_missing_inputs_and_accumulations_have_exact_errors(self) -> None:
        accumulation_message = (
            "`self.accumulation` is `None`. Please call "
            "`update_accumulation` before validating accumulation."
        )
        list_message = (
            "`self.accumulation_list` is `empty`. Please call "
            "`update_accumulation` before validating accumulation."
        )

        cv = CoefficientOfVariationLoss(1.0)
        self.assert_exact_exception(
            ValueError,
            "A valid input tensor is required when `loss_weight` > 0, "
            "for CoefficientOfVariationLoss instance.",
            lambda: cv.update_accumulation(None),
        )
        self.assert_exact_exception(
            ValueError,
            accumulation_message,
            cv._compute_loss,
        )

        switch = SwitchLoss(3, 1.0)
        self.assert_exact_exception(
            ValueError,
            "A valid input tensor is required when `loss_weight` > 0, "
            "for SwitchLoss instance.",
            lambda: switch.update_accumulation(None, torch.ones(1, 3)),
        )
        self.assert_exact_exception(
            ValueError,
            "A valid input tensor is required when `loss_weight` > 0, "
            "for SwitchLoss instance.",
            lambda: switch.update_accumulation(torch.ones(1, 3), None),
        )
        self.assert_exact_exception(
            ValueError,
            accumulation_message,
            switch._compute_loss,
        )

        zero = ZeroCentredLoss(1.0)
        self.assert_exact_exception(
            ValueError,
            "A valid input tensor is required when `loss_weight` > 0, "
            "for ZeroCentredLoss instance.",
            lambda: zero.update_accumulation(None),
        )
        self.assert_exact_exception(
            ValueError,
            accumulation_message,
            zero._compute_loss,
        )

        mutual = MutualInformationLoss(1.0)
        self.assert_exact_exception(
            ValueError,
            "A valid input tensor is required when `loss_weight` > 0, "
            "for MutualInformationLoss instance.",
            lambda: mutual.update_accumulation(
                None,
                torch.ones(1, 3),
                torch.ones(1, 1),
            ),
        )
        self.assert_exact_exception(
            ValueError,
            "A valid input tensor is required when `loss_weight` > 0, "
            "for MutualInformationLoss instance.",
            lambda: mutual.update_accumulation(
                torch.ones(1, 3),
                None,
                torch.ones(1, 1),
            ),
        )
        self.assert_exact_exception(
            ValueError,
            "A valid input tensor is required when `loss_weight` > 0, "
            "for MutualInformationLoss instance.",
            lambda: mutual.update_accumulation(
                torch.ones(1, 3),
                torch.ones(1, 3),
                None,
            ),
        )
        self.assert_exact_exception(
            ValueError,
            list_message,
            mutual._compute_loss,
        )

    def test_cv_matches_l1_equation_for_nonuniform_gates(self) -> None:
        loss = CoefficientOfVariationLoss(1.0)
        gates = torch.tensor(
            [[1.0, 0.0, 3.0], [0.0, 4.0, 4.0]],
            dtype=torch.float64,
        )
        loss.update_accumulation(gates)

        output = loss._compute_loss()
        accumulation = gates.sum(dim=0)
        probabilities = accumulation / accumulation.abs().sum()
        expected = probabilities.float().var() / (
            probabilities.float().mean().square() + 1e-10
        )

        torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)

    def test_cv_is_scale_invariant_below_unit_accumulated_mass(self) -> None:
        reference = CoefficientOfVariationLoss(1.0)
        scaled = CoefficientOfVariationLoss(1.0)
        gates = torch.tensor(
            [[1.0, 2.0, 0.0]],
            dtype=torch.float64,
        )
        reference.update_accumulation(gates)
        scaled.update_accumulation(gates * 1e-6)

        torch.testing.assert_close(
            scaled._compute_loss(),
            reference._compute_loss(),
            rtol=0.0,
            atol=0.0,
        )

    def test_cv_epsilon_stabilizes_a_large_sparse_expert_bank(self) -> None:
        num_experts = 100_000
        gates = torch.zeros(1, num_experts)
        gates[0, 0] = 1.0
        loss = CoefficientOfVariationLoss(1.0)
        loss.update_accumulation(gates)

        output = loss._compute_loss()
        probabilities = gates.sum(dim=0)
        expected = probabilities.var() / (probabilities.mean().square() + 1e-10)

        self.assertTrue(torch.isfinite(output))
        torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)

    def test_switch_loss_matches_nonuniform_probability_and_frequency_equation(
        self,
    ) -> None:
        probabilities = torch.tensor(
            [[0.7, 0.2, 0.1], [0.5, 0.1, 0.4]],
            dtype=torch.float64,
        )
        gates = torch.tensor(
            [[0.7, 0.0, 0.0], [0.5, 0.0, 0.4]],
            dtype=torch.float64,
        )
        loss = SwitchLoss(3, 1.0)
        loss.update_accumulation(probabilities, gates)

        output = loss._compute_loss()
        p = probabilities.sum(dim=0)
        f = (gates > 0).float().sum(dim=0)
        expected = 3 * (p / p.abs().sum() * (f / f.abs().sum())).sum()

        torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)

    def test_switch_loss_preserves_single_row_unit_mass_exactly(self) -> None:
        loss = SwitchLoss(3, 1.0)
        probabilities = torch.tensor(
            [[0.5, 0.25, 0.25]],
            dtype=torch.float64,
        )
        gates = torch.tensor(
            [[1.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        loss.update_accumulation(probabilities, gates)

        torch.testing.assert_close(
            loss._compute_loss(),
            torch.tensor(1.5, dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )

    def test_zero_centred_count_tracks_the_input_device(self) -> None:
        loss = ZeroCentredLoss(1.0)
        logits = torch.empty(2, 3, device="meta")

        loss.update_accumulation(logits)

        self.assertEqual(loss.count_accumulation.device.type, "meta")
        self.assertEqual(loss.count_accumulation.dtype, torch.long)
        self.assertEqual(
            loss.squared_log_sum_exp_accumulation.device.type,
            "meta",
        )

    def test_mutual_information_combines_rectangular_updates_exactly(self) -> None:
        first_logits = torch.tensor(
            [[2.0, 0.0, -1.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        second_logits = torch.tensor(
            [[0.0, 1.0, -2.0], [1.5, -0.5, 0.25]],
            dtype=torch.float64,
            requires_grad=True,
        )
        first_probabilities = torch.tensor(
            [[0.7, 0.2, 0.1]],
            dtype=torch.float64,
            requires_grad=True,
        )
        second_probabilities = torch.tensor(
            [[0.2, 0.6, 0.2], [0.5, 0.1, 0.4]],
            dtype=torch.float64,
            requires_grad=True,
        )
        first_mask = torch.tensor([[1.0]], dtype=torch.float64)
        second_mask = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        loss = MutualInformationLoss(1.0)

        loss.update_accumulation(
            first_logits,
            first_probabilities,
            first_mask,
        )
        loss.update_accumulation(
            second_logits,
            second_probabilities,
            second_mask,
        )
        output = loss._compute_loss()

        probabilities = torch.vstack(
            (first_probabilities.detach(), second_probabilities.detach())
        )
        log_probabilities = torch.vstack(
            (
                torch.log_softmax(first_logits.detach(), dim=1),
                torch.log_softmax(second_logits.detach(), dim=1),
            )
        )
        masks = torch.vstack((first_mask, second_mask))
        p_x = masks / (masks.sum() + 1e-12)
        p_e = (p_x * probabilities).sum(dim=0)
        positive_p_e = p_e[p_e > 0]
        expected = -(
            (p_x * probabilities * log_probabilities).sum()
            + (positive_p_e * positive_p_e.log()).sum()
        )

        torch.testing.assert_close(output, expected, rtol=0.0, atol=1e-15)
        output.backward()
        for gradient in (
            first_logits.grad,
            second_logits.grad,
            first_probabilities.grad,
            second_probabilities.grad,
        ):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)


class SamplerMonitoringContractTests(ExactExceptionMixin, unittest.TestCase):
    def test_callback_defaults_and_exact_option_errors(self) -> None:
        callback = SamplerMonitorCallback()
        self.assertEqual(callback.log_every_n_steps, 100)
        self.assertEqual(callback.history_size, 128)
        self.assertFalse(callback.log_per_expert_scalars)
        self.assertIsNone(callback._tracker_manager)

        cases = (
            (
                {"log_every_n_steps": 1.5},
                TypeError,
                "log_every_n_steps must be an integer, received float.",
            ),
            (
                {"history_size": 1.5},
                TypeError,
                "history_size must be an integer, received float.",
            ),
            (
                {"log_every_n_steps": 0},
                ValueError,
                "log_every_n_steps must be a positive integer, received 0.",
            ),
            (
                {"history_size": 0},
                ValueError,
                "history_size must be a positive integer, received 0.",
            ),
        )
        for kwargs, exception_type, message in cases:
            with self.subTest(kwargs=kwargs):
                self.assert_exact_exception(
                    exception_type,
                    message,
                    lambda kwargs=kwargs: SamplerMonitorCallback(**kwargs),
                )

    def test_usage_diagnostics_honor_small_totals(self) -> None:
        metrics = _SamplerDiagnostics.calculate_usage(
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.2, 0.3]),
        )

        torch.testing.assert_close(
            metrics.usage_fraction,
            torch.tensor([1.0, 0.0]),
        )
        torch.testing.assert_close(
            metrics.mass_fraction,
            torch.tensor([0.4, 0.6]),
        )

    def test_tracking_context_uses_missing_logger_and_step_defaults(self) -> None:
        sampler = SamplerModel(sampler_config())
        metrics = _SamplerDiagnostics.calculate_usage(
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            torch.tensor([0.6, 0.4, 0.0, 0.0]),
        )
        build_context = (
            SamplerMonitorCallback._SamplerMonitorCallback__build_tracking_context
        )

        context = build_context(
            object(),
            "sampler",
            sampler,
            metrics,
            metrics,
        )

        self.assertIsNone(context.experiment)
        self.assertEqual(context.global_step, 0)

    def test_missing_tracker_on_first_sampler_does_not_hide_later_samplers(
        self,
    ) -> None:
        first = SamplerModel(sampler_config(num_experts=3))
        second = SamplerModel(sampler_config(num_experts=3))
        module = _RecordingLightningModule(first=first, second=second)
        callback = SamplerMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(None, module)
        SamplerUsageTrackerManager().detach(first)
        second.usage_tracker.last_expert_usage_counts.copy_(
            torch.tensor([1.0, 1.0, 0.0])
        )
        second.usage_tracker.last_expert_usage_mass.copy_(torch.tensor([0.6, 0.4, 0.0]))

        callback.on_train_batch_end(None, module, None, None, 0)

        self.assertTrue(
            any(
                name == "second/batch/active_experts"
                for name, _ in module.logged_scalars
            )
        )
        callback.on_fit_end(None, module)

    def test_per_expert_batch_and_cumulative_values_are_exact(self) -> None:
        sampler = SamplerModel(sampler_config(num_experts=3))
        module = _RecordingLightningModule(sampler=sampler)
        callback = SamplerMonitorCallback(
            log_every_n_steps=1,
            log_per_expert_scalars=True,
        )
        callback.on_fit_start(None, module)
        sampler.usage_tracker.last_expert_usage_counts.copy_(
            torch.tensor([1.0, 1.0, 0.0])
        )
        sampler.usage_tracker.last_expert_usage_mass.copy_(
            torch.tensor([0.1, 0.6, 0.3])
        )
        sampler.usage_tracker.cumulative_expert_usage_counts.copy_(
            torch.tensor([1.0, 3.0, 0.0])
        )
        sampler.usage_tracker.cumulative_expert_usage_mass.copy_(
            torch.tensor([0.2, 0.3, 0.5])
        )

        callback.on_train_batch_end(None, module, None, None, 0)
        logged = dict(module.logged_scalars)

        expected = {
            "sampler/batch/expert_0/usage_fraction": 0.5,
            "sampler/batch/expert_1/usage_fraction": 0.5,
            "sampler/batch/expert_2/usage_fraction": 0.0,
            "sampler/batch/expert_0/probability_mass": 0.1,
            "sampler/batch/expert_1/probability_mass": 0.6,
            "sampler/batch/expert_2/probability_mass": 0.3,
            "sampler/cumulative/expert_0/usage_fraction": 0.25,
            "sampler/cumulative/expert_1/usage_fraction": 0.75,
            "sampler/cumulative/expert_2/usage_fraction": 0.0,
            "sampler/cumulative/expert_0/probability_mass": 0.2,
            "sampler/cumulative/expert_1/probability_mass": 0.3,
            "sampler/cumulative/expert_2/probability_mass": 0.5,
        }
        for name, value in expected.items():
            with self.subTest(name=name):
                torch.testing.assert_close(
                    logged[name],
                    torch.tensor(value),
                )

        callback.on_fit_end(None, module)


if __name__ == "__main__":
    unittest.main()
