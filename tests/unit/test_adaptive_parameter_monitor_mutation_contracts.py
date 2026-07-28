from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from emperor.augmentations.adaptive_parameters import (
    AdditiveDynamicBiasConfig,
    BankExpansionFactorOptions,
    DynamicDepthOptions,
    MaskDimensionOptions,
    MultiplicativeDynamicBiasConfig,
    PerAxisScoreMaskConfig,
    SoftWeightedBankDynamicWeightConfig,
    WeightDecayScheduleOptions,
)
from emperor.augmentations.adaptive_parameters._biases.variants.additive import (
    AdditiveDynamicBias,
)
from emperor.augmentations.adaptive_parameters._biases.variants.multiplicative import (
    MultiplicativeDynamicBias,
)
from emperor.augmentations.adaptive_parameters._masks.variants.per_axis import (
    PerAxisScoreMask,
)
from emperor.augmentations.adaptive_parameters._monitoring.adaptive_parameters import (
    AdaptiveParameterMonitorCallback,
    _AdaptiveParameterDiagnosticFacts,
    _AdaptiveParameterDiagnostics,
    _AdaptiveParameterObservation,
)
from emperor.augmentations.adaptive_parameters._weights.variants.soft_weighted_bank import (  # noqa: E501
    SoftWeightedBankDynamicWeight,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig


def generator_config(
    input_dim: int = 2,
    output_dim: int = 2,
) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=max(input_dim, output_dim),
        output_dim=output_dim,
        num_layers=1,
        apply_output_pipeline_flag=False,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        shared_gate_config=None,
        shared_halting_config=None,
        shared_memory_config=None,
        layer_config=LayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=ActivationOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=True,
            ),
        ),
    )


def soft_weight_bank() -> SoftWeightedBankDynamicWeight:
    return SoftWeightedBankDynamicWeight(
        SoftWeightedBankDynamicWeightConfig(
            input_dim=2,
            output_dim=2,
            generator_depth=DynamicDepthOptions.DEPTH_OF_ONE,
            decay_schedule=WeightDecayScheduleOptions.DISABLED,
            decay_rate=0.0,
            decay_warmup_batches=0,
            bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_TWO,
            model_config=generator_config(),
        )
    )


def additive_bias() -> AdditiveDynamicBias:
    return AdditiveDynamicBias(
        AdditiveDynamicBiasConfig(
            input_dim=2,
            output_dim=2,
            decay_schedule=WeightDecayScheduleOptions.DISABLED,
            decay_rate=0.0,
            decay_warmup_batches=0,
            model_config=generator_config(),
        )
    )


def multiplicative_bias() -> MultiplicativeDynamicBias:
    return MultiplicativeDynamicBias(
        MultiplicativeDynamicBiasConfig(
            input_dim=2,
            output_dim=2,
            decay_schedule=WeightDecayScheduleOptions.DISABLED,
            decay_rate=0.0,
            decay_warmup_batches=0,
            model_config=generator_config(),
        )
    )


def per_axis_mask() -> PerAxisScoreMask:
    return PerAxisScoreMask(
        PerAxisScoreMaskConfig(
            input_dim=2,
            output_dim=2,
            mask_dimension_option=MaskDimensionOptions.ROW,
            mask_threshold=0.5,
            mask_surrogate_scale=1.0,
            mask_floor=0.0,
            model_config=generator_config(),
        )
    )


def collect_facts(
    slot: str,
    option: torch.nn.Module,
    observation: _AdaptiveParameterObservation,
    *,
    include_internal_stats: bool = True,
    include_histograms: bool = False,
) -> _AdaptiveParameterDiagnosticFacts:
    return _AdaptiveParameterDiagnostics().collect(
        slot,  # type: ignore[arg-type]
        option,
        observation,
        include_internal_stats=include_internal_stats,
        include_histograms=include_histograms,
    )


def scalar_map(facts: _AdaptiveParameterDiagnosticFacts) -> dict[str, torch.Tensor]:
    return {metric.suffix: metric.value for metric in facts.scalars}


class AdaptiveParameterMonitorMutationContractTests(unittest.TestCase):
    def test_defaults_and_validation_message_are_exact(self) -> None:
        callback = AdaptiveParameterMonitorCallback()
        self.assertEqual(callback.log_every_n_steps, 100)
        self.assertFalse(callback.log_histograms)
        self.assertTrue(callback.log_internal_stats)
        self.assertEqual(callback.state_key, "AdaptiveParameterMonitorCallback")
        self.assertEqual(callback.state_dict(), {})

        with self.assertRaisesRegex(
            ValueError,
            r"^log_every_n_steps must be greater than 0\.$",
        ):
            AdaptiveParameterMonitorCallback(log_every_n_steps=0)

    def test_weight_facts_preserve_exact_common_adaptivity_and_internal_values(
        self,
    ) -> None:
        option = soft_weight_bank()
        bank_values = torch.tensor(
            [
                [
                    [1.0, -2.0],
                    [3.0, 0.5],
                    [-1.0, 4.0],
                    [2.0, -0.5],
                ]
            ]
        )
        with torch.no_grad():
            option.weight_bank.copy_(bank_values)
            option.decay_step.fill_(2.0)
            option.warmup_step.fill_(3.0)
            option.scale.fill_(4.0)
            option.clamp_limit.fill_(5.0)
        base = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        requested_delta = torch.tensor(
            [
                [[0.10, 0.00], [-0.05, 0.20]],
                [[0.00, 0.15], [0.05, -0.10]],
            ]
        )
        output = base + requested_delta
        observation = _AdaptiveParameterObservation.from_forward((base,), output)
        facts = collect_facts("weight", option, observation)
        actual = scalar_map(facts)

        delta = observation.delta
        self.assertIsNotNone(delta)
        common_expected = {
            "output_mean": output.float().mean(),
            "output_var": output.float().var(unbiased=False),
            "output_min": output.float().min(),
            "output_max": output.float().max(),
            "output_l2_norm": output.float().norm(),
            "output_max_abs": output.float().abs().max(),
            "base_mean": base.float().mean(),
            "base_var": base.float().var(unbiased=False),
            "delta_mean": delta.float().mean(),
            "delta_var": delta.float().var(unbiased=False),
            "delta_l2_norm": delta.float().norm(),
            "relative_delta_norm": (
                delta.float().norm() / base.float().norm().clamp_min(1.0e-6)
            ),
        }
        per_sample = delta.float().reshape(2, -1)
        centroid = per_sample.mean(dim=0)
        centered = per_sample - centroid
        adaptivity_expected = {
            "cross_sample_std": centered.pow(2).mean().sqrt(),
            "adaptivity_ratio": (
                centered.norm() / per_sample.norm().clamp_min(1.0e-12)
            ),
            "centroid_cosine_mean": (
                F.normalize(per_sample, dim=1) @ F.normalize(centroid, dim=0)
            ).mean(),
        }
        internal_expected = {
            "decay_step": torch.tensor(2.0),
            "warmup_step": torch.tensor(3.0),
            "scale": torch.tensor(4.0),
            "clamp_limit": torch.tensor(5.0),
            "weight_bank_mean": bank_values.mean(),
            "weight_bank_var": bank_values.var(unbiased=False),
            "weight_bank_l2_norm": bank_values.norm(),
        }
        for suffix, expected in {
            **common_expected,
            **adaptivity_expected,
            **internal_expected,
        }.items():
            with self.subTest(suffix=suffix):
                torch.testing.assert_close(actual[suffix], expected)

        self.assertEqual(
            [metric.suffix for metric in facts.scalars],
            [
                *common_expected,
                *adaptivity_expected,
                *internal_expected,
            ],
        )
        for forbidden_suffix in (
            "relative_output_norm",
            "attenuated_fraction",
            "near_zero_fraction",
        ):
            self.assertNotIn(forbidden_suffix, actual)

    def test_adaptivity_handles_scalar_rank_one_and_two_sample_boundary(self) -> None:
        option = soft_weight_bank()
        scalar = _AdaptiveParameterObservation(
            output=torch.tensor(2.0),
            base=None,
            delta=None,
        )
        scalar_facts = collect_facts(
            "weight",
            option,
            scalar,
            include_internal_stats=False,
        )
        self.assertNotIn("cross_sample_std", scalar_map(scalar_facts))

        rank_one_values = torch.tensor([0.1, 0.3])
        rank_one = _AdaptiveParameterObservation(
            output=rank_one_values,
            base=None,
            delta=None,
        )
        actual = scalar_map(
            collect_facts(
                "weight",
                option,
                rank_one,
                include_internal_stats=False,
            )
        )
        samples = rank_one_values.reshape(2, 1)
        centroid = samples.mean(dim=0)
        centered = samples - centroid
        torch.testing.assert_close(
            actual["cross_sample_std"],
            centered.pow(2).mean().sqrt(),
        )
        torch.testing.assert_close(
            actual["adaptivity_ratio"],
            centered.norm() / samples.norm().clamp_min(1.0e-12),
        )
        torch.testing.assert_close(
            actual["centroid_cosine_mean"],
            (F.normalize(samples, dim=1) @ F.normalize(centroid, dim=0)).mean(),
        )

    def test_effective_bias_scale_requires_every_guard_and_includes_boundary(
        self,
    ) -> None:
        base = torch.tensor([0.25, -0.5])
        output = torch.tensor([0.5, 1.5])
        observation = _AdaptiveParameterObservation.from_forward((base,), output)
        multiplicative = multiplicative_bias()

        actual = scalar_map(collect_facts("bias", multiplicative, observation))
        torch.testing.assert_close(actual["effective_scale_mean"], (output / base).mean())
        torch.testing.assert_close(
            actual["effective_scale_var"],
            (output / base).var(unbiased=False),
        )
        self.assertNotIn(
            "effective_scale_mean",
            scalar_map(collect_facts("weight", multiplicative, observation)),
        )
        self.assertNotIn(
            "effective_scale_mean",
            scalar_map(collect_facts("bias", additive_bias(), observation)),
        )

        threshold_base = torch.tensor([1.0e-6, 0.5])
        threshold_observation = _AdaptiveParameterObservation.from_forward(
            (threshold_base,),
            torch.ones(2),
        )
        self.assertNotIn(
            "effective_scale_mean",
            scalar_map(
                collect_facts("bias", multiplicative, threshold_observation)
            ),
        )

    def test_bias_facts_preserve_effective_scale_without_weight_internals(
        self,
    ) -> None:
        option = multiplicative_bias()
        with torch.no_grad():
            option.decay_step.fill_(8.0)
            option.warmup_step.fill_(9.0)
        base = torch.tensor([0.25, -0.5])
        output = torch.tensor([[0.5, 1.5], [1.0, -0.25]])
        observation = _AdaptiveParameterObservation.from_forward((base,), output)
        actual = scalar_map(collect_facts("bias", option, observation))

        effective_scale = output / base
        torch.testing.assert_close(
            actual["effective_scale_mean"],
            effective_scale.mean(),
        )
        torch.testing.assert_close(
            actual["effective_scale_var"],
            effective_scale.var(unbiased=False),
        )
        for forbidden in (
            "decay_step",
            "warmup_step",
            "scale",
            "clamp_limit",
        ):
            self.assertNotIn(forbidden, actual)

    def test_mask_metrics_use_exact_thresholds_and_respect_disabled_stats(
        self,
    ) -> None:
        base = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        output = torch.tensor([[0.1, 1.0e-6], [0.15, 0.8]])
        observation = _AdaptiveParameterObservation.from_forward((base,), output)
        option = per_axis_mask()
        actual = scalar_map(collect_facts("mask", option, observation))

        torch.testing.assert_close(
            actual["relative_output_norm"],
            output.norm() / base.norm().clamp_min(1.0e-6),
        )
        torch.testing.assert_close(
            actual["attenuated_fraction"],
            torch.tensor(0.5),
        )
        torch.testing.assert_close(
            actual["near_zero_fraction"],
            torch.tensor(0.25),
        )

        disabled = scalar_map(
            collect_facts(
                "mask",
                option,
                observation,
                include_internal_stats=False,
            )
        )
        for suffix in (
            "relative_output_norm",
            "attenuated_fraction",
            "near_zero_fraction",
        ):
            self.assertNotIn(suffix, disabled)

    def test_histogram_flags_gate_facts_and_preserve_payloads(self) -> None:
        base = torch.tensor([0.1, 0.2])
        output = torch.tensor([[0.4, -0.1], [0.2, 0.5]])
        observation = _AdaptiveParameterObservation.from_forward((base,), output)
        option = soft_weight_bank()

        disabled = collect_facts("weight", option, observation)
        self.assertEqual(disabled.histograms, ())

        enabled = collect_facts(
            "weight",
            option,
            observation,
            include_histograms=True,
        )
        self.assertEqual(
            [metric.suffix for metric in enabled.histograms],
            ["output", "delta"],
        )
        torch.testing.assert_close(enabled.histograms[0].value, output)
        torch.testing.assert_close(enabled.histograms[1].value, observation.delta)

        no_base_observation = _AdaptiveParameterObservation.from_forward((), output)
        no_base = collect_facts(
            "weight",
            option,
            no_base_observation,
            include_histograms=True,
        )
        self.assertEqual(
            [metric.suffix for metric in no_base.histograms],
            ["output"],
        )


if __name__ == "__main__":
    unittest.main()
