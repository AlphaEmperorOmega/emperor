from __future__ import annotations

import unittest

import torch
from lightning import LightningModule

from emperor.augmentations.adaptive_parameters import (
    BankExpansionFactorOptions,
    DynamicDepthOptions,
    SoftWeightedBankDynamicWeightConfig,
    WeightDecayScheduleOptions,
    WeightedBankDynamicBiasConfig,
)
from emperor.augmentations.adaptive_parameters._biases.variants.weighted_bank import (
    WeightedBankDynamicBias,
)
from emperor.augmentations.adaptive_parameters._monitoring.weight_banks import (
    WeightBankUtilizationMonitorCallback,
    _BankDistributionSummary,
    _WeightBankDiagnostics,
    _WeightBankTrackingContext,
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
from emperor.monitoring import MonitorTensorHistory


class RecordingExperiment:
    def __init__(self) -> None:
        self.histograms: list[tuple[str, torch.Tensor, int]] = []
        self.images: list[tuple[str, torch.Tensor, int, str]] = []

    def add_histogram(
        self,
        tag: str,
        values: torch.Tensor,
        step: int,
    ) -> None:
        self.histograms.append((tag, values.detach().clone(), step))

    def add_image(
        self,
        tag: str,
        image: torch.Tensor,
        step: int,
        *,
        dataformats: str,
    ) -> None:
        self.images.append((tag, image.detach().clone(), step, dataformats))


class RecordingLightningModule(LightningModule):
    def __init__(
        self,
        first_bank: WeightedBankDynamicBias | None = None,
        second_bank: WeightedBankDynamicBias | None = None,
    ) -> None:
        super().__init__()
        if first_bank is not None:
            self.first_bank = first_bank
        if second_bank is not None:
            self.second_bank = second_bank
        self.logged_values: list[tuple[str | None, object]] = []

    def log(
        self,
        name: str,
        value: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.logged_values.append((name, value))


def generator_config(
    input_dim: int,
    output_dim: int,
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


def weighted_bias(
    input_dim: int = 2,
    output_dim: int = 2,
) -> WeightedBankDynamicBias:
    return WeightedBankDynamicBias(
        WeightedBankDynamicBiasConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            decay_schedule=WeightDecayScheduleOptions.DISABLED,
            decay_rate=0.0,
            decay_warmup_batches=0,
            bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_TWO,
            model_config=generator_config(input_dim, output_dim),
        )
    )


def soft_weight_bank(
    input_dim: int = 3,
    output_dim: int = 2,
) -> SoftWeightedBankDynamicWeight:
    return SoftWeightedBankDynamicWeight(
        SoftWeightedBankDynamicWeightConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            generator_depth=DynamicDepthOptions.DEPTH_OF_ONE,
            decay_schedule=WeightDecayScheduleOptions.DISABLED,
            decay_rate=0.0,
            decay_warmup_batches=0,
            bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_TWO,
            model_config=generator_config(input_dim, output_dim),
        )
    )


def logged_map(module: RecordingLightningModule) -> dict[str | None, object]:
    return dict(module.logged_values)


class WeightBankMonitorMutationContractTests(unittest.TestCase):
    def test_defaults_and_both_validation_messages_are_exact(self) -> None:
        callback = WeightBankUtilizationMonitorCallback()
        self.assertEqual(callback.log_every_n_steps, 100)
        self.assertEqual(callback.history_size, 128)
        self.assertFalse(callback.log_per_slot_scalars)

        invalid_cases = (
            (
                {"log_every_n_steps": 0},
                r"^log_every_n_steps must be greater than 0\.$",
            ),
            (
                {"history_size": 0},
                r"^history_size must be greater than 0\.$",
            ),
        )
        for arguments, message in invalid_cases:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, message):
                    WeightBankUtilizationMonitorCallback(**arguments)

    def test_distribution_summaries_use_exact_axes_for_rectangular_inputs(
        self,
    ) -> None:
        soft_bank = soft_weight_bank(input_dim=3)
        soft_logits = torch.tensor(
            [
                [[2.0, -1.0, 0.5, 1.5, -0.25, 0.75]],
                [[-2.0, 1.0, 0.25, -0.75, 2.5, -1.5]],
            ]
        )
        soft_distribution = torch.softmax(
            soft_logits.view(2, 1, 3, 2),
            dim=-1,
        )
        soft_summary = _WeightBankDiagnostics.summarize(
            soft_bank,
            soft_logits,
        )
        self.assertIsNotNone(soft_summary)
        torch.testing.assert_close(
            soft_summary.per_slot_utilization,
            soft_distribution.mean(dim=(0, 1, 2)),
        )
        torch.testing.assert_close(
            soft_summary.mean_per_sample_entropy,
            (
                -(
                    soft_distribution.clamp_min(1.0e-9).log()
                    * soft_distribution
                )
                .sum(dim=-1)
                .mean()
            ),
        )

        bias_bank = weighted_bias(input_dim=3)
        bias_logits = torch.tensor(
            [[3.0, -1.0], [0.5, 2.0], [-2.0, 0.25]]
        )
        bias_distribution = torch.softmax(bias_logits, dim=-1)
        bias_summary = _WeightBankDiagnostics.summarize(
            bias_bank,
            bias_logits,
        )
        self.assertIsNotNone(bias_summary)
        torch.testing.assert_close(
            bias_summary.per_slot_utilization,
            bias_distribution.mean(dim=0),
        )
        torch.testing.assert_close(
            bias_summary.mean_per_sample_entropy,
            (
                -(
                    bias_distribution.clamp_min(1.0e-9).log()
                    * bias_distribution
                )
                .sum(dim=-1)
                .mean()
            ),
        )

    def test_utilization_and_logged_payloads_are_exact_at_dead_slot_floor(
        self,
    ) -> None:
        utilization = torch.tensor([0.0001, 0.1999, 0.8])
        mean_sample_entropy = torch.tensor(0.37)
        summary = _BankDistributionSummary(
            per_slot_utilization=utilization,
            mean_per_sample_entropy=mean_sample_entropy,
        )
        metrics = _WeightBankDiagnostics.calculate_utilization(
            summary,
            dead_slot_utilization_floor=0.0001,
        )
        expected_marginal_entropy = -(
            utilization.clamp_min(1.0e-9).log() * utilization
        ).sum()
        expected_coefficient = utilization.std() / utilization.mean().clamp_min(
            1.0e-6
        )
        expected = {
            "selection_entropy_marginal": expected_marginal_entropy,
            "selection_entropy_mean": mean_sample_entropy,
            "utilization_coefficient_of_variation": expected_coefficient,
            "active_slots": torch.tensor(2.0),
            "dead_slot_fraction": torch.tensor(1.0 / 3.0),
            "max_utilization": torch.tensor(0.8),
            "min_utilization": torch.tensor(0.0001),
            "slot_0/utilization": utilization[0],
            "slot_1/utilization": utilization[1],
            "slot_2/utilization": utilization[2],
        }
        module = RecordingLightningModule()
        callback = WeightBankUtilizationMonitorCallback(
            log_per_slot_scalars=True
        )
        context = (
            callback
            ._WeightBankUtilizationMonitorCallback__build_tracking_context(
                module,
                "dynamic_bank",
                metrics,
            )
        )
        self.assertEqual(context.global_step, 0)
        self.assertIs(context.metrics, metrics)
        callback._WeightBankUtilizationMonitorCallback__track_weight_bank_utilization(
            context
        )

        actual = logged_map(module)
        for suffix, expected_value in expected.items():
            with self.subTest(suffix=suffix):
                value = actual[f"dynamic_bank/bank/{suffix}"]
                self.assertIsInstance(value, torch.Tensor)
                torch.testing.assert_close(value, expected_value)
        self.assertNotIn(None, actual)

    def test_real_history_histogram_and_heatmap_preserve_global_step(self) -> None:
        utilization = torch.tensor([0.2, 0.3, 0.5])
        metrics = _WeightBankDiagnostics.calculate_utilization(
            _BankDistributionSummary(
                per_slot_utilization=utilization,
                mean_per_sample_entropy=torch.tensor(0.4),
            ),
            dead_slot_utilization_floor=0.0001,
        )
        experiment = RecordingExperiment()
        callback = WeightBankUtilizationMonitorCallback(
            history_size=3,
            log_per_slot_scalars=True,
        )
        callback._utilization_history["dynamic_bank"] = MonitorTensorHistory(3)
        context = _WeightBankTrackingContext(
            pl_module=RecordingLightningModule(),
            module_name="dynamic_bank",
            metrics=metrics,
            experiment=experiment,
            global_step=7,
        )

        callback._WeightBankUtilizationMonitorCallback__track_weight_bank_utilization(
            context
        )

        self.assertEqual(
            [tag for tag, _, _ in experiment.histograms],
            ["dynamic_bank/bank/histogram/utilization"],
        )
        torch.testing.assert_close(
            experiment.histograms[0][1],
            utilization,
        )
        self.assertEqual(experiment.histograms[0][2], 7)
        self.assertEqual(
            [tag for tag, _, _, _ in experiment.images],
            ["dynamic_bank/bank/heatmap/utilization"],
        )
        self.assertEqual(experiment.images[0][2], 7)
        self.assertEqual(experiment.images[0][3], "CHW")

    def test_batch_end_continues_past_unexecuted_real_bank(self) -> None:
        first = weighted_bias()
        second = weighted_bias()
        module = RecordingLightningModule(first, second)
        callback = WeightBankUtilizationMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(trainer=None, pl_module=module)
        self.assertEqual(
            [name for name, _ in callback._bank_modules],
            ["first_bank", "second_bank"],
        )

        second(
            torch.zeros(2),
            torch.tensor([[1.0, 2.0], [-1.0, 0.5]]),
        )
        callback.on_train_batch_end(
            trainer=None,
            pl_module=module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )

        names = set(logged_map(module))
        self.assertIn(
            "second_bank/bank/selection_entropy_marginal",
            names,
        )
        self.assertFalse(
            any(name and name.startswith("first_bank/") for name in names)
        )
        callback.on_fit_end(trainer=None, pl_module=module)


if __name__ == "__main__":
    unittest.main()
