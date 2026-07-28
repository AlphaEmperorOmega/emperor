from __future__ import annotations

import unittest

import torch
from torch import nn

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
    _WeightBankDiagnosticFacts,
    _WeightBankDiagnostics,
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


class RecordingLogger:
    def __init__(self, experiment: RecordingExperiment) -> None:
        self.experiment = experiment


class RecordingModule(nn.Module):
    def __init__(
        self,
        first_bank: WeightedBankDynamicBias | None = None,
        second_bank: WeightedBankDynamicBias | None = None,
        *,
        experiment: RecordingExperiment | None = None,
        global_step: int = 0,
    ) -> None:
        super().__init__()
        if first_bank is not None:
            self.first_bank = first_bank
        if second_bank is not None:
            self.second_bank = second_bank
        self.logger = RecordingLogger(experiment) if experiment is not None else None
        self.global_step = global_step
        self.logged_values: list[tuple[str, object]] = []

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
    bank_expansion_factor: BankExpansionFactorOptions = (
        BankExpansionFactorOptions.FACTOR_OF_TWO
    ),
) -> WeightedBankDynamicBias:
    return WeightedBankDynamicBias(
        WeightedBankDynamicBiasConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            decay_schedule=WeightDecayScheduleOptions.DISABLED,
            decay_rate=0.0,
            decay_warmup_batches=0,
            bank_expansion_factor=bank_expansion_factor,
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


def collect_facts(
    bank_module: nn.Module,
    bank_logits: torch.Tensor,
    *,
    dead_slot_utilization_floor: float = 1e-4,
    include_per_slot_scalars: bool = False,
) -> _WeightBankDiagnosticFacts:
    facts = _WeightBankDiagnostics().collect(
        bank_module,
        bank_logits,
        dead_slot_utilization_floor=dead_slot_utilization_floor,
        include_per_slot_scalars=include_per_slot_scalars,
    )
    if facts is None:
        raise AssertionError("Expected a recognized weighted-bank implementation.")
    return facts


def scalar_map(facts: _WeightBankDiagnosticFacts) -> dict[str, torch.Tensor]:
    return {metric.suffix: metric.value for metric in facts.scalars}


def logged_map(module: RecordingModule) -> dict[str, object]:
    return dict(module.logged_values)


class WeightBankMonitorMutationContractTests(unittest.TestCase):
    def test_defaults_and_both_validation_messages_are_exact(self) -> None:
        callback = WeightBankUtilizationMonitorCallback()
        self.assertEqual(callback.log_every_n_steps, 100)
        self.assertEqual(callback.history_size, 128)
        self.assertFalse(callback.log_per_slot_scalars)
        self.assertEqual(
            callback.state_key,
            "WeightBankUtilizationMonitorCallback",
        )
        self.assertEqual(callback.state_dict(), {})

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

    def test_distribution_facts_use_exact_axes_for_rectangular_inputs(self) -> None:
        soft_bank = soft_weight_bank(input_dim=3)
        soft_logits = torch.arange(36, dtype=torch.float32).view(2, 1, 18) / 10
        soft_distribution = torch.softmax(
            soft_logits.view(2, 1, 3, soft_bank.expanded_bank_row_count),
            dim=-1,
        )
        soft_facts = collect_facts(soft_bank, soft_logits)
        torch.testing.assert_close(
            soft_facts.per_slot_utilization,
            soft_distribution.mean(dim=(0, 1, 2)),
        )
        torch.testing.assert_close(
            scalar_map(soft_facts)["selection_entropy_mean"],
            (
                -(soft_distribution.clamp_min(1.0e-9).log() * soft_distribution)
                .sum(dim=-1)
                .mean()
            ),
        )

        bias_bank = weighted_bias(input_dim=3)
        bias_logits = torch.tensor([[3.0, -1.0], [0.5, 2.0], [-2.0, 0.25]])
        bias_distribution = torch.softmax(bias_logits, dim=-1)
        bias_facts = collect_facts(bias_bank, bias_logits)
        torch.testing.assert_close(
            bias_facts.per_slot_utilization,
            bias_distribution.mean(dim=0),
        )
        torch.testing.assert_close(
            scalar_map(bias_facts)["selection_entropy_mean"],
            (
                -(bias_distribution.clamp_min(1.0e-9).log() * bias_distribution)
                .sum(dim=-1)
                .mean()
            ),
        )

    def test_utilization_facts_are_exact_at_dead_slot_floor(self) -> None:
        bank = weighted_bias(
            input_dim=3,
            bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_THREE,
        )
        requested_utilization = torch.tensor([0.0001, 0.1999, 0.8])
        bank_logits = requested_utilization.log().unsqueeze(0)
        utilization = torch.softmax(bank_logits, dim=-1).squeeze(0)
        dead_slot_floor = float(utilization[0])
        facts = collect_facts(
            bank,
            bank_logits,
            dead_slot_utilization_floor=dead_slot_floor,
            include_per_slot_scalars=True,
        )
        actual = scalar_map(facts)
        expected = {
            "selection_entropy_marginal": -(
                utilization.clamp_min(1.0e-9).log() * utilization
            ).sum(),
            "selection_entropy_mean": -(
                utilization.clamp_min(1.0e-9).log() * utilization
            ).sum(),
            "utilization_coefficient_of_variation": (
                utilization.std() / utilization.mean().clamp_min(1.0e-6)
            ),
            "active_slots": torch.tensor(2.0),
            "dead_slot_fraction": torch.tensor(1.0 / 3.0),
            "max_utilization": utilization.max(),
            "min_utilization": utilization.min(),
            "slot_0/utilization": utilization[0],
            "slot_1/utilization": utilization[1],
            "slot_2/utilization": utilization[2],
        }
        self.assertEqual(
            [metric.suffix for metric in facts.scalars],
            list(expected),
        )
        for suffix, expected_value in expected.items():
            with self.subTest(suffix=suffix):
                torch.testing.assert_close(actual[suffix], expected_value)

    def test_real_history_histogram_and_heatmap_preserve_global_step(self) -> None:
        bank = weighted_bias()
        experiment = RecordingExperiment()
        module = RecordingModule(
            bank,
            experiment=experiment,
            global_step=7,
        )
        callback = WeightBankUtilizationMonitorCallback(
            log_every_n_steps=1,
            history_size=3,
            log_per_slot_scalars=True,
        )
        callback.on_fit_start(trainer=None, pl_module=module)
        bank(
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

        self.assertEqual(
            [tag for tag, _, _ in experiment.histograms],
            ["first_bank/bank/histogram/utilization"],
        )
        self.assertEqual(experiment.histograms[0][2], 7)
        self.assertEqual(
            [tag for tag, _, _, _ in experiment.images],
            ["first_bank/bank/heatmap/utilization"],
        )
        self.assertEqual(experiment.images[0][2], 7)
        self.assertEqual(experiment.images[0][3], "CHW")
        self.assertEqual(len(callback._utilization_history["first_bank"]), 1)
        callback.on_fit_end(trainer=None, pl_module=module)

    def test_batch_end_continues_past_unexecuted_real_bank(self) -> None:
        first = weighted_bias()
        second = weighted_bias()
        module = RecordingModule(first, second)
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
        self.assertFalse(any(name.startswith("first_bank/") for name in names))
        callback.on_fit_end(trainer=None, pl_module=module)


if __name__ == "__main__":
    unittest.main()
