import unittest

import torch
from torch import nn

from emperor.augmentations.adaptive_parameters import (
    BankExpansionFactorOptions,
    DynamicDepthOptions,
    LayeredWeightedBankDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
    WeightBankUtilizationMonitorCallback,
    WeightDecayScheduleOptions,
    WeightedBankDynamicBiasConfig,
)
from emperor.augmentations.adaptive_parameters._biases.variants.weighted_bank import (
    WeightedBankDynamicBias,
)
from emperor.augmentations.adaptive_parameters._monitoring.weight_banks import (
    _WeightBankDiagnostics,
)
from emperor.augmentations.adaptive_parameters._weights.variants.layered_weighted_bank import (  # noqa: E501
    LayeredWeightedBankDynamicWeight,
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
from support.monitor import orchestration_calls


class FakeExperiment:
    def __init__(self):
        self.histograms = []
        self.images = []

    def add_histogram(self, tag, values, step):
        self.histograms.append((tag, values.clone(), step))

    def add_image(self, tag, image, step, dataformats):
        self.images.append((tag, image.clone(), step, dataformats))


class FakeLogger:
    def __init__(self, experiment):
        self.experiment = experiment


class FakeLightningModule(nn.Module):
    def __init__(self, bank_module: nn.Module, experiment=None, global_step: int = 0):
        super().__init__()
        self.dynamic_bank = bank_module
        self.logger = FakeLogger(experiment) if experiment is not None else None
        self.global_step = global_step
        self.logged_scalars = []

    def log(self, name, value, *args, **kwargs):
        self.logged_scalars.append((name, value))


class TestWeightBankUtilizationMonitorCallback(unittest.TestCase):
    BANK_MODULE_PATH = "dynamic_bank"

    def test_tracking_orchestration_lists_each_tracked_fact(self):
        cls = WeightBankUtilizationMonitorCallback
        orchestration = (
            cls._WeightBankUtilizationMonitorCallback__track_weight_bank_utilization
        )

        self.assertEqual(
            orchestration_calls(orchestration),
            (
                "__track_marginal_selection_entropy",
                "__track_mean_per_sample_selection_entropy",
                "__track_utilization_coefficient_of_variation",
                "__track_active_slots",
                "__track_dead_slot_fraction",
                "__track_maximum_utilization",
                "__track_minimum_utilization",
                "__track_per_slot_utilization",
                "__track_utilization_history",
                "__track_utilization_histogram",
                "__track_utilization_heatmap",
            ),
        )

    def layer_stack_config(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 16,
    ) -> LayerStackConfig:
        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=2,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=True,
            layer_config=LayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                activation=ActivationOptions.RELU,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                residual_config=None,
                dropout_probability=0.0,
                gate_config=None,
                halting_config=None,
                layer_model_config=LinearLayerConfig(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    bias_flag=True,
                ),
            ),
        )

    def build_soft_weighted_bank_weight(
        self,
        input_dim: int = 6,
        output_dim: int = 8,
        bank_expansion_factor: BankExpansionFactorOptions = (
            BankExpansionFactorOptions.FACTOR_OF_THREE
        ),
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_TWO,
    ) -> SoftWeightedBankDynamicWeight:
        cfg = SoftWeightedBankDynamicWeightConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            generator_depth=generator_depth,
            decay_schedule=WeightDecayScheduleOptions.DISABLED,
            decay_rate=0.0,
            decay_warmup_batches=0,
            bank_expansion_factor=bank_expansion_factor,
            model_config=self.layer_stack_config(input_dim, output_dim),
        )
        return SoftWeightedBankDynamicWeight(cfg)

    def build_layered_weighted_bank_weight(
        self,
        input_dim: int = 6,
        output_dim: int = 8,
        bank_expansion_factor: BankExpansionFactorOptions = (
            BankExpansionFactorOptions.FACTOR_OF_THREE
        ),
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_TWO,
    ) -> LayeredWeightedBankDynamicWeight:
        cfg = LayeredWeightedBankDynamicWeightConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            generator_depth=generator_depth,
            decay_schedule=WeightDecayScheduleOptions.DISABLED,
            decay_rate=0.0,
            decay_warmup_batches=0,
            bank_expansion_factor=bank_expansion_factor,
            model_config=self.layer_stack_config(input_dim, output_dim),
        )
        return LayeredWeightedBankDynamicWeight(cfg)

    def build_weighted_bank_bias(
        self,
        input_dim: int = 6,
        output_dim: int = 8,
        bank_expansion_factor: BankExpansionFactorOptions = (
            BankExpansionFactorOptions.FACTOR_OF_THREE
        ),
    ) -> WeightedBankDynamicBias:
        cfg = WeightedBankDynamicBiasConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            decay_schedule=WeightDecayScheduleOptions.DISABLED,
            decay_rate=0.0,
            decay_warmup_batches=0,
            bank_expansion_factor=bank_expansion_factor,
            model_config=self.layer_stack_config(input_dim, output_dim),
        )
        return WeightedBankDynamicBias(cfg)

    def build_module(
        self,
        bank_module: nn.Module,
        experiment=None,
        global_step: int = 0,
    ) -> FakeLightningModule:
        return FakeLightningModule(
            bank_module, experiment=experiment, global_step=global_step
        )

    def primed_callback(
        self, module, **callback_kwargs
    ) -> WeightBankUtilizationMonitorCallback:
        callback = WeightBankUtilizationMonitorCallback(**callback_kwargs)
        callback.on_fit_start(trainer=None, pl_module=module)
        return callback

    def feed_bank(self, bank_module: nn.Module, batch_size: int = 4) -> None:
        input_dim = bank_module.input_dim
        output_dim = bank_module.output_dim
        logits = torch.randn(batch_size, input_dim)
        if isinstance(bank_module, WeightedBankDynamicBias):
            bias_params = torch.zeros(output_dim)
            bank_module(bias_params, logits)
        else:
            weight_params = torch.zeros(input_dim, output_dim)
            bank_module(weight_params, logits)

    def scalar_names(self, module: FakeLightningModule) -> set:
        return {name for name, _ in module.logged_scalars}

    def expected_scalar_suffixes(self) -> tuple:
        return (
            "selection_entropy_marginal",
            "selection_entropy_mean",
            "utilization_coefficient_of_variation",
            "active_slots",
            "dead_slot_fraction",
            "max_utilization",
            "min_utilization",
        )

    def test_rejects_non_positive_log_interval(self):
        with self.assertRaises(ValueError):
            WeightBankUtilizationMonitorCallback(log_every_n_steps=0)

    def test_rejects_non_positive_history_size(self):
        with self.assertRaises(ValueError):
            WeightBankUtilizationMonitorCallback(history_size=0)

    def test_on_fit_start_registers_hook_on_bank_generator(self):
        bank = self.build_soft_weighted_bank_weight()
        module = self.build_module(bank)

        callback = self.primed_callback(module)

        self.assertEqual(len(callback._bank_modules), 1)
        attached_name, attached_module = callback._bank_modules[0]
        self.assertEqual(attached_name, self.BANK_MODULE_PATH)
        self.assertIs(attached_module, bank)
        self.assertEqual(len(callback._hooks), 1)
        self.assertIn(self.BANK_MODULE_PATH, callback._utilization_history)

    def test_repeated_fit_start_replaces_generator_hook(self):
        bank = self.build_soft_weighted_bank_weight()
        module = self.build_module(bank)
        callback = self.primed_callback(module, log_every_n_steps=1)

        callback.on_fit_start(trainer=None, pl_module=module)
        self.feed_bank(bank)
        callback.on_train_batch_end(
            trainer=None,
            pl_module=module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )

        self.assertEqual(len(callback._hooks), 1)
        selection_logs = [
            name
            for name, _ in module.logged_scalars
            if name.endswith("/selection_entropy_marginal")
        ]
        self.assertEqual(len(selection_logs), 1)

    def test_on_train_batch_end_skips_when_not_at_logging_interval(self):
        bank = self.build_soft_weighted_bank_weight()
        module = self.build_module(bank)
        callback = self.primed_callback(module, log_every_n_steps=10)
        self.feed_bank(bank)

        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=3
        )

        self.assertEqual(module.logged_scalars, [])

    def test_skipped_batch_logits_are_not_reused_by_a_later_batch(self):
        bank = self.build_soft_weighted_bank_weight()
        module = self.build_module(bank)
        callback = self.primed_callback(module, log_every_n_steps=2)
        self.feed_bank(bank)

        callback.on_train_batch_end(
            trainer=None,
            pl_module=module,
            outputs=None,
            batch=None,
            batch_idx=1,
        )
        callback.on_train_batch_end(
            trainer=None,
            pl_module=module,
            outputs=None,
            batch=None,
            batch_idx=2,
        )

        self.assertEqual(module.logged_scalars, [])

    def test_logs_bank_scalars_for_each_bank_type(self):
        builders = (
            self.build_soft_weighted_bank_weight,
            self.build_layered_weighted_bank_weight,
            self.build_weighted_bank_bias,
        )
        for builder in builders:
            with self.subTest(builder=builder.__name__):
                bank = builder()
                module = self.build_module(bank)
                callback = self.primed_callback(module, log_every_n_steps=1)
                self.feed_bank(bank)

                callback.on_train_batch_end(
                    trainer=None,
                    pl_module=module,
                    outputs=None,
                    batch=None,
                    batch_idx=0,
                )

                names = self.scalar_names(module)
                for suffix in self.expected_scalar_suffixes():
                    self.assertIn(f"{self.BANK_MODULE_PATH}/bank/{suffix}", names)

    def test_logged_utilization_values_are_finite_and_in_range(self):
        bank = self.build_soft_weighted_bank_weight(
            bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_THREE
        )
        module = self.build_module(bank)
        callback = self.primed_callback(module, log_every_n_steps=1)
        self.feed_bank(bank)

        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
        )

        logged = dict(module.logged_scalars)
        active_slots = logged[f"{self.BANK_MODULE_PATH}/bank/active_slots"]
        dead_fraction = logged[f"{self.BANK_MODULE_PATH}/bank/dead_slot_fraction"]
        max_utilization = logged[f"{self.BANK_MODULE_PATH}/bank/max_utilization"]
        self.assertTrue(torch.isfinite(torch.as_tensor(max_utilization)))
        self.assertLessEqual(float(active_slots), bank.expanded_bank_row_count)
        self.assertGreaterEqual(float(dead_fraction), 0.0)
        self.assertLessEqual(float(dead_fraction), 1.0)

    def test_single_slot_bank_logs_finite_zero_coefficient_of_variation(self):
        bank = self.build_soft_weighted_bank_weight(
            input_dim=1, bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_ONE
        )
        module = self.build_module(bank)
        callback = self.primed_callback(module, log_every_n_steps=1)
        self.feed_bank(bank)

        callback.on_train_batch_end(
            trainer=None,
            pl_module=module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )

        logged = dict(module.logged_scalars)
        coefficient_of_variation = logged[
            f"{self.BANK_MODULE_PATH}/bank/utilization_coefficient_of_variation"
        ]
        self.assertTrue(torch.isfinite(coefficient_of_variation))
        torch.testing.assert_close(coefficient_of_variation, torch.zeros(()))

    def test_empty_bank_logits_do_not_emit_undefined_metrics(self):
        bank = self.build_weighted_bank_bias()
        module = self.build_module(bank)
        callback = self.primed_callback(module, log_every_n_steps=1)
        self.feed_bank(bank, batch_size=0)

        callback.on_train_batch_end(
            trainer=None,
            pl_module=module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )

        self.assertEqual(module.logged_scalars, [])
        self.assertNotIn(self.BANK_MODULE_PATH, callback._last_bank_logits)

    def test_distribution_summaries_match_exact_bank_reductions(self):
        batch_size = 2
        input_dim = 2
        bank_factor = BankExpansionFactorOptions.FACTOR_OF_TWO
        depth = DynamicDepthOptions.DEPTH_OF_ONE
        soft_bank = self.build_soft_weighted_bank_weight(
            input_dim=input_dim,
            output_dim=3,
            bank_expansion_factor=bank_factor,
            generator_depth=depth,
        )
        layered_bank = self.build_layered_weighted_bank_weight(
            input_dim=input_dim,
            output_dim=3,
            bank_expansion_factor=bank_factor,
            generator_depth=depth,
        )
        bias_bank = self.build_weighted_bank_bias(
            input_dim=input_dim,
            output_dim=3,
            bank_expansion_factor=bank_factor,
        )
        layered_weight_logits = torch.tensor(
            [[[2.0, -1.0, -0.5, 1.5]], [[-2.0, 1.0, 0.25, -0.75]]]
        )
        soft_weight_logits = torch.tensor(
            [
                [[2.0, -1.0, -0.5, 1.5, -2.0, 1.0, 0.25, -0.75]],
                [[0.75, -0.25, 2.5, -1.5, -0.5, 1.25, -2.0, 2.0]],
            ]
        )
        expanded_bank_rows = input_dim * bank_factor.value
        bias_logits = torch.tensor([[3.0, -1.0], [-2.0, 2.0]])

        cases = [
            (
                soft_bank,
                soft_weight_logits,
                torch.softmax(
                    soft_weight_logits.view(
                        batch_size,
                        depth.value,
                        input_dim,
                        expanded_bank_rows,
                    ),
                    dim=-1,
                ),
                torch.softmax(
                    soft_weight_logits.view(
                        batch_size,
                        depth.value,
                        input_dim,
                        expanded_bank_rows,
                    ),
                    dim=-1,
                ),
                lambda distribution: distribution.mean(dim=(0, 1, 2)),
            ),
            (
                layered_bank,
                layered_weight_logits,
                torch.softmax(layered_weight_logits, dim=-1).view(
                    batch_size, depth.value, input_dim, bank_factor.value
                ),
                torch.softmax(layered_weight_logits, dim=-1),
                lambda distribution: distribution.sum(dim=2).mean(dim=(0, 1)),
            ),
            (
                bias_bank,
                bias_logits,
                torch.softmax(bias_logits, dim=-1).reshape(-1, bank_factor.value),
                torch.softmax(bias_logits, dim=-1).reshape(-1, bank_factor.value),
                lambda distribution: distribution.mean(dim=0),
            ),
        ]

        for bank, logits, distribution, entropy_distribution, utilization_fn in cases:
            with self.subTest(bank=type(bank).__name__):
                summary = _WeightBankDiagnostics.summarize(bank, logits)
                expected_utilization = utilization_fn(distribution)
                expected_entropy = (
                    -(entropy_distribution.clamp_min(1e-9).log() * entropy_distribution)
                    .sum(dim=-1)
                    .mean()
                )

                torch.testing.assert_close(
                    summary.per_slot_utilization, expected_utilization
                )
                torch.testing.assert_close(
                    summary.mean_per_sample_entropy,
                    expected_entropy,
                )

    def test_distribution_summary_ignores_non_bank_modules(self):
        self.assertIsNone(
            _WeightBankDiagnostics.summarize(
                nn.Identity(),
                torch.ones(2, 3),
            )
        )

    def test_capture_hook_ignores_output_without_tensor_hidden_state(self):
        callback = WeightBankUtilizationMonitorCallback()
        hook = callback._WeightBankUtilizationMonitorCallback__make_bank_logits_capture_hook(
            "unknown"
        )

        hook(nn.Identity(), (), object())

        self.assertEqual(callback._last_bank_logits, {})

    def test_batch_end_ignores_unrecognized_registered_bank_module(self):
        callback = WeightBankUtilizationMonitorCallback(log_every_n_steps=1)
        module = self.build_module(nn.Identity())
        callback._bank_modules.append(("unknown", nn.Identity()))
        callback._last_bank_logits["unknown"] = torch.ones(2, 3)

        callback.on_train_batch_end(
            trainer=None,
            pl_module=module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )

        self.assertEqual(module.logged_scalars, [])
        self.assertEqual(callback._last_bank_logits, {})

    def test_per_slot_scalars_logged_when_enabled(self):
        bank = self.build_soft_weighted_bank_weight(
            bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_THREE
        )
        module = self.build_module(bank)
        callback = self.primed_callback(
            module, log_every_n_steps=1, log_per_slot_scalars=True
        )
        self.feed_bank(bank)

        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
        )

        names = self.scalar_names(module)
        for slot_index in range(bank.expanded_bank_row_count):
            self.assertIn(
                f"{self.BANK_MODULE_PATH}/bank/slot_{slot_index}/utilization", names
            )

    def test_per_slot_scalars_skipped_by_default(self):
        bank = self.build_soft_weighted_bank_weight()
        module = self.build_module(bank)
        callback = self.primed_callback(module, log_every_n_steps=1)
        self.feed_bank(bank)

        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
        )

        for name, _ in module.logged_scalars:
            self.assertNotIn("/slot_", name)

    def test_visual_summaries_skipped_when_no_experiment(self):
        bank = self.build_soft_weighted_bank_weight()
        module = self.build_module(bank, experiment=None)
        callback = self.primed_callback(module, log_every_n_steps=1)
        self.feed_bank(bank)

        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
        )

        self.assertEqual(
            len(callback._utilization_history[self.BANK_MODULE_PATH]),
            0,
        )

    def test_visual_summaries_log_histogram_and_heatmap(self):
        experiment = FakeExperiment()
        bank = self.build_soft_weighted_bank_weight()
        module = self.build_module(bank, experiment=experiment, global_step=11)
        callback = self.primed_callback(module, log_every_n_steps=1)
        self.feed_bank(bank)

        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
        )

        histogram_tags = {tag for tag, _, _ in experiment.histograms}
        self.assertIn(
            f"{self.BANK_MODULE_PATH}/bank/histogram/utilization", histogram_tags
        )
        image_tags = {tag for tag, _, _, _ in experiment.images}
        self.assertIn(f"{self.BANK_MODULE_PATH}/bank/heatmap/utilization", image_tags)
        for _, _, step in experiment.histograms:
            self.assertEqual(step, 11)

    def test_visual_history_bounded_by_history_size(self):
        experiment = FakeExperiment()
        bank = self.build_soft_weighted_bank_weight()
        module = self.build_module(bank, experiment=experiment)
        callback = self.primed_callback(module, log_every_n_steps=1, history_size=3)

        for batch_idx in range(5):
            self.feed_bank(bank)
            callback.on_train_batch_end(
                trainer=None,
                pl_module=module,
                outputs=None,
                batch=None,
                batch_idx=batch_idx,
            )

        self.assertEqual(len(callback._utilization_history[self.BANK_MODULE_PATH]), 3)

    def test_skips_modules_without_captured_logits(self):
        bank = self.build_soft_weighted_bank_weight()
        module = self.build_module(bank)
        callback = self.primed_callback(module, log_every_n_steps=1)

        callback.on_train_batch_end(
            trainer=None, pl_module=module, outputs=None, batch=None, batch_idx=0
        )

        self.assertEqual(module.logged_scalars, [])

    def test_on_fit_end_removes_hooks_and_clears_state(self):
        bank = self.build_soft_weighted_bank_weight()
        module = self.build_module(bank)
        callback = self.primed_callback(module)

        callback.on_fit_end(trainer=None, pl_module=module)
        callback._last_bank_logits["sentinel"] = torch.ones(1)
        self.feed_bank(bank)

        self.assertEqual(callback._hooks, [])
        self.assertEqual(callback._bank_modules, [])
        self.assertEqual(callback._utilization_history, {})
        self.assertEqual(set(callback._last_bank_logits), {"sentinel"})

    def test_on_exception_removes_hooks_and_clears_all_state(self):
        bank = self.build_soft_weighted_bank_weight()
        module = self.build_module(bank)
        callback = self.primed_callback(module)
        self.feed_bank(bank)
        self.assertTrue(callback._last_bank_logits)

        callback.on_exception(
            trainer=None,
            pl_module=module,
            exception=RuntimeError("deliberate failure"),
        )

        self.assertEqual(callback._hooks, [])
        self.assertEqual(callback._bank_modules, [])
        self.assertEqual(callback._utilization_history, {})
        self.assertEqual(callback._last_bank_logits, {})


if __name__ == "__main__":
    unittest.main()
