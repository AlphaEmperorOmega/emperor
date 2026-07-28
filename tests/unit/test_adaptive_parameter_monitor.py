import unittest

import torch
from torch import nn

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AdaptiveParameterGroupingScopeOptions,
    AdaptiveParameterMonitorCallback,
    AdditiveDynamicBiasConfig,
    DualModelDynamicWeightConfig,
    DynamicDepthOptions,
    MaskDimensionOptions,
    MultiplicativeDynamicBiasConfig,
    PerAxisScoreMaskConfig,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.augmentations.adaptive_parameters._augmentation import (
    AdaptiveParameterAugmentation,
)
from emperor.augmentations.adaptive_parameters._biases.variants.multiplicative import (
    MultiplicativeDynamicBias,
)
from emperor.augmentations.adaptive_parameters._linear_adapter import (
    AdaptiveLinearLayer,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    RowLayout,
)
from emperor.linears import LinearLayerConfig


class FakeExperiment:
    def __init__(self):
        self.histograms = []

    def add_histogram(self, tag, values, step):
        self.histograms.append((tag, values.clone(), step))


class IncompatibleExperiment:
    pass


class FakeLogger:
    def __init__(self, experiment):
        self.experiment = experiment


class FakeLightningModule(nn.Module):
    def __init__(self, adaptive, experiment=None, global_step: int = 0):
        super().__init__()
        self.adaptive = adaptive
        self.logger = FakeLogger(experiment) if experiment is not None else None
        self.global_step = global_step
        self.logged_scalars = []

    def log(self, name, value, *args, **kwargs):
        self.logged_scalars.append((name, value))


class AdditiveOption(nn.Module):
    def forward(self, parameters, logits):
        return parameters + 1.0


class BankOption(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_bank = nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))

    def forward(self, parameters, logits):
        return parameters + 1.0


class MaskOption(nn.Module):
    def forward(self, parameters, logits):
        mask = torch.tensor(
            [[1.0, 0.0, 0.5], [0.25, 1.0, 0.0]], device=parameters.device
        )
        return parameters * mask


class MultiplicativeBiasOption(MultiplicativeDynamicBias):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, parameters, logits):
        return parameters * 2.0


class ParentOption(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = AdditiveOption()

    def forward(self, parameters, logits):
        return self.child(parameters, logits)


class BatchedCollapsedOption(nn.Module):
    """Emits an identical generated tensor for every sample (non-adaptive)."""

    def forward(self, parameters, logits):
        batch_size = logits.shape[0]
        expanded = parameters.unsqueeze(0).expand(batch_size, *parameters.shape)
        return expanded + 1.0


class BatchedDiverseOption(nn.Module):
    """Emits a different generated direction per sample (input-adaptive)."""

    def forward(self, parameters, logits):
        batch_size = logits.shape[0]
        expanded = parameters.unsqueeze(0).expand(batch_size, *parameters.shape).clone()
        per_sample_offsets = torch.zeros_like(expanded)
        flat_offsets = per_sample_offsets.view(batch_size, -1)
        for index in range(batch_size):
            flat_offsets[index, index % flat_offsets.shape[1]] = float(index + 1)
        return expanded + per_sample_offsets


class TestAdaptiveParameterMonitorCallback(unittest.TestCase):
    def test_tracking_emits_scalar_facts_in_stable_order(self):
        adaptive = self.build_adaptive(weight_model=AdditiveOption())
        module = FakeLightningModule(adaptive)
        self.primed_callback(module, log_every_n_steps=1)

        self.feed_adaptive(adaptive)

        prefix = "adaptive/weight/batch"
        self.assertEqual(
            [name for name, _ in module.logged_scalars],
            [
                f"{prefix}/output_mean",
                f"{prefix}/output_var",
                f"{prefix}/output_min",
                f"{prefix}/output_max",
                f"{prefix}/output_l2_norm",
                f"{prefix}/output_max_abs",
                f"{prefix}/base_mean",
                f"{prefix}/base_var",
                f"{prefix}/delta_mean",
                f"{prefix}/delta_var",
                f"{prefix}/delta_l2_norm",
                f"{prefix}/relative_delta_norm",
                f"{prefix}/cross_sample_std",
                f"{prefix}/adaptivity_ratio",
                f"{prefix}/centroid_cosine_mean",
            ],
        )

    def build_adaptive(
        self,
        weight_model=None,
        diagonal_model=None,
        bias_model=None,
        mask_model=None,
    ) -> AdaptiveParameterAugmentation:
        adaptive = AdaptiveParameterAugmentation(
            AdaptiveParameterAugmentationConfig(
                input_dim=2,
                output_dim=3,
                grouping_scope=AdaptiveParameterGroupingScopeOptions.DISABLED,
                weight_config=None,
                diagonal_config=None,
                bias_config=None,
                mask_config=None,
                model_config=None,
            )
        )
        adaptive.weight_model = weight_model
        adaptive.diagonal_model = diagonal_model
        adaptive.bias_model = bias_model
        adaptive.mask_model = mask_model
        return adaptive

    def primed_callback(
        self, module: FakeLightningModule, **callback_kwargs
    ) -> AdaptiveParameterMonitorCallback:
        callback = AdaptiveParameterMonitorCallback(**callback_kwargs)
        callback.on_fit_start(trainer=None, pl_module=module)
        return callback

    def feed_adaptive(self, adaptive: AdaptiveParameterAugmentation) -> None:
        weights = torch.arange(1, 7, dtype=torch.float32).view(2, 3)
        bias = torch.tensor([1.0, 2.0, 3.0])
        logits = torch.ones(4, 2)
        adaptive(lambda weight, bias, input: weight, weights, bias, logits)

    def scalar_names(self, module: FakeLightningModule) -> set[str]:
        return {name for name, _ in module.logged_scalars}

    def layer_stack_config(
        self,
        input_dim: int = 2,
        hidden_dim: int = 4,
        output_dim: int = 3,
    ) -> LayerStackConfig:
        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=1,
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

    def real_adaptive_config(self) -> AdaptiveParameterAugmentationConfig:
        input_dim = 2
        output_dim = 3
        model_config = self.layer_stack_config(
            input_dim=input_dim, output_dim=output_dim
        )
        return AdaptiveParameterAugmentationConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            grouping_scope=AdaptiveParameterGroupingScopeOptions.DISABLED,
            weight_config=DualModelDynamicWeightConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                generator_depth=DynamicDepthOptions.DEPTH_OF_ONE,
                decay_schedule=WeightDecayScheduleOptions.DISABLED,
                decay_rate=0.0,
                decay_warmup_batches=0,
                normalization_option=WeightNormalizationOptions.DISABLED,
                normalization_position_option=WeightNormalizationPositionOptions.DISABLED,
                model_config=model_config,
            ),
            diagonal_config=None,
            bias_config=AdditiveDynamicBiasConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                decay_schedule=WeightDecayScheduleOptions.DISABLED,
                decay_rate=0.0,
                decay_warmup_batches=0,
                model_config=model_config,
            ),
            mask_config=PerAxisScoreMaskConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                mask_dimension_option=MaskDimensionOptions.ROW,
                mask_threshold=0.5,
                mask_surrogate_scale=1.0,
                mask_floor=0.0,
                model_config=model_config,
            ),
            model_config=None,
        )

    def test_on_fit_start_discovers_adaptive_parameter_modules(self):
        adaptive = self.build_adaptive(weight_model=AdditiveOption())
        module = FakeLightningModule(adaptive)

        callback = self.primed_callback(module)

        self.assertEqual(len(callback._hooks), 1)
        self.feed_adaptive(adaptive)
        self.assertIn("adaptive/weight/batch/output_mean", self.scalar_names(module))

    def test_hooks_attach_only_for_enabled_direct_option_slots(self):
        adaptive = self.build_adaptive(
            weight_model=AdditiveOption(),
            diagonal_model=ParentOption(),
            mask_model=MaskOption(),
        )
        module = FakeLightningModule(adaptive)

        callback = self.primed_callback(module)

        self.assertEqual(len(callback._hooks), 3)

        self.feed_adaptive(adaptive)

        names = self.scalar_names(module)
        self.assertIn("adaptive/diagonal/batch/output_mean", names)
        self.assertNotIn("adaptive/diagonal_model.child/batch/output_mean", names)

    def test_repeated_fit_start_replaces_existing_hooks(self):
        adaptive = self.build_adaptive(weight_model=AdditiveOption())
        module = FakeLightningModule(adaptive)
        callback = self.primed_callback(module, log_every_n_steps=1)

        callback.on_fit_start(trainer=None, pl_module=module)
        self.feed_adaptive(adaptive)

        self.assertEqual(len(callback._hooks), 1)
        output_mean_logs = [
            name
            for name, _ in module.logged_scalars
            if name == "adaptive/weight/batch/output_mean"
        ]
        self.assertEqual(len(output_mean_logs), 1)

    def test_multiple_augmentations_keep_hooks_and_metric_paths_isolated(self):
        first = self.build_adaptive(weight_model=AdditiveOption())
        second = self.build_adaptive(weight_model=AdditiveOption())
        module = FakeLightningModule(first)
        module.second_adaptive = second
        callback = self.primed_callback(module, log_every_n_steps=1)

        self.feed_adaptive(first)
        self.feed_adaptive(second)

        self.assertEqual(len(callback._hooks), 2)
        output_mean_tags = [
            name for name, _ in module.logged_scalars if name.endswith("/output_mean")
        ]
        self.assertEqual(
            output_mean_tags,
            [
                "adaptive/weight/batch/output_mean",
                "second_adaptive/weight/batch/output_mean",
            ],
        )

    def test_same_step_repeated_slot_calls_repeat_scalars_and_deduplicate_media(
        self,
    ):
        experiment = FakeExperiment()
        adaptive = self.build_adaptive(weight_model=AdditiveOption())
        module = FakeLightningModule(adaptive, experiment=experiment, global_step=7)
        self.primed_callback(
            module,
            log_every_n_steps=1,
            log_histograms=True,
        )

        self.feed_adaptive(adaptive)
        self.feed_adaptive(adaptive)

        output_mean_tags = [
            name for name, _ in module.logged_scalars if name.endswith("/output_mean")
        ]
        self.assertEqual(
            output_mean_tags,
            [
                "adaptive/weight/batch/output_mean",
                "adaptive/weight/batch/output_mean",
            ],
        )
        self.assertEqual(
            [tag for tag, _, _ in experiment.histograms],
            [
                "adaptive/weight/batch/output",
                "adaptive/weight/batch/delta",
            ],
        )
        self.assertEqual([step for _, _, step in experiment.histograms], [7, 7])

    def test_callback_reuse_clears_media_claims_between_fits(self):
        callback = AdaptiveParameterMonitorCallback(
            log_every_n_steps=1,
            log_histograms=True,
        )
        experiments = (FakeExperiment(), FakeExperiment())

        for experiment in experiments:
            adaptive = self.build_adaptive(weight_model=AdditiveOption())
            module = FakeLightningModule(
                adaptive,
                experiment=experiment,
                global_step=0,
            )
            callback.on_fit_start(trainer=None, pl_module=module)
            self.feed_adaptive(adaptive)
            callback.on_fit_end(trainer=None, pl_module=module)

        for experiment in experiments:
            self.assertEqual(
                [tag for tag, _, _ in experiment.histograms],
                [
                    "adaptive/weight/batch/output",
                    "adaptive/weight/batch/delta",
                ],
            )

    def test_forward_hook_skips_when_not_at_logging_interval(self):
        adaptive = self.build_adaptive(weight_model=AdditiveOption())
        module = FakeLightningModule(adaptive, global_step=3)
        self.primed_callback(module, log_every_n_steps=5)

        self.feed_adaptive(adaptive)

        self.assertEqual(module.logged_scalars, [])

    def test_forward_hook_ignores_non_tensor_output(self):
        module = FakeLightningModule(self.build_adaptive())
        callback = AdaptiveParameterMonitorCallback(log_every_n_steps=1)
        hook = callback._AdaptiveParameterMonitorCallback__make_forward_hook(
            "adaptive",
            "weight",
            module,
            suppress_input_adaptivity=False,
        )

        hook(nn.Identity(), (torch.ones(2, 3),), object())

        self.assertEqual(module.logged_scalars, [])

    def test_forward_hook_without_tensor_base_logs_only_output_metrics(self):
        module = FakeLightningModule(self.build_adaptive())
        callback = AdaptiveParameterMonitorCallback(log_every_n_steps=1)
        hook = callback._AdaptiveParameterMonitorCallback__make_forward_hook(
            "adaptive",
            "weight",
            module,
            suppress_input_adaptivity=False,
        )

        hook(
            nn.Identity(),
            (),
            torch.tensor(
                [
                    [1.0, -2.0],
                    [3.0, 4.0],
                ]
            ),
        )

        names = self.scalar_names(module)
        self.assertIn("adaptive/weight/batch/output_mean", names)
        for suffix in (
            "base_mean",
            "base_var",
            "delta_mean",
            "delta_var",
            "delta_l2_norm",
            "relative_delta_norm",
        ):
            self.assertNotIn(f"adaptive/weight/batch/{suffix}", names)

    def test_common_stats_are_logged_for_all_option_slots(self):
        adaptive = self.build_adaptive(
            weight_model=AdditiveOption(),
            diagonal_model=AdditiveOption(),
            bias_model=AdditiveOption(),
            mask_model=MaskOption(),
        )
        module = FakeLightningModule(adaptive, global_step=0)
        self.primed_callback(module, log_every_n_steps=1)

        self.feed_adaptive(adaptive)

        names = self.scalar_names(module)
        for slot in ("weight", "diagonal", "bias", "mask"):
            with self.subTest(slot=slot):
                for metric in (
                    "output_mean",
                    "output_var",
                    "output_min",
                    "output_max",
                    "output_l2_norm",
                    "output_max_abs",
                    "base_mean",
                    "base_var",
                    "delta_mean",
                    "delta_var",
                    "delta_l2_norm",
                    "relative_delta_norm",
                ):
                    self.assertIn(f"adaptive/{slot}/batch/{metric}", names)

    def test_real_weight_bias_and_mask_options_are_monitored(self):
        adaptive = AdaptiveParameterAugmentation(self.real_adaptive_config())
        module = FakeLightningModule(adaptive, global_step=0)
        callback = self.primed_callback(module, log_every_n_steps=1)

        self.feed_adaptive(adaptive)

        self.assertEqual(len(callback._hooks), 3)

        names = self.scalar_names(module)
        for slot in ("weight", "bias", "mask"):
            with self.subTest(slot=slot):
                self.assertIn(f"adaptive/{slot}/batch/output_mean", names)
                self.assertIn(f"adaptive/{slot}/batch/delta_l2_norm", names)
                self.assertIn(f"adaptive/{slot}/batch/relative_delta_norm", names)
        self.assertIn("adaptive/mask/batch/relative_output_norm", names)
        self.assertIn("adaptive/mask/batch/attenuated_fraction", names)
        self.assertIn("adaptive/mask/batch/near_zero_fraction", names)

    def test_weight_counter_metrics_observe_post_forward_decay_policy_state(self):
        config = self.real_adaptive_config()
        config.weight_config.decay_schedule = WeightDecayScheduleOptions.EXPONENTIAL
        config.weight_config.decay_rate = 0.1
        config.weight_config.decay_warmup_batches = 1
        adaptive = AdaptiveParameterAugmentation(config)
        module = FakeLightningModule(adaptive, global_step=0)
        self.primed_callback(module, log_every_n_steps=1)

        expected_counter_values = (
            (0.0, 1.0),
            (1.0, 1.0),
        )
        for expected_decay_step, expected_warmup_step in expected_counter_values:
            module.logged_scalars.clear()
            self.feed_adaptive(adaptive)
            logged = dict(module.logged_scalars)
            self.assertEqual(
                logged["adaptive/weight/batch/decay_step"].item(),
                expected_decay_step,
            )
            self.assertEqual(
                logged["adaptive/weight/batch/warmup_step"].item(),
                expected_warmup_step,
            )
            self.assertNotIn("adaptive/bias/batch/decay_step", logged)
            self.assertNotIn("adaptive/bias/batch/warmup_step", logged)

        adaptive.eval()
        module.logged_scalars.clear()
        self.feed_adaptive(adaptive)
        evaluation_logs = dict(module.logged_scalars)
        self.assertEqual(
            evaluation_logs["adaptive/weight/batch/decay_step"].item(),
            1.0,
        )
        self.assertEqual(
            evaluation_logs["adaptive/weight/batch/warmup_step"].item(),
            1.0,
        )

    def test_diagnostic_values_are_detached_without_changing_source_gradients(self):
        module = FakeLightningModule(self.build_adaptive())
        callback = AdaptiveParameterMonitorCallback(log_every_n_steps=1)
        hook = callback._AdaptiveParameterMonitorCallback__make_forward_hook(
            "adaptive",
            "weight",
            module,
            suppress_input_adaptivity=False,
        )
        base = torch.tensor([[1.0, -2.0], [0.5, 3.0]], requires_grad=True)
        output = base * 2.0

        hook(nn.Identity(), (base,), output)

        for _, value in module.logged_scalars:
            self.assertIsNone(value.grad_fn)
        output.sum().backward()
        torch.testing.assert_close(base.grad, torch.full_like(base, 2.0))

    def test_bank_stats_are_logged_for_weight_and_bias_options(self):
        adaptive = self.build_adaptive(
            weight_model=BankOption(),
            bias_model=BankOption(),
        )
        module = FakeLightningModule(adaptive)
        self.primed_callback(module, log_every_n_steps=1)

        self.feed_adaptive(adaptive)

        names = self.scalar_names(module)
        for slot in ("weight", "bias"):
            with self.subTest(slot=slot):
                self.assertIn(f"adaptive/{slot}/batch/weight_bank_mean", names)
                self.assertIn(f"adaptive/{slot}/batch/weight_bank_var", names)
                self.assertIn(f"adaptive/{slot}/batch/weight_bank_l2_norm", names)

    def test_internal_stats_can_be_disabled(self):
        adaptive = self.build_adaptive(
            weight_model=BankOption(),
            bias_model=MultiplicativeBiasOption(),
            mask_model=MaskOption(),
        )
        module = FakeLightningModule(adaptive)
        self.primed_callback(module, log_every_n_steps=1, log_internal_stats=False)

        self.feed_adaptive(adaptive)

        names = self.scalar_names(module)
        self.assertIn("adaptive/weight/batch/output_mean", names)
        self.assertNotIn("adaptive/weight/batch/weight_bank_mean", names)
        self.assertNotIn("adaptive/bias/batch/effective_scale_mean", names)
        self.assertNotIn("adaptive/mask/batch/relative_output_norm", names)

    def test_effective_scale_logged_for_multiplicative_bias_options(self):
        adaptive = self.build_adaptive(bias_model=MultiplicativeBiasOption())
        module = FakeLightningModule(adaptive)
        self.primed_callback(module, log_every_n_steps=1)

        self.feed_adaptive(adaptive)

        scalars = dict(module.logged_scalars)
        torch.testing.assert_close(
            scalars["adaptive/bias/batch/effective_scale_mean"],
            torch.tensor(2.0),
        )
        torch.testing.assert_close(
            scalars["adaptive/bias/batch/effective_scale_var"],
            torch.tensor(0.0),
        )

    def test_effective_scale_skips_real_multiplicative_bias_with_zero_base(self):
        adaptive = AdaptiveParameterAugmentation(
            AdaptiveParameterAugmentationConfig(
                input_dim=2,
                output_dim=3,
                grouping_scope=AdaptiveParameterGroupingScopeOptions.DISABLED,
                bias_config=MultiplicativeDynamicBiasConfig(
                    input_dim=2,
                    output_dim=3,
                    decay_schedule=WeightDecayScheduleOptions.DISABLED,
                    decay_rate=0.0,
                    decay_warmup_batches=0,
                    model_config=self.layer_stack_config(
                        input_dim=2,
                        output_dim=3,
                    ),
                ),
            )
        )
        module = FakeLightningModule(adaptive)
        self.primed_callback(module, log_every_n_steps=1)
        weights = torch.arange(1, 7, dtype=torch.float32).view(2, 3)
        bias = torch.tensor([0.0, 2.0, 3.0])
        inputs = torch.ones(4, 2)

        adaptive(
            lambda adjusted_weight, adjusted_bias, batch: adjusted_weight,
            weights,
            bias,
            inputs,
        )

        names = self.scalar_names(module)
        self.assertIn("adaptive/bias/batch/output_mean", names)
        self.assertNotIn("adaptive/bias/batch/effective_scale_mean", names)
        self.assertNotIn("adaptive/bias/batch/effective_scale_var", names)

    def test_mask_stats_are_logged_from_input_and_output_weights(self):
        adaptive = self.build_adaptive(mask_model=MaskOption())
        module = FakeLightningModule(adaptive)
        self.primed_callback(module, log_every_n_steps=1)

        self.feed_adaptive(adaptive)

        names = self.scalar_names(module)
        self.assertIn("adaptive/mask/batch/relative_output_norm", names)
        self.assertIn("adaptive/mask/batch/attenuated_fraction", names)
        self.assertIn("adaptive/mask/batch/near_zero_fraction", names)

    def test_histograms_are_logged_when_enabled_and_supported(self):
        experiment = FakeExperiment()
        adaptive = self.build_adaptive(weight_model=AdditiveOption())
        module = FakeLightningModule(adaptive, experiment=experiment, global_step=7)
        self.primed_callback(module, log_every_n_steps=1, log_histograms=True)

        self.feed_adaptive(adaptive)

        histogram_tags = {tag for tag, _, _ in experiment.histograms}
        self.assertIn("adaptive/weight/batch/output", histogram_tags)
        self.assertIn("adaptive/weight/batch/delta", histogram_tags)
        for _, _, step in experiment.histograms:
            self.assertEqual(step, 7)

    def test_histograms_are_not_logged_when_disabled_with_supported_experiment(self):
        experiment = FakeExperiment()
        adaptive = self.build_adaptive(weight_model=AdditiveOption())
        module = FakeLightningModule(adaptive, experiment=experiment, global_step=7)
        self.primed_callback(module, log_every_n_steps=1, log_histograms=False)

        self.feed_adaptive(adaptive)

        self.assertGreater(len(module.logged_scalars), 0)
        self.assertEqual(experiment.histograms, [])

    def test_histogram_logging_skips_without_compatible_experiment(self):
        adaptive = self.build_adaptive(weight_model=AdditiveOption())
        module = FakeLightningModule(
            adaptive, experiment=IncompatibleExperiment(), global_step=0
        )
        self.primed_callback(module, log_every_n_steps=1, log_histograms=True)

        self.feed_adaptive(adaptive)

        self.assertGreater(len(module.logged_scalars), 0)

    def test_on_fit_end_removes_hooks_and_clears_state(self):
        adaptive = self.build_adaptive(weight_model=AdditiveOption())
        module = FakeLightningModule(adaptive)
        callback = self.primed_callback(module, log_every_n_steps=1)

        callback.on_fit_end(trainer=None, pl_module=module)
        self.feed_adaptive(adaptive)

        self.assertEqual(module.logged_scalars, [])
        self.assertEqual(callback._hooks, [])

    def test_on_exception_removes_hooks_and_clears_state(self):
        adaptive = self.build_adaptive(weight_model=AdditiveOption())
        module = FakeLightningModule(adaptive)
        callback = self.primed_callback(module, log_every_n_steps=1)

        callback.on_exception(
            trainer=None,
            pl_module=module,
            exception=RuntimeError("deliberate failure"),
        )
        self.feed_adaptive(adaptive)

        self.assertEqual(module.logged_scalars, [])
        self.assertEqual(callback._hooks, [])

    def test_invalid_log_every_n_steps_raises(self):
        with self.assertRaises(ValueError):
            AdaptiveParameterMonitorCallback(log_every_n_steps=0)

    def feed_adaptive_with_batch(
        self, adaptive: AdaptiveParameterAugmentation, batch_size: int
    ) -> None:
        weights = torch.arange(1, 7, dtype=torch.float32).view(2, 3)
        bias = torch.tensor([1.0, 2.0, 3.0])
        logits = torch.ones(batch_size, 2)
        adaptive(lambda weight, bias, input: weight, weights, bias, logits)

    def test_input_adaptivity_metrics_logged_for_real_options(self):
        adaptive = AdaptiveParameterAugmentation(self.real_adaptive_config())
        module = FakeLightningModule(adaptive, global_step=0)
        self.primed_callback(module, log_every_n_steps=1)

        self.feed_adaptive(adaptive)

        names = self.scalar_names(module)
        for slot in ("weight", "bias", "mask"):
            with self.subTest(slot=slot):
                self.assertIn(f"adaptive/{slot}/batch/cross_sample_std", names)
                self.assertIn(f"adaptive/{slot}/batch/adaptivity_ratio", names)
                self.assertIn(f"adaptive/{slot}/batch/centroid_cosine_mean", names)
        for name, value in module.logged_scalars:
            self.assertTrue(torch.isfinite(value).all(), f"{name} not finite")

    def test_input_adaptivity_zero_for_collapsed_delta(self):
        adaptive = self.build_adaptive(weight_model=BatchedCollapsedOption())
        module = FakeLightningModule(adaptive, global_step=0)
        self.primed_callback(module, log_every_n_steps=1)

        self.feed_adaptive(adaptive)

        scalars = dict(module.logged_scalars)
        self.assertEqual(scalars["adaptive/weight/batch/cross_sample_std"].item(), 0.0)
        self.assertEqual(scalars["adaptive/weight/batch/adaptivity_ratio"].item(), 0.0)
        self.assertAlmostEqual(
            scalars["adaptive/weight/batch/centroid_cosine_mean"].item(),
            1.0,
            places=5,
        )

    def test_input_adaptivity_positive_for_per_sample_variation(self):
        adaptive = self.build_adaptive(weight_model=BatchedDiverseOption())
        module = FakeLightningModule(adaptive, global_step=0)
        self.primed_callback(module, log_every_n_steps=1)

        self.feed_adaptive(adaptive)

        scalars = dict(module.logged_scalars)
        self.assertGreater(
            scalars["adaptive/weight/batch/cross_sample_std"].item(), 0.0
        )
        self.assertGreater(
            scalars["adaptive/weight/batch/adaptivity_ratio"].item(), 0.0
        )
        self.assertLess(
            scalars["adaptive/weight/batch/centroid_cosine_mean"].item(), 1.0
        )

    def test_input_adaptivity_skipped_for_single_sample(self):
        adaptive = AdaptiveParameterAugmentation(self.real_adaptive_config())
        module = FakeLightningModule(adaptive, global_step=0)
        self.primed_callback(module, log_every_n_steps=1)

        self.feed_adaptive_with_batch(adaptive, batch_size=1)

        names = self.scalar_names(module)
        self.assertNotIn("adaptive/weight/batch/cross_sample_std", names)

    def test_grouped_owner_suppresses_only_legacy_sample_adaptivity_metrics(self):
        adaptive = AdaptiveLinearLayer(
            AdaptiveLinearLayerConfig(
                input_dim=2,
                output_dim=2,
                bias_flag=True,
                adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                    grouping_scope=AdaptiveParameterGroupingScopeOptions.ROWS,
                    group_count=2,
                    bias_config=AdditiveDynamicBiasConfig(
                        decay_schedule=WeightDecayScheduleOptions.DISABLED,
                        decay_rate=0.0,
                        decay_warmup_batches=0,
                        model_config=self.layer_stack_config(
                            input_dim=2,
                            output_dim=2,
                        ),
                    ),
                ),
            )
        )
        module = FakeLightningModule(adaptive, global_step=0)
        self.primed_callback(module, log_every_n_steps=1)

        adaptive(
            torch.ones(4, 2),
            row_layout=RowLayout.rows(
                4,
                context_sharing_restricted=False,
            ),
        )

        prefix = "adaptive.adaptive_behaviour/bias/batch"
        names = self.scalar_names(module)
        self.assertIn(f"{prefix}/output_mean", names)
        self.assertNotIn(f"{prefix}/cross_sample_std", names)
        self.assertNotIn(f"{prefix}/adaptivity_ratio", names)
        self.assertNotIn(f"{prefix}/centroid_cosine_mean", names)


if __name__ == "__main__":
    unittest.main()
