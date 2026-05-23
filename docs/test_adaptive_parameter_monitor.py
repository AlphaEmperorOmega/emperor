import torch
import unittest

from torch import nn

from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.base.layer.config import LayerConfig, LayerStackConfig
from emperor.linears.core.config import LinearLayerConfig
from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterAugmentation,
    AdaptiveParameterAugmentationConfig,
    AdaptiveParameterMonitorCallback,
    DynamicDepthOptions,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.augmentations.adaptive_parameters.core.bias import (
    AdditiveDynamicBiasConfig,
)
from emperor.augmentations.adaptive_parameters.core.mask import (
    PerAxisScoreMaskConfig,
)
from emperor.augmentations.adaptive_parameters.core.weight import (
    DualModelDynamicWeightConfig,
)


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


class ParentOption(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = AdditiveOption()

    def forward(self, parameters, logits):
        return self.child(parameters, logits)


class TestAdaptiveParameterMonitorCallback(unittest.TestCase):
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
                residual_flag=False,
                dropout_probability=0.0,
                gate_config=None,
                halting_config=None,
                shared_halting_flag=False,
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

        self.assertEqual(len(callback._adaptive_modules), 1)
        attached_name, attached_module = callback._adaptive_modules[0]
        self.assertEqual(attached_name, "adaptive")
        self.assertIs(attached_module, adaptive)

    def test_hooks_attach_only_for_enabled_direct_option_slots(self):
        adaptive = self.build_adaptive(
            weight_model=AdditiveOption(),
            diagonal_model=ParentOption(),
            mask_model=MaskOption(),
        )
        module = FakeLightningModule(adaptive)

        callback = self.primed_callback(module)

        monitored_slots = [slot for _, slot, _ in callback._monitored_options]
        self.assertEqual(monitored_slots, ["weight", "diagonal", "mask"])
        self.assertEqual(len(callback._hooks), 3)

        self.feed_adaptive(adaptive)

        names = self.scalar_names(module)
        self.assertIn("adaptive/diagonal/batch/output_mean", names)
        self.assertNotIn("adaptive/diagonal_model.child/batch/output_mean", names)

    def test_forward_hook_skips_when_not_at_logging_interval(self):
        adaptive = self.build_adaptive(weight_model=AdditiveOption())
        module = FakeLightningModule(adaptive, global_step=3)
        self.primed_callback(module, log_every_n_steps=5)

        self.feed_adaptive(adaptive)

        self.assertEqual(module.logged_scalars, [])

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

        monitored_slots = [slot for _, slot, _ in callback._monitored_options]
        self.assertEqual(monitored_slots, ["weight", "bias", "mask"])

        names = self.scalar_names(module)
        for slot in ("weight", "bias", "mask"):
            with self.subTest(slot=slot):
                self.assertIn(f"adaptive/{slot}/batch/output_mean", names)
                self.assertIn(f"adaptive/{slot}/batch/delta_l2_norm", names)
                self.assertIn(f"adaptive/{slot}/batch/relative_delta_norm", names)
        self.assertIn("adaptive/mask/batch/relative_output_norm", names)
        self.assertIn("adaptive/mask/batch/attenuated_fraction", names)
        self.assertIn("adaptive/mask/batch/near_zero_fraction", names)

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
        self.assertEqual(callback._adaptive_modules, [])
        self.assertEqual(callback._monitored_options, [])

    def test_invalid_log_every_n_steps_raises(self):
        with self.assertRaises(ValueError):
            AdaptiveParameterMonitorCallback(log_every_n_steps=0)


if __name__ == "__main__":
    unittest.main()
