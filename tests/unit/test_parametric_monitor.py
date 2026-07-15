import unittest

import torch
from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.base.layer import LayerConfig, LayerStackConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.linears.core.config import LinearLayerConfig
from emperor.parametric import (
    AdaptiveRouterOptions,
    ClipParameterOptions,
    MatrixBiasMixtureConfig,
    MatrixWeightsMixtureConfig,
    ParametricLayer,
    ParametricLayerConfig,
)
from emperor.parametric.core.monitor import ParametricLayerMonitorCallback
from emperor.sampler.core.config import RouterConfig, SamplerConfig

from support.monitor import (
    CaptureLightningModule,
    NoExperimentLightningModule,
    TrainerStub,
    orchestration_calls,
    same_bound_method,
)


class TestParametricLayerMonitorCallback(unittest.TestCase):
    def test_tracking_orchestration_lists_each_tracked_fact(self):
        cls = ParametricLayerMonitorCallback
        orchestration = (
            cls._ParametricLayerMonitorCallback__track_parametric_diagnostics
        )
        routed_slot_calls = (
            "__track_router_auxiliary_loss",
            "__track_router_entropy",
            "__track_active_slots",
            "__track_dead_slot_fraction",
            "__track_maximum_utilization",
            "__track_minimum_utilization",
            "__track_per_slot_utilization",
            "__track_utilization_history",
            "__track_utilization_histogram",
            "__track_utilization_heatmap",
        )

        self.assertEqual(
            orchestration_calls(orchestration),
            (
                "__track_generated_parameter_norm",
                "__track_generated_parameter_norm",
                "__track_clip_saturation_fraction",
                "__track_clip_saturation_fraction",
                "__track_auxiliary_loss",
                "__track_skip_fraction",
                "__track_drop_fraction",
                "__track_affine_output_norm",
                "__track_affine_relative_output_norm",
                "__track_affine_delta_norm",
                "__track_affine_relative_delta_norm",
                *routed_slot_calls,
                *routed_slot_calls,
            ),
        )

    def layer_stack_config(self, input_dim: int = 4, output_dim: int = 3):
        hidden_dim = max(input_dim, output_dim)
        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            layer_config=LayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                activation=ActivationOptions.DISABLED,
                residual_connection_option=ResidualConnectionOptions.DISABLED,
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

    def parametric_config(self) -> ParametricLayerConfig:
        top_k = 2
        num_experts = 4
        input_dim = 4
        output_dim = 4
        mixture_kwargs = dict(
            input_dim=input_dim,
            output_dim=output_dim,
            top_k=top_k,
            num_experts=num_experts,
            weighted_parameters_flag=True,
            clip_parameter_option=ClipParameterOptions.DISABLED,
            clip_range=1.0,
        )
        router_config = RouterConfig(
            input_dim=input_dim,
            num_experts=num_experts,
            noisy_topk_flag=False,
            model_config=self.layer_stack_config(
                input_dim=input_dim,
                output_dim=num_experts,
            ),
        )
        return ParametricLayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            weight_mixture_config=MatrixWeightsMixtureConfig(**mixture_kwargs),
            bias_mixture_config=MatrixBiasMixtureConfig(**mixture_kwargs),
            routing_initialization_mode=AdaptiveRouterOptions.INDEPENDENT_ROUTER,
            router_config=router_config,
            sampler_config=SamplerConfig(
                top_k=top_k,
                threshold=0.0,
                filter_above_threshold=False,
                num_topk_samples=0,
                normalize_probabilities_flag=False,
                noisy_topk_flag=False,
                num_experts=num_experts,
                coefficient_of_variation_loss_weight=0.0,
                switch_loss_weight=0.0,
                zero_centred_loss_weight=0.0,
                mutual_information_loss_weight=0.0,
                router_config=router_config,
            ),
            adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                weight_config=None,
                bias_config=None,
                diagonal_config=None,
                mask_config=None,
                model_config=None,
            ),
        )

    def layer(self) -> ParametricLayer:
        return ParametricLayer(self.parametric_config())

    def input(self):
        return torch.randn(3, 4)

    def test_rejects_non_positive_cadence(self):
        for bad in (0, -1):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    ParametricLayerMonitorCallback(log_every_n_steps=bad)

    def test_rejects_non_positive_history_size(self):
        for bad in (0, -1):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    ParametricLayerMonitorCallback(history_size=bad)

    def test_discovers_only_parametric_layers(self):
        module = CaptureLightningModule(
            parametric=self.layer(), other=torch.nn.Linear(4, 4)
        )
        callback = ParametricLayerMonitorCallback(log_every_n_steps=1)

        callback.on_fit_start(TrainerStub(), module)

        self.assertEqual(
            set(callback._utilization_histories),
            {("parametric", "weight"), ("parametric", "bias")},
        )
        callback.on_fit_end(TrainerStub(), module)

    def test_respects_global_step_cadence(self):
        layer = self.layer()
        module = CaptureLightningModule(parametric=layer)
        callback = ParametricLayerMonitorCallback(log_every_n_steps=2)
        callback.on_fit_start(TrainerStub(), module)

        module.global_step = 1
        layer(self.input())
        self.assertEqual(module.logged, [])

        module.global_step = 2
        layer(self.input())
        self.assertIn("parametric/parametric/generated_weight_norm", module.logged_tags)
        callback.on_fit_end(TrainerStub(), module)

    def test_repeated_fit_start_replaces_existing_wrappers(self):
        layer = self.layer()
        original_forward = layer.forward
        module = CaptureLightningModule(parametric=layer)
        callback = ParametricLayerMonitorCallback(log_every_n_steps=1)

        callback.on_fit_start(TrainerStub(), module)
        first_wrapper_count = len(callback._wrapped_methods)
        callback.on_fit_start(TrainerStub(), module)
        layer(self.input())

        self.assertEqual(len(callback._wrapped_methods), first_wrapper_count)
        self.assertEqual(
            module.logged_tags.count("parametric/parametric/generated_weight_norm"),
            1,
        )
        callback.on_fit_end(TrainerStub(), module)
        self.assertTrue(same_bound_method(layer.forward, original_forward))

    def test_logs_expected_finite_scalars(self):
        layer = self.layer()
        module = CaptureLightningModule(parametric=layer)
        callback = ParametricLayerMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        layer(self.input())

        expected_tags = {
            "parametric/parametric/generated_weight_norm",
            "parametric/parametric/generated_bias_norm",
            "parametric/parametric/auxiliary_loss",
            "parametric/parametric/affine/output_norm",
            "parametric/parametric/affine/delta_norm",
            "parametric/router/weight_entropy",
            "parametric/router/bias_entropy",
            "parametric/mixture/weight_active_slots",
            "parametric/mixture/bias_active_slots",
            "parametric/mixture/weight_dead_slot_fraction",
            "parametric/mixture/bias_dead_slot_fraction",
        }
        self.assertTrue(expected_tags.issubset(set(module.logged_tags)))
        for tag in expected_tags:
            self.assertTrue(
                torch.isfinite(torch.as_tensor(module.logged_value(tag))).all(), tag
            )
        callback.on_fit_end(TrainerStub(), module)

    def test_emits_histograms_and_images_when_experiment_supports_them(self):
        layer = self.layer()
        module = CaptureLightningModule(parametric=layer)
        callback = ParametricLayerMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        layer(self.input())

        experiment = module.logger.experiment
        self.assertTrue(
            any(
                tag == "parametric/mixture/histogram/weight_utilization"
                for tag, _, _ in experiment.histograms
            )
        )
        self.assertTrue(
            any(
                tag == "parametric/mixture/heatmap/weight_utilization"
                for tag, _, _, _ in experiment.images
            )
        )
        callback.on_fit_end(TrainerStub(), module)

    def test_skips_visual_summaries_without_experiment(self):
        layer = self.layer()
        module = NoExperimentLightningModule(parametric=layer)
        callback = ParametricLayerMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        layer(self.input())

        self.assertIn("parametric/parametric/generated_weight_norm", module.logged_tags)
        callback.on_fit_end(TrainerStub(), module)

    def test_utilization_histories_are_bounded_and_detached(self):
        layer = self.layer()
        module = CaptureLightningModule(parametric=layer)
        callback = ParametricLayerMonitorCallback(
            log_every_n_steps=1,
            history_size=2,
        )
        callback.on_fit_start(TrainerStub(), module)

        for global_step in range(3):
            module.global_step = global_step
            layer(self.input())

        for history in callback._utilization_histories.values():
            self.assertEqual(len(history), 2)
            for tensor in history.tensors:
                self.assertEqual(tensor.device.type, "cpu")
                self.assertFalse(tensor.requires_grad)
        callback.on_fit_end(TrainerStub(), module)

    def test_restores_wrappers_and_clears_state_on_fit_end(self):
        layer = self.layer()
        original_forward = layer.forward
        original_generate = layer._generate_parameters
        original_affine = layer._compute_affine_transformation_callback
        module = CaptureLightningModule(parametric=layer)
        callback = ParametricLayerMonitorCallback(log_every_n_steps=1)

        callback.on_fit_start(TrainerStub(), module)
        self.assertIsNot(layer.forward, original_forward)
        self.assertIsNot(layer._generate_parameters, original_generate)
        self.assertIsNot(layer._compute_affine_transformation_callback, original_affine)

        callback.on_fit_end(TrainerStub(), module)

        self.assertTrue(same_bound_method(layer.forward, original_forward))
        self.assertTrue(
            same_bound_method(layer._generate_parameters, original_generate)
        )
        self.assertTrue(
            same_bound_method(
                layer._compute_affine_transformation_callback,
                original_affine,
            )
        )
        self.assertEqual(callback._wrapped_methods, [])
        self.assertEqual(callback._observations, {})
        self.assertEqual(callback._utilization_histories, {})


if __name__ == "__main__":
    unittest.main()
