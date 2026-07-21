import unittest
from dataclasses import FrozenInstanceError

import torch

from emperor.linears import LinearMonitorCallback
from emperor.monitoring import (
    MonitorEmissionPolicy,
    MonitorOption,
    MonitorSettings,
    MonitorTensorHistory,
)


class CaptureExperiment:
    def __init__(self):
        self.histograms = []
        self.images = []

    def add_histogram(self, tag, values, step):
        self.histograms.append((tag, values.detach().clone(), step))

    def add_image(self, tag, image, step=None, global_step=None, dataformats=None):
        resolved_step = global_step if global_step is not None else step
        self.images.append((tag, image.detach().clone(), resolved_step, dataformats))


class TestMonitorEmissionPolicy(unittest.TestCase):
    def test_monitor_settings_defaults_validation_and_immutability(self):
        settings = MonitorSettings()

        self.assertEqual(settings.log_every_n_steps, 100)
        with self.assertRaisesRegex(
            ValueError,
            "monitor log cadence must be at least one step.",
        ):
            MonitorSettings(log_every_n_steps=0)
        with self.assertRaisesRegex(
            ValueError,
            "monitor log cadence must be at least one step.",
        ):
            MonitorSettings(log_every_n_steps=-3)
        with self.assertRaises(FrozenInstanceError):
            settings.log_every_n_steps = 5

    def test_monitor_option_builds_real_callbacks_and_serializes_api_metadata(self):
        def callback_factory(settings):
            return LinearMonitorCallback(
                log_every_n_steps=settings.log_every_n_steps,
                log_weight_conditioning=False,
            )

        option = MonitorOption(
            name="linear-health",
            label="Linear health",
            description="Track real linear diagnostics.",
            kinds=("scalar", "histogram"),
            callback_factory=callback_factory,
            default_enabled=True,
        )

        default_callback = option.build_callback()
        configured_callback = option.build_callback(
            MonitorSettings(log_every_n_steps=7)
        )
        api = option.to_api()

        self.assertIsInstance(default_callback, LinearMonitorCallback)
        self.assertIsInstance(configured_callback, LinearMonitorCallback)
        self.assertIsNot(default_callback, configured_callback)
        self.assertEqual(default_callback.log_every_n_steps, 100)
        self.assertEqual(configured_callback.log_every_n_steps, 7)
        self.assertFalse(default_callback.log_weight_conditioning)
        self.assertEqual(
            api,
            {
                "name": "linear-health",
                "label": "Linear health",
                "description": "Track real linear diagnostics.",
                "kinds": ["scalar", "histogram"],
                "defaultEnabled": True,
            },
        )
        api["kinds"].append("image")
        self.assertEqual(option.kinds, ("scalar", "histogram"))

    def test_tensor_history_validates_constructor_and_empty_states(self):
        with self.assertRaises(ValueError) as max_entries_error:
            MonitorTensorHistory(max_entries=0)
        self.assertEqual(
            str(max_entries_error.exception),
            "max_entries must be greater than 0.",
        )

        with self.assertRaises(ValueError) as normalization_error:
            MonitorTensorHistory(max_entries=1, normalization="invalid")
        self.assertEqual(
            str(normalization_error.exception),
            "normalization must be either 'maximum' or 'unit_interval'.",
        )

        history = MonitorTensorHistory(max_entries=2)
        self.assertFalse(history)
        self.assertEqual(len(history), 0)
        self.assertEqual(history.tensors, ())
        self.assertIsNone(history.render_heatmap())

        history.append(torch.empty(0))
        self.assertTrue(history)
        self.assertEqual(len(history), 1)
        self.assertIsNone(history.render_heatmap())

        history.clear()
        self.assertFalse(history)
        self.assertEqual(history.tensors, ())

    def test_tensor_history_stores_an_isolated_flat_float_snapshot(self):
        history = MonitorTensorHistory(max_entries=2)
        source = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
            requires_grad=True,
        )

        history.append(source)
        with torch.no_grad():
            source.add_(10.0)

        stored = history.tensors[0]
        self.assertEqual(stored.dtype, torch.float32)
        self.assertEqual(stored.device.type, "cpu")
        self.assertFalse(stored.requires_grad)
        self.assertTrue(stored.is_contiguous())
        torch.testing.assert_close(stored, torch.tensor([1.0, 2.0, 3.0, 4.0]))

    def test_tensor_history_bounds_and_detaches_cpu_values(self):
        history = MonitorTensorHistory(max_entries=2)
        source = torch.tensor([1.0, 2.0], requires_grad=True)

        history.append(source)
        history.append(torch.tensor([3.0]))
        history.append(torch.tensor([4.0, 5.0, 6.0]))

        self.assertEqual(len(history), 2)
        self.assertEqual(
            [tensor.tolist() for tensor in history.tensors], [[3.0], [4.0, 5.0, 6.0]]
        )
        for stored_tensor in history.tensors:
            self.assertEqual(stored_tensor.device.type, "cpu")
            self.assertFalse(stored_tensor.requires_grad)

    def test_tensor_history_renders_ragged_maximum_normalized_heatmap(self):
        history = MonitorTensorHistory(max_entries=3, normalization="maximum")
        history.append(torch.tensor([1.0, 2.0]))
        history.append(torch.tensor([4.0]))

        heatmap = history.render_heatmap()

        self.assertIsNotNone(heatmap)
        torch.testing.assert_close(
            heatmap,
            torch.tensor([[[0.25, 1.0], [0.5, 0.0]]]),
        )

    def test_tensor_history_renders_unit_interval_heatmap(self):
        history = MonitorTensorHistory(max_entries=2, normalization="unit_interval")
        history.append(torch.tensor([-0.5, 0.25, 2.0]))

        heatmap = history.render_heatmap()

        self.assertIsNotNone(heatmap)
        torch.testing.assert_close(
            heatmap,
            torch.tensor([[[0.0], [0.25], [1.0]]]),
        )

    def test_maximum_normalization_scales_sub_unit_values_to_one(self):
        history = MonitorTensorHistory(max_entries=2, normalization="maximum")
        history.append(torch.tensor([0.25, 0.5]))

        heatmap = history.render_heatmap()

        torch.testing.assert_close(
            heatmap,
            torch.tensor([[[0.5], [1.0]]]),
        )

    def test_media_cadence_uses_global_step_precedence_and_minimum_one(self):
        every_step = MonitorEmissionPolicy(media_every_n_steps=0)
        every_third_step = MonitorEmissionPolicy(media_every_n_steps=3)

        self.assertTrue(every_step.should_emit_media())
        self.assertTrue(every_step.should_emit_media(step=7))
        self.assertFalse(every_third_step.should_emit_media(step=3, global_step=4))
        self.assertTrue(every_third_step.should_emit_media(step=4, global_step=6))

    def test_missing_experiment_methods_do_not_claim_emission_keys(self):
        policy = MonitorEmissionPolicy()
        experiment = CaptureExperiment()
        no_writer = object()

        self.assertFalse(
            policy.emit_histogram(
                no_writer,
                "layer/histogram/values",
                torch.tensor([1.0]),
                2,
            )
        )
        self.assertFalse(
            policy.emit_image(
                no_writer,
                "layer/image/values",
                torch.ones(2, 2),
                2,
            )
        )
        self.assertTrue(
            policy.emit_histogram(
                experiment,
                "layer/histogram/values",
                torch.tensor([1.0]),
                2,
            )
        )
        self.assertTrue(
            policy.emit_image(
                experiment,
                "layer/image/values",
                torch.ones(2, 2),
                2,
            )
        )

    def test_default_module_key_and_zero_step_are_recorded_exactly(self):
        experiment = CaptureExperiment()
        policy = MonitorEmissionPolicy()

        self.assertTrue(
            policy.emit_histogram(
                experiment,
                "outer/inner/histogram",
                torch.tensor([1.0]),
                0,
            )
        )
        self.assertTrue(
            policy.emit_image(
                experiment,
                "standalone",
                torch.ones(2, 2),
            )
        )

        self.assertEqual(
            list(policy._emitted),
            [
                ("histogram", "outer", "outer/inner/histogram", 0),
                ("image", "standalone", "standalone", 0),
            ],
        )
        self.assertEqual(experiment.images[0][2], 0)

    def test_histogram_preserves_small_payload_and_clamps_zero_limit_to_one(self):
        experiment = CaptureExperiment()
        unchanged_policy = MonitorEmissionPolicy(histogram_max_elements=10)
        clamped_policy = MonitorEmissionPolicy(histogram_max_elements=0)
        values = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float64,
            requires_grad=True,
        ).T

        self.assertTrue(
            unchanged_policy.emit_histogram(
                experiment,
                "first/histogram",
                values,
                3,
            )
        )
        self.assertTrue(
            clamped_policy.emit_histogram(
                experiment,
                "second/histogram",
                values,
                3,
            )
        )

        unchanged = experiment.histograms[0][1]
        clamped = experiment.histograms[1][1]
        self.assertEqual(unchanged.dtype, torch.float32)
        self.assertEqual(unchanged.device.type, "cpu")
        self.assertFalse(unchanged.requires_grad)
        torch.testing.assert_close(unchanged, torch.tensor([1.0, 3.0, 2.0, 4.0]))
        torch.testing.assert_close(clamped, torch.tensor([1.0]))

    def test_caps_histogram_elements_and_dedupes_same_step_tag(self):
        experiment = CaptureExperiment()
        policy = MonitorEmissionPolicy(histogram_max_elements=5)

        first = policy.emit_histogram(
            experiment,
            "layer/histogram/values",
            torch.arange(20),
            10,
            module_key="layer",
        )
        second = policy.emit_histogram(
            experiment,
            "layer/histogram/values",
            torch.arange(20),
            10,
            module_key="layer",
        )

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(len(experiment.histograms), 1)
        _, values, step = experiment.histograms[0]
        self.assertEqual(step, 10)
        self.assertEqual(values.numel(), 5)
        self.assertEqual(values[-1].item(), 19)

    def test_emission_history_evicts_oldest_key_and_clear_reenables_all_keys(self):
        experiment = CaptureExperiment()
        policy = MonitorEmissionPolicy(emitted_max_entries=1)

        self.assertTrue(
            policy.emit_histogram(
                experiment,
                "first/histogram",
                torch.tensor([1.0]),
                1,
            )
        )
        self.assertTrue(
            policy.emit_histogram(
                experiment,
                "second/histogram",
                torch.tensor([2.0]),
                1,
            )
        )
        self.assertTrue(
            policy.emit_histogram(
                experiment,
                "first/histogram",
                torch.tensor([3.0]),
                1,
            )
        )
        self.assertFalse(
            policy.emit_histogram(
                experiment,
                "first/histogram",
                torch.tensor([4.0]),
                1,
            )
        )

        policy.clear()

        self.assertTrue(
            policy.emit_histogram(
                experiment,
                "first/histogram",
                torch.tensor([5.0]),
                1,
            )
        )
        self.assertEqual(
            [values.item() for _, values, _ in experiment.histograms],
            [1.0, 2.0, 3.0, 5.0],
        )

    def test_caps_image_payload_and_applies_media_cadence(self):
        experiment = CaptureExperiment()
        policy = MonitorEmissionPolicy(
            image_max_raw_bytes=256,
            image_max_side=64,
            media_every_n_steps=2,
        )

        skipped = policy.emit_image(
            experiment,
            "layer/heatmap/large",
            torch.ones(1, 128, 128),
            3,
            dataformats="CHW",
        )
        emitted = policy.emit_image(
            experiment,
            "layer/heatmap/large",
            torch.ones(1, 128, 128),
            4,
            dataformats="CHW",
        )

        self.assertFalse(skipped)
        self.assertTrue(emitted)
        self.assertEqual(len(experiment.images), 1)
        _, image, step, dataformats = experiment.images[0]
        self.assertEqual(step, 4)
        self.assertEqual(dataformats, "CHW")
        self.assertLessEqual(image.numel() * image.element_size(), 256)
        self.assertLessEqual(max(image.shape[-2:]), 64)

    def test_image_without_explicit_format_uses_keyword_global_step_and_snapshot(self):
        experiment = CaptureExperiment()
        policy = MonitorEmissionPolicy()
        source = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float64,
            requires_grad=True,
        )

        emitted = policy.emit_image(
            experiment,
            "layer/image/small",
            source,
            global_step=5,
        )
        with torch.no_grad():
            source.add_(10.0)

        self.assertTrue(emitted)
        tag, image, step, dataformats = experiment.images[0]
        self.assertEqual(tag, "layer/image/small")
        self.assertEqual(step, 5)
        self.assertIsNone(dataformats)
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(image.device.type, "cpu")
        self.assertFalse(image.requires_grad)
        torch.testing.assert_close(image, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def test_image_scaling_preserves_each_supported_layout_and_values(self):
        cases = (
            ("HW", (4, 6), (1, 2)),
            ("CHW", (1, 4, 6), (1, 1, 2)),
            ("NCHW", (2, 1, 4, 6), (2, 1, 1, 2)),
            ("HWC", (4, 6, 1), (1, 2, 1)),
            ("NHWC", (2, 4, 6, 1), (2, 1, 2, 1)),
            ("XY", (4, 6), (1, 2)),
        )

        for index, (dataformats, shape, expected_shape) in enumerate(cases):
            with self.subTest(dataformats=dataformats):
                experiment = CaptureExperiment()
                policy = MonitorEmissionPolicy(
                    image_max_raw_bytes=1_000_000,
                    image_max_side=2,
                )

                emitted = policy.emit_image(
                    experiment,
                    f"layer/image/{index}",
                    torch.ones(shape),
                    step=index,
                    dataformats=dataformats,
                )

                self.assertTrue(emitted)
                image = experiment.images[0][1]
                self.assertEqual(tuple(image.shape), expected_shape)
                torch.testing.assert_close(image, torch.ones(expected_shape))

    def test_image_format_inference_covers_matrix_channel_and_batch_layouts(self):
        cases = (
            ((4, 6), (1, 2)),
            ((1, 4, 6), (1, 1, 2)),
            ((2, 1, 4, 6), (2, 1, 1, 2)),
        )

        for index, (shape, expected_shape) in enumerate(cases):
            with self.subTest(shape=shape):
                experiment = CaptureExperiment()
                policy = MonitorEmissionPolicy(image_max_side=2)

                self.assertTrue(
                    policy.emit_image(
                        experiment,
                        f"inferred/image/{index}",
                        torch.ones(shape),
                        step=index,
                    )
                )

                self.assertEqual(
                    tuple(experiment.images[0][1].shape),
                    expected_shape,
                )

    def test_unsupported_vector_image_stops_scaling_without_corrupting_shape(self):
        experiment = CaptureExperiment()
        side_limited = MonitorEmissionPolicy(
            image_max_raw_bytes=1_000_000,
            image_max_side=1,
        )
        byte_limited = MonitorEmissionPolicy(
            image_max_raw_bytes=1,
            image_max_side=1,
        )

        self.assertTrue(
            side_limited.emit_image(
                experiment,
                "vector/image/side",
                torch.arange(4),
                step=1,
            )
        )
        self.assertTrue(
            byte_limited.emit_image(
                experiment,
                "vector/image/bytes",
                torch.tensor([7.0]),
                step=1,
            )
        )

        torch.testing.assert_close(experiment.images[0][1], torch.arange(4).float())
        torch.testing.assert_close(experiment.images[1][1], torch.tensor([7.0]))

    def test_empty_history_does_not_emit_image(self):
        experiment = CaptureExperiment()
        policy = MonitorEmissionPolicy()
        history = MonitorTensorHistory(max_entries=2)

        self.assertFalse(
            policy.emit_history_heatmap(
                experiment,
                "layer/heatmap/empty",
                history,
                step=2,
            )
        )
        self.assertEqual(experiment.images, [])

    def test_emits_history_heatmap_through_image_policy(self):
        experiment = CaptureExperiment()
        policy = MonitorEmissionPolicy()
        history = MonitorTensorHistory(max_entries=2)
        history.append(torch.tensor([1.0, 2.0]))

        emitted = policy.emit_history_heatmap(
            experiment,
            "layer/heatmap/history",
            history,
            global_step=6,
        )
        duplicate = policy.emit_history_heatmap(
            experiment,
            "layer/heatmap/history",
            history,
            global_step=6,
        )

        self.assertTrue(emitted)
        self.assertFalse(duplicate)
        self.assertEqual(len(experiment.images), 1)
        tag, image, step, dataformats = experiment.images[0]
        self.assertEqual(tag, "layer/heatmap/history")
        self.assertEqual(step, 6)
        self.assertEqual(dataformats, "CHW")
        self.assertEqual(image.shape, (1, 2, 1))


if __name__ == "__main__":
    unittest.main()
