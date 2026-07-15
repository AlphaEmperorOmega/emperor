import unittest

import torch
from emperor.experiments.monitor_policy import (
    MonitorEmissionPolicy,
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
