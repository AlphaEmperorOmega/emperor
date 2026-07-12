import unittest

import torch

from emperor.experiments.monitor_policy import MonitorEmissionPolicy


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


if __name__ == "__main__":
    unittest.main()
