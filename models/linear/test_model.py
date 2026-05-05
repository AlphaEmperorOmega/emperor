import unittest

import torch

import models.linear.config as config

from models.linear.model import Model
from models.linear.presets import ExperimentOptions, ExperimentPresets


class TestLinearModel(unittest.TestCase):
    def test_all_options_forward_one_batch_per_dataset(self):
        batch_size = 4
        presets = ExperimentPresets()

        for dataset in config.DATASET_OPTIONS:
            for option in ExperimentOptions:
                message = f"dataset={dataset.__name__}, option={option.name}"
                with self.subTest(msg=message):
                    cfg = presets.get_config(option, dataset)[0]
                    model = Model(cfg)
                    X = self._fake_batch(dataset, batch_size)

                    output = model(X)
                    logits = output[0] if isinstance(output, tuple) else output

                    self.assertEqual(logits.shape, (batch_size, dataset.num_classes))

    def _fake_batch(self, dataset: type, batch_size: int) -> torch.Tensor:
        return torch.randn(
            batch_size,
            dataset.num_channels,
            dataset.default_height,
            dataset.default_width,
        )


if __name__ == "__main__":
    unittest.main()
