import unittest

import torch
from emperor.convs.core._validator import Conv2dLayerValidator
from emperor.convs.core.config import Conv2dLayerConfig
from emperor.convs.core.layers import Conv2dLayer


def make_config(**overrides) -> Conv2dLayerConfig:
    values = {
        "input_dim": 3,
        "output_dim": 5,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "bias_flag": True,
    }
    values.update(overrides)
    return Conv2dLayerConfig(**values)


class TestConv2dLayerValidator(unittest.TestCase):
    def test_module_exposes_validator_adapter(self):
        self.assertIs(Conv2dLayer.VALIDATOR, Conv2dLayerValidator)

    def test_valid_configuration_and_forward_input(self):
        model = Conv2dLayer(make_config())
        output = model(torch.randn(2, 3, 8, 8))

        self.assertEqual(output.shape, (2, 5, 8, 8))

    def test_rejects_invalid_kernel_size(self):
        with self.assertRaisesRegex(
            ValueError, "kernel_size must be >= 1, received 0"
        ):
            Conv2dLayer(make_config(kernel_size=0))

    def test_rejects_invalid_field_type(self):
        with self.assertRaisesRegex(
            TypeError,
            "input_dim must be int for Conv2dLayerConfig, got float",
        ):
            Conv2dLayer(make_config(input_dim=3.0))

    def test_rejects_non_tensor_forward_input(self):
        model = Conv2dLayer(make_config())

        with self.assertRaisesRegex(
            TypeError,
            "forward input must be a Tensor, received list",
        ):
            model([])

    def test_rejects_non_four_dimensional_tensor(self):
        model = Conv2dLayer(make_config())

        with self.assertRaisesRegex(
            ValueError,
            "Conv2dLayer expects a 4D input tensor",
        ):
            model(torch.randn(2, 3, 8))


if __name__ == "__main__":
    unittest.main()
