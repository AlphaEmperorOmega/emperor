import json
import subprocess
import sys
import unittest

import torch
from emperor.convs import Conv2dLayerConfig
from emperor.convs._layer import Conv2dLayer
from emperor.convs._validation import Conv2dLayerValidator


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


class Conv2dInterfaceTests(unittest.TestCase):
    def test_interface_exports_only_config_without_dynamic_shortcuts(self) -> None:
        import emperor.convs as convs

        self.assertEqual(convs.__all__, ("Conv2dLayerConfig",))
        self.assertIs(convs.Conv2dLayerConfig, Conv2dLayerConfig)
        self.assertFalse(hasattr(convs, "Conv2dLayer"))
        self.assertFalse(hasattr(convs, "__getattr__"))
        self.assertFalse(hasattr(convs, "_LAZY_EXPORTS"))

    def test_explicit_interface_is_lightweight_and_build_loads_implementation(
        self,
    ) -> None:
        script = """\
import json
import sys

import emperor.convs as convs

after_interface_import = sorted(
    name for name in sys.modules if name.startswith("emperor.convs.")
)
runtime_before_build = {
    "layer": "emperor.convs._layer" in sys.modules,
    "validation": "emperor.convs._validation" in sys.modules,
    "torch": "torch" in sys.modules,
}
config = convs.Conv2dLayerConfig(
    input_dim=2,
    output_dim=3,
    kernel_size=1,
    stride=1,
    padding=0,
    bias_flag=True,
)
model = config.build()
print(json.dumps({
    "all": convs.__all__,
    "after_interface_import": after_interface_import,
    "runtime_before_build": runtime_before_build,
    "runtime_after_build": {
        "layer": "emperor.convs._layer" in sys.modules,
        "validation": "emperor.convs._validation" in sys.modules,
        "torch": "torch" in sys.modules,
    },
    "config_module": convs.Conv2dLayerConfig.__module__,
    "model_module": type(model).__module__,
    "shortcut_attributes": {
        "__getattr__": hasattr(convs, "__getattr__"),
        "_LAZY_EXPORTS": hasattr(convs, "_LAZY_EXPORTS"),
    },
}))
"""

        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)

        self.assertEqual(
            result,
            {
                "all": ["Conv2dLayerConfig"],
                "after_interface_import": ["emperor.convs._config"],
                "runtime_before_build": {
                    "layer": False,
                    "validation": False,
                    "torch": False,
                },
                "runtime_after_build": {
                    "layer": True,
                    "validation": True,
                    "torch": True,
                },
                "config_module": "emperor.convs._config",
                "model_module": "emperor.convs._layer",
                "shortcut_attributes": {
                    "__getattr__": False,
                    "_LAZY_EXPORTS": False,
                },
            },
        )

    def test_removed_implementation_import_fails(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "from emperor.convs import Conv2dLayer",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("ImportError", completed.stderr)


if __name__ == "__main__":
    unittest.main()
