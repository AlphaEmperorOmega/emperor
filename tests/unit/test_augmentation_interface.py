import json
import subprocess
import sys
import unittest

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
)

EXPECTED_EXPORTS = (
    "AdaptiveParameterAugmentationConfig",
    "AdaptiveLinearLayerConfig",
    "DynamicWeightConfig",
    "SingleModelDynamicWeightConfig",
    "DualModelDynamicWeightConfig",
    "LowRankDynamicWeightConfig",
    "HypernetworkDynamicWeightConfig",
    "LayeredWeightedBankDynamicWeightConfig",
    "SoftWeightedBankDynamicWeightConfig",
    "DynamicBiasConfig",
    "AffineTransformDynamicBiasConfig",
    "AdditiveDynamicBiasConfig",
    "MultiplicativeDynamicBiasConfig",
    "SigmoidGatedDynamicBiasConfig",
    "TanhGatedDynamicBiasConfig",
    "GeneratorDynamicBiasConfig",
    "WeightedBankDynamicBiasConfig",
    "DynamicDiagonalConfig",
    "StandardDynamicDiagonalConfig",
    "AntiDynamicDiagonalConfig",
    "CombinedDynamicDiagonalConfig",
    "AxisMaskConfig",
    "WeightInformedScoreAxisMaskConfig",
    "PerAxisScoreMaskConfig",
    "TopSliceAxisMaskConfig",
    "OuterProductMaskConfig",
    "DiagonalAxisMaskConfig",
    "BankExpansionFactorOptions",
    "DynamicDepthOptions",
    "MaskDimensionOptions",
    "WeightDecayScheduleOptions",
    "WeightNormalizationOptions",
    "WeightNormalizationPositionOptions",
)


class TestAugmentationPublicInterface(unittest.TestCase):
    def test_interfaces_eagerly_export_only_lightweight_configuration(self):
        script = """\
import json
import sys

import emperor.augmentations as augmentations
import emperor.augmentations.adaptive_parameters as adaptive_parameters

expected_eager_modules = (
    "emperor.augmentations.adaptive_parameters._config",
    "emperor.augmentations.adaptive_parameters._options",
    "emperor.augmentations.adaptive_parameters._biases.config",
    "emperor.augmentations.adaptive_parameters._diagonals.config",
    "emperor.augmentations.adaptive_parameters._masks.config",
    "emperor.augmentations.adaptive_parameters._weights.config",
)
heavy_modules = (
    "emperor.augmentations.adaptive_parameters._augmentation",
    "emperor.augmentations.adaptive_parameters._linear_adapter",
    "emperor.augmentations.adaptive_parameters._validation",
    "emperor.augmentations.adaptive_parameters._biases.variants",
    "emperor.augmentations.adaptive_parameters._biases.variants.additive",
    "emperor.augmentations.adaptive_parameters._biases.variants.affine",
    "emperor.augmentations.adaptive_parameters._biases.base",
    "emperor.augmentations.adaptive_parameters._biases.variants.gated",
    "emperor.augmentations.adaptive_parameters._biases.variants.generator",
    "emperor.augmentations.adaptive_parameters._biases.variants.multiplicative",
    "emperor.augmentations.adaptive_parameters._biases.validation",
    "emperor.augmentations.adaptive_parameters._biases.variants.weighted_bank",
    "emperor.augmentations.adaptive_parameters._diagonals.variants",
    "emperor.augmentations.adaptive_parameters._diagonals.variants.anti",
    "emperor.augmentations.adaptive_parameters._diagonals.base",
    "emperor.augmentations.adaptive_parameters._diagonals.variants.combined",
    "emperor.augmentations.adaptive_parameters._diagonals.variants.standard",
    "emperor.augmentations.adaptive_parameters._diagonals.validation",
    "emperor.augmentations.adaptive_parameters._masks.variants",
    "emperor.augmentations.adaptive_parameters._masks.base",
    "emperor.augmentations.adaptive_parameters._masks.variants.diagonal",
    "emperor.augmentations.adaptive_parameters._masks.variants.outer_product",
    "emperor.augmentations.adaptive_parameters._masks.variants.per_axis",
    "emperor.augmentations.adaptive_parameters._masks.variants.top_slice",
    "emperor.augmentations.adaptive_parameters._masks.validation",
    "emperor.augmentations.adaptive_parameters._masks.variants.weight_informed",
    "emperor.augmentations.adaptive_parameters._monitoring.adaptive_parameters",
    "emperor.augmentations.adaptive_parameters._monitoring.weight_banks",
    "emperor.augmentations.adaptive_parameters._weights.variants",
    "emperor.augmentations.adaptive_parameters._weights.base",
    "emperor.augmentations.adaptive_parameters._weights.depth_mapping",
    "emperor.augmentations.adaptive_parameters._weights.variants.dual_model",
    "emperor.augmentations.adaptive_parameters._weights.variants.hypernetwork",
    "emperor.augmentations.adaptive_parameters._weights.variants.layered_weighted_bank",
    "emperor.augmentations.adaptive_parameters._weights.variants.low_rank",
    "emperor.augmentations.adaptive_parameters._weights.variants.single_model",
    "emperor.augmentations.adaptive_parameters._weights.variants.soft_weighted_bank",
    "emperor.augmentations.adaptive_parameters._weights.validation",
)

print(json.dumps({
    "root_all": augmentations.__all__,
    "adaptive_all": adaptive_parameters.__all__,
    "root_owns_child": augmentations.adaptive_parameters is adaptive_parameters,
    "expected_eager_modules": {
        name: name in sys.modules for name in expected_eager_modules
    },
    "heavy_modules": {name: name in sys.modules for name in heavy_modules},
    "private_exports": {
        name: hasattr(adaptive_parameters, name)
        for name in (
            "AdaptiveLinearLayer",
            "AdaptiveParameterAugmentation",
            "AdaptiveParameterMonitorCallback",
            "WeightBankUtilizationMonitorCallback",
        )
    },
    "runtime_loaded": {
        "lightning": "lightning" in sys.modules,
        "torch": "torch" in sys.modules,
    },
    "shortcut_attributes": {
        "root___getattr__": hasattr(augmentations, "__getattr__"),
        "adaptive___getattr__": hasattr(adaptive_parameters, "__getattr__"),
        "adaptive__LAZY_EXPORTS": hasattr(adaptive_parameters, "_LAZY_EXPORTS"),
    },
}))
"""

        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)

        self.assertEqual(tuple(result["root_all"]), ("adaptive_parameters",))
        self.assertEqual(tuple(result["adaptive_all"]), EXPECTED_EXPORTS)
        self.assertTrue(result["root_owns_child"])
        self.assertEqual(
            result["expected_eager_modules"],
            dict.fromkeys(result["expected_eager_modules"], True),
        )
        self.assertEqual(
            result["heavy_modules"],
            dict.fromkeys(result["heavy_modules"], False),
        )
        self.assertEqual(
            result["private_exports"],
            dict.fromkeys(result["private_exports"], False),
        )
        self.assertEqual(
            result["runtime_loaded"],
            {"lightning": False, "torch": False},
        )
        self.assertEqual(
            result["shortcut_attributes"],
            {
                "root___getattr__": False,
                "adaptive___getattr__": False,
                "adaptive__LAZY_EXPORTS": False,
            },
        )

    def test_removed_implementation_imports_fail(self):
        for name in (
            "AdaptiveLinearLayer",
            "AdaptiveParameterAugmentation",
            "AdaptiveParameterMonitorCallback",
            "WeightBankUtilizationMonitorCallback",
        ):
            with self.subTest(name=name):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "from emperor.augmentations.adaptive_parameters "
                            f"import {name}"
                        ),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("ImportError", completed.stderr)

    def test_configs_build_the_private_implementations(self):
        augmentation_config = AdaptiveParameterAugmentationConfig(
            input_dim=2,
            output_dim=3,
        )
        augmentation = augmentation_config.build()
        self.assertEqual(
            type(augmentation).__module__,
            "emperor.augmentations.adaptive_parameters._augmentation",
        )

        linear_config = AdaptiveLinearLayerConfig(
            input_dim=2,
            output_dim=3,
            bias_flag=True,
            adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(),
        )
        linear = linear_config.build()
        self.assertEqual(
            type(linear).__module__,
            "emperor.augmentations.adaptive_parameters._linear_adapter",
        )

    def test_monitoring_has_its_own_explicit_interface(self):
        adaptive_parameters = __import__(
            "emperor.augmentations.adaptive_parameters",
            fromlist=["adaptive_parameters"],
        )
        monitoring = __import__(
            "emperor.augmentations.adaptive_parameters.monitoring",
            fromlist=["monitoring"],
        )

        self.assertEqual(
            monitoring.__all__,
            (
                "AdaptiveParameterMonitorCallback",
                "WeightBankUtilizationMonitorCallback",
            ),
        )
        self.assertFalse(
            hasattr(adaptive_parameters, "AdaptiveParameterMonitorCallback")
        )
        self.assertFalse(
            hasattr(adaptive_parameters, "WeightBankUtilizationMonitorCallback")
        )


if __name__ == "__main__":
    unittest.main()
