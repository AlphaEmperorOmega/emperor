from __future__ import annotations

import json
import subprocess
import sys
import unittest


class EmbeddingInterfaceBehaviorTests(unittest.TestCase):
    def test_interfaces_export_only_configuration_without_dynamic_shortcuts(
        self,
    ) -> None:
        import emperor.embedding as embedding
        import emperor.embedding.absolute as absolute
        import emperor.embedding.relative as relative

        self.assertEqual(embedding.__all__, ("absolute", "relative"))
        self.assertIs(embedding.absolute, absolute)
        self.assertIs(embedding.relative, relative)
        self.assertEqual(
            absolute.__all__,
            (
                "AbsolutePositionalEmbeddingConfig",
                "TextLearnedPositionalEmbeddingConfig",
                "ImageLearnedPositionalEmbeddingConfig",
                "TextSinusoidalPositionalEmbeddingConfig",
                "ImageSinusoidalPositionalEmbeddingConfig",
            ),
        )
        self.assertEqual(
            relative.__all__,
            (
                "RelativePositionalEmbeddingConfig",
                "DynamicPositionalBiasConfig",
            ),
        )
        for module in (embedding, absolute, relative):
            with self.subTest(module=module.__name__):
                self.assertFalse(hasattr(module, "__getattr__"))
                self.assertFalse(hasattr(module, "_LAZY_EXPORTS"))

    def test_explicit_interfaces_eagerly_load_only_configuration(
        self,
    ) -> None:
        script = """\
import json
import sys

import emperor.embedding as embedding

eager_modules = sorted(
    name for name in sys.modules if name.startswith("emperor.embedding.")
)
print(json.dumps({
    "root_all": embedding.__all__,
    "absolute_all": embedding.absolute.__all__,
    "relative_all": embedding.relative.__all__,
    "eager_modules": eager_modules,
    "heavy_modules": {
        name: name in sys.modules
        for name in (
            "emperor.embedding.absolute._base",
            "emperor.embedding.absolute._learned",
            "emperor.embedding.absolute._sinusoidal",
            "emperor.embedding.absolute._validation",
            "emperor.embedding.relative._bias",
            "emperor.embedding.relative._validation",
        )
    },
    "private_exports": {
        "AbsolutePositionalEmbeddingBase": hasattr(
            embedding.absolute, "AbsolutePositionalEmbeddingBase"
        ),
        "LearnedPositionalEmbedding": hasattr(
            embedding.absolute, "LearnedPositionalEmbedding"
        ),
        "AbsolutePositionalEmbeddingValidator": hasattr(
            embedding.absolute, "AbsolutePositionalEmbeddingValidator"
        ),
        "DynamicPositionalBias": hasattr(
            embedding.relative, "DynamicPositionalBias"
        ),
        "RelativePositionalEmbeddingValidator": hasattr(
            embedding.relative, "RelativePositionalEmbeddingValidator"
        ),
    },
    "runtime_loaded": {
        "torch": "torch" in sys.modules,
        "lightning": "lightning" in sys.modules,
    },
    "shortcut_attributes": {
        "root___getattr__": hasattr(embedding, "__getattr__"),
        "absolute___getattr__": hasattr(embedding.absolute, "__getattr__"),
        "relative___getattr__": hasattr(embedding.relative, "__getattr__"),
        "absolute__LAZY_EXPORTS": hasattr(embedding.absolute, "_LAZY_EXPORTS"),
        "relative__LAZY_EXPORTS": hasattr(embedding.relative, "_LAZY_EXPORTS"),
    },
}))
"""

        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            json.loads(completed.stdout),
            {
                "root_all": ["absolute", "relative"],
                "absolute_all": [
                    "AbsolutePositionalEmbeddingConfig",
                    "TextLearnedPositionalEmbeddingConfig",
                    "ImageLearnedPositionalEmbeddingConfig",
                    "TextSinusoidalPositionalEmbeddingConfig",
                    "ImageSinusoidalPositionalEmbeddingConfig",
                ],
                "relative_all": [
                    "RelativePositionalEmbeddingConfig",
                    "DynamicPositionalBiasConfig",
                ],
                "eager_modules": [
                    "emperor.embedding.absolute",
                    "emperor.embedding.absolute._config",
                    "emperor.embedding.relative",
                    "emperor.embedding.relative._config",
                ],
                "heavy_modules": {
                    "emperor.embedding.absolute._base": False,
                    "emperor.embedding.absolute._learned": False,
                    "emperor.embedding.absolute._sinusoidal": False,
                    "emperor.embedding.absolute._validation": False,
                    "emperor.embedding.relative._bias": False,
                    "emperor.embedding.relative._validation": False,
                },
                "private_exports": {
                    "AbsolutePositionalEmbeddingBase": False,
                    "LearnedPositionalEmbedding": False,
                    "AbsolutePositionalEmbeddingValidator": False,
                    "DynamicPositionalBias": False,
                    "RelativePositionalEmbeddingValidator": False,
                },
                "runtime_loaded": {
                    "torch": False,
                    "lightning": False,
                },
                "shortcut_attributes": {
                    "root___getattr__": False,
                    "absolute___getattr__": False,
                    "relative___getattr__": False,
                    "absolute__LAZY_EXPORTS": False,
                    "relative__LAZY_EXPORTS": False,
                },
            },
        )

    def test_removed_implementation_imports_fail(self) -> None:
        removed_exports = {
            "emperor.embedding.absolute": (
                "AbsolutePositionalEmbeddingBase",
                "LearnedPositionalEmbedding",
                "TextLearnedPositionalEmbedding",
                "ImageLearnedPositionalEmbedding",
                "SinusoidalPositionalEmbedding",
                "TextSinusoidalPositionalEmbedding",
                "ImageSinusoidalPositionalEmbedding",
                "AbsolutePositionalEmbeddingValidator",
            ),
            "emperor.embedding.relative": (
                "DynamicPositionalBias",
                "RelativePositionalEmbeddingValidator",
            ),
        }
        for module_name, export_names in removed_exports.items():
            for export_name in export_names:
                with self.subTest(module=module_name, export=export_name):
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-c",
                            f"from {module_name} import {export_name}",
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn("ImportError", completed.stderr)


if __name__ == "__main__":
    unittest.main()
