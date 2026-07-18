from __future__ import annotations

import json
import subprocess
import sys
import unittest

import torch
from emperor.embedding.absolute import (
    ImageLearnedPositionalEmbeddingConfig,
    ImageSinusoidalPositionalEmbeddingConfig,
    TextLearnedPositionalEmbeddingConfig,
    TextSinusoidalPositionalEmbeddingConfig,
)


def text_learned_config(**overrides: object) -> TextLearnedPositionalEmbeddingConfig:
    values: dict[str, object] = {
        "num_embeddings": 4,
        "embedding_dim": 2,
        "init_size": 4,
        "padding_idx": 0,
        "auto_expand_flag": False,
    }
    values.update(overrides)
    return TextLearnedPositionalEmbeddingConfig(**values)


class LearnedEmbeddingBehaviorTests(unittest.TestCase):
    def test_no_padding_table_positions_and_device_follow_input(self) -> None:
        model = text_learned_config(
            num_embeddings=4,
            padding_idx=None,
        ).build()
        weights = torch.tensor(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
            ]
        )
        with torch.no_grad():
            model.embedding_model.weight.copy_(weights)

        output = model(torch.tensor([[9, 8, 7], [6, 5, 4]]))

        self.assertEqual(model.embedding_model.num_embeddings, 4)
        torch.testing.assert_close(
            output,
            weights[torch.tensor([[0, 1, 2], [0, 1, 2]])],
            rtol=0,
            atol=0,
        )

        meta_tokens = torch.empty(
            (2, 3),
            dtype=torch.long,
            device="meta",
        )
        meta_positions = model._make_positions(meta_tokens)
        self.assertEqual(meta_positions.device.type, "meta")
        self.assertEqual(meta_positions.dtype, torch.long)
        self.assertEqual(meta_positions.shape, meta_tokens.shape)

    def test_image_learned_padding_row_stays_zero_after_optimizer_step(
        self,
    ) -> None:
        model = ImageLearnedPositionalEmbeddingConfig(
            num_embeddings=2,
            embedding_dim=2,
            init_size=2,
            padding_idx=0,
            auto_expand_flag=False,
            class_token_flag=True,
        ).build()
        initial_weights = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
        with torch.no_grad():
            model.embedding_model.weight.copy_(initial_weights)
        patches = torch.zeros(2, 3, 2, requires_grad=True)

        output = model(patches)
        output.sum().backward()

        torch.testing.assert_close(
            output,
            initial_weights.unsqueeze(0).expand(2, -1, -1),
            rtol=0,
            atol=0,
        )
        gradient = model.embedding_model.weight.grad
        assert gradient is not None
        torch.testing.assert_close(
            gradient,
            torch.tensor([[0.0, 0.0], [2.0, 2.0], [2.0, 2.0]]),
            rtol=0,
            atol=0,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
        optimizer.step()
        torch.testing.assert_close(
            model.embedding_model.weight,
            torch.tensor([[0.0, 0.0], [0.5, 1.5], [2.5, 3.5]]),
            rtol=0,
            atol=0,
        )

    def test_image_learned_indices_follow_input_not_global_default_device(
        self,
    ) -> None:
        model = ImageLearnedPositionalEmbeddingConfig(
            num_embeddings=2,
            embedding_dim=2,
            init_size=2,
            padding_idx=None,
            auto_expand_flag=False,
            class_token_flag=True,
        ).build()
        weights = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        with torch.no_grad():
            model.embedding_model.weight.copy_(weights)
        patches = torch.zeros(2, 3, 2)

        with torch.device("meta"):
            output = model(patches)

        self.assertEqual(output.device.type, "cpu")
        torch.testing.assert_close(
            output,
            weights.unsqueeze(0).expand(2, -1, -1),
            rtol=0,
            atol=0,
        )


class SinusoidalEmbeddingBehaviorTests(unittest.TestCase):
    def test_sinusoidal_table_dtype_is_independent_of_global_default_dtype(
        self,
    ) -> None:
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            model = TextSinusoidalPositionalEmbeddingConfig(
                num_embeddings=3,
                embedding_dim=4,
                init_size=3,
                padding_idx=None,
                auto_expand_flag=False,
            ).build()
        finally:
            torch.set_default_dtype(previous_dtype)

        self.assertEqual(model.weights.dtype, torch.float32)
        expected = torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [
                    torch.sin(torch.tensor(1.0)),
                    torch.sin(torch.tensor(0.0001)),
                    torch.cos(torch.tensor(1.0)),
                    torch.cos(torch.tensor(0.0001)),
                ],
                [
                    torch.sin(torch.tensor(2.0)),
                    torch.sin(torch.tensor(0.0002)),
                    torch.cos(torch.tensor(2.0)),
                    torch.cos(torch.tensor(0.0002)),
                ],
            ]
        )
        torch.testing.assert_close(model.weights, expected)

    def test_text_table_sizes_follow_none_and_nonzero_padding_contracts(
        self,
    ) -> None:
        no_padding = TextSinusoidalPositionalEmbeddingConfig(
            num_embeddings=4,
            embedding_dim=4,
            init_size=4,
            padding_idx=None,
            auto_expand_flag=False,
        ).build()
        nonzero_padding = TextSinusoidalPositionalEmbeddingConfig(
            num_embeddings=4,
            embedding_dim=4,
            init_size=4,
            padding_idx=2,
            auto_expand_flag=False,
        ).build()

        self.assertEqual(
            (
                no_padding.position_offset,
                no_padding.init_size,
                no_padding.weights.shape,
            ),
            (0, 4, torch.Size((4, 4))),
        )
        self.assertEqual(
            (
                nonzero_padding.position_offset,
                nonzero_padding.init_size,
                nonzero_padding.weights.shape,
            ),
            (2, 7, torch.Size((7, 4))),
        )
        torch.testing.assert_close(
            nonzero_padding.weights[2],
            torch.zeros(4),
            rtol=0,
            atol=0,
        )

    def test_image_sinusoidal_class_token_has_zero_positional_offset(
        self,
    ) -> None:
        model = ImageSinusoidalPositionalEmbeddingConfig(
            num_embeddings=2,
            embedding_dim=2,
            init_size=2,
            padding_idx=None,
            auto_expand_flag=False,
            class_token_flag=True,
        ).build()
        patches = torch.tensor([[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]])

        output = model(patches)

        expected = patches.clone()
        expected[:, 1, 0] += torch.sin(torch.tensor(1.0))
        expected[:, 1, 1] += torch.cos(torch.tensor(1.0))
        expected[:, 2, 0] += torch.sin(torch.tensor(2.0))
        expected[:, 2, 1] += torch.cos(torch.tensor(2.0))
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(
            model.weights[0],
            torch.zeros(2),
            rtol=0,
            atol=0,
        )


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
