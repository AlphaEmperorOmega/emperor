import importlib
import inspect
import pkgutil
import unittest

import torch
from emperor.attention.core.handlers.reshaper import ReshaperBase
from emperor.attention.core.layers import MultiHeadAttentionAbstract
from emperor.attention.core.runtime import QKV, AttentionRuntimeShape
from emperor.attention.core.state import AttentionLayerState

from support.attention import build_attention_config
from support.attention_contract_manifest import ATTENTION_CONTRACT_MANIFEST

EXPECTED_EXPORTS = {
    "emperor.attention": (
        "MultiHeadAttentionConfig",
        "SelfAttentionConfig",
        "SelfAttentionProjectionStrategy",
        "IndependentAttentionConfig",
        "MixtureOfAttentionHeadsConfig",
        "MultiHeadAttentionAbstract",
        "AttentionMonitorCallback",
        "AttentionLayerState",
        "AttentionRuntimeShape",
        "SelfAttention",
        "IndependentAttention",
        "MixtureOfAttentionHeads",
    ),
    "emperor.attention.core": (
        "MultiHeadAttentionConfig",
        "MultiHeadAttentionAbstract",
        "AttentionMonitorCallback",
        "AttentionLayerState",
        "AttentionRuntimeShape",
        "AttentionValidatorBase",
        "MultiHeadAttentionValidator",
        "SelfAttentionConfig",
        "SelfAttentionProjectionStrategy",
        "IndependentAttentionConfig",
        "MixtureOfAttentionHeadsConfig",
        "SelfAttention",
        "IndependentAttention",
        "MixtureOfAttentionHeads",
    ),
    "emperor.attention.core.variants": (
        "SelfAttention",
        "SelfAttentionConfig",
        "SelfAttentionProjectionStrategy",
        "SelfAttentionProcessor",
        "SelfAttentionProjector",
        "SelfAttentionValidator",
        "IndependentAttention",
        "IndependentAttentionConfig",
        "IndependentProcessor",
        "IndependentProjector",
        "IndependentAttentionValidator",
        "MixtureOfAttentionHeads",
        "MixtureOfAttentionHeadsConfig",
        "MixtureOfAttentionHeadsKeyValueBias",
        "MixtureOfAttentionHeadsMask",
        "MixtureOfAttentionHeadsProcessor",
        "MixtureOfAttentionHeadsProjector",
        "MixtureOfAttentionHeadsReshaper",
        "MixtureOfAttentionHeadsValidator",
        "MixtureOfAttentionHeadsZeroAttention",
    ),
    "emperor.attention.core.variants.independent_attention": (
        "IndependentAttentionConfig",
        "IndependentAttention",
        "IndependentProcessor",
        "IndependentProjector",
        "IndependentAttentionValidator",
    ),
    "emperor.attention.core.variants.mixture_of_attention_heads": (
        "MixtureOfAttentionHeadsConfig",
        "MixtureOfAttentionHeadsKeyValueBias",
        "MixtureOfAttentionHeads",
        "MixtureOfAttentionHeadsMask",
        "MixtureOfAttentionHeadsProcessor",
        "MixtureOfAttentionHeadsProjector",
        "MixtureOfAttentionHeadsReshaper",
        "MixtureOfAttentionHeadsValidator",
        "MixtureOfAttentionHeadsZeroAttention",
    ),
    "emperor.attention.core.variants.self_attention": (
        "SelfAttentionConfig",
        "SelfAttentionProjectionStrategy",
        "SelfAttention",
        "SelfAttentionProcessor",
        "SelfAttentionProjector",
        "SelfAttentionValidator",
    ),
}


def discover_attention_modules():
    package = importlib.import_module("emperor.attention")
    names = {package.__name__}
    names.update(
        module.name
        for module in pkgutil.walk_packages(
            package.__path__, prefix=f"{package.__name__}."
        )
    )
    return {name: importlib.import_module(name) for name in names}


class TestAttentionContractManifest(unittest.TestCase):
    def test_manifest_exactly_matches_production_modules_and_classes(self):
        modules = discover_attention_modules()

        self.assertSetEqual(set(ATTENTION_CONTRACT_MANIFEST), set(modules))
        for module_name, module in modules.items():
            with self.subTest(module=module_name):
                discovered_classes = {
                    name
                    for name, value in vars(module).items()
                    if inspect.isclass(value) and value.__module__ == module_name
                }
                manifest_classes = set(
                    ATTENTION_CONTRACT_MANIFEST[module_name]["classes"]
                )
                self.assertSetEqual(manifest_classes, discovered_classes)

    def test_every_contract_has_responsibilities_and_loadable_tests(self):
        for module_name, contract in ATTENTION_CONTRACT_MANIFEST.items():
            records = ((module_name, contract), *contract["classes"].items())
            for record_name, record in records:
                with self.subTest(module=module_name, contract=record_name):
                    self.assertTrue(record["responsibilities"])
                    self.assertTrue(record["tests"])
                    for test_id in record["tests"]:
                        loader = unittest.TestLoader()
                        suite = loader.loadTestsFromName(test_id)
                        self.assertFalse(loader.errors, loader.errors)
                        self.assertGreater(suite.countTestCases(), 0, test_id)


class TestAttentionExports(unittest.TestCase):
    def test_public_packages_have_exact_exports_and_reexport_identity(self):
        for module_name, expected_exports in EXPECTED_EXPORTS.items():
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertTupleEqual(tuple(module.__all__), expected_exports)
                for name in expected_exports:
                    exported = getattr(module, name)
                    defining_module = importlib.import_module(exported.__module__)
                    self.assertIs(exported, getattr(defining_module, name))

    def test_runtime_value_objects_remain_intentionally_private(self):
        for module_name in ("emperor.attention", "emperor.attention.core"):
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertFalse(hasattr(module, "QKV"))
                self.assertFalse(hasattr(module, "AttentionMasks"))

    def test_handlers_package_has_no_public_namespace(self):
        module = importlib.import_module("emperor.attention.core.handlers")

        self.assertFalse(hasattr(module, "__all__"))
        self.assertSetEqual(
            {name for name in vars(module) if not name.startswith("_")},
            {
                "batch",
                "bias",
                "mask",
                "processor",
                "projector",
                "reshaper",
                "zero_attention",
            },
        )


class TestAttentionBaseContracts(unittest.TestCase):
    def setUp(self):
        self.cfg = build_attention_config(
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=2,
            source_sequence_length=3,
        )

    def test_reshaper_base_identity_and_abstract_guard(self):
        reshaper = ReshaperBase(self.cfg)
        tensor = torch.randn(1, 2, 2)
        qkv = QKV(query=tensor, key=tensor, value=tensor)

        self.assertIs(reshaper.reshape_before_attention(qkv), qkv)
        with self.assertRaises(NotImplementedError) as caught:
            reshaper.reshape_qkv_for_attention(qkv)
        self.assertEqual(
            str(caught.exception),
            "reshape_qkv_for_attention must be implemented by subclass.",
        )

    def test_abstract_attention_layer_rejects_direct_construction(self):
        with self.assertRaises(NotImplementedError) as caught:
            MultiHeadAttentionAbstract(self.cfg)
        self.assertEqual(
            str(caught.exception),
            "_build_attention_components must be implemented by subclass.",
        )

    def test_runtime_shape_helpers_preserve_real_source_topology(self):
        shape = AttentionRuntimeShape(
            batch_size=2,
            target_sequence_length=3,
            source_sequence_length=5,
        )

        self.assertEqual(shape.branch_count(num_heads=4), 8)
        extended = shape.with_source_extension(2)
        self.assertEqual(extended.source_sequence_length, 7)
        self.assertEqual(extended.source_extension_count, 2)
        self.assertEqual(extended.real_source_sequence_length, 5)
        self.assertEqual(shape.source_sequence_length, 5)

    def test_attention_layer_state_preserves_all_state_fields(self):
        hidden = torch.randn(2, 3)
        loss = torch.tensor(1.5)
        halting_state = object()
        key_padding_mask = torch.zeros(2, 3, dtype=torch.bool)
        attention_mask = torch.zeros(3, 3)

        state = AttentionLayerState(
            hidden=hidden,
            loss=loss,
            halting_state=halting_state,
            key_padding_mask=key_padding_mask,
            attention_mask=attention_mask,
        )

        self.assertIs(state.hidden, hidden)
        self.assertIs(state.loss, loss)
        self.assertIs(state.halting_state, halting_state)
        self.assertIs(state.key_padding_mask, key_padding_mask)
        self.assertIs(state.attention_mask, attention_mask)


if __name__ == "__main__":
    unittest.main()
