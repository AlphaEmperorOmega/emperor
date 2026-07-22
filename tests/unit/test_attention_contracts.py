import importlib
import inspect
import pkgutil
import unittest

import torch

from emperor.attention import AttentionLayerState
from emperor.attention._base import MultiHeadAttentionAbstract
from emperor.attention._ops.reshaping import ReshaperBase
from emperor.attention._runtime import QKV, AttentionRuntimeLayout
from support.attention import build_attention_config
from support.attention_contract_manifest import ATTENTION_CONTRACT_MANIFEST

EXPECTED_EXPORTS = (
    "MultiHeadAttentionConfig",
    "SelfAttentionConfig",
    "SelfAttentionProjectionStrategy",
    "IndependentAttentionConfig",
    "MixtureOfAttentionHeadsConfig",
    "MixerAttentionConfig",
    "AttentionLayerState",
    "AttentionMonitorCallback",
)


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
    def test_root_package_has_exact_exports_and_reexport_identity(self):
        module = importlib.import_module("emperor.attention")

        self.assertTupleEqual(tuple(module.__all__), EXPECTED_EXPORTS)
        for name in EXPECTED_EXPORTS:
            exported = getattr(module, name)
            defining_module = importlib.import_module(exported.__module__)
            self.assertIs(exported, getattr(defining_module, name))

    def test_monitoring_package_has_exact_export_and_reexport_identity(self):
        module = importlib.import_module("emperor.attention.monitoring")

        self.assertTupleEqual(module.__all__, ("AttentionMonitorCallback",))
        exported = module.AttentionMonitorCallback
        defining_module = importlib.import_module(exported.__module__)
        self.assertIs(exported, defining_module.AttentionMonitorCallback)

    def test_runtime_implementation_remains_intentionally_private(self):
        module = importlib.import_module("emperor.attention")

        for name in (
            "AttentionMasks",
            "AttentionRuntimeLayout",
            "IndependentAttention",
            "MixtureOfAttentionHeads",
            "MixerAttention",
            "MultiHeadAttentionAbstract",
            "QKV",
            "SelfAttention",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(module, name))

    def test_private_namespace_packages_have_no_interface(self):
        for module_name in (
            "emperor.attention._monitoring",
            "emperor.attention._ops",
            "emperor.attention._variants",
            "emperor.attention._variants.independent",
            "emperor.attention._variants.mixture",
            "emperor.attention._variants.mixer",
            "emperor.attention._variants.self_attention",
        ):
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertFalse(hasattr(module, "__all__"))


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

    def test_runtime_layout_helpers_preserve_real_source_topology(self):
        shape = AttentionRuntimeLayout(
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
