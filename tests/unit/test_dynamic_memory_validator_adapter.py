import unittest

import torch
from emperor.memory.config import (
    AttentionDynamicMemoryConfig,
    DynamicMemoryConfig,
)
from emperor.memory.core import (
    AttentionDynamicMemory,
    DynamicMemoryAbstract,
    ElementWiseWeightedDynamicMemory,
    GatedResidualDynamicMemory,
    WeightedDynamicMemory,
)
from emperor.memory.core._validator import DynamicMemoryValidator


class TestDynamicMemoryValidatorAdapter(unittest.TestCase):
    def test_memory_modules_share_the_base_owner_adapter(self):
        module_types = (
            DynamicMemoryAbstract,
            GatedResidualDynamicMemory,
            WeightedDynamicMemory,
            ElementWiseWeightedDynamicMemory,
            AttentionDynamicMemory,
        )

        for module_type in module_types:
            with self.subTest(module_type=module_type.__name__):
                self.assertIs(module_type.VALIDATOR, DynamicMemoryValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(DynamicMemoryValidator):
            @staticmethod
            def validate_required_fields(cfg):
                raise RuntimeError("substituted construction validator was called")

        class TrackingMemory(DynamicMemoryAbstract):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingMemory(DynamicMemoryConfig())

    def test_specialized_construction_rule_dispatches_through_adapter(self):
        class TrackingValidator(DynamicMemoryValidator):
            @classmethod
            def validate(cls, model):
                return None

            @staticmethod
            def validate_attention_num_memory_slots(cfg):
                raise RuntimeError("substituted attention validator was called")

        class TrackingAttention(AttentionDynamicMemory):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted attention validator was called",
        ):
            TrackingAttention(AttentionDynamicMemoryConfig())

    def test_runtime_dispatches_through_substituted_validator(self):
        class RejectingValidator(DynamicMemoryValidator):
            @staticmethod
            def validate_forward_inputs(logits, expected_dim):
                raise RuntimeError("substituted runtime validator was called")

        class RejectingMemory(GatedResidualDynamicMemory):
            VALIDATOR = RejectingValidator

        model = RejectingMemory.__new__(RejectingMemory)
        torch.nn.Module.__init__(model)
        model.memory_dim = 3

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted runtime validator was called",
        ):
            model(torch.ones(1, 3))


if __name__ == "__main__":
    unittest.main()
