import unittest
from types import SimpleNamespace

import torch

from emperor.attention._ops.bias import KeyValueBias
from emperor.attention._ops.masking import Mask
from emperor.attention._ops.processing import ProcessorBase
from emperor.attention._ops.reshaping import (
    AttentionReshaper,
    ReshaperBase,
)
from emperor.attention._runtime import QKV, AttentionMasks, AttentionRuntimeLayout
from emperor.attention._validation import AttentionValidatorBase
from support.attention import build_attention_config


class TestAttentionValidatorBaseAdapter(unittest.TestCase):
    def test_attention_handlers_expose_the_shared_validator_adapter(self):
        handler_types = (
            ReshaperBase,
            AttentionReshaper,
            Mask,
            KeyValueBias,
            ProcessorBase,
        )

        for handler_type in handler_types:
            with self.subTest(handler_type=handler_type.__name__):
                self.assertIs(handler_type.VALIDATOR, AttentionValidatorBase)

    def test_static_projection_orchestration_dispatches_through_subclass(self):
        validated_tensors = []

        class TrackingValidator(AttentionValidatorBase):
            @staticmethod
            def _validate_static_projection_shape(
                model,
                static_tensor,
                tensor_name,
                runtime_layout=None,
            ):
                validated_tensors.append((static_tensor, tensor_name, runtime_layout))

        runtime_layout = object()
        model = SimpleNamespace()
        static_keys = object()
        static_values = object()

        TrackingValidator.validate_static_projection_shapes(
            model,
            static_keys,
            static_values,
            runtime_layout,
        )

        self.assertEqual(
            validated_tensors,
            [
                (static_keys, "static_keys", runtime_layout),
                (static_values, "static_values", runtime_layout),
            ],
        )

    def test_mask_shape_validation_dispatches_through_substituted_validator(self):
        class RejectingValidator(AttentionValidatorBase):
            @staticmethod
            def validate_mask_shapes(*args, **kwargs):
                raise RuntimeError("substituted mask-shape validator was called")

        class RejectingMask(Mask):
            VALIDATOR = RejectingValidator

        cfg = build_attention_config(
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=2,
            source_sequence_length=2,
        )
        model = RejectingMask(cfg)
        query = torch.zeros(2, 1, 2)
        runtime_layout = AttentionRuntimeLayout(
            batch_size=1,
            target_sequence_length=2,
            source_sequence_length=2,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted mask-shape validator was called",
        ):
            model.prepare_attention_masks(query, AttentionMasks(), runtime_layout)

    def test_key_value_bias_dispatches_through_substituted_validator(self):
        class RejectingValidator(AttentionValidatorBase):
            @staticmethod
            def validate_attention_ready_projection_branch_count(
                branch_count,
                base_branch_count,
            ):
                raise RuntimeError("substituted branch-count validator was called")

        class RejectingKeyValueBias(KeyValueBias):
            VALIDATOR = RejectingValidator

        cfg = build_attention_config(
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            query_key_projection_dim=2,
            value_projection_dim=2,
            add_key_value_bias_flag=True,
        )
        model = RejectingKeyValueBias(cfg)
        projection = torch.zeros(1, 2, 2)

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted branch-count validator was called",
        ):
            model.add_kv_learnable_bias_vectors(
                QKV(query=projection, key=projection, value=projection),
                AttentionMasks(),
            )


if __name__ == "__main__":
    unittest.main()
