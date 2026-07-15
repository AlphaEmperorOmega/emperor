import unittest
from types import SimpleNamespace

import torch
from emperor.transformer.config import TransformerConfig
from emperor.transformer.core._validator import TransformerValidator
from emperor.transformer.core.layers import (
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from emperor.transformer.core.stack import (
    TransformerDecoderStack,
    TransformerEncoderStack,
)
from emperor.transformer.model import Transformer


class TestTransformerValidatorAdapter(unittest.TestCase):
    def test_transformer_modules_share_the_validator_adapter(self):
        module_types = (
            Transformer,
            TransformerEncoderLayer,
            TransformerDecoderLayer,
            TransformerEncoderStack,
            TransformerDecoderStack,
        )

        for module_type in module_types:
            with self.subTest(module_type=module_type.__name__):
                self.assertIs(module_type.VALIDATOR, TransformerValidator)

    def test_build_orchestration_dispatches_through_subclass(self):
        class TrackingValidator(TransformerValidator):
            @staticmethod
            def _validate_decoder_cross_attention_has_encoder(
                encoder_stack_config,
                decoder_stack_config,
            ):
                raise RuntimeError("substituted structural validator was called")

        model = SimpleNamespace(
            cfg=SimpleNamespace(
                encoder_stack_config=object(),
                decoder_stack_config=None,
            )
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted structural validator was called",
        ):
            TrackingValidator.validate_transformer(model)

    def test_construction_dispatches_through_module_adapter(self):
        class TrackingValidator(TransformerValidator):
            @classmethod
            def validate_transformer(cls, model):
                raise RuntimeError("substituted construction validator was called")

        class TrackingTransformer(Transformer):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingTransformer(TransformerConfig())

    def test_runtime_dispatches_through_module_adapter(self):
        class RejectingValidator(TransformerValidator):
            @staticmethod
            def validate_transformer_forward_inputs(
                model,
                source_token_embeddings,
                target_token_embeddings,
            ):
                raise RuntimeError("substituted runtime validator was called")

        class RejectingTransformer(Transformer):
            VALIDATOR = RejectingValidator

        model = RejectingTransformer.__new__(RejectingTransformer)
        torch.nn.Module.__init__(model)

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted runtime validator was called",
        ):
            model()


if __name__ == "__main__":
    unittest.main()
