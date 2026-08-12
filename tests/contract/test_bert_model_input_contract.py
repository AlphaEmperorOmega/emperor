from __future__ import annotations

import unittest

import torch

from models.catalog import model_package

_BERT_PACKAGES = (
    "bert/linear",
    "bert/linear_adaptive",
    "bert/expert_linear",
    "bert/expert_linear_adaptive",
)
_BERT_OVERRIDES = {
    "batch_size": 1,
    "input_dim": 32,
    "output_dim": 32,
    "hidden_dim": 8,
    "sequence_length": 4,
    "stack_num_layers": 1,
    "attn_num_heads": 2,
    "embedding_dropout_probability": 0.5,
}


class BertModelInputContractTests(unittest.TestCase):
    def test_embedding_ids_reject_unsupported_integer_dtypes_before_dropout(
        self,
    ) -> None:
        for catalog_key in _BERT_PACKAGES:
            package = model_package(catalog_key)
            self.assertIsNotNone(package)
            assert package is not None
            configuration = package.build_configuration(
                config_overrides=_BERT_OVERRIDES
            )
            model = package.build_model(configuration).train()

            for dtype in (torch.uint8, torch.int8, torch.int16):
                valid_ids = torch.tensor([[2, 3, 4, 5]])
                invalid_ids = valid_ids.to(dtype=dtype)
                invalid_token_types = torch.tensor([[0, 1, 0, 1]], dtype=dtype)
                for input_ids, token_type_ids, message in (
                    (
                        invalid_ids,
                        None,
                        "input_ids must use torch.int32 or torch.int64",
                    ),
                    (
                        valid_ids,
                        invalid_token_types,
                        "token_type_ids must use torch.int32 or torch.int64",
                    ),
                ):
                    rng_before = torch.random.get_rng_state().clone()
                    with self.subTest(
                        model_package=catalog_key,
                        dtype=dtype,
                        field=message.split()[0],
                    ):
                        with self.assertRaisesRegex(ValueError, message):
                            model(input_ids, token_type_ids=token_type_ids)
                        torch.testing.assert_close(
                            torch.random.get_rng_state(),
                            rng_before,
                        )

    def test_embedding_ids_accept_pytorch_index_dtypes(self) -> None:
        for catalog_key in _BERT_PACKAGES:
            package = model_package(catalog_key)
            self.assertIsNotNone(package)
            assert package is not None
            configuration = package.build_configuration(
                config_overrides=_BERT_OVERRIDES
            )
            model = package.build_model(configuration).eval()

            for dtype in (torch.int32, torch.int64):
                input_ids = torch.tensor([[2, 3, 4, 5]], dtype=dtype)
                token_type_ids = torch.tensor([[0, 1, 0, 1]], dtype=dtype)
                with self.subTest(model_package=catalog_key, dtype=dtype):
                    mlm_logits, nsp_logits, _ = model(
                        input_ids,
                        token_type_ids=token_type_ids,
                    )
                    self.assertEqual(tuple(mlm_logits.shape), (1, 4, 32))
                    self.assertEqual(tuple(nsp_logits.shape), (1, 2))

    def test_malformed_attention_mask_fails_before_dropout_consumes_rng(self) -> None:
        input_ids = torch.tensor([[2, 3, 4, 5]])
        malformed_attention_mask = torch.ones((1, 3), dtype=torch.long)

        for catalog_key in _BERT_PACKAGES:
            package = model_package(catalog_key)
            self.assertIsNotNone(package)
            assert package is not None
            configuration = package.build_configuration(
                config_overrides=_BERT_OVERRIDES,
            )
            model = package.build_model(configuration).train()
            rng_before = torch.random.get_rng_state().clone()

            with self.subTest(model_package=catalog_key):
                with self.assertRaisesRegex(
                    ValueError,
                    r"attention_mask must have shape \(1, 4\)",
                ):
                    model(input_ids, attention_mask=malformed_attention_mask)
                torch.testing.assert_close(torch.random.get_rng_state(), rng_before)


if __name__ == "__main__":
    unittest.main()
