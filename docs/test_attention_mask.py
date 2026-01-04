import torch
import unittest

from dataclasses import asdict
from docs.config import default_unittest_config
from Emperor.attention.utils.handlers.maks import Mask
from Emperor.attention.utils.layer import MultiHeadAttentionConfig


class TestMask(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.model = None
        self.batch_size = None
        self.embedding_dim = None
        self.target_sequence_length = None
        self.num_heads = None
        self.head_dim = None
        self.query_model = None
        self.key_model = None
        self.value_model = None
        self.qkv_model = None

    def rebuild_presets(self, config: MultiHeadAttentionConfig | None = None):
        self.cfg = default_unittest_config()
        self.config = self.cfg.multi_head_attention_model_config
        if config is not None:
            for k in asdict(config):
                if hasattr(self.config, k) and getattr(config, k) is not None:
                    setattr(self.config, k, getattr(config, k))

        self.model = Mask(self.config)

        self.batch_size = self.config.batch_size
        self.embedding_dim = self.config.embedding_dim
        self.target_sequence_length = self.config.target_sequence_length
        self.source_sequence_length = self.config.source_sequence_length
        self.num_heads = self.config.num_heads
        self.head_dim = self.embedding_dim // self.num_heads


class Test___canonical_mask(TestMask):
    def test_mask_set_as_none_tensor(self):
        mask = None
        mask_name = ""
        other_type = None
        other_name = ""
        target_type = torch.float32
        check_other = False

        output = self.model._Mask__canonical_mask(
            mask, mask_name, other_type, other_name, target_type, check_other
        )
        self.assertIsNone(output)

    def test_check_if_boolean_mask_is_converted_to_mask_filled_with_zero_and_negative_infinity(
        self,
    ):
        mask = torch.randn(
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.target_sequence_length,
        )
        mask = mask > 0
        mask_name = "maks_to_test"
        other_type = torch.bool
        other_name = "required_mask_dtype"
        target_type = torch.float32
        check_other = True

        output = self.model._Mask__canonical_mask(
            mask, mask_name, other_type, other_name, target_type, check_other
        )

        self.assertTrue(output.dtype == torch.float32)
        self.assertFalse(output.dtype == mask.dtype)

    def test_if_same_mask_is_returned_when_type_is_floating_point(self):
        mask = torch.randn(
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.target_sequence_length,
        )
        mask_name = "maks_to_test"
        other_type = torch.float32
        other_name = "required_mask_dtype"
        target_type = torch.float32
        check_other = True

        output = self.model._Mask__canonical_mask(
            mask, mask_name, other_type, other_name, target_type, check_other
        )

        self.assertTrue(torch.equal(output, mask))


class Test___validate_attention_mask(TestMask):
    def test_only_attention_mask_as_input(self):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=False,
            causal_attention_mask_flag=False,
        )
        self.rebuild_presets(config)
        key_padding_mask = None
        attention_mask = torch.randn(
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.target_sequence_length,
        )
        attention_mask = attention_mask > 0

        output = self.model._Mask__validate_attention_mask(
            key_padding_mask,
            attention_mask,
        )

        self.assertIsInstance(output, torch.Tensor)
        self.assertNotEqual(output.dtype, torch.bool)
        self.assertEqual(output.dtype, self.config.target_dtype)
        self.assertFalse(self.model.causal_attention_mask_flag)

    def test_only_key_padding_mask_as_input(self):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=False,
            causal_attention_mask_flag=False,
        )
        self.rebuild_presets(config)
        key_padding_mask = torch.randint(
            0, 2, (self.batch_size, self.source_sequence_length)
        )
        attention_mask = None

        output = self.model._Mask__validate_attention_mask(
            key_padding_mask,
            attention_mask,
        )
        self.assertIsNone(output)

    def test_attention_and_padding_mask_as_input_with_all_flags_true(self):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=True,
            causal_attention_mask_flag=True,
        )
        self.rebuild_presets(config)
        key_padding_mask = torch.randint(
            0, 2, (self.batch_size, self.source_sequence_length)
        )
        attention_mask = torch.randn(
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.target_sequence_length,
        )
        attention_mask = attention_mask > 0

        output = self.model._Mask__validate_attention_mask(
            key_padding_mask,
            attention_mask,
        )

        self.assertIsInstance(output, torch.Tensor)
        self.assertNotEqual(output.dtype, torch.bool)
        self.assertEqual(output.dtype, self.config.target_dtype)
        self.assertFalse(self.model.causal_attention_mask_flag)


class Test___should_skip_attention_mask(TestMask):
    def test_if_attention_is_skipped_when_flags_are_false_and_no_padding_mask_is_given(
        self,
    ):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=False,
            causal_attention_mask_flag=False,
        )
        self.rebuild_presets(config)

        key_padding_mask = None
        output = self.model._Mask__should_skip_attention_mask(
            key_padding_mask,
        )

        self.assertFalse(output)

    def test_if_attention_is_skipped_when_padding_mask_is_given(self):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=False,
            causal_attention_mask_flag=False,
        )
        self.rebuild_presets(config)

        key_padding_mask = torch.randint(
            0, 2, (self.batch_size, self.source_sequence_length)
        )
        output = self.model._Mask__should_skip_attention_mask(
            key_padding_mask,
        )

        self.assertFalse(output)

    def test_if_attention_is_skipped_when_padding_mask_is_given_and_flags_are_true(
        self,
    ):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=True,
            causal_attention_mask_flag=True,
        )
        self.rebuild_presets(config)

        key_padding_mask = torch.randint(
            0, 2, (self.batch_size, self.source_sequence_length)
        )
        output = self.model._Mask__should_skip_attention_mask(
            key_padding_mask,
        )

        self.assertFalse(output)

    def test_only_case_attention_is_skipped(self):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=False,
            causal_attention_mask_flag=True,
        )
        self.rebuild_presets(config)

        key_padding_mask = None
        output = self.model._Mask__should_skip_attention_mask(
            key_padding_mask,
        )

        self.assertTrue(output)


class Test_process_attention_masks(TestMask):
    def test_no_inputs_and_flags_set_to_fasle(self):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=False,
            causal_attention_mask_flag=False,
        )
        self.rebuild_presets(config)

        key_padding_mask = None
        attention_mask = None

        key_padding_mask, attention_mask = self.model.process_attention_masks(
            key_padding_mask,
            attention_mask,
        )

        self.assertIsNone(key_padding_mask)
        self.assertIsNone(attention_mask)

    def test_no_inputs_and_return_weight_flag_set_to_true(self):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=True,
            causal_attention_mask_flag=False,
        )
        self.rebuild_presets(config)

        key_padding_mask = None
        attention_mask = None

        key_padding_mask, attention_mask = self.model.process_attention_masks(
            key_padding_mask,
            attention_mask,
        )

        self.assertIsNone(key_padding_mask)
        self.assertIsNone(attention_mask)

    def test_no_inputs_and_both_flags_set_to_true(self):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=True,
            causal_attention_mask_flag=True,
        )
        self.rebuild_presets(config)

        key_padding_mask = None
        attention_mask = None

        with self.assertRaises(RuntimeError):
            key_padding_mask, attention_mask = self.model.process_attention_masks(
                key_padding_mask,
                attention_mask,
            )

    def test_only_key_padding_mask_input_with_both_flags_set_to_false(self):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=False,
            causal_attention_mask_flag=False,
        )
        self.rebuild_presets(config)

        key_padding_mask = (
            torch.randint(0, 2, (self.batch_size, self.source_sequence_length)) > 0
        )
        attention_mask = None

        output_key_padding_mask, output_attention_mask = (
            self.model.process_attention_masks(
                key_padding_mask,
                attention_mask,
            )
        )

        self.assertIsInstance(output_key_padding_mask, torch.Tensor)
        self.assertEqual(output_key_padding_mask.dtype, torch.float32)
        self.assertIsNone(output_attention_mask)

    def test_only_key_padding_mask_input_with_return_attention_weights_flag_set_to_true(
        self,
    ):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=True,
            causal_attention_mask_flag=False,
        )
        self.rebuild_presets(config)

        key_padding_mask = (
            torch.randint(0, 2, (self.batch_size, self.source_sequence_length)) > 0
        )
        attention_mask = None

        output_key_padding_mask, output_attention_mask = (
            self.model.process_attention_masks(
                key_padding_mask,
                attention_mask,
            )
        )

        self.assertIsInstance(output_key_padding_mask, torch.Tensor)
        self.assertEqual(output_key_padding_mask.dtype, torch.float32)
        self.assertIsNone(output_attention_mask)

    def test_only_key_padding_mask_input_with_both_flags_set_to_true(
        self,
    ):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=True,
            causal_attention_mask_flag=True,
        )
        self.rebuild_presets(config)

        key_padding_mask = (
            torch.randint(0, 2, (self.batch_size, self.source_sequence_length)) > 0
        )
        attention_mask = None

        with self.assertRaises(RuntimeError):
            output_key_padding_mask, output_attention_mask = (
                self.model.process_attention_masks(
                    key_padding_mask,
                    attention_mask,
                )
            )

    def test_only_attention_mask_input_with_both_flags_set_to_false(self):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=False,
            causal_attention_mask_flag=False,
            target_dtype=torch.float64,
        )
        self.rebuild_presets(config)

        key_padding_mask = None
        attention_mask = (
            torch.randn(
                self.batch_size * self.num_heads,
                self.source_sequence_length,
                self.target_sequence_length,
            )
            > 0
        )
        key_padding_mask, attention_mask = self.model.process_attention_masks(
            key_padding_mask,
            attention_mask,
        )

        self.assertIsNone(key_padding_mask)
        self.assertIsInstance(attention_mask, torch.Tensor)
        self.assertEqual(attention_mask.dtype, config.target_dtype)

    def test_only_attention_mask_input_with_return_attention_weights_flag_set_to_true(
        self,
    ):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=True,
            causal_attention_mask_flag=False,
            target_dtype=torch.float64,
        )
        self.rebuild_presets(config)

        key_padding_mask = None
        attention_mask = (
            torch.randn(
                self.batch_size * self.num_heads,
                self.source_sequence_length,
                self.target_sequence_length,
            )
            > 0
        )
        key_padding_mask, attention_mask = self.model.process_attention_masks(
            key_padding_mask,
            attention_mask,
        )

        self.assertIsNone(key_padding_mask)
        self.assertIsInstance(attention_mask, torch.Tensor)
        self.assertEqual(attention_mask.dtype, config.target_dtype)

    def test_only_attention_mask_input_with_both_flags_set_to_true(
        self,
    ):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=True,
            causal_attention_mask_flag=True,
            target_dtype=torch.float64,
        )
        self.rebuild_presets(config)

        key_padding_mask = None
        attention_mask = (
            torch.randn(
                self.batch_size * self.num_heads,
                self.source_sequence_length,
                self.target_sequence_length,
            )
            > 0
        )
        key_padding_mask, attention_mask = self.model.process_attention_masks(
            key_padding_mask,
            attention_mask,
        )

        self.assertIsNone(key_padding_mask)
        self.assertIsInstance(attention_mask, torch.Tensor)
        self.assertEqual(attention_mask.dtype, config.target_dtype)

    def test_attention_and_padding_mask_with_both_masks_set_to_false(self):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=False,
            causal_attention_mask_flag=False,
        )
        self.rebuild_presets(config)

        key_padding_mask = (
            torch.randint(0, 2, (self.batch_size, self.source_sequence_length)) > 0
        )
        attention_mask = (
            torch.randn(
                self.batch_size * self.num_heads,
                self.source_sequence_length,
                self.target_sequence_length,
            )
            > 0
        )

        key_padding_mask, attention_mask = self.model.process_attention_masks(
            key_padding_mask,
            attention_mask,
        )

        self.assertIsInstance(key_padding_mask, torch.Tensor)
        self.assertEqual(key_padding_mask.dtype, torch.float32)
        self.assertIsInstance(attention_mask, torch.Tensor)
        self.assertEqual(attention_mask.dtype, torch.float32)

    def test_attention_and_padding_mask_with_return_attention_weights_flag_set_to_true(
        self,
    ):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=True,
            causal_attention_mask_flag=False,
        )
        self.rebuild_presets(config)

        key_padding_mask = (
            torch.randint(0, 2, (self.batch_size, self.source_sequence_length)) > 0
        )
        attention_mask = (
            torch.randn(
                self.batch_size * self.num_heads,
                self.source_sequence_length,
                self.target_sequence_length,
            )
            > 0
        )

        key_padding_mask, attention_mask = self.model.process_attention_masks(
            key_padding_mask,
            attention_mask,
        )

        self.assertIsInstance(key_padding_mask, torch.Tensor)
        self.assertEqual(key_padding_mask.dtype, torch.float32)
        self.assertIsInstance(attention_mask, torch.Tensor)
        self.assertEqual(attention_mask.dtype, torch.float32)

    def test_attention_and_padding_mask_with_both_flags_set_to_true(
        self,
    ):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=True,
            causal_attention_mask_flag=True,
        )
        self.rebuild_presets(config)

        key_padding_mask = (
            torch.randint(0, 2, (self.batch_size, self.source_sequence_length)) > 0
        )
        attention_mask = (
            torch.randn(
                self.batch_size * self.num_heads,
                self.source_sequence_length,
                self.target_sequence_length,
            )
            > 0
        )

        key_padding_mask, attention_mask = self.model.process_attention_masks(
            key_padding_mask,
            attention_mask,
        )

        self.assertIsInstance(key_padding_mask, torch.Tensor)
        self.assertEqual(key_padding_mask.dtype, torch.float32)
        self.assertIsInstance(attention_mask, torch.Tensor)
        self.assertEqual(attention_mask.dtype, torch.float32)

    def test_key_and_padding_mask_input_as_floating_point_mask(self):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=True,
            causal_attention_mask_flag=True,
        )
        self.rebuild_presets(config)

        key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
        attention_mask = torch.randn(
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.target_sequence_length,
        )

        output_key_padding_mask, output_attention_mask = (
            self.model.process_attention_masks(
                key_padding_mask,
                attention_mask,
            )
        )

        self.assertTrue(torch.equal(output_key_padding_mask, key_padding_mask))
        self.assertTrue(torch.equal(output_attention_mask, attention_mask))
