import torch
import unittest

from dataclasses import asdict
from Emperor.attention.utils.maks_handler import Mask
from Emperor.attention.utils.validation_handler import Validator
from Emperor.attention.attention import MultiHeadAttentionConfig
from docs.utils import default_unittest_config


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

        validator = Validator(self.config)
        self.model = Mask(self.config, validator)

        self.batch_size = self.config.batch_size
        self.embedding_dim = self.config.embedding_dim
        self.target_sequence_length = self.config.target_sequence_length
        self.source_sequence_length = self.config.source_sequence_length
        self.num_heads = self.config.num_heads
        self.head_dim = self.embedding_dim // self.num_heads


class Test___canonical_mask(TestMask):
    def test__input_as_None(self):
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

    def test__boolean_mask_input(self):
        mask = torch.randn(10, 10, dtype=torch.float32) > 0
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

    def test__float_mask_input(self):
        mask = torch.randn(10, 10, dtype=torch.float32)
        mask_name = "maks_to_test"
        other_type = torch.float32
        other_name = "required_mask_dtype"
        target_type = torch.float32
        check_other = True

        output = self.model._Mask__canonical_mask(
            mask, mask_name, other_type, other_name, target_type, check_other
        )

        self.assertTrue(torch.equal(output, mask))


class Test_validate_attention_mask(TestMask):
    def test__key_padding_mask__None(self):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=False,
            causal_attention_mask_flag=True,
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
        self.assertIsNone(output)

    def test__boolean_attention_mask(self):
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

    def test__float_attention_mask(self):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=True,
            causal_attention_mask_flag=True,
        )
        self.rebuild_presets(config)

        key_padding_mask = torch.randint(
            0, 2, (self.batch_size, self.source_sequence_length)
        )

        key_padding_mask = torch.randint(
            0, 2, (self.batch_size, self.source_sequence_length)
        )
        attention_mask = torch.randn(
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.target_sequence_length,
        )

        output = self.model._Mask__validate_attention_mask(
            key_padding_mask,
            attention_mask,
        )

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.dtype, attention_mask.dtype)
        self.assertTrue(torch.equal(output, attention_mask))
        self.assertFalse(self.model.causal_attention_mask_flag)


class Test_process_attention_masks(TestMask):
    def test__inputs_as_None(self):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=True,
            causal_attention_mask_flag=True,
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

    def test__only_key_padding_mask_input(self):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=True,
            causal_attention_mask_flag=True,
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

    def test__only_attention_mask_input(self):
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

    def test__key_padding_mask__and__attention_mask_input(self):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=False,
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

    def test__key_padding_mask__and__attention_mask__as__float_inputs(self):
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


class Test_is_mask_float_or_bool(TestMask):
    def test_check_if_error_is_thrown_when_integer_mask_is_given(self):
        mask_shape = (
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.target_sequence_length,
        )
        integer_mask = torch.randint(0, 20, mask_shape)
        maks_name = "test_mask"

        with self.assertRaises(RuntimeError) as context:
            self.model._Mask__ensure_mask_is_float_or_bool(integer_mask, maks_name)

    def test_ensure_no_error_is_thrown_when_float_mask_is_given(self):
        mask_shape = (
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.target_sequence_length,
        )
        float_mask = torch.randn(mask_shape)
        maks_name = "test_mask"

        output = self.model._Mask__ensure_mask_is_float_or_bool(float_mask, maks_name)
        self.assertIsNone(output)

    def test_ensure_no_error_is_thrown_when_boolean_mask_is_given(self):
        mask_shape = (
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.target_sequence_length,
        )
        boolean_mask = torch.randn(mask_shape) > 0
        maks_name = "test_mask"

        output = self.model._Mask__ensure_mask_is_float_or_bool(boolean_mask, maks_name)
        self.assertIsNone(output)


# class Test_is_mask_correct_dtype(TestMask):
#     def test__incorrect__other_dtype__check_other__True(self):
#         c = copy.deepcopy(self.cfg)
#         config = c.multi_head_attention_model_config
#         m = Validator(config)
#         m.batched_input_flag = False
#
#         mask = torch.randn(10, 10, dtype=torch.float64)
#         maks_name = "test_mask"
#         other_type = torch.float32
#         other_name = "real_maks_dtype"
#         check_other = True
#
#         with self.assertRaises(RuntimeError) as context:
#             m.is_mask_correct_dtype(
#                 mask, maks_name, other_type, other_name, check_other
#             )
#
#     def test__incorrect__other_dtype__check_other__False(self):
#         c = copy.deepcopy(self.cfg)
#         config = c.multi_head_attention_model_config
#         m = Validator(config)
#         m.batched_input_flag = False
#
#         mask = torch.randn(10, 10, dtype=torch.float64)
#         maks_name = "test_mask"
#         other_type = torch.float32
#         other_name = "real_maks_dtype"
#         check_other = False
#
#         output = m.is_mask_correct_dtype(
#             mask, maks_name, other_type, other_name, check_other
#         )
#         self.assertIsNone(output)
#
#     def test__mask__and__other_type__same_dtype(self):
#         c = copy.deepcopy(self.cfg)
#         config = c.multi_head_attention_model_config
#         m = Validator(config)
#         m.batched_input_flag = False
#
#         mask = torch.randn(10, 10, dtype=torch.float32)
#         maks_name = "test_mask"
#         other_type = torch.float32
#         other_name = "real_maks_dtype"
#         check_other = True
#
#         output = m.is_mask_correct_dtype(
#             mask, maks_name, other_type, other_name, check_other
#         )
#         self.assertIsNone(output)
#
#     def test__other_type__None__check_other__True(self):
#         c = copy.deepcopy(self.cfg)
#         config = c.multi_head_attention_model_config
#         m = Validator(config)
#         m.batched_input_flag = False
#
#         mask = torch.randn(10, 10, dtype=torch.float32)
#         maks_name = "test_mask"
#         other_type = None
#         other_name = "real_maks_dtype"
#         check_other = True
#
#         output = m.is_mask_correct_dtype(
#             mask, maks_name, other_type, other_name, check_other
#         )
#         self.assertIsNone(output)
