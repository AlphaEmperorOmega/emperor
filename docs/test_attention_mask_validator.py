import torch
import unittest

from dataclasses import asdict
from docs.config import default_unittest_config
from Emperor.attention.utils.handlers.maks import Mask
from Emperor.attention.utils.layer import MultiHeadAttentionConfig
from Emperor.attention.utils.handlers._validator import MaskValidator


class TestMaskValidator(unittest.TestCase):
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

        model = Mask(self.config)
        self.model = MaskValidator(model)

        self.batch_size = self.config.batch_size
        self.embedding_dim = self.config.embedding_dim
        self.target_sequence_length = self.config.target_sequence_length
        self.source_sequence_length = self.config.source_sequence_length
        self.num_heads = self.config.num_heads
        self.head_dim = self.embedding_dim // self.num_heads


class Test_ensure_mask_is_float_or_bool(TestMaskValidator):
    def test_check_if_error_is_thrown_when_integer_mask_is_given(self):
        mask_shape = (
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.target_sequence_length,
        )
        integer_mask = torch.randint(0, 20, mask_shape)
        maks_name = "test_mask"

        with self.assertRaises(RuntimeError) as context:
            self.model.ensure_mask_is_float_or_bool(integer_mask, maks_name)

    def test_ensure_no_error_is_thrown_when_float_mask_is_given(self):
        mask_shape = (
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.target_sequence_length,
        )
        float_mask = torch.randn(mask_shape)
        maks_name = "test_mask"

        output = self.model.ensure_mask_is_float_or_bool(float_mask, maks_name)
        self.assertIsNone(output)

    def test_ensure_no_error_is_thrown_when_boolean_mask_is_given(self):
        mask_shape = (
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.target_sequence_length,
        )
        boolean_mask = torch.randn(mask_shape) > 0
        maks_name = "test_mask"

        output = self.model.ensure_mask_is_float_or_bool(boolean_mask, maks_name)
        self.assertIsNone(output)


class Test_ensure_mask_is_correct_dtype(TestMaskValidator):
    def test_ensure_errror_is_thrown_when_incorrect_dtype_is_given_and_check_other_flag_is_set_to_True(
        self,
    ):
        mask_shape = (
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.target_sequence_length,
        )
        mask = torch.randn(mask_shape, dtype=torch.float64)
        maks_name = "test_mask"
        other_type = torch.float32
        other_name = "real_maks_dtype"
        check_other = True

        with self.assertRaises(RuntimeError) as context:
            self.model.ensure_mask_is_correct_dtype(
                mask, maks_name, other_type, other_name, check_other
            )

    def test_ensure_nothing_happends_when_incorrect_dtype_is_given_and_check_other_flag_is_set_to_False(
        self,
    ):
        mask_shape = (
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.target_sequence_length,
        )

        mask = torch.randn(mask_shape, dtype=torch.float64)
        maks_name = "test_mask"
        other_type = torch.float32
        other_name = "real_maks_dtype"
        check_other = False

        output = self.model.ensure_mask_is_correct_dtype(
            mask, maks_name, other_type, other_name, check_other
        )
        self.assertIsNone(output)

    def test_ensure_no_error_is_thrown_when_given_dtype_and_mask_dtype_match(self):
        mask_shape = (
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.target_sequence_length,
        )

        mask = torch.randn(mask_shape, dtype=torch.float32)
        maks_name = "test_mask"
        other_type = torch.float32
        other_name = "real_maks_dtype"
        check_other = True

        output = self.model.ensure_mask_is_correct_dtype(
            mask, maks_name, other_type, other_name, check_other
        )
        self.assertIsNone(output)

    def test_ensure_no_error_is_thrown_when_no_dtype_is_given_but_check_other_flag_is_set_to_True(
        self,
    ):
        mask_shape = (
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.target_sequence_length,
        )

        mask = torch.randn(mask_shape, dtype=torch.float32)
        maks_name = "test_mask"
        other_type = None
        other_name = "real_maks_dtype"
        check_other = True

        output = self.model.ensure_mask_is_correct_dtype(
            mask, maks_name, other_type, other_name, check_other
        )
        self.assertIsNone(output)


class Test_ensure_attention_mask_for_required_causal_mask(TestMaskValidator):
    def test_no_input_with_causal_mask_flag_set_to_True(self):
        attention_mask = None
        causal_attention_mask_flag = True
        with self.assertRaises(RuntimeError) as context:
            self.model.ensure_attention_mask_for_required_causal_mask(
                attention_mask, causal_attention_mask_flag
            )

    def test_no_input_with_causal_mask_flag_set_to_False(self):
        attention_mask = None
        causal_attention_mask_flag = False
        output = self.model.ensure_attention_mask_for_required_causal_mask(
            attention_mask, causal_attention_mask_flag
        )
        self.assertIsNone(output)

    def test_attention_mask_input_with_causal_mask_flag_set_to_False(self):
        attention_mask = torch.randn(
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.target_sequence_length,
        )

        output = self.model.ensure_attention_mask_for_required_causal_mask(
            attention_mask, False
        )
        self.assertIsNone(output)

    def test_attention_mask_input_with_causal_mask_flag_set_to_True(self):
        attention_mask = torch.randn(
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.target_sequence_length,
        )

        output = self.model.ensure_attention_mask_for_required_causal_mask(
            attention_mask, True
        )
        self.assertIsNone(output)
