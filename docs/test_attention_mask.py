import copy
import torch
import unittest
from dataclasses import asdict
from Emperor.attention.utils.utils import (
    AttentionProjector,
    AttentionValidator,
    AttentionMask,
)
from Emperor.attention.attention import MultiHeadAttention, MultiHeadAttentionConfig
from docs.utils import default_unittest_config


class TestAttentionMask(unittest.TestCase):
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

        main_model = MultiHeadAttention(self.cfg)
        validator = AttentionValidator(self.config)
        self.query_model = main_model.query_model
        self.key_model = main_model.key_model
        self.value_model = main_model.value_model
        self.qkv_model = main_model.qkv_model

        self.model = AttentionProjector(
            self.config,
            validator,
            self.qkv_model,
            self.query_model,
            self.key_model,
            self.value_model,
        )

        self.batch_size = self.config.batch_size
        self.embedding_dim = self.config.embedding_dim
        self.target_sequence_length = self.config.target_sequence_length
        self.source_sequence_length = self.config.source_sequence_length
        self.num_heads = self.config.num_heads
        self.head_dim = self.embedding_dim // self.num_heads


class Test___canonical_mask(TestAttentionMask):
    def test__input_as_None(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)

        mask = None
        mask_name = ""
        other_type = None
        other_name = ""
        target_type = torch.float32
        check_other = False

        output = m._AttentionMask__canonical_mask(
            mask, mask_name, other_type, other_name, target_type, check_other
        )
        self.assertIsNone(output)

    def test__boolean_mask_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)

        mask = torch.randn(10, 10, dtype=torch.float32) > 0
        mask_name = "maks_to_test"
        other_type = torch.bool
        other_name = "required_mask_dtype"
        target_type = torch.float32
        check_other = True

        output = m._AttentionMask__canonical_mask(
            mask, mask_name, other_type, other_name, target_type, check_other
        )

        self.assertTrue(output.dtype == torch.float32)
        self.assertFalse(output.dtype == mask.dtype)

    def test__float_mask_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)

        mask = torch.randn(10, 10, dtype=torch.float32)
        mask_name = "maks_to_test"
        other_type = torch.float32
        other_name = "required_mask_dtype"
        target_type = torch.float32
        check_other = True

        output = m._AttentionMask__canonical_mask(
            mask, mask_name, other_type, other_name, target_type, check_other
        )

        self.assertTrue(torch.equal(output, mask))


class Test_validate_attention_mask(TestAttentionMask):
    def test__key_padding_mask__None(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)
        m.causal_attention_mask_flag = True

        batch_size = config.batch_size
        num_heads = config.num_heads
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.target_sequence_length

        key_padding_mask = None
        attention_mask = torch.randn(
            batch_size * num_heads, source_sequence_length, target_sequence_length
        )
        attention_mask = attention_mask > 0
        need_weights = False

        output = m._AttentionMask__validate_attention_mask(
            key_padding_mask,
            attention_mask,
            need_weights,
        )

        self.assertIsNone(output)

    def test__boolean_attention_mask(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)
        m.causal_attention_mask_flag = True

        batch_size = config.batch_size
        num_heads = config.num_heads
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.target_sequence_length

        key_padding_mask = torch.randint(0, 2, (batch_size, source_sequence_length))
        attention_mask = torch.randn(
            batch_size * num_heads, source_sequence_length, target_sequence_length
        )
        attention_mask = attention_mask > 0
        need_weights = True

        output = m._AttentionMask__validate_attention_mask(
            key_padding_mask,
            attention_mask,
            need_weights,
        )

        self.assertIsInstance(output, torch.Tensor)
        self.assertNotEqual(output.dtype, torch.bool)
        self.assertEqual(output.dtype, config.target_dtype)
        self.assertFalse(m.causal_attention_mask_flag)

    def test__float_attention_mask(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)
        m.causal_attention_mask_flag = True

        batch_size = config.batch_size
        num_heads = config.num_heads
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.target_sequence_length

        key_padding_mask = torch.randint(0, 2, (batch_size, source_sequence_length))
        attention_mask = torch.randn(
            batch_size * num_heads, source_sequence_length, target_sequence_length
        )
        need_weights = True

        output = m._AttentionMask__validate_attention_mask(
            key_padding_mask,
            attention_mask,
            need_weights,
        )

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.dtype, attention_mask.dtype)
        self.assertTrue(torch.equal(output, attention_mask))
        self.assertFalse(m.causal_attention_mask_flag)


class Test_validate_padding_and_attention_masks(TestAttentionMask):
    def test__inputs_as_None(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)
        m.causal_attention_mask_flag = True

        key_padding_mask = None
        attention_mask = None
        need_weights = True

        key_padding_mask, attention_mask = m.validate_padding_and_attention_masks(
            key_padding_mask,
            attention_mask,
            need_weights,
        )

        self.assertIsNone(key_padding_mask)
        self.assertIsNone(attention_mask)

    def test__only_key_padding_mask_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)
        m.causal_attention_mask_flag = True

        batch_size = config.batch_size
        source_sequence_length = config.source_sequence_length

        key_padding_mask = torch.randint(0, 2, (batch_size, source_sequence_length)) > 0
        attention_mask = None
        need_weights = True

        output_key_padding_mask, output_attention_mask = (
            m.validate_padding_and_attention_masks(
                key_padding_mask,
                attention_mask,
                need_weights,
            )
        )

        self.assertIsInstance(output_key_padding_mask, torch.Tensor)
        self.assertEqual(output_key_padding_mask.dtype, torch.float32)
        self.assertIsNone(output_attention_mask)

    def test__only_attention_mask_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)
        m.causal_attention_mask_flag = True

        batch_size = config.batch_size
        num_heads = config.num_heads
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.target_sequence_length

        key_padding_mask = None
        attention_mask = (
            torch.randn(
                batch_size * num_heads, source_sequence_length, target_sequence_length
            )
            > 0
        )
        need_weights = True

        key_padding_mask, attention_mask = m.validate_padding_and_attention_masks(
            key_padding_mask,
            attention_mask,
            need_weights,
        )

        self.assertIsNone(key_padding_mask)
        self.assertIsInstance(attention_mask, torch.Tensor)
        self.assertEqual(attention_mask.dtype, config.target_dtype)

    def test__key_padding_mask__and__attention_mask_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)
        m.causal_attention_mask_flag = True

        batch_size = config.batch_size
        num_heads = config.num_heads
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.target_sequence_length

        key_padding_mask = torch.randint(0, 2, (batch_size, source_sequence_length)) > 0
        attention_mask = (
            torch.randn(
                batch_size * num_heads, source_sequence_length, target_sequence_length
            )
            > 0
        )
        need_weights = True

        key_padding_mask, attention_mask = m.validate_padding_and_attention_masks(
            key_padding_mask,
            attention_mask,
            need_weights,
        )

        self.assertIsInstance(key_padding_mask, torch.Tensor)
        self.assertEqual(key_padding_mask.dtype, torch.float32)
        self.assertIsInstance(attention_mask, torch.Tensor)
        self.assertEqual(attention_mask.dtype, torch.float32)

    def test__key_padding_mask__and__attention_mask__as__float_inputs(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)
        m.causal_attention_mask_flag = True

        batch_size = config.batch_size
        num_heads = config.num_heads
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.target_sequence_length

        key_padding_mask = torch.randn(batch_size, source_sequence_length)
        attention_mask = torch.randn(
            batch_size * num_heads, source_sequence_length, target_sequence_length
        )
        need_weights = True

        output_key_padding_mask, output_attention_mask = (
            m.validate_padding_and_attention_masks(
                key_padding_mask,
                attention_mask,
                need_weights,
            )
        )

        self.assertTrue(torch.equal(output_key_padding_mask, key_padding_mask))
        self.assertTrue(torch.equal(output_attention_mask, attention_mask))
