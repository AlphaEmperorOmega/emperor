import torch
import unittest
import itertools

from dataclasses import asdict
from Emperor.attention.utils._validator import MultiHeadAttentionValidator
from Emperor.attention.utils.enums import AttentionOptions
from Emperor.attention.utils.handlers.batch import BatchDimensionManager
from Emperor.attention.utils.handlers.bias import KeyValueBias
from Emperor.attention.utils.utils import Utils
from Emperor.attention.utils.handlers.maks import Mask
from Emperor.linears.options import LinearLayerStackOptions
from Emperor.adaptive.options import AdaptiveLayerStackOptions
from Emperor.attention.utils.handlers.processor import ProcessorBase
from Emperor.attention.utils.handlers.projector import ProjectorBase
from Emperor.attention.utils.presets import MultiHeadAttentionPresets
from Emperor.attention.utils.layer import MultiHeadAttention, MultiHeadAttentionConfig


def create_key_padding_mask(
    batch_size: int, source_sequence_length: int
) -> torch.Tensor:
    key_padding_mask_shape = (
        batch_size,
        source_sequence_length,
    )
    key_padding_mask = torch.randint(0, 2, key_padding_mask_shape)
    key_padding_mask = torch.where(
        key_padding_mask > 0,
        torch.tensor(float("-inf")),
        torch.tensor(0.0),
    )
    return key_padding_mask


def create_attention_mask(
    target_sequence_length: int,
    source_sequence_length: int,
    attention_mask_repeat: int = 1,
) -> torch.Tensor:
    attention_mask_shape = (
        1,
        target_sequence_length,
        source_sequence_length,
    )
    attention_mask = torch.randint(0, 2, attention_mask_shape)
    attention_mask = torch.where(
        attention_mask > 0,
        torch.tensor(float("-inf")),
        torch.tensor(0.0),
    )
    attention_mask = attention_mask.repeat(attention_mask_repeat, 1, 1)
    return attention_mask


def create_qkv_tensors(
    target_sequence_length: int,
    source_sequence_length: int,
    batch_size: int,
    embedding_dim: int,
    attention_option: AttentionOptions,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.randn(
        target_sequence_length,
        batch_size,
        embedding_dim,
    )
    key = torch.randn(
        source_sequence_length,
        batch_size,
        embedding_dim,
    )
    value = torch.randn(
        source_sequence_length,
        batch_size,
        embedding_dim,
    )

    if attention_option == AttentionOptions.SELF_ATTENTION:
        query = key = value = query

    return query, key, value


class TestAttention(unittest.TestCase):
    def test__init_input_layer_with_default_config(self):
        c = MultiHeadAttentionPresets.multi_head_attention_preset(
            embedding_dim=12,
            query_key_projection_dim=16,
            value_projection_dim=20,
        )
        m = MultiHeadAttention(c)
        self.assertEqual(m.batch_size, c.batch_size)
        self.assertEqual(m.model_type, c.model_type)
        self.assertEqual(m.num_heads, c.num_heads)
        self.assertEqual(m.embedding_dim, c.embedding_dim)
        self.assertEqual(m.target_dtype, c.target_dtype)
        self.assertEqual(m.target_sequence_length, c.target_sequence_length)
        self.assertEqual(m.source_sequence_length, c.source_sequence_length)
        self.assertEqual(m.attention_option, c.attention_option)
        self.assertEqual(m.dropout_probability, c.dropout_probability)
        self.assertEqual(m.key_value_bias_flag, c.key_value_bias_flag)
        self.assertEqual(m.zero_attention_flag, c.zero_attention_flag)
        self.assertEqual(m.query_key_projection_dim, c.query_key_projection_dim)
        self.assertEqual(m.value_projection_dim, c.value_projection_dim)
        self.assertIsInstance(m.validator, MultiHeadAttentionValidator)
        self.assertIsInstance(m.masks, Mask)
        self.assertIsInstance(m.projector, ProjectorBase)
        self.assertIsInstance(m.processor, ProcessorBase)
        self.assertIsInstance(m.bias, KeyValueBias)
        self.assertIsInstance(m.utils, Utils)
        self.assertIsInstance(m.batch_utils, BatchDimensionManager)

    # def test_init_with_different_configation_options(self):
    #     attention_options = [
    #         AttentionOptions.SELF_ATTENTION,
    #         AttentionOptions.INDEPENDENT,
    #     ]
    #     model_types = list(LinearLayerStackOptions) + list(AdaptiveLayerStackOptions)
    #
    #     for model_type in model_types:
    #         for attention_option in attention_options:
    #             for key_value_bias_flag in [True, False]:
    #                 for zero_attention_flag in [True, False]:
    #                     for causal_attention_mask_flag in [True, False]:
    #                         for add_key_value_bias_flag in [True, False]:
    #                             for average_attention_weights_flag in [True, False]:
    #                                 for return_attention_weights_flag in [True, False]:
    #                                     message = f"Test failed for options - model_type: {model_type}, attention_option: {attention_option}, key_value_bias_flag: {key_value_bias_flag}, zero_attention_flag: {zero_attention_flag}, causal_attention_mask_flag: {causal_attention_mask_flag}, add_key_value_bias_flag: {add_key_value_bias_flag}, average_attention_weights_flag: {average_attention_weights_flag}, return_attention_weights_flag: {return_attention_weights_flag}"
    #                                     with self.subTest(message=message):
    #                                         c = MultiHeadAttentionPresets.multi_head_attention_preset(
    #                                             model_type=model_type,
    #                                             embedding_dim=12,
    #                                             query_key_projection_dim=12,
    #                                             value_projection_dim=12,
    #                                             attention_option=attention_option,
    #                                             key_value_bias_flag=key_value_bias_flag,
    #                                             zero_attention_flag=zero_attention_flag,
    #                                             causal_attention_mask_flag=causal_attention_mask_flag,
    #                                             add_key_value_bias_flag=add_key_value_bias_flag,
    #                                             average_attention_weights_flag=average_attention_weights_flag,
    #                                             return_attention_weights_flag=return_attention_weights_flag,
    #                                         )
    #                                         m = MultiHeadAttention(c)
    #
    #                                         self.assertIsInstance(m, MultiHeadAttention)
    #                                         self.assertEqual(m.model_type, model_type)
    #                                         self.assertEqual(
    #                                             m.attention_option, attention_option
    #                                         )
    #                                         self.assertEqual(m.embedding_dim, 12)
    #                                         self.assertEqual(
    #                                             m.key_value_bias_flag,
    #                                             key_value_bias_flag,
    #                                         )
    #                                         self.assertEqual(
    #                                             m.zero_attention_flag,
    #                                             zero_attention_flag,
    #                                         )
    #                                         self.assertEqual(
    #                                             m.causal_attention_mask_flag,
    #                                             causal_attention_mask_flag,
    #                                         )
    #                                         self.assertEqual(
    #                                             m.add_key_value_bias_flag,
    #                                             add_key_value_bias_flag,
    #                                         )
    #                                         self.assertEqual(
    #                                             m.average_attention_weights_flag,
    #                                             average_attention_weights_flag,
    #                                         )
    #                                         self.assertEqual(
    #                                             m.return_attention_weights_flag,
    #                                             return_attention_weights_flag,
    #                                         )

    def test__attention_option_with_different_qkv_tensors(self):
        batch_size = 4
        target_sequence_length = 8
        source_sequence_length = 10
        embeddimd_dim = 12
        qkv_dimensions = [0, 16, 20]

        for query_key_projection_dim in qkv_dimensions:
            for value_projection_dim in qkv_dimensions:
                for attention_option in AttentionOptions:
                    message = f"Test failed for the inputs: attention_option={attention_option}, query_key_projection_dim={query_key_projection_dim}, value_projection_dim={value_projection_dim}"
                    with self.subTest(i=message):
                        if attention_option == AttentionOptions.SELF_ATTENTION:
                            continue

                        c = MultiHeadAttentionPresets.multi_head_attention_preset(
                            batch_size=batch_size,
                            embedding_dim=embeddimd_dim,
                            target_sequence_length=target_sequence_length,
                            source_sequence_length=source_sequence_length,
                            attention_option=attention_option,
                            query_key_projection_dim=query_key_projection_dim,
                            value_projection_dim=value_projection_dim,
                        )
                        m = MultiHeadAttention(c)

                        query, key, value = create_qkv_tensors(
                            target_sequence_length,
                            source_sequence_length,
                            batch_size,
                            m.embedding_dim,
                            attention_option,
                        )

                        key_padding_mask = None
                        attention_mask = None
                        static_key = None
                        static_values = None

                        attention_output, attention_weights = m.forward(
                            query,
                            key,
                            value,
                            key_padding_mask,
                            attention_mask,
                            static_key,
                            static_values,
                        )

                        expected_shape = (
                            target_sequence_length,
                            batch_size,
                            m.embedding_dim,
                        )

                        expected_attention_weights_shape = (
                            batch_size,
                            m.num_heads,
                            source_sequence_length,
                            source_sequence_length,
                        )
                        self.assertIsInstance(attention_output, torch.Tensor)
                        if (
                            attention_option == AttentionOptions.SELF_ATTENTION
                            and m.return_attention_weights_flag
                        ):
                            self.assertIsInstance(attention_weights, torch.Tensor)
                            self.assertEqual(
                                attention_weights.shape,
                                expected_attention_weights_shape,
                            )
                        else:
                            self.assertIsNone(attention_weights)
                        self.assertEqual(attention_output.shape, expected_shape)

    def test__qkv_tensors_and_key_padding_mask(self):
        batch_size = 4
        sequence_lengths = [8, 10]
        embeddimd_dim = 12
        qkv_dimensions = [0, 16, 20]

        for target_sequence_length in sequence_lengths:
            for source_sequence_length in sequence_lengths:
                for query_key_projection_dim in qkv_dimensions:
                    for value_projection_dim in qkv_dimensions:
                        for attention_option in AttentionOptions:
                            message = f"Test failed for the inputs: attention_option={attention_option}, query_key_projection_dim={query_key_projection_dim}, value_projection_dim={value_projection_dim}"
                            with self.subTest(i=message):
                                if attention_option == AttentionOptions.SELF_ATTENTION:
                                    query_key_projection_dim = embeddimd_dim
                                    value_projection_dim = embeddimd_dim
                                    source_sequence_length = target_sequence_length

                                c = MultiHeadAttentionPresets.multi_head_attention_preset(
                                    batch_size=batch_size,
                                    embedding_dim=embeddimd_dim,
                                    target_sequence_length=target_sequence_length,
                                    source_sequence_length=source_sequence_length,
                                    attention_option=attention_option,
                                    query_key_projection_dim=query_key_projection_dim,
                                    value_projection_dim=value_projection_dim,
                                )

                                m = MultiHeadAttention(c)

                                query, key, value = create_qkv_tensors(
                                    target_sequence_length,
                                    source_sequence_length,
                                    batch_size,
                                    m.embedding_dim,
                                    attention_option,
                                )

                                if attention_option == AttentionOptions.SELF_ATTENTION:
                                    query = key = value = query

                                key_padding_mask = create_key_padding_mask(
                                    batch_size, source_sequence_length
                                )
                                attention_mask = None
                                static_key = None
                                static_values = None

                                attention_output, attention_weights = m.forward(
                                    query,
                                    key,
                                    value,
                                    key_padding_mask,
                                    attention_mask,
                                    static_key,
                                    static_values,
                                )

                                expected_shape = (
                                    target_sequence_length,
                                    batch_size,
                                    m.embedding_dim,
                                )

                                expected_attention_weights_shape = (
                                    batch_size,
                                    m.num_heads,
                                    source_sequence_length,
                                    source_sequence_length,
                                )
                                self.assertIsInstance(attention_output, torch.Tensor)
                                if (
                                    attention_option == AttentionOptions.SELF_ATTENTION
                                    and m.return_attention_weights_flag
                                ):
                                    self.assertIsInstance(
                                        attention_weights, torch.Tensor
                                    )
                                    self.assertEqual(
                                        attention_weights.shape,
                                        expected_attention_weights_shape,
                                    )
                                else:
                                    self.assertIsNone(attention_weights)
                                self.assertEqual(attention_output.shape, expected_shape)

    def test__qkv_tensors_and_attention_mask(self):
        batch_size = 4
        sequence_lengths = [8, 10]
        embeddimd_dim = 12
        qkv_dimensions = [0, 16, 20]

        for target_sequence_length in sequence_lengths:
            for source_sequence_length in sequence_lengths:
                for query_key_projection_dim in qkv_dimensions:
                    for value_projection_dim in qkv_dimensions:
                        for attention_option in AttentionOptions:
                            message = f"Test failed for the inputs: attention_option={attention_option}, query_key_projection_dim={query_key_projection_dim}, value_projection_dim={value_projection_dim}"
                            with self.subTest(i=message):
                                if attention_option == AttentionOptions.SELF_ATTENTION:
                                    query_key_projection_dim = embeddimd_dim
                                    value_projection_dim = embeddimd_dim
                                    source_sequence_length = target_sequence_length

                                c = MultiHeadAttentionPresets.multi_head_attention_preset(
                                    batch_size=batch_size,
                                    embedding_dim=embeddimd_dim,
                                    target_sequence_length=target_sequence_length,
                                    source_sequence_length=source_sequence_length,
                                    attention_option=attention_option,
                                    query_key_projection_dim=query_key_projection_dim,
                                    value_projection_dim=value_projection_dim,
                                )

                                m = MultiHeadAttention(c)

                                query, key, value = create_qkv_tensors(
                                    target_sequence_length,
                                    source_sequence_length,
                                    batch_size,
                                    m.embedding_dim,
                                    attention_option,
                                )

                                key_padding_mask = None

                                attention_mask_repeat = batch_size * m.num_heads
                                if (
                                    attention_option
                                    == AttentionOptions.MIXTURE_OF_ATTENTION_HEADS
                                ):
                                    attention_mask_repeat = (
                                        batch_size
                                        * m.num_heads
                                        * c.experts_config.top_k
                                    )

                                attention_mask = create_attention_mask(
                                    target_sequence_length,
                                    source_sequence_length,
                                    attention_mask_repeat,
                                )
                                static_key = None
                                static_values = None

                                attention_output, attention_weights = m.forward(
                                    query,
                                    key,
                                    value,
                                    key_padding_mask,
                                    attention_mask,
                                    static_key,
                                    static_values,
                                )

                                expected_shape = (
                                    target_sequence_length,
                                    batch_size,
                                    m.embedding_dim,
                                )
                                expected_attention_weights_shape = (
                                    batch_size,
                                    m.num_heads,
                                    source_sequence_length,
                                    source_sequence_length,
                                )
                                self.assertIsInstance(attention_output, torch.Tensor)
                                if (
                                    attention_option == AttentionOptions.SELF_ATTENTION
                                    and m.return_attention_weights_flag
                                ):
                                    self.assertIsInstance(
                                        attention_weights, torch.Tensor
                                    )
                                    self.assertEqual(
                                        attention_weights.shape,
                                        expected_attention_weights_shape,
                                    )
                                else:
                                    self.assertIsNone(attention_weights)
                                self.assertEqual(attention_output.shape, expected_shape)

    def test__qkv_tensors_with_combined_key_and_attention_masks(self):
        batch_size = 4
        sequence_lengths = [8, 10]
        embeddimd_dim = 12
        qkv_dimensions = [0, 16, 20]

        for target_sequence_length in sequence_lengths:
            for source_sequence_length in sequence_lengths:
                for query_key_projection_dim in qkv_dimensions:
                    for value_projection_dim in qkv_dimensions:
                        for attention_option in AttentionOptions:
                            message = f"Test failed for the inputs: attention_option={attention_option}, query_key_projection_dim={query_key_projection_dim}, value_projection_dim={value_projection_dim}"
                            with self.subTest(i=message):
                                if attention_option == AttentionOptions.SELF_ATTENTION:
                                    query_key_projection_dim = embeddimd_dim
                                    value_projection_dim = embeddimd_dim
                                    source_sequence_length = target_sequence_length

                                c = MultiHeadAttentionPresets.multi_head_attention_preset(
                                    batch_size=batch_size,
                                    embedding_dim=embeddimd_dim,
                                    target_sequence_length=target_sequence_length,
                                    source_sequence_length=source_sequence_length,
                                    attention_option=attention_option,
                                    query_key_projection_dim=query_key_projection_dim,
                                    value_projection_dim=value_projection_dim,
                                )

                                m = MultiHeadAttention(c)

                                query, key, value = create_qkv_tensors(
                                    target_sequence_length,
                                    source_sequence_length,
                                    batch_size,
                                    m.embedding_dim,
                                    attention_option,
                                )

                                key_padding_mask = create_key_padding_mask(
                                    batch_size, source_sequence_length
                                )

                                attention_mask_repeat = batch_size * m.num_heads
                                if (
                                    attention_option
                                    == AttentionOptions.MIXTURE_OF_ATTENTION_HEADS
                                ):
                                    attention_mask_repeat = (
                                        batch_size
                                        * m.num_heads
                                        * c.experts_config.top_k
                                    )

                                attention_mask = create_attention_mask(
                                    target_sequence_length,
                                    source_sequence_length,
                                    attention_mask_repeat,
                                )
                                static_key = None
                                static_values = None

                                attention_output, attention_weights = m.forward(
                                    query,
                                    key,
                                    value,
                                    key_padding_mask,
                                    attention_mask,
                                    static_key,
                                    static_values,
                                )

                                expected_shape = (
                                    target_sequence_length,
                                    batch_size,
                                    m.embedding_dim,
                                )
                                expected_attention_weights_shape = (
                                    batch_size,
                                    m.num_heads,
                                    source_sequence_length,
                                    source_sequence_length,
                                )
                                self.assertIsInstance(attention_output, torch.Tensor)
                                if (
                                    attention_option == AttentionOptions.SELF_ATTENTION
                                    and m.return_attention_weights_flag
                                ):
                                    self.assertIsInstance(
                                        attention_weights, torch.Tensor
                                    )
                                    self.assertEqual(
                                        attention_weights.shape,
                                        expected_attention_weights_shape,
                                    )
                                else:
                                    self.assertIsNone(attention_weights)
                                self.assertEqual(attention_output.shape, expected_shape)

    def test__return_attention_weights_flag(self):
        batch_size = 4
        sequence_lengths = [8, 10]
        embeddimd_dim = 12
        qkv_dimensions = [0, 16, 20]
        bool_options = [True, False]

        for target_sequence_length in sequence_lengths:
            for source_sequence_length in sequence_lengths:
                for query_key_projection_dim in qkv_dimensions:
                    for value_projection_dim in qkv_dimensions:
                        for attention_option in AttentionOptions:
                            for return_attention_weights_flag in bool_options:
                                message = f"Test failed for the inputs: attention_option={attention_option}, query_key_projection_dim={query_key_projection_dim}, value_projection_dim={value_projection_dim}, return_attention_weights_flag={return_attention_weights_flag}"
                                with self.subTest(i=message):
                                    return_attention_weights_flag = False
                                    if (
                                        attention_option
                                        == AttentionOptions.SELF_ATTENTION
                                    ):
                                        query_key_projection_dim = embeddimd_dim
                                        value_projection_dim = embeddimd_dim
                                        source_sequence_length = target_sequence_length
                                        return_attention_weights_flag = (
                                            return_attention_weights_flag
                                        )

                                    c = MultiHeadAttentionPresets.multi_head_attention_preset(
                                        batch_size=batch_size,
                                        embedding_dim=embeddimd_dim,
                                        target_sequence_length=target_sequence_length,
                                        source_sequence_length=source_sequence_length,
                                        attention_option=attention_option,
                                        query_key_projection_dim=query_key_projection_dim,
                                        value_projection_dim=value_projection_dim,
                                        return_attention_weights_flag=return_attention_weights_flag,
                                    )

                                    m = MultiHeadAttention(c)

                                    query, key, value = create_qkv_tensors(
                                        target_sequence_length,
                                        source_sequence_length,
                                        batch_size,
                                        m.embedding_dim,
                                        attention_option,
                                    )

                                    key_padding_mask = create_key_padding_mask(
                                        batch_size, source_sequence_length
                                    )

                                    attention_mask_repeat = batch_size * m.num_heads
                                    if (
                                        attention_option
                                        == AttentionOptions.MIXTURE_OF_ATTENTION_HEADS
                                    ):
                                        attention_mask_repeat = (
                                            batch_size
                                            * m.num_heads
                                            * c.experts_config.top_k
                                        )

                                    attention_mask = create_attention_mask(
                                        target_sequence_length,
                                        source_sequence_length,
                                        attention_mask_repeat,
                                    )
                                    static_key = None
                                    static_values = None

                                    attention_output, attention_weights = m.forward(
                                        query,
                                        key,
                                        value,
                                        key_padding_mask,
                                        attention_mask,
                                        static_key,
                                        static_values,
                                    )

                                    expected_shape = (
                                        target_sequence_length,
                                        batch_size,
                                        m.embedding_dim,
                                    )
                                    self.assertIsInstance(
                                        attention_output, torch.Tensor
                                    )
                                    if (
                                        attention_option
                                        == AttentionOptions.SELF_ATTENTION
                                        and return_attention_weights_flag
                                    ):
                                        self.assertIsInstance(
                                            attention_weights, torch.Tensor
                                        )
                                    else:
                                        self.assertIsNone(attention_weights)
                                    self.assertEqual(
                                        attention_output.shape, expected_shape
                                    )

    def test__zero_attention_flag(self):
        batch_size = 4
        sequence_lengths = [8, 10]
        embeddimd_dim = 12
        qkv_dimensions = [0, 16, 20]
        bool_options = [True, False]

        for target_sequence_length in sequence_lengths:
            for source_sequence_length in sequence_lengths:
                for query_key_projection_dim in qkv_dimensions:
                    for value_projection_dim in qkv_dimensions:
                        for attention_option in AttentionOptions:
                            for zero_attention_flag in bool_options:
                                message = f"Test failed for the inputs: attention_option={attention_option}, query_key_projection_dim={query_key_projection_dim}, value_projection_dim={value_projection_dim}, zero_attention_flag={zero_attention_flag}"
                                with self.subTest(i=message):
                                    if (
                                        attention_option
                                        == AttentionOptions.SELF_ATTENTION
                                    ):
                                        query_key_projection_dim = embeddimd_dim
                                        value_projection_dim = embeddimd_dim
                                        source_sequence_length = target_sequence_length

                                    c = MultiHeadAttentionPresets.multi_head_attention_preset(
                                        batch_size=batch_size,
                                        embedding_dim=embeddimd_dim,
                                        target_sequence_length=target_sequence_length,
                                        source_sequence_length=source_sequence_length,
                                        attention_option=attention_option,
                                        query_key_projection_dim=query_key_projection_dim,
                                        value_projection_dim=value_projection_dim,
                                        zero_attention_flag=zero_attention_flag,
                                    )

                                    m = MultiHeadAttention(c)

                                    query, key, value = create_qkv_tensors(
                                        target_sequence_length,
                                        source_sequence_length,
                                        batch_size,
                                        m.embedding_dim,
                                        attention_option,
                                    )

                                    key_padding_mask = create_key_padding_mask(
                                        batch_size, source_sequence_length
                                    )

                                    attention_mask_repeat = batch_size * m.num_heads
                                    if (
                                        attention_option
                                        == AttentionOptions.MIXTURE_OF_ATTENTION_HEADS
                                    ):
                                        attention_mask_repeat = (
                                            batch_size
                                            * m.num_heads
                                            * c.experts_config.top_k
                                        )

                                    attention_mask = create_attention_mask(
                                        target_sequence_length,
                                        source_sequence_length,
                                        attention_mask_repeat,
                                    )
                                    static_key = None
                                    static_values = None

                                    attention_output, attention_weights = m.forward(
                                        query,
                                        key,
                                        value,
                                        key_padding_mask,
                                        attention_mask,
                                        static_key,
                                        static_values,
                                    )

                                    expected_shape = (
                                        target_sequence_length,
                                        batch_size,
                                        m.embedding_dim,
                                    )
                                    self.assertIsInstance(
                                        attention_output, torch.Tensor
                                    )
                                    self.assertIsNone(attention_weights)
                                    self.assertEqual(
                                        attention_output.shape, expected_shape
                                    )

    def test__causal_attention_mask_flag(self):
        batch_size = 4
        sequence_lengths = [8, 10]
        embeddimd_dim = 12
        qkv_dimensions = [0, 16, 20]
        bool_options = [True, False]

        for target_sequence_length in sequence_lengths:
            for source_sequence_length in sequence_lengths:
                for query_key_projection_dim in qkv_dimensions:
                    for value_projection_dim in qkv_dimensions:
                        for attention_option in AttentionOptions:
                            for causal_attention_mask_flag in bool_options:
                                message = f"Test failed for the inputs: attention_option={attention_option}, query_key_projection_dim={query_key_projection_dim}, value_projection_dim={value_projection_dim}, causal_attention_mask_flag={causal_attention_mask_flag}"
                                with self.subTest(i=message):
                                    if (
                                        attention_option
                                        == AttentionOptions.SELF_ATTENTION
                                    ):
                                        query_key_projection_dim = embeddimd_dim
                                        value_projection_dim = embeddimd_dim
                                        source_sequence_length = target_sequence_length

                                    c = MultiHeadAttentionPresets.multi_head_attention_preset(
                                        batch_size=batch_size,
                                        embedding_dim=embeddimd_dim,
                                        target_sequence_length=target_sequence_length,
                                        source_sequence_length=source_sequence_length,
                                        attention_option=attention_option,
                                        query_key_projection_dim=query_key_projection_dim,
                                        value_projection_dim=value_projection_dim,
                                        causal_attention_mask_flag=causal_attention_mask_flag,
                                    )

                                    m = MultiHeadAttention(c)

                                    query, key, value = create_qkv_tensors(
                                        target_sequence_length,
                                        source_sequence_length,
                                        batch_size,
                                        m.embedding_dim,
                                        attention_option,
                                    )

                                    key_padding_mask = create_key_padding_mask(
                                        batch_size, source_sequence_length
                                    )

                                    attention_mask_repeat = batch_size * m.num_heads
                                    if (
                                        attention_option
                                        == AttentionOptions.MIXTURE_OF_ATTENTION_HEADS
                                    ):
                                        attention_mask_repeat = (
                                            batch_size
                                            * m.num_heads
                                            * c.experts_config.top_k
                                        )

                                    attention_mask = create_attention_mask(
                                        target_sequence_length,
                                        source_sequence_length,
                                        attention_mask_repeat,
                                    )
                                    static_key = None
                                    static_values = None

                                    attention_output, attention_weights = m.forward(
                                        query,
                                        key,
                                        value,
                                        key_padding_mask,
                                        attention_mask,
                                        static_key,
                                        static_values,
                                    )

                                    expected_shape = (
                                        target_sequence_length,
                                        batch_size,
                                        m.embedding_dim,
                                    )
                                    self.assertIsInstance(
                                        attention_output, torch.Tensor
                                    )
                                    self.assertIsNone(attention_weights)
                                    self.assertEqual(
                                        attention_output.shape, expected_shape
                                    )

    def test__add_key_value_bias_flag(self):
        batch_size = 4
        sequence_lengths = [8, 10]
        embeddimd_dim = 12
        qkv_dimensions = [0, 16, 20]
        bool_options = [True, False]

        for target_sequence_length in sequence_lengths:
            for source_sequence_length in sequence_lengths:
                for query_key_projection_dim in qkv_dimensions:
                    for value_projection_dim in qkv_dimensions:
                        for attention_option in AttentionOptions:
                            for add_key_value_bias_flag in bool_options:
                                message = f"Test failed for the inputs: attention_option={attention_option}, query_key_projection_dim={query_key_projection_dim}, value_projection_dim={value_projection_dim}, add_key_value_bias_flag={add_key_value_bias_flag}"
                                with self.subTest(i=message):
                                    if (
                                        attention_option
                                        == AttentionOptions.SELF_ATTENTION
                                    ):
                                        query_key_projection_dim = embeddimd_dim
                                        value_projection_dim = embeddimd_dim
                                        source_sequence_length = target_sequence_length

                                    c = MultiHeadAttentionPresets.multi_head_attention_preset(
                                        batch_size=batch_size,
                                        embedding_dim=embeddimd_dim,
                                        target_sequence_length=target_sequence_length,
                                        source_sequence_length=source_sequence_length,
                                        attention_option=attention_option,
                                        query_key_projection_dim=query_key_projection_dim,
                                        value_projection_dim=value_projection_dim,
                                        add_key_value_bias_flag=add_key_value_bias_flag,
                                    )

                                    m = MultiHeadAttention(c)

                                    query, key, value = create_qkv_tensors(
                                        target_sequence_length,
                                        source_sequence_length,
                                        batch_size,
                                        m.embedding_dim,
                                        attention_option,
                                    )

                                    key_padding_mask = create_key_padding_mask(
                                        batch_size, source_sequence_length
                                    )

                                    attention_mask_repeat = batch_size * m.num_heads
                                    if (
                                        attention_option
                                        == AttentionOptions.MIXTURE_OF_ATTENTION_HEADS
                                    ):
                                        attention_mask_repeat = (
                                            batch_size
                                            * m.num_heads
                                            * c.experts_config.top_k
                                        )

                                    attention_mask = create_attention_mask(
                                        target_sequence_length,
                                        source_sequence_length,
                                        attention_mask_repeat,
                                    )
                                    static_key = None
                                    static_values = None

                                    attention_output, attention_weights = m.forward(
                                        query,
                                        key,
                                        value,
                                        key_padding_mask,
                                        attention_mask,
                                        static_key,
                                        static_values,
                                    )

                                    expected_shape = (
                                        target_sequence_length,
                                        batch_size,
                                        m.embedding_dim,
                                    )
                                    self.assertIsInstance(
                                        attention_output, torch.Tensor
                                    )
                                    self.assertIsNone(attention_weights)
                                    self.assertEqual(
                                        attention_output.shape, expected_shape
                                    )

    def test__average_attention_weights_flag(self):
        batch_size = 4
        sequence_lengths = [8, 10]
        embeddimd_dim = 12
        qkv_dimensions = [0, 16, 20]
        bool_options = [True, False]

        for target_sequence_length in sequence_lengths:
            for source_sequence_length in sequence_lengths:
                for query_key_projection_dim in qkv_dimensions:
                    for value_projection_dim in qkv_dimensions:
                        for attention_option in AttentionOptions:
                            for average_attention_weights_flag in bool_options:
                                message = f"Test failed for the inputs: attention_option={attention_option}, query_key_projection_dim={query_key_projection_dim}, value_projection_dim={value_projection_dim}, average_attention_weights_flag={average_attention_weights_flag}"
                                with self.subTest(i=message):
                                    if (
                                        attention_option
                                        == AttentionOptions.SELF_ATTENTION
                                    ):
                                        query_key_projection_dim = embeddimd_dim
                                        value_projection_dim = embeddimd_dim
                                        source_sequence_length = target_sequence_length

                                    c = MultiHeadAttentionPresets.multi_head_attention_preset(
                                        batch_size=batch_size,
                                        embedding_dim=embeddimd_dim,
                                        target_sequence_length=target_sequence_length,
                                        source_sequence_length=source_sequence_length,
                                        attention_option=attention_option,
                                        query_key_projection_dim=query_key_projection_dim,
                                        value_projection_dim=value_projection_dim,
                                        return_attention_weights_flag=True,
                                        average_attention_weights_flag=average_attention_weights_flag,
                                    )

                                    m = MultiHeadAttention(c)

                                    query, key, value = create_qkv_tensors(
                                        target_sequence_length,
                                        source_sequence_length,
                                        batch_size,
                                        m.embedding_dim,
                                        attention_option,
                                    )

                                    key_padding_mask = create_key_padding_mask(
                                        batch_size, source_sequence_length
                                    )

                                    attention_mask_repeat = batch_size * m.num_heads
                                    if (
                                        attention_option
                                        == AttentionOptions.MIXTURE_OF_ATTENTION_HEADS
                                    ):
                                        attention_mask_repeat = (
                                            batch_size
                                            * m.num_heads
                                            * c.experts_config.top_k
                                        )

                                    attention_mask = create_attention_mask(
                                        target_sequence_length,
                                        source_sequence_length,
                                        attention_mask_repeat,
                                    )
                                    static_key = None
                                    static_values = None

                                    if (
                                        attention_option
                                        != AttentionOptions.SELF_ATTENTION
                                    ):
                                        with self.assertRaises(RuntimeError):
                                            m.forward(
                                                query,
                                                key,
                                                value,
                                                key_padding_mask,
                                                attention_mask,
                                                static_key,
                                                static_values,
                                            )
                                        continue

                                    attention_output, attention_weights = m.forward(
                                        query,
                                        key,
                                        value,
                                        key_padding_mask,
                                        attention_mask,
                                        static_key,
                                        static_values,
                                    )

                                    expected_shape = (
                                        target_sequence_length,
                                        batch_size,
                                        m.embedding_dim,
                                    )
                                    self.assertIsInstance(
                                        attention_output, torch.Tensor
                                    )
                                    self.assertIsInstance(
                                        attention_weights, torch.Tensor
                                    )
                                    self.assertEqual(
                                        attention_output.shape, expected_shape
                                    )
                                    if average_attention_weights_flag:
                                        self.assertEqual(attention_weights.dim(), 3)
                                    else:
                                        self.assertEqual(attention_weights.dim(), 4)

    def test__add_key_value_bias_and_zero_attention_flags(self):
        batch_size = 4
        sequence_lengths = [8, 10]
        embeddimd_dim = 12
        qkv_dimensions = [0, 16, 20]
        bool_options = [True, False]

        for target_sequence_length in sequence_lengths:
            for source_sequence_length in sequence_lengths:
                for query_key_projection_dim in qkv_dimensions:
                    for value_projection_dim in qkv_dimensions:
                        for attention_option in AttentionOptions:
                            for add_key_value_bias_flag in bool_options:
                                for zero_attention_flag in bool_options:
                                    message = f"Test failed for the inputs: attention_option={attention_option}, query_key_projection_dim={query_key_projection_dim}, value_projection_dim={value_projection_dim}, add_key_value_bias_flag={add_key_value_bias_flag}, zero_attention_flag={zero_attention_flag}"
                                    with self.subTest(i=message):
                                        if (
                                            attention_option
                                            == AttentionOptions.SELF_ATTENTION
                                        ):
                                            query_key_projection_dim = embeddimd_dim
                                            value_projection_dim = embeddimd_dim
                                            source_sequence_length = (
                                                target_sequence_length
                                            )

                                        c = MultiHeadAttentionPresets.multi_head_attention_preset(
                                            batch_size=batch_size,
                                            embedding_dim=embeddimd_dim,
                                            target_sequence_length=target_sequence_length,
                                            source_sequence_length=source_sequence_length,
                                            attention_option=attention_option,
                                            query_key_projection_dim=query_key_projection_dim,
                                            value_projection_dim=value_projection_dim,
                                            add_key_value_bias_flag=add_key_value_bias_flag,
                                            zero_attention_flag=zero_attention_flag,
                                        )

                                        m = MultiHeadAttention(c)

                                        query, key, value = create_qkv_tensors(
                                            target_sequence_length,
                                            source_sequence_length,
                                            batch_size,
                                            m.embedding_dim,
                                            attention_option,
                                        )

                                        key_padding_mask = create_key_padding_mask(
                                            batch_size, source_sequence_length
                                        )

                                        attention_mask_repeat = batch_size * m.num_heads
                                        if (
                                            attention_option
                                            == AttentionOptions.MIXTURE_OF_ATTENTION_HEADS
                                        ):
                                            attention_mask_repeat = (
                                                batch_size
                                                * m.num_heads
                                                * c.experts_config.top_k
                                            )

                                        attention_mask = create_attention_mask(
                                            target_sequence_length,
                                            source_sequence_length,
                                            attention_mask_repeat,
                                        )
                                        static_key = None
                                        static_values = None

                                        attention_output, attention_weights = m.forward(
                                            query,
                                            key,
                                            value,
                                            key_padding_mask,
                                            attention_mask,
                                            static_key,
                                            static_values,
                                        )

                                        expected_shape = (
                                            target_sequence_length,
                                            batch_size,
                                            m.embedding_dim,
                                        )
                                        self.assertIsInstance(
                                            attention_output, torch.Tensor
                                        )
                                        self.assertIsNone(attention_weights)
                                        self.assertEqual(
                                            attention_output.shape, expected_shape
                                        )

    def test__self_attention_possible_flag_combinations(self):
        batch_size = 4
        sequence_lengths = [8, 10]
        embeddimd_dim = 12
        qkv_dimensions = [0, 16, 20]
        bool_options = [True, False]

        for target_sequence_length in sequence_lengths:
            for source_sequence_length in sequence_lengths:
                for query_key_projection_dim in qkv_dimensions:
                    for value_projection_dim in qkv_dimensions:
                        for add_key_value_bias_flag in bool_options:
                            for zero_attention_flag in bool_options:
                                for average_attention_weights_flag in bool_options:
                                    for causal_attention_mask_flag in bool_options:
                                        for (
                                            return_attention_weights_flag
                                        ) in bool_options:
                                            message = f"Test failed for the inputs: query_key_projection_dim={query_key_projection_dim}, value_projection_dim={value_projection_dim}, add_key_value_bias_flag={add_key_value_bias_flag}, zero_attention_flag={zero_attention_flag}, average_attention_weights_flag={average_attention_weights_flag}, causal_attention_mask_flag={causal_attention_mask_flag}"
                                            with self.subTest(i=message):
                                                query_key_projection_dim = embeddimd_dim
                                                value_projection_dim = embeddimd_dim
                                                source_sequence_length = (
                                                    target_sequence_length
                                                )

                                                c = MultiHeadAttentionPresets.multi_head_attention_preset(
                                                    batch_size=batch_size,
                                                    embedding_dim=embeddimd_dim,
                                                    target_sequence_length=target_sequence_length,
                                                    source_sequence_length=source_sequence_length,
                                                    attention_option=AttentionOptions.SELF_ATTENTION,
                                                    query_key_projection_dim=query_key_projection_dim,
                                                    value_projection_dim=value_projection_dim,
                                                    add_key_value_bias_flag=add_key_value_bias_flag,
                                                    zero_attention_flag=zero_attention_flag,
                                                    average_attention_weights_flag=average_attention_weights_flag,
                                                    causal_attention_mask_flag=causal_attention_mask_flag,
                                                    return_attention_weights_flag=return_attention_weights_flag,
                                                )

                                                m = MultiHeadAttention(c)

                                                query, key, value = create_qkv_tensors(
                                                    target_sequence_length,
                                                    source_sequence_length,
                                                    batch_size,
                                                    m.embedding_dim,
                                                    AttentionOptions.SELF_ATTENTION,
                                                )

                                                key_padding_mask = (
                                                    create_key_padding_mask(
                                                        batch_size,
                                                        source_sequence_length,
                                                    )
                                                )

                                                attention_mask_repeat = (
                                                    batch_size * m.num_heads
                                                )

                                                attention_mask = create_attention_mask(
                                                    target_sequence_length,
                                                    source_sequence_length,
                                                    attention_mask_repeat,
                                                )
                                                static_key = None
                                                static_values = None

                                                attention_output, attention_weights = (
                                                    m.forward(
                                                        query,
                                                        key,
                                                        value,
                                                        key_padding_mask,
                                                        attention_mask,
                                                        static_key,
                                                        static_values,
                                                    )
                                                )

                                                expected_shape = (
                                                    target_sequence_length,
                                                    batch_size,
                                                    m.embedding_dim,
                                                )
                                                self.assertIsInstance(
                                                    attention_output, torch.Tensor
                                                )
                                                self.assertEqual(
                                                    attention_output.shape,
                                                    expected_shape,
                                                )
                                                if return_attention_weights_flag:
                                                    self.assertIsInstance(
                                                        attention_weights, torch.Tensor
                                                    )
                                                    if average_attention_weights_flag:
                                                        self.assertEqual(
                                                            attention_weights.dim(), 3
                                                        )
                                                    else:
                                                        self.assertEqual(
                                                            attention_weights.dim(), 4
                                                        )
                                                else:
                                                    self.assertIsNone(attention_weights)

    def test__independent_possible_flag_combinations(self):
        batch_size = 4
        sequence_lengths = [8, 10]
        embeddimd_dim = 12
        qkv_dimensions = [0, 16, 20]
        bool_options = [True, False]

        for target_sequence_length in sequence_lengths:
            for source_sequence_length in sequence_lengths:
                for query_key_projection_dim in qkv_dimensions:
                    for value_projection_dim in qkv_dimensions:
                        for add_key_value_bias_flag in bool_options:
                            for zero_attention_flag in bool_options:
                                for average_attention_weights_flag in bool_options:
                                    for causal_attention_mask_flag in bool_options:
                                        for (
                                            return_attention_weights_flag
                                        ) in bool_options:
                                            message = f"Test failed for the inputs: query_key_projection_dim={query_key_projection_dim}, value_projection_dim={value_projection_dim}, add_key_value_bias_flag={add_key_value_bias_flag}, zero_attention_flag={zero_attention_flag}, average_attention_weights_flag={average_attention_weights_flag}, causal_attention_mask_flag={causal_attention_mask_flag}"
                                            with self.subTest(i=message):
                                                query_key_projection_dim = embeddimd_dim
                                                value_projection_dim = embeddimd_dim

                                                c = MultiHeadAttentionPresets.multi_head_attention_preset(
                                                    batch_size=batch_size,
                                                    embedding_dim=embeddimd_dim,
                                                    target_sequence_length=target_sequence_length,
                                                    source_sequence_length=source_sequence_length,
                                                    attention_option=AttentionOptions.INDEPENDENT,
                                                    query_key_projection_dim=query_key_projection_dim,
                                                    value_projection_dim=value_projection_dim,
                                                    add_key_value_bias_flag=add_key_value_bias_flag,
                                                    zero_attention_flag=zero_attention_flag,
                                                    average_attention_weights_flag=average_attention_weights_flag,
                                                    causal_attention_mask_flag=causal_attention_mask_flag,
                                                    return_attention_weights_flag=return_attention_weights_flag,
                                                )

                                                m = MultiHeadAttention(c)

                                                query, key, value = create_qkv_tensors(
                                                    target_sequence_length,
                                                    source_sequence_length,
                                                    batch_size,
                                                    m.embedding_dim,
                                                    AttentionOptions.INDEPENDENT,
                                                )

                                                key_padding_mask = (
                                                    create_key_padding_mask(
                                                        batch_size,
                                                        source_sequence_length,
                                                    )
                                                )

                                                attention_mask_repeat = (
                                                    batch_size * m.num_heads
                                                )

                                                attention_mask = create_attention_mask(
                                                    target_sequence_length,
                                                    source_sequence_length,
                                                    attention_mask_repeat,
                                                )
                                                static_key = None
                                                static_values = None

                                                attention_output, attention_weights = (
                                                    m.forward(
                                                        query,
                                                        key,
                                                        value,
                                                        key_padding_mask,
                                                        attention_mask,
                                                        static_key,
                                                        static_values,
                                                    )
                                                )

                                                expected_shape = (
                                                    target_sequence_length,
                                                    batch_size,
                                                    m.embedding_dim,
                                                )
                                                self.assertIsInstance(
                                                    attention_output, torch.Tensor
                                                )
                                                self.assertEqual(
                                                    attention_output.shape,
                                                    expected_shape,
                                                )
                                                self.assertIsNone(attention_weights)

    # def test__all_layer_types_and_flags_and_different_batch_sizes(self):
    #     flags = [
    #         "add_key_value_bias_flag",
    #         "zero_attention_flag",
    #         # "return_attention_weights_flag",
    #         "average_attention_weights_flag",
    #         "causal_attention_mask_flag",
    #         # "attention_option",
    #     ]
    #     combinations = list(itertools.product([False, True], repeat=len(flags)))
    #     tests = [dict(zip(flags, combo)) for combo in combinations]
    #     batch_sizes = [2, 4, 8]  # Pass
    #
    #     attention_options = [LinearLayerStackOptions, AdaptiveLayerStackOptions]
    #
    #     for attention_option in attention_options:
    #         for model_type in attention_option:
    #             for test in tests:
    #                 for batch_size in batch_sizes:
    #                     message = f"Failed for model_type: {model_type}, batch_size: {batch_size}, test: {test}"
    #                     with self.subTest(msg=message):
    #                         c = MultiHeadAttentionPresets.multi_head_attention_preset(
    #                             model_type=model_type,
    #                             batch_size=batch_size,
    #                             embedding_dim=12,
    #                             query_key_projection_dim=12,
    #                             value_projection_dim=12,
    #                             target_sequence_length=24,
    #                             source_sequence_length=24,
    #                             attention_option=AttentionOptions.INDEPENDENT,
    #                             **test,
    #                         )
    #                         m = MultiHeadAttention(c)
    #                         query = torch.randn(
    #                             c.target_sequence_length,
    #                             c.batch_size,
    #                             c.embedding_dim,
    #                         )
    #                         key = torch.randn(
    #                             c.target_sequence_length,
    #                             c.batch_size,
    #                             c.embedding_dim,
    #                         )
    #                         value = torch.randn(
    #                             c.target_sequence_length,
    #                             c.batch_size,
    #                             c.embedding_dim,
    #                         )
    #                         key_padding_mask = torch.randn(
    #                             c.batch_size, c.source_sequence_length
    #                         )
    #                         attention_mask = torch.randn(
    #                             1,
    #                             c.target_sequence_length,
    #                             c.source_sequence_length,
    #                         )
    #                         attention_mask = torch.where(
    #                             attention_mask > 0,
    #                             torch.tensor(float("-inf")),
    #                             torch.tensor(0.0),
    #                         )
    #                         attention_mask = attention_mask.repeat(
    #                             c.batch_size * c.num_heads, 1, 1
    #                         )
    #                         static_key = None
    #                         static_value = None
    #
    #                         attention_output, attention_weights = m.forward(
    #                             query,
    #                             key,
    #                             value,
    #                             key_padding_mask,
    #                             attention_mask,
    #                             static_key,
    #                             static_value,
    #                         )
    #                         self.assertIsInstance(attention_output, torch.Tensor)
    #
    # # FIX: In the case of this test embedding dimension is only updated in the
    # # `MultiHeadAttentionConfig` and but not the `ModelConfig`.
    # #
    # # def test__all_layer_types_and_flags_and_different_embedding_dims(self):
    # #     ...
    #
    # # BUG: `query_key_projection_dim` and `value_projection_dim` not working if not equel to `embedding_dim`. Fix this later!
    # #
    # # def test__all_layer_types_and_flags_and_different_query_key_projection_dim(self):
    # #     ...
    #
    # # BUG: `target_sequence_length` and `source_sequence_length` not working if not equal. Fix this later!
    # #
    # # def test__all_layer_types_and_flags_and_different_target_sequence_length(self):
    # #     ...
    #
    # def test_forward(self):
    #     attention_options = [
    #         AttentionOptions.SELF_ATTENTION,
    #         AttentionOptions.INDEPENDENT,
    #     ]
    #     model_types = list(LinearLayerStackOptions) + list(AdaptiveLayerStackOptions)
    #     batch_sizes = [2, 4]
    #     num_heads_options = [2, 4]
    #
    #     for model_type in model_types:
    #         for attention_option in attention_options:
    #             for batch_size in batch_sizes:
    #                 for num_heads in num_heads_options:
    #                     for key_value_bias_flag in [True, False]:
    #                         for zero_attention_flag in [True, False]:
    #                             for causal_attention_mask_flag in [True, False]:
    #                                 for add_key_value_bias_flag in [True, False]:
    #                                     for return_attention_weights_flag in [
    #                                         True,
    #                                         False,
    #                                     ]:
    #                                         for average_attention_weights_flag in [
    #                                             True,
    #                                             False,
    #                                         ]:
    #                                             message = f"Test failed for options - model_type: {model_type}, attention_option: {attention_option}, batch_size: {batch_size}, num_heads: {num_heads}, key_value_bias_flag: {key_value_bias_flag}, zero_attention_flag: {zero_attention_flag}, causal_attention_mask_flag: {causal_attention_mask_flag}, add_key_value_bias_flag: {add_key_value_bias_flag}, return_attention_weights_flag: {return_attention_weights_flag}, average_attention_weights_flag: {average_attention_weights_flag}"
    #                                             with self.subTest(message=message):
    #                                                 c = MultiHeadAttentionPresets.multi_head_attention_preset(
    #                                                     model_type=model_type,
    #                                                     batch_size=batch_size,
    #                                                     num_heads=num_heads,
    #                                                     embedding_dim=12,
    #                                                     query_key_projection_dim=12,
    #                                                     value_projection_dim=12,
    #                                                     target_sequence_length=24,
    #                                                     source_sequence_length=24,
    #                                                     attention_option=attention_option,
    #                                                     key_value_bias_flag=key_value_bias_flag,
    #                                                     zero_attention_flag=zero_attention_flag,
    #                                                     causal_attention_mask_flag=causal_attention_mask_flag,
    #                                                     add_key_value_bias_flag=add_key_value_bias_flag,
    #                                                     return_attention_weights_flag=return_attention_weights_flag,
    #                                                     average_attention_weights_flag=average_attention_weights_flag,
    #                                                 )
    #                                                 m = MultiHeadAttention(c)
    #
    #                                                 if (
    #                                                     attention_option
    #                                                     == AttentionOptions.SELF_ATTENTION
    #                                                 ):
    #                                                     query = key = value = (
    #                                                         torch.randn(
    #                                                             c.target_sequence_length,
    #                                                             batch_size,
    #                                                             c.embedding_dim,
    #                                                         )
    #                                                     )
    #                                                 else:
    #                                                     query = torch.randn(
    #                                                         c.target_sequence_length,
    #                                                         batch_size,
    #                                                         c.embedding_dim,
    #                                                     )
    #                                                     key = torch.randn(
    #                                                         c.source_sequence_length,
    #                                                         batch_size,
    #                                                         c.embedding_dim,
    #                                                     )
    #                                                     value = torch.randn(
    #                                                         c.source_sequence_length,
    #                                                         batch_size,
    #                                                         c.embedding_dim,
    #                                                     )
    #
    #                                                 key_padding_mask = torch.randn(
    #                                                     batch_size,
    #                                                     c.source_sequence_length,
    #                                                 )
    #                                                 attention_mask = torch.randn(
    #                                                     1,
    #                                                     c.target_sequence_length,
    #                                                     c.source_sequence_length,
    #                                                 )
    #                                                 attention_mask = torch.where(
    #                                                     attention_mask > 0,
    #                                                     torch.tensor(float("-inf")),
    #                                                     torch.tensor(0.0),
    #                                                 )
    #                                                 attention_mask = (
    #                                                     attention_mask.repeat(
    #                                                         batch_size * num_heads,
    #                                                         1,
    #                                                         1,
    #                                                     )
    #                                                 )
    #
    #                                                 (
    #                                                     attention_output,
    #                                                     attention_weights,
    #                                                 ) = m.forward(
    #                                                     query,
    #                                                     key,
    #                                                     value,
    #                                                     key_padding_mask,
    #                                                     attention_mask,
    #                                                     None,
    #                                                     None,
    #                                                 )
    #
    #                                                 self.assertIsInstance(
    #                                                     attention_output, torch.Tensor
    #                                                 )
    #                                                 self.assertEqual(
    #                                                     attention_output.shape,
    #                                                     (
    #                                                         c.target_sequence_length,
    #                                                         batch_size,
    #                                                         c.embedding_dim,
    #                                                     ),
    #                                                 )
    #                                                 if return_attention_weights_flag:
    #                                                     self.assertIsInstance(
    #                                                         attention_weights,
    #                                                         torch.Tensor,
    #                                                     )
    #                                                     if average_attention_weights_flag:
    #                                                         self.assertEqual(
    #                                                             attention_weights.dim(),
    #                                                             3,
    #                                                         )
    #                                                     else:
    #                                                         self.assertEqual(
    #                                                             attention_weights.dim(),
    #                                                             4,
    #                                                         )
    #                                                 else:
    #                                                     self.assertIsNone(
    #                                                         attention_weights
    #                                                     )
    #
    # # ------------------------------------------------------------------ #
    # #  gradient tests
    # # ------------------------------------------------------------------ #
    #
    # def test_gradients_flow_through_multi_head_attention(self):
    #     attention_options = [
    #         AttentionOptions.SELF_ATTENTION,
    #         AttentionOptions.INDEPENDENT,
    #     ]
    #     model_types = list(LinearLayerStackOptions) + list(AdaptiveLayerStackOptions)
    #
    #     for model_type in model_types:
    #         for attention_option in attention_options:
    #             for key_value_bias_flag in [True, False]:
    #                 for zero_attention_flag in [True, False]:
    #                     for causal_attention_mask_flag in [True, False]:
    #                         for add_key_value_bias_flag in [True, False]:
    #                             message = f"Test failed for options - model_type: {model_type}, attention_option: {attention_option}, key_value_bias_flag: {key_value_bias_flag}, zero_attention_flag: {zero_attention_flag}, causal_attention_mask_flag: {causal_attention_mask_flag}, add_key_value_bias_flag: {add_key_value_bias_flag}"
    #                             with self.subTest(message=message):
    #                                 c = MultiHeadAttentionPresets.multi_head_attention_preset(
    #                                     model_type=model_type,
    #                                     embedding_dim=12,
    #                                     query_key_projection_dim=12,
    #                                     value_projection_dim=12,
    #                                     target_sequence_length=16,
    #                                     source_sequence_length=16,
    #                                     attention_option=attention_option,
    #                                     key_value_bias_flag=key_value_bias_flag,
    #                                     zero_attention_flag=zero_attention_flag,
    #                                     causal_attention_mask_flag=causal_attention_mask_flag,
    #                                     add_key_value_bias_flag=add_key_value_bias_flag,
    #                                 )
    #                                 m = MultiHeadAttention(c)
    #
    #                                 if (
    #                                     attention_option
    #                                     == AttentionOptions.SELF_ATTENTION
    #                                 ):
    #                                     query = key = value = torch.randn(
    #                                         c.target_sequence_length,
    #                                         c.batch_size,
    #                                         c.embedding_dim,
    #                                         requires_grad=True,
    #                                     )
    #                                 else:
    #                                     query = torch.randn(
    #                                         c.target_sequence_length,
    #                                         c.batch_size,
    #                                         c.embedding_dim,
    #                                         requires_grad=True,
    #                                     )
    #                                     key = torch.randn(
    #                                         c.source_sequence_length,
    #                                         c.batch_size,
    #                                         c.embedding_dim,
    #                                         requires_grad=True,
    #                                     )
    #                                     value = torch.randn(
    #                                         c.source_sequence_length,
    #                                         c.batch_size,
    #                                         c.embedding_dim,
    #                                         requires_grad=True,
    #                                     )
    #
    #                                 attention_output, _ = m.forward(
    #                                     query,
    #                                     key,
    #                                     value,
    #                                     None,
    #                                     None,
    #                                     None,
    #                                     None,
    #                                 )
    #                                 attention_output.sum().backward()
    #
    #                                 grads = [
    #                                     p.grad
    #                                     for p in m.parameters()
    #                                     if p.requires_grad
    #                                 ]
    #                                 non_none_grads = [g for g in grads if g is not None]
    #                                 self.assertTrue(len(non_none_grads) > 0)
