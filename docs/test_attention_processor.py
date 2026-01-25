from dataclasses import asdict
import unittest
import torch

from Emperor.attention.utils.enums import AttentionOptions
from Emperor.linears.options import LinearLayerStackOptions
from Emperor.adaptive.options import AdaptiveLayerStackOptions
from Emperor.attention.utils.layer import MultiHeadAttentionConfig
from Emperor.attention.utils.presets import MultiHeadAttentionPresets
from Emperor.attention.utils.handlers.projector import (
    IndependentProjector,
    ProjectorBuilder,
)
from Emperor.attention.utils._validator import MultiHeadAttentionValidator
from Emperor.attention.utils.handlers.processor import (
    IndependentProcessor,
    MixtureOfAttentionHeadsProcessor,
    ProcessorBuilder,
    SelfAttentionProcessor,
)


class TestIndependentProjector(unittest.TestCase):
    def setUp(self):
        self.cfg = MultiHeadAttentionPresets.multi_head_attention_preset()

    def test_init(self):
        attention_options = [LinearLayerStackOptions, AdaptiveLayerStackOptions]

        for attention_option in attention_options:
            for model_type in attention_option:
                message = f"Testing configuration: model_type: {model_type}"
                with self.subTest(i=message):
                    c = MultiHeadAttentionPresets.multi_head_attention_preset(
                        attention_option=AttentionOptions.INDEPENDENT,
                        model_type=model_type,
                    )
                    projector = IndependentProjector(c)
                    m = IndependentProcessor(c, projector)
                    self.assertIsInstance(m, IndependentProcessor)
                    self.assertIsInstance(m.projector, IndependentProjector)

    def test__prepare_attnetion_mask(self):
        c = MultiHeadAttentionPresets.multi_head_attention_preset(
            attention_option=AttentionOptions.INDEPENDENT,
        )
        rand_tensor = torch.randn(1, c.target_sequence_length, c.source_sequence_length)
        attention_mask = torch.where(
            rand_tensor > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
        )
        unbatched_attention_mask = attention_mask
        batched_attention_mask = attention_mask.repeat(c.batch_size * c.num_heads, 1, 1)

        attention_mask_options = {
            "none": None,
            "batched": batched_attention_mask,
            "unbatched": unbatched_attention_mask,
        }

        for mask_name, mask_option in attention_mask_options.items():
            message = f"Testing configuration: mask_option: {mask_name}"
            with self.subTest(i=message):
                projector = IndependentProjector(c)
                m = IndependentProcessor(c, projector)
                output_attention_mask = m._IndependentProcessor__prepare_attnetion_mask(
                    mask_option
                )

                if mask_name == "none":
                    self.assertIsNone(output_attention_mask)
                elif mask_name == "batched":
                    expected_shape = (
                        m.batch_size,
                        m.num_heads,
                        m.target_sequence_length,
                        m.source_sequence_length,
                    )
                    self.assertEqual(output_attention_mask.shape, expected_shape)
                elif mask_name == "unbatched":
                    expected_shape = (
                        1,
                        1,
                        m.target_sequence_length,
                        m.source_sequence_length,
                    )
                    self.assertEqual(output_attention_mask.shape, expected_shape)

    def test__compute_weighted_values(self):
        c = MultiHeadAttentionPresets.multi_head_attention_preset(
            attention_option=AttentionOptions.INDEPENDENT,
            query_key_projection_dim=0,
            value_projection_dim=0,
            source_sequence_length=16,
            target_sequence_length=16,
        )
        head_dim = c.embedding_dim // c.num_heads
        query = torch.randn(
            c.batch_size, c.num_heads, c.target_sequence_length, head_dim
        )
        key = value = torch.randn(
            c.batch_size, c.num_heads, c.source_sequence_length, head_dim
        )

        attention_mask_raw = torch.randn(
            c.batch_size,
            c.num_heads,
            c.target_sequence_length,
            c.source_sequence_length,
        )
        attention_mask = torch.where(
            attention_mask_raw > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
        )
        attention_mask_options = [
            None,
            attention_mask,
        ]

        boolean_options = [True, False]

        for causal_attention_mask_flag in boolean_options:
            for attention_mask in attention_mask_options:
                mask_name = "none" if attention_mask is None else "batched"
                message = f"Testing configuration: causal_attention_mask_flag: {causal_attention_mask_flag}, mask_option: {mask_name}"
                with self.subTest(i=message):
                    c.causal_attention_mask_flag = causal_attention_mask_flag
                    projector = IndependentProjector(c)
                    m = IndependentProcessor(c, projector)

                    weighted_values = m._IndependentProcessor__compute_weighted_values(
                        query, key, value, attention_mask
                    )

                    expected_shape = (
                        m.batch_size * m.target_sequence_length,
                        m.embedding_dim,
                    )
                    self.assertIsInstance(weighted_values, torch.Tensor)
                    self.assertEqual(weighted_values.shape, expected_shape)

    def test_compute_attention(self):
        c = MultiHeadAttentionPresets.multi_head_attention_preset(
            attention_option=AttentionOptions.INDEPENDENT,
            embedding_dim=12,
            query_key_projection_dim=0,
            value_projection_dim=0,
            source_sequence_length=16,
            target_sequence_length=16,
        )
        head_dim = c.embedding_dim // c.num_heads
        query = torch.randn(
            c.batch_size * c.num_heads, c.source_sequence_length, head_dim
        )
        key = value = torch.randn(
            c.batch_size * c.num_heads, c.target_sequence_length, head_dim
        )

        attention_mask_raw = torch.randn(
            1,
            c.target_sequence_length,
            c.source_sequence_length,
        )
        attention_mask = torch.where(
            attention_mask_raw > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
        )
        attention_mask = attention_mask.repeat(c.batch_size * c.num_heads, 1, 1)
        attention_mask_options = [
            None,
            attention_mask,
        ]

        boolean_options = [True, False]

        for causal_attention_mask_flag in boolean_options:
            for attention_mask in attention_mask_options:
                mask_name = "none" if attention_mask is None else "batched"
                message = f"Testing configuration: causal_attention_mask_flag: {causal_attention_mask_flag}, mask_option: {mask_name}"
                with self.subTest(i=message):
                    c.causal_attention_mask_flag = causal_attention_mask_flag
                    projector = IndependentProjector(c)
                    m = IndependentProcessor(c, projector)

                    output_attention_output, _ = m.compute_attention(
                        query, key, value, attention_mask
                    )

                    expected_shape = (
                        m.target_sequence_length,
                        m.batch_size,
                        m.embedding_dim,
                    )

                    self.assertIsInstance(output_attention_output, torch.Tensor)
                    self.assertIsNone(_)
                    self.assertEqual(output_attention_output.shape, expected_shape)


# class TestProcessor(unittest.TestCase):
#     def setUp(self):
#         self.rebuild_presets()
#
#     def tearDown(self):
#         self.cfg = None
#         self.config = None
#         self.model = None
#         self.batch_size = None
#         self.embedding_dim = None
#         self.target_sequence_length = None
#         self.num_heads = None
#         self.head_dim = None
#
#     def rebuild_presets(self, config: MultiHeadAttentionConfig | None = None):
#         self.config = MultiHeadAttentionPresets.multi_head_attention_preset(
#             embedding_dim=12,
#             query_key_projection_dim=12,
#             value_projection_dim=12,
#         )
#         if config is not None:
#             for k in asdict(config):
#                 if hasattr(self.config, k) and getattr(config, k) is not None:
#                     setattr(self.config, k, getattr(config, k))
#
#         validator = MultiHeadAttentionValidator(self.config)
#         self.model = ProcessorBuilder(self.config, validator, projector)
#
#         self.batch_size = self.config.batch_size
#         self.embedding_dim = self.config.embedding_dim
#         self.target_sequence_length = self.config.target_sequence_length
#         self.source_sequence_length = self.config.source_sequence_length
#         self.num_heads = self.config.num_heads
#         self.head_dim = self.embedding_dim // self.num_heads
#
#
# class Test__init(TestProcessor):
#     def test__init(self):
#         self.assertIsInstance(self.model, Processor)
#         self.assertIsInstance(
#             self.model.processor,
#             (ProcessorDefault, ProcessorWithReturnedWeights),
#         )
#
#
# class Test____create_processor(TestProcessor):
#     def test__return_attention_weights_flag__False(self):
#         config = MultiHeadAttentionConfig(
#             return_attention_weights_flag=False,
#         )
#         self.rebuild_presets(config)
#         self.assertIsInstance(self.model.processor, ProcessorDefault)
#
#     def test__return_attention_weights_flag__True(self):
#         config = MultiHeadAttentionConfig(
#             return_attention_weights_flag=True,
#         )
#         self.rebuild_presets(config)
#         self.assertIsInstance(self.model.processor, ProcessorWithReturnedWeights)
#
#
# class Test____compute_weighted_values_default(TestProcessor):
#     def test__return_attention_weights_flag__True(self):
#         config = MultiHeadAttentionConfig(
#             source_sequence_length=32,
#             target_sequence_length=32,
#             return_attention_weights_flag=True,
#         )
#         self.rebuild_presets(config)
#
#         query = torch.randn(
#             self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
#         )
#         key = torch.randn(
#             self.batch_size * self.num_heads, self.target_sequence_length, self.head_dim
#         )
#         value = torch.randn(
#             self.batch_size * self.num_heads, self.target_sequence_length, self.head_dim
#         )
#         attention_mask = torch.randn(
#             1, self.target_sequence_length, self.source_sequence_length
#         )
#         attention_mask = torch.where(
#             attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
#         )
#         attention_mask = attention_mask.repeat(self.batch_size * self.num_heads, 1, 1)
#
#         output_attention_output, output_attention_weights = (
#             self.model.compute_attention(query, key, value, attention_mask)
#         )
#
#         self.assertIsInstance(output_attention_output, torch.Tensor)
#         self.assertIsInstance(output_attention_weights, torch.Tensor)
#         self.assertEqual(
#             output_attention_output.shape,
#             (self.target_sequence_length, self.batch_size, self.embedding_dim),
#         )
#         self.assertEqual(
#             output_attention_weights.shape,
#             (
#                 self.batch_size,
#                 self.num_heads,
#                 self.target_sequence_length,
#                 self.source_sequence_length,
#             ),
#         )
#
#     def test__return_attention_weights_flag__True__average_attention_weights_flag__True(
#         self,
#     ):
#         config = MultiHeadAttentionConfig(
#             source_sequence_length=32,
#             target_sequence_length=32,
#             return_attention_weights_flag=True,
#             average_attention_weights_flag=True,
#         )
#         self.rebuild_presets(config)
#
#         query = torch.randn(
#             self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
#         )
#         key = torch.randn(
#             self.batch_size * self.num_heads, self.target_sequence_length, self.head_dim
#         )
#         value = torch.randn(
#             self.batch_size * self.num_heads, self.target_sequence_length, self.head_dim
#         )
#         attention_mask = torch.randn(
#             1, self.target_sequence_length, self.source_sequence_length
#         )
#         attention_mask = torch.where(
#             attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
#         )
#         attention_mask = attention_mask.repeat(self.batch_size * self.num_heads, 1, 1)
#
#         output_attention_output, output_attention_weights = (
#             self.model.compute_attention(query, key, value, attention_mask)
#         )
#
#         self.assertIsInstance(output_attention_output, torch.Tensor)
#         self.assertIsInstance(output_attention_weights, torch.Tensor)
#         self.assertEqual(
#             output_attention_output.shape,
#             (self.target_sequence_length, self.batch_size, self.embedding_dim),
#         )
#         self.assertEqual(
#             output_attention_weights.shape,
#             (
#                 self.batch_size,
#                 self.target_sequence_length,
#                 self.source_sequence_length,
#             ),
#         )
#
#     def test__return_attention_weights_flag__False(self):
#         config = MultiHeadAttentionConfig(
#             source_sequence_length=32,
#             target_sequence_length=32,
#             return_attention_weights_flag=False,
#         )
#         self.rebuild_presets(config)
#
#         query = torch.randn(
#             self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
#         )
#         key = torch.randn(
#             self.batch_size * self.num_heads, self.target_sequence_length, self.head_dim
#         )
#         value = torch.randn(
#             self.batch_size * self.num_heads, self.target_sequence_length, self.head_dim
#         )
#         attention_mask = torch.randn(
#             1, self.target_sequence_length, self.source_sequence_length
#         )
#         attention_mask = torch.where(
#             attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
#         )
#         attention_mask = attention_mask.repeat(self.batch_size * self.num_heads, 1, 1)
#
#         output_attention_output, output_attention_weights = (
#             self.model.compute_attention(query, key, value, attention_mask)
#         )
#
#         self.assertIsInstance(output_attention_output, torch.Tensor)
#         self.assertIsNone(output_attention_weights)
#         self.assertEqual(
#             output_attention_output.shape,
#             (self.target_sequence_length, self.batch_size, self.embedding_dim),
#         )
