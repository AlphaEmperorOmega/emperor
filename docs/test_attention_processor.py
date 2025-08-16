from dataclasses import asdict
import math
import copy
import unittest
from unittest.mock import MagicMock
import torch
import torch.nn as nn
import torch.nn.functional as F
from Emperor.attention.utils.utils import (
    AttentionProcessor,
    AttentionProcessorDefault,
    AttentionProcessorWithReturnedWeights,
    AttentionValidator,
)
from Emperor.layers.utils.base import LayerBlock
from Emperor.attention.attention import MultiHeadAttention, MultiHeadAttentionConfig
from docs.utils import default_unittest_config


class TestAttentionProcessor(unittest.TestCase):
    def setUp(self):
        self.cfg = default_unittest_config()
        self.config = self.cfg.multi_head_attention_model_config

        model = MultiHeadAttention(self.cfg)
        output_model = model.output_model

        validator = AttentionValidator(self.config)
        self.model = AttentionProcessor(self.config, validator, output_model)

        self.batch_size = self.config.batch_size
        self.embedding_dim = self.config.embedding_dim
        self.target_sequence_length = self.config.target_sequence_length
        self.source_sequence_length = self.config.source_sequence_length
        self.num_heads = self.config.num_heads
        self.head_dim = self.embedding_dim // self.num_heads

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.model = None
        self.batch_size = None
        self.embedding_dim = None
        self.target_sequence_length = None
        self.num_heads = None
        self.head_dim = None

    def rebuild_presets(self, config: MultiHeadAttentionConfig):
        self.cfg = default_unittest_config()
        self.config = self.cfg.multi_head_attention_model_config
        for k in asdict(config):
            if hasattr(self.config, k) and getattr(config, k) is not None:
                setattr(self.config, k, getattr(config, k))

        model = MultiHeadAttention(self.cfg)
        output_model = model.output_model

        validator = AttentionValidator(self.config)
        self.model = AttentionProcessor(self.config, validator, output_model)

        self.batch_size = self.config.batch_size
        self.embedding_dim = self.config.embedding_dim
        self.target_sequence_length = self.config.target_sequence_length
        self.source_sequence_length = self.config.source_sequence_length
        self.num_heads = self.config.num_heads
        self.head_dim = self.embedding_dim // self.num_heads


class Test__init(TestAttentionProcessor):
    def test__init(self):
        self.assertIsInstance(self.model, AttentionProcessor)
        self.assertIsInstance(
            self.model.processor,
            (AttentionProcessorDefault, AttentionProcessorWithReturnedWeights),
        )


class Test____create_processor(TestAttentionProcessor):
    def test__return_attention_weights_flag__False(self):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=False,
        )
        self.rebuild_presets(config)
        self.assertIsInstance(self.model.processor, AttentionProcessorDefault)

    def test__return_attention_weights_flag__True(self):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=True,
        )
        self.rebuild_presets(config)
        self.assertIsInstance(
            self.model.processor, AttentionProcessorWithReturnedWeights
        )


# class Test____compute_weighted_values_default(TestAttentionProcessor):
#     def test__method(self):
#         pass
