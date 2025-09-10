import unittest

from dataclasses import asdict
from Emperor.feedForward.feed_forward import FeedForward, FeedForwardConfig
from docs.utils import default_unittest_config


class TestFeedForward(unittest.TestCase):
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

    def rebuild_presets(self, config: FeedForwardConfig | None = None):
        self.cfg = default_unittest_config()
        self.config = self.cfg.transformer_feed_forward_config
        if config is not None:
            for k in asdict(config):
                if hasattr(self.config, k) and getattr(config, k) is not None:
                    setattr(self.config, k, getattr(config, k))

        self.model = FeedForward(self.cfg)

        self.batch_size = self.cfg.batch_size
        # self.embedding_dim = self.cfg.embedding_dim
        # self.target_sequence_length = self.cfg.target_sequence_length
        # self.source_sequence_length = self.cfg.source_sequence_length
        # self.num_heads = self.cfg.num_heads
        # self.head_dim = self.embedding_dim // self.num_heads


class Test___init(TestFeedForward):
    def test__init_input_layer_with_default_config(self):
        config = FeedForwardConfig(
            num_layers=2,
        )
        self.rebuild_presets(config)
        self.assertIsInstance(self.model, FeedForward)
        self.assertEqual(self.model.num_layers, self.config.num_layers)
        self.assertEqual(len(self.model.model), self.config.num_layers)
