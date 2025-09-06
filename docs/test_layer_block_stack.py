import unittest

from dataclasses import asdict
from Emperor.layers.utils.base import LayerBlockStack, LayerBlockStackConfig
from docs.utils import default_unittest_config


class TestLayerBlockStack(unittest.TestCase):
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

    def rebuild_presets(self, config: LayerBlockStackConfig | None = None):
        self.cfg = default_unittest_config()
        self.config = self.cfg.linear_block_stack_config
        if config is not None:
            for k in asdict(config):
                if hasattr(self.config, k) and getattr(config, k) is not None:
                    setattr(self.config, k, getattr(config, k))

        self.model = LayerBlockStack(self.cfg)

        # self.batch_size = self.cfg.batch_size
        # self.embedding_dim = self.cfg.embedding_dim
        # self.target_sequence_length = self.cfg.target_sequence_length
        # self.source_sequence_length = self.cfg.source_sequence_length
        # self.num_heads = self.cfg.num_heads
        # self.head_dim = self.embedding_dim // self.num_heads


class TestFeedForward__init(TestLayerBlockStack):
    def test__init_input_layer_with_default_config(self):
        self.assertIsInstance(self.model, LayerBlockStack)

        self.assertEqual(self.model.input_dim, self.config.input_dim)
        self.assertEqual(self.model.hidden_dim, self.config.hidden_dim)
        self.assertEqual(self.model.output_dim, self.config.output_dim)
        self.assertEqual(self.model.num_layers, self.config.num_layers)
        self.assertEqual(self.model.activation, self.config.activation)
        self.assertEqual(self.model.layer_type, self.config.layer_type.value)
        self.assertEqual(self.model.layer_norm_flag, self.config.layer_norm_flag)
        self.assertEqual(
            self.model.layer_form_first_flag, self.config.layer_form_first_flag
        )
        self.assertEqual(self.model.residual_flag, self.config.residual_flag)
        self.assertEqual(
            self.model.adaptive_computation_flag, self.config.adaptive_computation_flag
        )
