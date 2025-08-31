import torch
import unittest
import itertools
from dataclasses import asdict
from Emperor.attention.utils.utils import Mask, Processor, Projector, Utils, Validator
from Emperor.attention.attention import MultiHeadAttention, MultiHeadAttentionConfig
from Emperor.layers.utils.enums import LayerTypes
from docs.utils import default_unittest_config


class TestAttention(unittest.TestCase):
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

    def rebuild_presets(self, config: MultiHeadAttentionConfig | None = None):
        self.cfg = default_unittest_config()
        self.config = self.cfg.multi_head_attention_model_config
        if config is not None:
            for k in asdict(config):
                if hasattr(self.config, k) and getattr(config, k) is not None:
                    setattr(self.config, k, getattr(config, k))

        self.model = MultiHeadAttention(self.cfg)

        self.batch_size = self.config.batch_size
        self.embedding_dim = self.config.embedding_dim
        self.target_sequence_length = self.config.target_sequence_length
        self.source_sequence_length = self.config.source_sequence_length
        self.num_heads = self.config.num_heads
        self.head_dim = self.embedding_dim // self.num_heads


class TestMultiHeadAttention__init(TestAttention):
    def test__init_input_layer_with_default_config(self):
        self.assertIsInstance(self.model, MultiHeadAttention)
        self.assertEqual(self.model.batch_size, self.config.batch_size)
        self.assertEqual(self.model.model_type, self.config.model_type)
        self.assertEqual(self.model.num_heads, self.config.num_heads)
        self.assertEqual(self.model.embedding_dim, self.config.embedding_dim)
        self.assertEqual(self.model.target_dtype, self.config.target_dtype)
        self.assertEqual(
            self.model.target_sequence_length, self.config.target_sequence_length
        )
        self.assertEqual(
            self.model.source_sequence_length, self.config.source_sequence_length
        )
        self.assertEqual(
            self.model.use_separate_projection_weight_flag,
            self.config.use_separate_projection_weight_flag,
        )
        self.assertEqual(
            self.model.dropout_probability, self.config.dropout_probability
        )
        self.assertEqual(
            self.model.key_value_bias_flag, self.config.key_value_bias_flag
        )
        self.assertEqual(
            self.model.zero_attention_flag, self.config.zero_attention_flag
        )
        self.assertEqual(
            self.model.query_key_projection_dim, self.config.query_key_projection_dim
        )
        self.assertEqual(
            self.model.value_projection_dim, self.config.value_projection_dim
        )


class TestMultIHeadAttention____initialize_attention_components(TestAttention):
    def test__ensure_componets_are_initialzied(self):
        self.assertIsInstance(self.model.validator, Validator)
        self.assertIsInstance(self.model.masks, Mask)
        self.assertIsInstance(self.model.projector, Projector)
        self.assertIsInstance(self.model.processor, Processor)
        self.assertIsInstance(self.model.utils, Utils)


class TestMultIHeadAttention_forward(TestAttention):
    def test__use_separate_projection_weight_flag_with_same_qkv_tensors(self):
        tests = [
            {"use_separate_projection_weight_flag": False},
            {"use_separate_projection_weight_flag": True},
        ]
        for test in tests:
            config = MultiHeadAttentionConfig(
                target_sequence_length=32,
                source_sequence_length=32,
                **test,
            )
            self.rebuild_presets(config)

            query = key = value = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )

            key_padding_mask = None
            attention_mask = None
            static_key = None
            static_values = None

            attention_output, attention_weights = self.model.forward(
                query,
                key,
                value,
                key_padding_mask,
                attention_mask,
                static_key,
                static_values,
            )

            self.assertIsInstance(attention_output, torch.Tensor)
            self.assertIsNone(attention_weights)
            self.assertEqual(
                attention_output.shape,
                (self.target_sequence_length, self.batch_size, self.embedding_dim),
            )

    def test__use_separate_projection_weight_flag_with_different_qkv_tensors(self):
        config = MultiHeadAttentionConfig(
            target_sequence_length=32,
            source_sequence_length=32,
            use_separate_projection_weight_flag=True,
        )
        self.rebuild_presets(config)

        query = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        key = torch.randn(
            self.source_sequence_length, self.batch_size, self.embedding_dim
        )
        value = torch.randn(
            self.source_sequence_length, self.batch_size, self.embedding_dim
        )

        key_padding_mask = None
        attention_mask = None
        static_key = None
        static_values = None

        attention_output, attention_weights = self.model.forward(
            query,
            key,
            value,
            key_padding_mask,
            attention_mask,
            static_key,
            static_values,
        )

        self.assertIsInstance(attention_output, torch.Tensor)
        self.assertIsNone(attention_weights)
        self.assertEqual(
            attention_output.shape,
            (self.target_sequence_length, self.batch_size, self.embedding_dim),
        )

    def test__qkv_tensors_and_key_padding_mask(self):
        config = MultiHeadAttentionConfig(
            target_sequence_length=32,
            source_sequence_length=32,
            use_separate_projection_weight_flag=True,
        )
        self.rebuild_presets(config)

        query = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        key = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        value = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )

        key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
        attention_mask = None
        static_key = None
        static_values = None

        attention_output, attention_weights = self.model.forward(
            query,
            key,
            value,
            key_padding_mask,
            attention_mask,
            static_key,
            static_values,
        )

        self.assertIsInstance(attention_output, torch.Tensor)
        self.assertIsNone(attention_weights)

    def test__qkv_tensors_and_attention_mask(self):
        config = MultiHeadAttentionConfig(
            target_sequence_length=32,
            source_sequence_length=32,
            use_separate_projection_weight_flag=True,
        )
        self.rebuild_presets(config)

        query = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        key = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        value = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )

        key_padding_mask = None
        attention_mask = torch.randn(
            1, self.target_sequence_length, self.source_sequence_length
        )
        attention_mask = torch.where(
            attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
        )
        attention_mask = attention_mask.repeat(self.batch_size * self.num_heads, 1, 1)
        static_key = None
        static_values = None

        attention_output, attention_weights = self.model.forward(
            query,
            key,
            value,
            key_padding_mask,
            attention_mask,
            static_key,
            static_values,
        )

        self.assertIsInstance(attention_output, torch.Tensor)
        self.assertIsNone(attention_weights)

    def test__qkv_tensors_and_key_padding_mask_and_attention_mask(self):
        config = MultiHeadAttentionConfig(
            target_sequence_length=32,
            source_sequence_length=32,
            use_separate_projection_weight_flag=True,
        )
        self.rebuild_presets(config)

        query = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        key = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        value = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
        attention_mask = torch.randn(
            1, self.target_sequence_length, self.source_sequence_length
        )
        attention_mask = torch.where(
            attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
        )
        attention_mask = attention_mask.repeat(self.batch_size * self.num_heads, 1, 1)
        static_key = None
        static_values = None

        attention_output, attention_weights = self.model.forward(
            query,
            key,
            value,
            key_padding_mask,
            attention_mask,
            static_key,
            static_values,
        )

        self.assertIsInstance(attention_output, torch.Tensor)
        self.assertIsNone(attention_weights)

    def test__return_attention_weights_flag(self):
        tests = [
            {"return_attention_weights_flag": False},
            {"return_attention_weights_flag": True},
        ]

        for test in tests:
            config = MultiHeadAttentionConfig(
                target_sequence_length=32,
                source_sequence_length=32,
                use_separate_projection_weight_flag=False,
                **test,
            )
            self.rebuild_presets(config)

            query = key = value = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
            attention_mask = torch.randn(
                1, self.target_sequence_length, self.source_sequence_length
            )
            attention_mask = torch.where(
                attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
            )
            attention_mask = attention_mask.repeat(
                self.batch_size * self.num_heads, 1, 1
            )
            static_key = None
            static_values = None

            attention_output, attention_weights = self.model.forward(
                query,
                key,
                value,
                key_padding_mask,
                attention_mask,
                static_key,
                static_values,
            )

            self.assertIsInstance(attention_output, torch.Tensor)
            if test["return_attention_weights_flag"]:
                self.assertIsInstance(attention_weights, torch.Tensor)
            else:
                self.assertIsNone(attention_weights)

    def test__use_separate_projection_weight_flag(self):
        tests = [
            {"use_separate_projection_weight_flag": False},
            {"use_separate_projection_weight_flag": True},
        ]

        for test in tests:
            config = MultiHeadAttentionConfig(
                target_sequence_length=32,
                source_sequence_length=32,
                **test,
            )
            self.rebuild_presets(config)

            query = key = value = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
            attention_mask = torch.randn(
                1, self.target_sequence_length, self.source_sequence_length
            )
            attention_mask = torch.where(
                attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
            )
            attention_mask = attention_mask.repeat(
                self.batch_size * self.num_heads, 1, 1
            )
            static_key = None
            static_value = None

            attention_output, attention_weights = self.model.forward(
                query,
                key,
                value,
                key_padding_mask,
                attention_mask,
                static_key,
                static_value,
            )

            self.assertIsInstance(attention_output, torch.Tensor)
            self.assertIsNone(attention_weights)

    def test__zero_attention_flag(self):
        tests = [
            {"zero_attention_flag": False},
            {"zero_attention_flag": True},
        ]

        for test in tests:
            config = MultiHeadAttentionConfig(
                target_sequence_length=32,
                source_sequence_length=32,
                use_separate_projection_weight_flag=True,
                **test,
            )
            self.rebuild_presets(config)

            query = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            key = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            value = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )

            key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
            attention_mask = torch.randn(
                1, self.target_sequence_length, self.source_sequence_length
            )
            attention_mask = torch.where(
                attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
            )
            attention_mask = attention_mask.repeat(
                self.batch_size * self.num_heads, 1, 1
            )
            static_key = None
            static_value = None

            attention_output, attention_weights = self.model.forward(
                query,
                key,
                value,
                key_padding_mask,
                attention_mask,
                static_key,
                static_value,
            )

            self.assertIsInstance(attention_output, torch.Tensor)
            self.assertIsNone(attention_weights)

    def test__causal_attention_mask_flag(self):
        tests = [
            {"causal_attention_mask_flag": False},
            {"causal_attention_mask_flag": True},
        ]

        for test in tests:
            config = MultiHeadAttentionConfig(
                target_sequence_length=32,
                source_sequence_length=32,
                use_separate_projection_weight_flag=True,
                **test,
            )
            self.rebuild_presets(config)

            query = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            key = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            value = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
            attention_mask = torch.randn(
                1, self.target_sequence_length, self.source_sequence_length
            )
            attention_mask = torch.where(
                attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
            )
            attention_mask = attention_mask.repeat(
                self.batch_size * self.num_heads, 1, 1
            )
            static_key = None
            static_value = None

            attention_output, attention_weights = self.model.forward(
                query,
                key,
                value,
                key_padding_mask,
                attention_mask,
                static_key,
                static_value,
            )

            self.assertIsInstance(attention_output, torch.Tensor)
            self.assertIsNone(attention_weights)

    def test__add_key_value_bias_flag(self):
        tests = [
            {"add_key_value_bias_flag": False},
            {"add_key_value_bias_flag": True},
        ]

        for test in tests:
            config = MultiHeadAttentionConfig(
                target_sequence_length=32,
                source_sequence_length=32,
                use_separate_projection_weight_flag=True,
                **test,
            )
            self.rebuild_presets(config)

            query = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            key = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            value = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
            attention_mask = torch.randn(
                1, self.target_sequence_length, self.source_sequence_length
            )
            attention_mask = torch.where(
                attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
            )
            attention_mask = attention_mask.repeat(
                self.batch_size * self.num_heads, 1, 1
            )
            static_key = None
            static_value = None

            attention_output, attention_weights = self.model.forward(
                query,
                key,
                value,
                key_padding_mask,
                attention_mask,
                static_key,
                static_value,
            )
            self.assertIsInstance(attention_output, torch.Tensor)
            self.assertIsNone(attention_weights)

    def test__average_attention_weights_flag(self):
        tests = [
            {"average_attention_weights_flag": False},
            {"average_attention_weights_flag": True},
        ]

        for test in tests:
            config = MultiHeadAttentionConfig(
                target_sequence_length=32,
                source_sequence_length=32,
                return_attention_weights_flag=True,
                use_separate_projection_weight_flag=False,
                **test,
            )
            self.rebuild_presets(config)

            query = key = value = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
            attention_mask = torch.randn(
                1, self.target_sequence_length, self.source_sequence_length
            )
            attention_mask = torch.where(
                attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
            )
            attention_mask = attention_mask.repeat(
                self.batch_size * self.num_heads, 1, 1
            )
            static_key = None
            static_value = None

            attention_output, attention_weights = self.model.forward(
                query,
                key,
                value,
                key_padding_mask,
                attention_mask,
                static_key,
                static_value,
            )
            self.assertIsInstance(attention_output, torch.Tensor)
            self.assertIsInstance(attention_weights, torch.Tensor)
            if test["average_attention_weights_flag"]:
                self.assertEqual(attention_weights.dim(), 3)
            else:
                self.assertEqual(attention_weights.dim(), 4)

    # TODO: This flag has been removed but ensure you test the new condition
    # def test__batch_first_flag(self):
    #     tests = [
    #         {"batch_first_flag": False},
    #         {"batch_first_flag": True},
    #     ]
    #
    #     for test in tests:
    #         config = MultiHeadAttentionConfig(
    #             target_sequence_length=32,
    #             source_sequence_length=32,
    #             use_separate_projection_weight_flag=True,
    #             **test,
    #         )
    #         self.rebuild_presets(config)
    #
    #         if test["batch_first_flag"]:
    #             q_shape = (
    #                 self.batch_size,
    #                 self.source_sequence_length,
    #                 self.embedding_dim,
    #             )
    #             kv_shape = (
    #                 self.batch_size,
    #                 self.target_sequence_length,
    #                 self.embedding_dim,
    #             )
    #         else:
    #             q_shape = (
    #                 self.source_sequence_length,
    #                 self.batch_size,
    #                 self.embedding_dim,
    #             )
    #             kv_shape = (
    #                 self.target_sequence_length,
    #                 self.batch_size,
    #                 self.embedding_dim,
    #             )
    #
    #         query = torch.randn(q_shape)
    #         key = torch.randn(kv_shape)
    #         value = torch.randn(kv_shape)
    #         key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
    #         attention_mask = torch.randn(
    #             1, self.target_sequence_length, self.source_sequence_length
    #         )
    #         attention_mask = torch.where(
    #             attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
    #         )
    #         attention_mask = attention_mask.repeat(
    #             self.batch_size * self.num_heads, 1, 1
    #         )
    #         static_key = None
    #         static_value = None
    #
    #         attention_output, attention_weights = self.model.forward(
    #             query,
    #             key,
    #             value,
    #             key_padding_mask,
    #             attention_mask,
    #             static_key,
    #             static_value,
    #         )
    #
    #         self.assertIsInstance(attention_output, torch.Tensor)
    #         self.assertEqual(attention_output.shape, kv_shape)
    #         self.assertIsNone(attention_weights)

    def test__add_key_value_bias_flag__and__zero_attention_flag(self):
        tests = [
            {"add_key_value_bias_flag": False, "zero_attention_flag": False},
            {"add_key_value_bias_flag": True, "zero_attention_flag": False},
            {"add_key_value_bias_flag": False, "zero_attention_flag": True},
            {"add_key_value_bias_flag": True, "zero_attention_flag": True},
        ]

        for test in tests:
            config = MultiHeadAttentionConfig(
                target_sequence_length=32,
                source_sequence_length=32,
                use_separate_projection_weight_flag=True,
                **test,
            )
            self.rebuild_presets(config)

            query = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            key = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            value = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
            attention_mask = torch.randn(
                1, self.target_sequence_length, self.source_sequence_length
            )
            attention_mask = torch.where(
                attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
            )
            attention_mask = attention_mask.repeat(
                self.batch_size * self.num_heads, 1, 1
            )
            static_key = None
            static_value = None

            attention_output, attention_weights = self.model.forward(
                query,
                key,
                value,
                key_padding_mask,
                attention_mask,
                static_key,
                static_value,
            )
            self.assertIsInstance(attention_output, torch.Tensor)
            self.assertIsNone(attention_weights)

    def test__self_attention__possible_flag_combinations(self):
        flags = [
            "add_key_value_bias_flag",
            "zero_attention_flag",
            # "return_attention_weights_flag",
            "average_attention_weights_flag",
            "causal_attention_mask_flag",
            "use_separate_projection_weight_flag",
        ]
        combinations = list(itertools.product([False, True], repeat=len(flags)))
        tests = [dict(zip(flags, combo)) for combo in combinations]

        for test in tests:
            config = MultiHeadAttentionConfig(
                target_sequence_length=32,
                source_sequence_length=32,
                **test,
            )
            self.rebuild_presets(config)

            query = key = value = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
            attention_mask = torch.randn(
                1, self.target_sequence_length, self.source_sequence_length
            )
            attention_mask = torch.where(
                attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
            )
            attention_mask = attention_mask.repeat(
                self.batch_size * self.num_heads, 1, 1
            )
            static_key = None
            static_value = None

            attention_output, attention_weights = self.model.forward(
                query,
                key,
                value,
                key_padding_mask,
                attention_mask,
                static_key,
                static_value,
            )
            self.assertIsInstance(attention_output, torch.Tensor)
            # self.assertIsNone(attention_weights)

    def test__differetn_input_tensors__possible_flag_combinations(self):
        flags = [
            "add_key_value_bias_flag",
            "zero_attention_flag",
            # "return_attention_weights_flag",
            "average_attention_weights_flag",
            "causal_attention_mask_flag",
            "use_separate_projection_weight_flag",
        ]
        combinations = list(itertools.product([False, True], repeat=len(flags)))
        tests = [dict(zip(flags, combo)) for combo in combinations]

        for test in tests:
            config = MultiHeadAttentionConfig(
                target_sequence_length=32,
                source_sequence_length=32,
                **test,
            )
            self.rebuild_presets(config)

            query = key = value = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )

            key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
            attention_mask = torch.randn(
                1, self.target_sequence_length, self.source_sequence_length
            )
            attention_mask = torch.where(
                attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
            )
            attention_mask = attention_mask.repeat(
                self.batch_size * self.num_heads, 1, 1
            )
            static_key = None
            static_value = None

            attention_output, attention_weights = self.model.forward(
                query,
                key,
                value,
                key_padding_mask,
                attention_mask,
                static_key,
                static_value,
            )
            self.assertIsInstance(attention_output, torch.Tensor)

    def test__all_layer_types_and_flags_and_different_batch_sizes(self):
        flags = [
            "add_key_value_bias_flag",
            "zero_attention_flag",
            # "return_attention_weights_flag",
            "average_attention_weights_flag",
            "causal_attention_mask_flag",
            # "use_separate_projection_weight_flag",
        ]
        combinations = list(itertools.product([False, True], repeat=len(flags)))
        tests = [dict(zip(flags, combo)) for combo in combinations]
        batch_sizes = [1, 2, 4, 8]  # Pass
        for test in tests:
            for batch_size in batch_sizes:
                for model_type in LayerTypes:
                    config = MultiHeadAttentionConfig(
                        model_type=model_type,
                        batch_size=batch_size,
                        target_sequence_length=32,
                        source_sequence_length=32,
                        use_separate_projection_weight_flag=True,
                        **test,
                    )
                    self.rebuild_presets(config)
                    query = torch.randn(
                        self.target_sequence_length, self.batch_size, self.embedding_dim
                    )
                    key = torch.randn(
                        self.target_sequence_length, self.batch_size, self.embedding_dim
                    )
                    value = torch.randn(
                        self.target_sequence_length, self.batch_size, self.embedding_dim
                    )
                    key_padding_mask = torch.randn(
                        self.batch_size, self.source_sequence_length
                    )
                    attention_mask = torch.randn(
                        1, self.target_sequence_length, self.source_sequence_length
                    )
                    attention_mask = torch.where(
                        attention_mask > 0,
                        torch.tensor(float("-inf")),
                        torch.tensor(0.0),
                    )
                    attention_mask = attention_mask.repeat(
                        self.batch_size * self.num_heads, 1, 1
                    )
                    static_key = None
                    static_value = None

                    attention_output, attention_weights = self.model.forward(
                        query,
                        key,
                        value,
                        key_padding_mask,
                        attention_mask,
                        static_key,
                        static_value,
                    )
                    self.assertIsInstance(attention_output, torch.Tensor)

    # FIX: In the case of this test embedding dimension is only updated in the
    # `MultiHeadAttentionConfig` and but not the `ModelConfig`.
    #
    # def test__all_layer_types_and_flags_and_different_embedding_dims(self):
    #     flags = [
    #         "add_key_value_bias_flag",
    #         "zero_attention_flag",
    #         # "return_attention_weights_flag",
    #         "average_attention_weights_flag",
    #         "causal_attention_mask_flag",
    #         # "use_separate_projection_weight_flag",
    #     ]
    #     combinations = list(itertools.product([False, True], repeat=len(flags)))
    #     tests = [dict(zip(flags, combo)) for combo in combinations]
    #     embedding_dims = [16, 32, 64]
    #     for test in tests:
    #         for embedding_dim in embedding_dims:
    #             for model_type in LayerTypes:
    #                 config = MultiHeadAttentionConfig(
    #                     model_type=model_type,
    #                     embedding_dim=embedding_dim,
    #                     target_sequence_length=32,
    #                     source_sequence_length=32,
    #                     use_separate_projection_weight_flag=True,
    #                     **test,
    #                 )
    #                 self.rebuild_presets(config)
    #
    #                 query = torch.randn(
    #                     self.target_sequence_length, self.batch_size, self.embedding_dim
    #                 )
    #                 key = torch.randn(
    #                     self.target_sequence_length, self.batch_size, self.embedding_dim
    #                 )
    #                 value = torch.randn(
    #                     self.target_sequence_length, self.batch_size, self.embedding_dim
    #                 )
    #                 key_padding_mask = torch.randn(
    #                     self.batch_size, self.source_sequence_length
    #                 )
    #                 attention_mask = torch.randn(
    #                     1, self.target_sequence_length, self.source_sequence_length
    #                 )
    #                 attention_mask = torch.where(
    #                     attention_mask > 0,
    #                     torch.tensor(float("-inf")),
    #                     torch.tensor(0.0),
    #                 )
    #                 attention_mask = attention_mask.repeat(
    #                     self.batch_size * self.num_heads, 1, 1
    #                 )
    #                 static_key = None
    #                 static_value = None
    #
    #                 attention_output, attention_weights = self.model.forward(
    #                     query,
    #                     key,
    #                     value,
    #                     key_padding_mask,
    #                     attention_mask,
    #                     static_key,
    #                     static_value,
    #                 )
    #                 self.assertIsInstance(attention_output, torch.Tensor)

    # BUG: `query_key_projection_dim` and `value_projection_dim` not working if not equel to `embedding_dim`. Fix this later!
    #
    # def test__all_layer_types_and_flags_and_different_query_key_projection_dim(self):
    #     flags = [
    #         "add_key_value_bias_flag",
    #         "zero_attention_flag",
    #         # "return_attention_weights_flag",
    #         "average_attention_weights_flag",
    #         "causal_attention_mask_flag",
    #         # "use_separate_projection_weight_flag",
    #     ]
    #     combinations = list(itertools.product([False, True], repeat=len(flags)))
    #     tests = [dict(zip(flags, combo)) for combo in combinations]
    #     query_key_value_projection_dims = [16, 32, 64]
    #     value_projection_dims = [16, 32, 64]
    #     target_sequence_lengths = [16, 32, 64]
    #     for test in tests:
    #         for query_key_value_projection_dim in query_key_value_projection_dims:
    #             for model_type in LayerTypes:
    #                 config = MultiHeadAttentionConfig(
    #                     model_type=model_type,
    #                     query_key_projection_dim=query_key_value_projection_dim,
    #                     value_projection_dim=query_key_value_projection_dim,
    #                     target_sequence_length=32,
    #                     source_sequence_length=32,
    #                     use_separate_projection_weight_flag=True,
    #                     **test,
    #                 )
    #                 self.rebuild_presets(config)
    #
    #                 query = torch.randn(
    #                     self.target_sequence_length, self.batch_size, self.embedding_dim
    #                 )
    #                 key = torch.randn(
    #                     self.target_sequence_length, self.batch_size, self.embedding_dim
    #                 )
    #                 value = torch.randn(
    #                     self.target_sequence_length, self.batch_size, self.embedding_dim
    #                 )
    #                 key_padding_mask = torch.randn(
    #                     self.batch_size, self.source_sequence_length
    #                 )
    #                 attention_mask = torch.randn(
    #                     1, self.target_sequence_length, self.source_sequence_length
    #                 )
    #                 attention_mask = torch.where(
    #                     attention_mask > 0,
    #                     torch.tensor(float("-inf")),
    #                     torch.tensor(0.0),
    #                 )
    #                 attention_mask = attention_mask.repeat(
    #                     self.batch_size * self.num_heads, 1, 1
    #                 )
    #                 static_key = None
    #                 static_value = None
    #
    #                 attention_output, attention_weights = self.model.forward(
    #                     query,
    #                     key,
    #                     value,
    #                     key_padding_mask,
    #                     attention_mask,
    #                     static_key,
    #                     static_value,
    #                 )
    #                 self.assertIsInstance(attention_output, torch.Tensor)

    # BUG: `target_sequence_length` and `source_sequence_length` not working if not equal. Fix this later!
    #
    # def test__all_layer_types_and_flags_and_different_target_sequence_length(self):
    #     flags = [
    #         "add_key_value_bias_flag",
    #         "zero_attention_flag",
    #         # "return_attention_weights_flag",
    #         "average_attention_weights_flag",
    #         "causal_attention_mask_flag",
    #         # "use_separate_projection_weight_flag",
    #     ]
    #     combinations = list(itertools.product([False, True], repeat=len(flags)))
    #     tests = [dict(zip(flags, combo)) for combo in combinations]
    #     target_sequence_lengths = [16, 32, 64]
    #     for test in tests:
    #         for target_sequence_length in target_sequence_lengths:
    #             for model_type in LayerTypes:
    #                 config = MultiHeadAttentionConfig(
    #                     model_type=model_type,
    #                     target_sequence_length=target_sequence_length,
    #                     source_sequence_length=32,
    #                     use_separate_projection_weight_flag=True,
    #                     **test,
    #                 )
    #                 self.rebuild_presets(config)
    #
    #                 query = torch.randn(
    #                     self.target_sequence_length, self.batch_size, self.embedding_dim
    #                 )
    #                 key = torch.randn(
    #                     self.target_sequence_length, self.batch_size, self.embedding_dim
    #                 )
    #                 value = torch.randn(
    #                     self.target_sequence_length, self.batch_size, self.embedding_dim
    #                 )
    #                 key_padding_mask = torch.randn(
    #                     self.batch_size, self.source_sequence_length
    #                 )
    #                 attention_mask = torch.randn(
    #                     1, self.target_sequence_length, self.source_sequence_length
    #                 )
    #                 attention_mask = torch.where(
    #                     attention_mask > 0,
    #                     torch.tensor(float("-inf")),
    #                     torch.tensor(0.0),
    #                 )
    #                 attention_mask = attention_mask.repeat(
    #                     self.batch_size * self.num_heads, 1, 1
    #                 )
    #                 static_key = None
    #                 static_value = None
    #
    #                 attention_output, attention_weights = self.model.forward(
    #                     query,
    #                     key,
    #                     value,
    #                     key_padding_mask,
    #                     attention_mask,
    #                     static_key,
    #                     static_value,
    #                 )
    #                 self.assertIsInstance(attention_output, torch.Tensor)
