from emperor.base.layer.residual import ResidualConnectionOptions
import torch
import itertools
import unittest

from torch.nn import LayerNorm, ModuleList
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.base.layer import LayerConfig, LayerStackConfig
from emperor.linears.core.config import LinearLayerConfig
from emperor.attention.core.variants.self_attention.config import SelfAttentionConfig
from emperor.attention.core.variants.independent_attention.config import IndependentAttentionConfig
from emperor.attention.core.variants.mixture_of_attention_heads.config import (
    MixtureOfAttentionHeadsConfig,
)
from emperor.transformer.feed_forward.core.config import FeedForwardConfig
from emperor.transformer.config import TransformerConfig
from emperor.transformer.model import Transformer
from emperor.transformer.core.config import (
    TransformerEncoderLayerConfig,
    TransformerDecoderLayerConfig,
    TransformerEncoderStackConfig,
    TransformerDecoderStackConfig,
)
from emperor.transformer.core.stack import (
    TransformerEncoderStack,
    TransformerDecoderStack,
)
from _attention_test_helpers import (
    build_attention_config,
    make_adaptive_projection_model_config,
    make_mixture_of_experts_model_config,
)

BATCH_SIZE = 4
NUM_HEADS = 2
EMBEDDING_DIM = 10
NUM_LAYERS = 2
SEQUENCE_LENGTH = 6
EXPERTS_TOP_K = 3


def create_key_padding_mask(
    batch_size: int, source_sequence_length: int
) -> torch.Tensor:
    key_padding_mask = torch.randint(0, 2, (batch_size, source_sequence_length))
    return torch.where(
        key_padding_mask > 0,
        torch.tensor(float("-inf")),
        torch.tensor(0.0),
    )


def create_attention_mask(
    target_sequence_length: int,
    source_sequence_length: int,
    attention_mask_repeat: int = 1,
) -> torch.Tensor:
    attention_mask = torch.triu(
        torch.full((target_sequence_length, source_sequence_length), float("-inf")),
        diagonal=1,
    )
    return attention_mask.unsqueeze(0).repeat(attention_mask_repeat, 1, 1)


def feed_forward_config(
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=20,
    num_layers=2,
    dropout_probability=0.0,
    feed_forward_kind="base",
):
    if feed_forward_kind == "moe":
        stack_config = make_mixture_of_experts_model_config(
            input_dim=embedding_dim, output_dim=embedding_dim
        )
    elif feed_forward_kind == "adaptive":
        stack_config = make_adaptive_projection_model_config()
    else:
        stack_config = LayerStackConfig(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_layers,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            layer_config=LayerConfig(
                activation=ActivationOptions.RELU,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                residual_connection_option=ResidualConnectionOptions.DISABLED,
                dropout_probability=dropout_probability,
                halting_config=None,
                gate_config=None,
                layer_model_config=LinearLayerConfig(bias_flag=True),
            ),
        )
    return FeedForwardConfig(
        input_dim=embedding_dim,
        output_dim=embedding_dim,
        stack_config=stack_config,
    )


def encoder_layer_config(
    sequence_length=SEQUENCE_LENGTH,
    embedding_dim=EMBEDDING_DIM,
    batch_size=BATCH_SIZE,
    num_heads=NUM_HEADS,
    attention_config_class=SelfAttentionConfig,
    query_key_projection_dim=0,
    value_projection_dim=0,
    add_key_value_bias_flag=False,
    zero_attention_flag=False,
    average_attention_weights_flag=False,
    causal_attention_mask_flag=False,
    dropout_probability=0.0,
    projection_kind="base",
    feed_forward_kind="base",
):
    attention_config = build_attention_config(
        config_class=attention_config_class,
        batch_size=batch_size,
        num_heads=num_heads,
        embedding_dim=embedding_dim,
        query_key_projection_dim=query_key_projection_dim,
        value_projection_dim=value_projection_dim,
        target_sequence_length=sequence_length,
        source_sequence_length=sequence_length,
        dropout_probability=dropout_probability,
        zero_attention_flag=zero_attention_flag,
        causal_attention_mask_flag=causal_attention_mask_flag,
        add_key_value_bias_flag=add_key_value_bias_flag,
        average_attention_weights_flag=average_attention_weights_flag,
        return_attention_weights_flag=attention_config_class is SelfAttentionConfig,
        projection_kind=projection_kind,
        experts_top_k=EXPERTS_TOP_K,
    )
    return TransformerEncoderLayerConfig(
        embedding_dim=embedding_dim,
        layer_norm_position=LayerNormPositionOptions.DEFAULT,
        dropout_probability=dropout_probability,
        residual_connection_option=ResidualConnectionOptions.RESIDUAL,
        causal_attention_mask_flag=causal_attention_mask_flag,
        attention_config=attention_config,
        feed_forward_config=feed_forward_config(
            embedding_dim,
            dropout_probability=dropout_probability,
            feed_forward_kind=feed_forward_kind,
        ),
    )


def decoder_layer_config(
    sequence_length=SEQUENCE_LENGTH,
    cross_attention=True,
    embedding_dim=EMBEDDING_DIM,
    batch_size=BATCH_SIZE,
    num_heads=NUM_HEADS,
    target_sequence_length=None,
    source_sequence_length=None,
    self_attention_config_class=SelfAttentionConfig,
    cross_attention_config_class=IndependentAttentionConfig,
    query_key_projection_dim=0,
    value_projection_dim=0,
    add_key_value_bias_flag=False,
    zero_attention_flag=False,
    average_attention_weights_flag=False,
    causal_attention_mask_flag=False,
    dropout_probability=0.0,
    projection_kind="base",
    feed_forward_kind="base",
):
    target_sequence_length = target_sequence_length or sequence_length
    source_sequence_length = source_sequence_length or sequence_length

    self_attention_config = build_attention_config(
        config_class=self_attention_config_class,
        batch_size=batch_size,
        num_heads=num_heads,
        embedding_dim=embedding_dim,
        query_key_projection_dim=query_key_projection_dim,
        value_projection_dim=value_projection_dim,
        target_sequence_length=target_sequence_length,
        source_sequence_length=target_sequence_length,
        dropout_probability=dropout_probability,
        zero_attention_flag=zero_attention_flag,
        causal_attention_mask_flag=causal_attention_mask_flag,
        add_key_value_bias_flag=add_key_value_bias_flag,
        average_attention_weights_flag=average_attention_weights_flag,
        return_attention_weights_flag=self_attention_config_class
        is SelfAttentionConfig,
        projection_kind=projection_kind,
        experts_top_k=EXPERTS_TOP_K,
    )

    cross_attention_config = None
    if cross_attention:
        cross_attention_config = build_attention_config(
            config_class=cross_attention_config_class,
            batch_size=batch_size,
            num_heads=num_heads,
            embedding_dim=embedding_dim,
            query_key_projection_dim=query_key_projection_dim,
            value_projection_dim=value_projection_dim,
            target_sequence_length=target_sequence_length,
            source_sequence_length=source_sequence_length,
            dropout_probability=dropout_probability,
            zero_attention_flag=zero_attention_flag,
            causal_attention_mask_flag=causal_attention_mask_flag,
            add_key_value_bias_flag=add_key_value_bias_flag,
            average_attention_weights_flag=average_attention_weights_flag,
            return_attention_weights_flag=False,
            projection_kind=projection_kind,
            experts_top_k=EXPERTS_TOP_K,
        )

    return TransformerDecoderLayerConfig(
        embedding_dim=embedding_dim,
        layer_norm_position=LayerNormPositionOptions.DEFAULT,
        dropout_probability=dropout_probability,
        residual_connection_option=ResidualConnectionOptions.RESIDUAL,
        causal_attention_mask_flag=causal_attention_mask_flag,
        self_attention_config=self_attention_config,
        cross_attention_config=cross_attention_config,
        feed_forward_config=feed_forward_config(
            embedding_dim,
            dropout_probability=dropout_probability,
            feed_forward_kind=feed_forward_kind,
        ),
    )


def encoder_stack_config(
    causal_attention_mask_flag=False,
    num_layers=NUM_LAYERS,
    embedding_dim=EMBEDDING_DIM,
    batch_size=BATCH_SIZE,
    num_heads=NUM_HEADS,
    sequence_length=SEQUENCE_LENGTH,
    attention_config_class=SelfAttentionConfig,
    query_key_projection_dim=0,
    value_projection_dim=0,
    add_key_value_bias_flag=False,
    zero_attention_flag=False,
    average_attention_weights_flag=False,
    projection_kind="base",
    feed_forward_kind="base",
    layer_config=None,
):
    if layer_config is None:
        layer_config = encoder_layer_config(
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            num_heads=num_heads,
            attention_config_class=attention_config_class,
            query_key_projection_dim=query_key_projection_dim,
            value_projection_dim=value_projection_dim,
            add_key_value_bias_flag=add_key_value_bias_flag,
            zero_attention_flag=zero_attention_flag,
            average_attention_weights_flag=average_attention_weights_flag,
            causal_attention_mask_flag=causal_attention_mask_flag,
            projection_kind=projection_kind,
            feed_forward_kind=feed_forward_kind,
        )

    return TransformerEncoderStackConfig(
        num_layers=num_layers,
        embedding_dim=embedding_dim,
        source_sequence_length=sequence_length,
        target_sequence_length=sequence_length,
        causal_attention_mask_flag=causal_attention_mask_flag,
        layer_config=layer_config,
    )


def decoder_stack_config(
    cross_attention=True,
    causal_attention_mask_flag=False,
    num_layers=NUM_LAYERS,
    embedding_dim=EMBEDDING_DIM,
    batch_size=BATCH_SIZE,
    num_heads=NUM_HEADS,
    target_sequence_length=SEQUENCE_LENGTH,
    source_sequence_length=SEQUENCE_LENGTH,
    self_attention_config_class=SelfAttentionConfig,
    cross_attention_config_class=IndependentAttentionConfig,
    query_key_projection_dim=0,
    value_projection_dim=0,
    add_key_value_bias_flag=False,
    zero_attention_flag=False,
    average_attention_weights_flag=False,
    projection_kind="base",
    feed_forward_kind="base",
    layer_config=None,
):
    if layer_config is None:
        layer_config = decoder_layer_config(
            cross_attention=cross_attention,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            num_heads=num_heads,
            target_sequence_length=target_sequence_length,
            source_sequence_length=source_sequence_length,
            self_attention_config_class=self_attention_config_class,
            cross_attention_config_class=cross_attention_config_class,
            query_key_projection_dim=query_key_projection_dim,
            value_projection_dim=value_projection_dim,
            add_key_value_bias_flag=add_key_value_bias_flag,
            zero_attention_flag=zero_attention_flag,
            average_attention_weights_flag=average_attention_weights_flag,
            causal_attention_mask_flag=causal_attention_mask_flag,
            projection_kind=projection_kind,
            feed_forward_kind=feed_forward_kind,
        )

    return TransformerDecoderStackConfig(
        num_layers=num_layers,
        embedding_dim=embedding_dim,
        source_sequence_length=source_sequence_length,
        target_sequence_length=target_sequence_length,
        causal_attention_mask_flag=causal_attention_mask_flag,
        layer_config=layer_config,
    )


def embeddings(sequence_length=SEQUENCE_LENGTH):
    return torch.randn(sequence_length, BATCH_SIZE, EMBEDDING_DIM)


make_encoder_stack_config = encoder_stack_config
make_decoder_stack_config = decoder_stack_config


ATTENTION_CONFIG_CLASSES = (
    SelfAttentionConfig,
    IndependentAttentionConfig,
    MixtureOfAttentionHeadsConfig,
)
CROSS_ATTENTION_CONFIG_CLASSES = (
    IndependentAttentionConfig,
    MixtureOfAttentionHeadsConfig,
)
FEED_FORWARD_KINDS = ("base", "adaptive", "moe")


def attention_mask_repeat_for(attention_config_class):
    """Mixture-of-attention-heads expands masks across experts, so its attention
    mask repeats batch * heads * top_k times; the others repeat batch * heads."""
    repeat = BATCH_SIZE * NUM_HEADS
    if attention_config_class is MixtureOfAttentionHeadsConfig:
        repeat *= EXPERTS_TOP_K
    return repeat


class TestTransformerEncoderStack(unittest.TestCase):
    def preset(
        self,
        num_layers=NUM_LAYERS,
        embedding_dim=EMBEDDING_DIM,
        batch_size=BATCH_SIZE,
        num_heads=NUM_HEADS,
        sequence_length=SEQUENCE_LENGTH,
        attention_config_class=SelfAttentionConfig,
        query_key_projection_dim=0,
        value_projection_dim=0,
        add_key_value_bias_flag=False,
        zero_attention_flag=False,
        average_attention_weights_flag=False,
        causal_attention_mask_flag=False,
        projection_kind="base",
        feed_forward_kind="base",
        layer_config=None,
    ):
        return encoder_stack_config(
            causal_attention_mask_flag=causal_attention_mask_flag,
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            num_heads=num_heads,
            sequence_length=sequence_length,
            attention_config_class=attention_config_class,
            query_key_projection_dim=query_key_projection_dim,
            value_projection_dim=value_projection_dim,
            add_key_value_bias_flag=add_key_value_bias_flag,
            zero_attention_flag=zero_attention_flag,
            average_attention_weights_flag=average_attention_weights_flag,
            projection_kind=projection_kind,
            feed_forward_kind=feed_forward_kind,
            layer_config=layer_config,
        )

    def test_init(self):
        model = TransformerEncoderStack(encoder_stack_config())

        self.assertEqual(model.num_layers, NUM_LAYERS)
        self.assertEqual(model.source_sequence_length, SEQUENCE_LENGTH)
        self.assertEqual(model.target_sequence_length, SEQUENCE_LENGTH)
        self.assertIsInstance(model.layer_norm_module, LayerNorm)
        self.assertIsInstance(model.layers, ModuleList)
        self.assertEqual(len(model.layers), NUM_LAYERS)

    def test_config_build_returns_encoder_stack(self):
        config = self.preset()
        model = config.build()

        self.assertIsInstance(model, TransformerEncoderStack)
        self.assertIsInstance(model, config._registry_owner())
        self.assertEqual(model.num_layers, config.num_layers)
        self.assertEqual(model.source_sequence_length, config.source_sequence_length)
        self.assertEqual(model.target_sequence_length, config.target_sequence_length)
        self.assertEqual(len(model.layers), config.num_layers)

    def test_config_build_applies_overrides(self):
        config = self.preset(num_layers=1, sequence_length=SEQUENCE_LENGTH)
        overrides = self.preset(
            num_layers=3,
            sequence_length=4,
            causal_attention_mask_flag=True,
        )
        model = config.build(overrides)

        self.assertIsInstance(model, TransformerEncoderStack)
        self.assertEqual(model.num_layers, overrides.num_layers)
        self.assertEqual(model.source_sequence_length, overrides.source_sequence_length)
        self.assertEqual(model.target_sequence_length, overrides.target_sequence_length)
        self.assertEqual(
            model.causal_attention_mask_flag, overrides.causal_attention_mask_flag
        )
        self.assertEqual(len(model.layers), overrides.num_layers)

    def test_partial_overrides_keep_unset_encoder_stack_fields(self):
        config = self.preset(
            num_layers=2,
            sequence_length=SEQUENCE_LENGTH,
            causal_attention_mask_flag=False,
        )
        overrides = TransformerEncoderStackConfig(num_layers=1)
        model = config.build(overrides)

        self.assertIsInstance(model, TransformerEncoderStack)
        self.assertEqual(model.num_layers, overrides.num_layers)
        self.assertEqual(model.source_sequence_length, config.source_sequence_length)
        self.assertEqual(model.target_sequence_length, config.target_sequence_length)
        self.assertEqual(
            model.causal_attention_mask_flag, config.causal_attention_mask_flag
        )
        self.assertEqual(len(model.layers), overrides.num_layers)

    def test_forward_with_mask_combinations(self):
        model = TransformerEncoderStack(encoder_stack_config())
        source = embeddings()
        key_padding_mask_options = (
            None,
            create_key_padding_mask(BATCH_SIZE, SEQUENCE_LENGTH),
        )
        attention_mask_options = (
            None,
            create_attention_mask(
                SEQUENCE_LENGTH, SEQUENCE_LENGTH, BATCH_SIZE * NUM_HEADS
            ),
        )

        for key_padding_mask, attention_mask in itertools.product(
            key_padding_mask_options, attention_mask_options
        ):
            with self.subTest(
                padding=key_padding_mask is not None,
                attention=attention_mask is not None,
            ):
                output, loss = model(
                    source_token_embeddings=source,
                    source_key_padding_mask=key_padding_mask,
                    attention_mask=attention_mask,
                )
                self.assertEqual(
                    output.shape, (SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)
                )
                self.assertIsInstance(loss, torch.Tensor)

    def test_causal_flag_changes_output(self):
        causal = TransformerEncoderStack(encoder_stack_config(True))
        plain = TransformerEncoderStack(encoder_stack_config(False))
        plain.load_state_dict(causal.state_dict())
        source = embeddings()

        causal_output, _ = causal(source_token_embeddings=source)
        plain_output, _ = plain(source_token_embeddings=source)

        self.assertFalse(torch.allclose(causal_output, plain_output))

    def test_forward_rejects_wrong_embedding_dim(self):
        model = TransformerEncoderStack(encoder_stack_config())
        with self.assertRaises(ValueError):
            model(source_token_embeddings=torch.randn(SEQUENCE_LENGTH, BATCH_SIZE, 3))

    def test_forward_with_all_option_combinations(self):
        sequence_lengths = (6, 10)

        for attention_config_class in ATTENTION_CONFIG_CLASSES:
            for sequence_length in sequence_lengths:
                config = self.preset(
                    sequence_length=sequence_length,
                    attention_config_class=attention_config_class,
                )
                model = TransformerEncoderStack(config)
                source = torch.randn(sequence_length, BATCH_SIZE, EMBEDDING_DIM)
                key_padding_mask_options = (
                    None,
                    create_key_padding_mask(BATCH_SIZE, sequence_length),
                )
                attention_mask_options = (
                    None,
                    create_attention_mask(
                        sequence_length,
                        sequence_length,
                        attention_mask_repeat_for(attention_config_class),
                    ),
                )

                for key_padding_mask, attention_mask in itertools.product(
                    key_padding_mask_options, attention_mask_options
                ):
                    with self.subTest(
                        attention_type=attention_config_class.__name__,
                        sequence_length=sequence_length,
                        padding=key_padding_mask is not None,
                        attention=attention_mask is not None,
                    ):
                        output, loss = model(
                            source_token_embeddings=source,
                            source_key_padding_mask=key_padding_mask,
                            attention_mask=attention_mask,
                        )
                        self.assertEqual(
                            output.shape,
                            (sequence_length, BATCH_SIZE, EMBEDDING_DIM),
                        )
                        self.assertIsInstance(loss, torch.Tensor)

    def test_self_attention_rejects_mismatched_projection_dims(self):
        for query_key_projection_dim, value_projection_dim in (
            (16, 0),
            (0, 16),
            (16, 20),
        ):
            with self.subTest(
                query_key_projection_dim=query_key_projection_dim,
                value_projection_dim=value_projection_dim,
            ):
                config = self.preset(
                    attention_config_class=SelfAttentionConfig,
                    query_key_projection_dim=query_key_projection_dim,
                    value_projection_dim=value_projection_dim,
                )
                with self.assertRaises(RuntimeError):
                    TransformerEncoderStack(config)

    def test_forward_with_feed_forward_variants(self):
        for feed_forward_kind in FEED_FORWARD_KINDS:
            with self.subTest(feed_forward_kind=feed_forward_kind):
                model = TransformerEncoderStack(
                    self.preset(feed_forward_kind=feed_forward_kind)
                )
                output, loss = model(source_token_embeddings=embeddings())
                self.assertEqual(
                    output.shape, (SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)
                )
                self.assertIsInstance(loss, torch.Tensor)


class TestTransformerDecoderStack(unittest.TestCase):
    def preset(
        self,
        num_layers=NUM_LAYERS,
        embedding_dim=EMBEDDING_DIM,
        batch_size=BATCH_SIZE,
        num_heads=NUM_HEADS,
        target_sequence_length=SEQUENCE_LENGTH,
        source_sequence_length=SEQUENCE_LENGTH,
        cross_attention=True,
        self_attention_config_class=SelfAttentionConfig,
        cross_attention_config_class=IndependentAttentionConfig,
        query_key_projection_dim=0,
        value_projection_dim=0,
        add_key_value_bias_flag=False,
        zero_attention_flag=False,
        average_attention_weights_flag=False,
        causal_attention_mask_flag=False,
        projection_kind="base",
        feed_forward_kind="base",
        layer_config=None,
    ):
        return decoder_stack_config(
            cross_attention=cross_attention,
            causal_attention_mask_flag=causal_attention_mask_flag,
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            num_heads=num_heads,
            target_sequence_length=target_sequence_length,
            source_sequence_length=source_sequence_length,
            self_attention_config_class=self_attention_config_class,
            cross_attention_config_class=cross_attention_config_class,
            query_key_projection_dim=query_key_projection_dim,
            value_projection_dim=value_projection_dim,
            add_key_value_bias_flag=add_key_value_bias_flag,
            zero_attention_flag=zero_attention_flag,
            average_attention_weights_flag=average_attention_weights_flag,
            projection_kind=projection_kind,
            feed_forward_kind=feed_forward_kind,
            layer_config=layer_config,
        )

    def test_init(self):
        model = TransformerDecoderStack(decoder_stack_config())

        self.assertEqual(model.num_layers, NUM_LAYERS)
        self.assertIsInstance(model.layer_norm_module, LayerNorm)
        self.assertIsInstance(model.layers, ModuleList)
        self.assertEqual(len(model.layers), NUM_LAYERS)

    def test_config_build_returns_decoder_stack(self):
        config = self.preset()
        model = config.build()

        self.assertIsInstance(model, TransformerDecoderStack)
        self.assertIsInstance(model, config._registry_owner())
        self.assertEqual(model.num_layers, config.num_layers)
        self.assertEqual(model.source_sequence_length, config.source_sequence_length)
        self.assertEqual(model.target_sequence_length, config.target_sequence_length)
        self.assertEqual(len(model.layers), config.num_layers)

    def test_config_build_applies_overrides(self):
        config = self.preset(num_layers=1)
        overrides = self.preset(
            num_layers=3,
            source_sequence_length=5,
            target_sequence_length=4,
            causal_attention_mask_flag=True,
        )
        model = config.build(overrides)

        self.assertIsInstance(model, TransformerDecoderStack)
        self.assertEqual(model.num_layers, overrides.num_layers)
        self.assertEqual(model.source_sequence_length, overrides.source_sequence_length)
        self.assertEqual(model.target_sequence_length, overrides.target_sequence_length)
        self.assertEqual(
            model.causal_attention_mask_flag, overrides.causal_attention_mask_flag
        )
        self.assertEqual(len(model.layers), overrides.num_layers)

    def test_partial_overrides_keep_unset_decoder_stack_fields(self):
        config = self.preset(
            num_layers=2,
            source_sequence_length=SEQUENCE_LENGTH,
            target_sequence_length=SEQUENCE_LENGTH,
            causal_attention_mask_flag=False,
        )
        overrides = TransformerDecoderStackConfig(num_layers=1)
        model = config.build(overrides)

        self.assertIsInstance(model, TransformerDecoderStack)
        self.assertEqual(model.num_layers, overrides.num_layers)
        self.assertEqual(model.source_sequence_length, config.source_sequence_length)
        self.assertEqual(model.target_sequence_length, config.target_sequence_length)
        self.assertEqual(
            model.causal_attention_mask_flag, config.causal_attention_mask_flag
        )
        self.assertEqual(len(model.layers), overrides.num_layers)

    def test_forward_with_cross_attention(self):
        model = TransformerDecoderStack(decoder_stack_config(cross_attention=True))
        target = embeddings()
        encoder_output = embeddings()
        key_padding_mask_options = (
            None,
            create_key_padding_mask(BATCH_SIZE, SEQUENCE_LENGTH),
        )
        attention_mask_options = (
            None,
            create_attention_mask(
                SEQUENCE_LENGTH, SEQUENCE_LENGTH, BATCH_SIZE * NUM_HEADS
            ),
        )

        for key_padding_mask, attention_mask in itertools.product(
            key_padding_mask_options, attention_mask_options
        ):
            with self.subTest(
                padding=key_padding_mask is not None,
                attention=attention_mask is not None,
            ):
                output, loss = model(
                    target_token_embeddings=target,
                    encoder_output=encoder_output,
                    target_key_padding_mask=key_padding_mask,
                    encoder_key_padding_mask=key_padding_mask,
                    attention_mask=attention_mask,
                    encoder_attention_mask=attention_mask,
                )
                self.assertEqual(
                    output.shape, (SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)
                )
                self.assertIsInstance(loss, torch.Tensor)

    def test_forward_decoder_only_without_encoder_output(self):
        model = TransformerDecoderStack(decoder_stack_config(cross_attention=False))
        output, loss = model(target_token_embeddings=embeddings())

        self.assertEqual(output.shape, (SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM))
        self.assertIsInstance(loss, torch.Tensor)

    def test_cross_attention_requires_encoder_output(self):
        model = TransformerDecoderStack(decoder_stack_config(cross_attention=True))
        with self.assertRaises(ValueError):
            model(target_token_embeddings=embeddings(), encoder_output=None)

    def test_forward_with_all_option_combinations(self):
        sequence_lengths = (6, 10)

        for self_attention_config_class in ATTENTION_CONFIG_CLASSES:
            for cross_attention_config_class in CROSS_ATTENTION_CONFIG_CLASSES:
                for target_sequence_length in sequence_lengths:
                    for source_sequence_length in sequence_lengths:
                        self._assert_decoder_combination(
                            self_attention_config_class=self_attention_config_class,
                            cross_attention_config_class=cross_attention_config_class,
                            target_sequence_length=target_sequence_length,
                            source_sequence_length=source_sequence_length,
                        )

    def _assert_decoder_combination(
        self,
        self_attention_config_class,
        cross_attention_config_class,
        target_sequence_length,
        source_sequence_length,
    ):
        config = self.preset(
            target_sequence_length=target_sequence_length,
            source_sequence_length=source_sequence_length,
            self_attention_config_class=self_attention_config_class,
            cross_attention_config_class=cross_attention_config_class,
        )
        description = dict(
            self_attention_type=self_attention_config_class.__name__,
            cross_attention_type=cross_attention_config_class.__name__,
            target_sequence_length=target_sequence_length,
            source_sequence_length=source_sequence_length,
        )

        model = TransformerDecoderStack(config)
        target = torch.randn(target_sequence_length, BATCH_SIZE, EMBEDDING_DIM)
        encoder_output = torch.randn(source_sequence_length, BATCH_SIZE, EMBEDDING_DIM)
        target_key_padding_mask_options = (
            None,
            create_key_padding_mask(BATCH_SIZE, target_sequence_length),
        )
        encoder_key_padding_mask_options = (
            None,
            create_key_padding_mask(BATCH_SIZE, source_sequence_length),
        )
        attention_mask_options = (
            None,
            create_attention_mask(
                target_sequence_length,
                target_sequence_length,
                attention_mask_repeat_for(self_attention_config_class),
            ),
        )
        encoder_attention_mask_options = (
            None,
            create_attention_mask(
                target_sequence_length,
                source_sequence_length,
                attention_mask_repeat_for(cross_attention_config_class),
            ),
        )

        for (
            target_key_padding_mask,
            encoder_key_padding_mask,
            attention_mask,
            encoder_attention_mask,
        ) in itertools.product(
            target_key_padding_mask_options,
            encoder_key_padding_mask_options,
            attention_mask_options,
            encoder_attention_mask_options,
        ):
            with self.subTest(
                target_padding=target_key_padding_mask is not None,
                encoder_padding=encoder_key_padding_mask is not None,
                attention=attention_mask is not None,
                encoder_attention=encoder_attention_mask is not None,
                **description,
            ):
                output, loss = model(
                    target_token_embeddings=target,
                    encoder_output=encoder_output,
                    target_key_padding_mask=target_key_padding_mask,
                    encoder_key_padding_mask=encoder_key_padding_mask,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
                self.assertEqual(
                    output.shape,
                    (target_sequence_length, BATCH_SIZE, EMBEDDING_DIM),
                )
                self.assertIsInstance(loss, torch.Tensor)

    def test_self_attention_rejects_mismatched_projection_dims(self):
        for query_key_projection_dim, value_projection_dim in (
            (16, 0),
            (0, 16),
            (16, 20),
        ):
            with self.subTest(
                query_key_projection_dim=query_key_projection_dim,
                value_projection_dim=value_projection_dim,
            ):
                config = self.preset(
                    self_attention_config_class=SelfAttentionConfig,
                    query_key_projection_dim=query_key_projection_dim,
                    value_projection_dim=value_projection_dim,
                )
                with self.assertRaises(RuntimeError):
                    TransformerDecoderStack(config)

    def test_forward_with_feed_forward_variants(self):
        for feed_forward_kind in FEED_FORWARD_KINDS:
            with self.subTest(feed_forward_kind=feed_forward_kind):
                model = TransformerDecoderStack(
                    self.preset(feed_forward_kind=feed_forward_kind)
                )
                output, loss = model(
                    target_token_embeddings=embeddings(),
                    encoder_output=embeddings(),
                )
                self.assertEqual(
                    output.shape, (SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)
                )
                self.assertIsInstance(loss, torch.Tensor)


class TestTransformer(unittest.TestCase):
    def preset(
        self,
        num_layers=NUM_LAYERS,
        embedding_dim=EMBEDDING_DIM,
        batch_size=BATCH_SIZE,
        num_heads=NUM_HEADS,
        target_sequence_length=SEQUENCE_LENGTH,
        source_sequence_length=SEQUENCE_LENGTH,
        self_attention_config_class=SelfAttentionConfig,
        cross_attention_config_class=IndependentAttentionConfig,
        query_key_projection_dim=0,
        value_projection_dim=0,
        add_key_value_bias_flag=False,
        zero_attention_flag=False,
        average_attention_weights_flag=False,
        causal_attention_mask_flag=False,
        projection_kind="base",
        feed_forward_kind="base",
        encoder_stack_config=None,
        decoder_stack_config=None,
        encoder_stack_enabled=True,
        decoder_stack_enabled=True,
        decoder_cross_attention=True,
    ):
        if encoder_stack_config is None and encoder_stack_enabled:
            encoder_stack_config = make_encoder_stack_config(
                causal_attention_mask_flag=causal_attention_mask_flag,
                num_layers=num_layers,
                embedding_dim=embedding_dim,
                batch_size=batch_size,
                num_heads=num_heads,
                sequence_length=source_sequence_length,
                attention_config_class=self_attention_config_class,
                query_key_projection_dim=query_key_projection_dim,
                value_projection_dim=value_projection_dim,
                add_key_value_bias_flag=add_key_value_bias_flag,
                zero_attention_flag=zero_attention_flag,
                average_attention_weights_flag=average_attention_weights_flag,
                projection_kind=projection_kind,
                feed_forward_kind=feed_forward_kind,
            )

        if decoder_stack_config is None and decoder_stack_enabled:
            decoder_stack_config = make_decoder_stack_config(
                cross_attention=decoder_cross_attention,
                causal_attention_mask_flag=causal_attention_mask_flag,
                num_layers=num_layers,
                embedding_dim=embedding_dim,
                batch_size=batch_size,
                num_heads=num_heads,
                target_sequence_length=target_sequence_length,
                source_sequence_length=source_sequence_length,
                self_attention_config_class=self_attention_config_class,
                cross_attention_config_class=cross_attention_config_class,
                query_key_projection_dim=query_key_projection_dim,
                value_projection_dim=value_projection_dim,
                add_key_value_bias_flag=add_key_value_bias_flag,
                zero_attention_flag=zero_attention_flag,
                average_attention_weights_flag=average_attention_weights_flag,
                projection_kind=projection_kind,
                feed_forward_kind=feed_forward_kind,
            )

        return TransformerConfig(
            encoder_stack_config=encoder_stack_config,
            decoder_stack_config=decoder_stack_config,
        )

    def test_init_encoder_and_decoder(self):
        model = Transformer(self.preset())
        self.assertIsInstance(model.encoder_model, TransformerEncoderStack)
        self.assertIsInstance(model.decoder_model, TransformerDecoderStack)

    def test_config_build_returns_transformer(self):
        config = self.preset()
        model = config.build()

        self.assertIsInstance(model, Transformer)
        self.assertIsInstance(model, config._registry_owner())
        self.assertIsInstance(model.encoder_model, TransformerEncoderStack)
        self.assertIsInstance(model.decoder_model, TransformerDecoderStack)

    def test_config_build_applies_overrides(self):
        config = self.preset(decoder_stack_enabled=False)
        decoder_config = make_decoder_stack_config(cross_attention=False)
        overrides = self.preset(
            encoder_stack_enabled=False,
            decoder_stack_config=decoder_config,
        )
        model = config.build(overrides)

        self.assertIsInstance(model, Transformer)
        self.assertIsInstance(model.encoder_model, TransformerEncoderStack)
        self.assertIsInstance(model.decoder_model, TransformerDecoderStack)
        self.assertIs(model.decoder_model.cfg, decoder_config)
        self.assertIsNone(model.decoder_model.layers[0].cross_attention_model)

    def test_partial_overrides_keep_unset_transformer_fields(self):
        encoder_config = make_encoder_stack_config(num_layers=1)
        decoder_config = make_decoder_stack_config(cross_attention=False, num_layers=3)
        config = self.preset(
            encoder_stack_config=encoder_config,
            decoder_stack_config=make_decoder_stack_config(cross_attention=False),
        )
        overrides = self.preset(
            encoder_stack_enabled=False,
            decoder_stack_config=decoder_config,
        )
        model = config.build(overrides)

        self.assertIsInstance(model, Transformer)
        self.assertIsInstance(model.encoder_model, TransformerEncoderStack)
        self.assertIsInstance(model.decoder_model, TransformerDecoderStack)
        self.assertEqual(model.encoder_model.num_layers, encoder_config.num_layers)
        self.assertEqual(model.decoder_model.num_layers, decoder_config.num_layers)

    def test_encoder_only(self):
        model = Transformer(self.preset(decoder_stack_enabled=False))
        self.assertIsNone(model.decoder_model)

        output, loss = model(source_token_embeddings=embeddings())
        self.assertEqual(output.shape, (SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM))
        self.assertIsInstance(loss, torch.Tensor)

    def test_decoder_only(self):
        model = Transformer(
            self.preset(
                encoder_stack_enabled=False,
                decoder_cross_attention=False,
            )
        )
        self.assertIsNone(model.encoder_model)

        output, loss = model(target_token_embeddings=embeddings())
        self.assertEqual(output.shape, (SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM))
        self.assertIsInstance(loss, torch.Tensor)

    def test_encoder_and_decoder(self):
        model = Transformer(self.preset())
        output, loss = model(
            source_token_embeddings=embeddings(),
            target_token_embeddings=embeddings(),
        )
        self.assertEqual(output.shape, (SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM))
        self.assertIsInstance(loss, torch.Tensor)

    def test_requires_at_least_one_stack(self):
        with self.assertRaises(ValueError):
            Transformer(
                self.preset(
                    encoder_stack_enabled=False,
                    decoder_stack_enabled=False,
                )
            )

    def test_decoder_only_with_cross_attention_is_rejected(self):
        with self.assertRaises(ValueError):
            Transformer(
                self.preset(
                    encoder_stack_enabled=False,
                    decoder_cross_attention=True,
                )
            )

    def test_encoder_requires_source_embeddings(self):
        model = Transformer(self.preset(decoder_stack_enabled=False))
        with self.assertRaises(ValueError):
            model(source_token_embeddings=None)

    def test_decoder_requires_target_embeddings(self):
        model = Transformer(
            self.preset(
                encoder_stack_enabled=False,
                decoder_cross_attention=False,
            )
        )
        with self.assertRaises(ValueError):
            model(target_token_embeddings=None)

    def test_all_possible_inputs(self):
        sequence_lengths = (6, 10)

        for self_attention_config_class in ATTENTION_CONFIG_CLASSES:
            for cross_attention_config_class in CROSS_ATTENTION_CONFIG_CLASSES:
                for target_sequence_length in sequence_lengths:
                    for source_sequence_length in sequence_lengths:
                        self._assert_transformer_combination(
                            self_attention_config_class=self_attention_config_class,
                            cross_attention_config_class=cross_attention_config_class,
                            target_sequence_length=target_sequence_length,
                            source_sequence_length=source_sequence_length,
                        )

    def _assert_transformer_combination(
        self,
        self_attention_config_class,
        cross_attention_config_class,
        target_sequence_length,
        source_sequence_length,
    ):
        config = self.preset(
            target_sequence_length=target_sequence_length,
            source_sequence_length=source_sequence_length,
            self_attention_config_class=self_attention_config_class,
            cross_attention_config_class=cross_attention_config_class,
        )
        description = dict(
            self_attention_type=self_attention_config_class.__name__,
            cross_attention_type=cross_attention_config_class.__name__,
            target_sequence_length=target_sequence_length,
            source_sequence_length=source_sequence_length,
        )

        model = Transformer(config)
        source = torch.randn(source_sequence_length, BATCH_SIZE, EMBEDDING_DIM)
        target = torch.randn(target_sequence_length, BATCH_SIZE, EMBEDDING_DIM)
        source_key_padding_mask_options = (
            None,
            create_key_padding_mask(BATCH_SIZE, source_sequence_length),
        )
        target_key_padding_mask_options = (
            None,
            create_key_padding_mask(BATCH_SIZE, target_sequence_length),
        )
        encoder_key_padding_mask_options = (
            None,
            create_key_padding_mask(BATCH_SIZE, source_sequence_length),
        )
        source_attention_mask_options = (
            None,
            create_attention_mask(
                source_sequence_length,
                source_sequence_length,
                attention_mask_repeat_for(self_attention_config_class),
            ),
        )
        target_attention_mask_options = (
            None,
            create_attention_mask(
                target_sequence_length,
                target_sequence_length,
                attention_mask_repeat_for(self_attention_config_class),
            ),
        )
        encoder_attention_mask_options = (
            None,
            create_attention_mask(
                target_sequence_length,
                source_sequence_length,
                attention_mask_repeat_for(cross_attention_config_class),
            ),
        )

        for (
            source_key_padding_mask,
            target_key_padding_mask,
            encoder_key_padding_mask,
            source_attention_mask,
            target_attention_mask,
            encoder_attention_mask,
        ) in itertools.product(
            source_key_padding_mask_options,
            target_key_padding_mask_options,
            encoder_key_padding_mask_options,
            source_attention_mask_options,
            target_attention_mask_options,
            encoder_attention_mask_options,
        ):
            with self.subTest(
                source_padding=source_key_padding_mask is not None,
                target_padding=target_key_padding_mask is not None,
                encoder_padding=encoder_key_padding_mask is not None,
                source_attention=source_attention_mask is not None,
                target_attention=target_attention_mask is not None,
                encoder_attention=encoder_attention_mask is not None,
                **description,
            ):
                output, loss = model(
                    source_token_embeddings=source,
                    target_token_embeddings=target,
                    source_attention_mask=source_attention_mask,
                    target_attention_mask=target_attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    source_key_padding_mask=source_key_padding_mask,
                    target_key_padding_mask=target_key_padding_mask,
                    encoder_key_padding_mask=encoder_key_padding_mask,
                )
                self.assertEqual(
                    output.shape,
                    (target_sequence_length, BATCH_SIZE, EMBEDDING_DIM),
                )
                self.assertIsInstance(loss, torch.Tensor)

    def test_self_attention_rejects_mismatched_projection_dims(self):
        for query_key_projection_dim, value_projection_dim in (
            (16, 0),
            (0, 16),
            (16, 20),
        ):
            with self.subTest(
                query_key_projection_dim=query_key_projection_dim,
                value_projection_dim=value_projection_dim,
            ):
                config = self.preset(
                    self_attention_config_class=SelfAttentionConfig,
                    query_key_projection_dim=query_key_projection_dim,
                    value_projection_dim=value_projection_dim,
                )
                with self.assertRaises(RuntimeError):
                    Transformer(config)

    def test_forward_with_feed_forward_variants(self):
        for feed_forward_kind in FEED_FORWARD_KINDS:
            with self.subTest(feed_forward_kind=feed_forward_kind):
                model = Transformer(self.preset(feed_forward_kind=feed_forward_kind))
                output, loss = model(
                    source_token_embeddings=embeddings(),
                    target_token_embeddings=embeddings(),
                )
                self.assertEqual(
                    output.shape, (SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)
                )
                self.assertIsInstance(loss, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
