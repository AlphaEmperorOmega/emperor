import unittest
from dataclasses import dataclass

import torch

from emperor.attention import (
    IndependentAttentionConfig,
    MixerAttentionConfig,
)
from emperor.attention._base import MultiHeadAttentionAbstract
from emperor.attention._variants.mixer.layer import MixerAttention
from emperor.attention.monitoring import AttentionMonitorCallback
from emperor.config import ConfigBase, optional_field
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    LayerState,
    RecurrentLayer,
    RecurrentLayerConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.nn import Module
from emperor.transformer import (
    FeedForward,
    FeedForwardConfig,
    TransformerDecoderLayer,
    TransformerDecoderLayerConfig,
    TransformerEncoderLayer,
    TransformerEncoderLayerConfig,
)
from support.attention import build_attention_config


def _linear_stack(
    input_dim: int,
    output_dim: int,
    *,
    hidden_dim: int | None = None,
    num_layers: int = 1,
) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=input_dim if hidden_dim is None else hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        apply_output_pipeline_flag=False,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        shared_gate_config=None,
        shared_halting_config=None,
        shared_memory_config=None,
        layer_config=LayerConfig(
            activation=ActivationOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=True),
        ),
    )


def _recurrent_stack(sequence_length: int, *, max_steps: int = 2):
    return RecurrentLayerConfig(
        input_dim=sequence_length,
        output_dim=sequence_length,
        max_steps=max_steps,
        recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
        block_config=_linear_stack(sequence_length, sequence_length),
        gate_config=None,
        residual_config=None,
        halting_config=None,
        memory_config=None,
    )


def _mixer_config(
    *,
    embedding_dim: int = 4,
    sequence_length: int = 3,
    batch_first_flag: bool = True,
    causal_attention_mask_flag: bool = False,
    mixing_model_config: LayerStackConfig | RecurrentLayerConfig | None = None,
) -> MixerAttentionConfig:
    if mixing_model_config is None:
        mixing_model_config = _linear_stack(sequence_length, sequence_length)
    return MixerAttentionConfig(
        embedding_dim=embedding_dim,
        sequence_length=sequence_length,
        batch_first_flag=batch_first_flag,
        causal_attention_mask_flag=causal_attention_mask_flag,
        mixing_model_config=mixing_model_config,
    )


def _feed_forward_config(embedding_dim: int) -> FeedForwardConfig:
    return FeedForwardConfig(
        input_dim=embedding_dim,
        output_dim=embedding_dim,
        stack_config=_linear_stack(embedding_dim, embedding_dim),
    )


def _cross_attention_config(
    *,
    embedding_dim: int,
    batch_size: int,
    target_sequence_length: int,
    source_sequence_length: int,
) -> IndependentAttentionConfig:
    config = build_attention_config(
        config_class=IndependentAttentionConfig,
        batch_size=batch_size,
        num_heads=2,
        embedding_dim=embedding_dim,
        target_sequence_length=target_sequence_length,
        source_sequence_length=source_sequence_length,
    )
    config.batch_first_flag = True
    return config


def _set_linear_parameters(
    model: MixerAttention,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> None:
    linear = model.mixing_model.layers[0].model
    with torch.no_grad():
        linear.weight_params.copy_(weight)
        linear.bias_params.copy_(bias)


def _direct_token_axis_reference(
    values: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    batch_first_flag: bool,
) -> torch.Tensor:
    sequence_axis = 1 if batch_first_flag else 0
    token_vectors = values.movedim(sequence_axis, -1)
    leading_shape = token_vectors.shape[:-1]
    flattened = token_vectors.reshape(-1, token_vectors.size(-1))
    mixed = flattened @ weight + bias
    return mixed.reshape(*leading_shape, token_vectors.size(-1)).movedim(
        -1,
        sequence_axis,
    )


@dataclass
class _LossBlockConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")

    def _registry_owner(self) -> type:
        return _LossBlock


class _LossBlock(Module):
    def __init__(
        self,
        cfg: _LossBlockConfig,
        overrides: _LossBlockConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)

    def forward(self, state: LayerState) -> LayerState:
        contribution = state.hidden.new_tensor(0.25)
        state.hidden = state.hidden + 1.0
        state.loss = contribution if state.loss is None else state.loss + contribution
        return state


class _WrongShapeModel(torch.nn.Module):
    def forward(self, state: LayerState) -> LayerState:
        state.hidden = state.hidden[..., :-1]
        return state


class _InvalidNestedResultModel(torch.nn.Module):
    def __init__(self, failure: str) -> None:
        super().__init__()
        self.failure = failure

    def forward(self, state: LayerState):
        if self.failure == "state":
            return state.hidden
        if self.failure == "hidden":
            state.hidden = object()
        elif self.failure == "dtype":
            state.hidden = state.hidden.to(dtype=torch.float64)
        elif self.failure == "device":
            state.hidden = torch.empty_like(state.hidden, device="meta")
        return state


class _MonitorHost(torch.nn.Module):
    def __init__(self, attention: MixerAttention) -> None:
        super().__init__()
        self.attention = attention
        self.global_step = 0
        self.logger = None
        self.logged: dict[str, torch.Tensor] = {}

    def log(self, name: str, value: torch.Tensor) -> None:
        self.logged[name] = value


class TestMixerAttention(unittest.TestCase):
    def test_config_builds_registered_attention_variant_and_exact_inner_dims(self):
        config = _mixer_config(sequence_length=3)

        model = config.build()

        self.assertIsInstance(model, MixerAttention)
        self.assertIsInstance(model, MultiHeadAttentionAbstract)
        self.assertIs(config._registry_owner(), MixerAttention)
        self.assertEqual(model.mixing_model.input_dim, 3)
        self.assertEqual(model.mixing_model.output_dim, 3)
        self.assertFalse(hasattr(model, "projector"))
        self.assertFalse(hasattr(model, "processor"))

    def test_batch_and_sequence_first_match_direct_token_axis_reference(self):
        weight = torch.tensor(
            [
                [1.0, 0.5, -0.25],
                [-0.5, 2.0, 0.75],
                [0.25, -1.0, 1.5],
            ]
        )
        bias = torch.tensor([0.2, -0.3, 0.4])
        batch_first_values = torch.tensor(
            [
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
                [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
            ]
        )
        batch_model = _mixer_config(embedding_dim=2).build()
        sequence_model = _mixer_config(
            embedding_dim=2,
            batch_first_flag=False,
        ).build()
        _set_linear_parameters(batch_model, weight, bias)
        sequence_model.load_state_dict(batch_model.state_dict(), strict=True)

        batch_output, batch_weights, batch_loss = batch_model(
            batch_first_values,
            batch_first_values,
            batch_first_values,
        )
        sequence_values = batch_first_values.transpose(0, 1)
        sequence_output, sequence_weights, sequence_loss = sequence_model(
            sequence_values,
            sequence_values,
            sequence_values,
        )

        expected = _direct_token_axis_reference(
            batch_first_values,
            weight,
            bias,
            batch_first_flag=True,
        )
        torch.testing.assert_close(batch_output, expected)
        torch.testing.assert_close(sequence_output.transpose(0, 1), expected)
        self.assertIsNone(batch_weights)
        self.assertIsNone(sequence_weights)
        self.assertIsNone(batch_loss)
        self.assertIsNone(sequence_loss)

    def test_flattens_batch_and_channels_and_shares_model_without_batch_leakage(self):
        model = _mixer_config(embedding_dim=2).build()
        values = torch.tensor(
            [
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
            ]
        )
        captured_inputs: list[torch.Tensor] = []

        def capture_flattened_input(_module, args) -> None:
            captured_inputs.append(args[0].hidden.detach().clone())

        handle = model.mixing_model.register_forward_pre_hook(capture_flattened_input)
        output, _, _ = model(values, values, values)
        handle.remove()

        expected_flattened = values.movedim(1, -1).reshape(4, 3)
        torch.testing.assert_close(captured_inputs[0], expected_flattened)
        torch.testing.assert_close(output[..., 0], output[..., 1])

        changed = values.clone()
        changed[0] += 100.0
        changed_output, _, _ = model(changed, changed, changed)
        torch.testing.assert_close(changed_output[1], output[1])
        self.assertFalse(torch.allclose(changed_output[0], output[0]))

    def test_non_contiguous_input_preserves_shape_dtype_device_and_gradients(self):
        model = _mixer_config(embedding_dim=2).build().to(dtype=torch.float64)
        contiguous_source = torch.randn(2, 2, 3, dtype=torch.float64)
        values = contiguous_source.transpose(1, 2).requires_grad_()
        self.assertFalse(values.is_contiguous())
        linear = model.mixing_model.layers[0].model

        output, weights, loss = model(values, values, values)
        output.square().sum().backward()

        self.assertEqual(output.shape, values.shape)
        self.assertEqual(output.dtype, values.dtype)
        self.assertEqual(output.device, values.device)
        self.assertIsNone(weights)
        self.assertIsNone(loss)
        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertGreater(torch.count_nonzero(values.grad).item(), 0)
        self.assertIsNotNone(linear.weight_params.grad)
        self.assertGreater(torch.count_nonzero(linear.weight_params.grad).item(), 0)

    def test_state_dict_loads_strictly_and_preserves_outputs(self):
        torch.manual_seed(112)
        source = _mixer_config().build().eval()
        restored = _mixer_config().build().eval()
        values = torch.randn(2, 3, 4)

        restored.load_state_dict(source.state_dict(), strict=True)
        source_output, _, _ = source(values, values, values)
        restored_output, _, _ = restored(values, values, values)

        torch.testing.assert_close(restored_output, source_output)

    def test_recurrent_mixing_reuses_one_model_for_each_step(self):
        model = _mixer_config(
            embedding_dim=2,
            mixing_model_config=_recurrent_stack(3),
        ).build()
        self.assertIsInstance(model.mixing_model, RecurrentLayer)
        weight = torch.diag(torch.tensor([0.5, 1.5, 2.0]))
        linear = model.mixing_model.block_model.layers[0].model
        with torch.no_grad():
            linear.weight_params.copy_(weight)
            linear.bias_params.zero_()
        values = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])

        output, _, loss = model(values, values, values)

        expected = _direct_token_axis_reference(
            _direct_token_axis_reference(
                values,
                weight,
                torch.zeros(3),
                batch_first_flag=True,
            ),
            weight,
            torch.zeros(3),
            batch_first_flag=True,
        )
        torch.testing.assert_close(output, expected)
        self.assertIsNone(loss)

    def test_recurrent_layer_state_auxiliary_loss_is_returned_unchanged(self):
        recurrent_config = RecurrentLayerConfig(
            input_dim=3,
            output_dim=3,
            max_steps=2,
            recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
            block_config=_LossBlockConfig(input_dim=3, output_dim=3),
            gate_config=None,
            residual_config=None,
            halting_config=None,
            memory_config=None,
        )
        model = _mixer_config(
            embedding_dim=2,
            mixing_model_config=recurrent_config,
        ).build()
        values = torch.zeros(1, 3, 2)

        output, weights, loss = model(values, values, values)

        torch.testing.assert_close(output, torch.full_like(values, 2.0))
        torch.testing.assert_close(loss, torch.tensor(0.5))
        self.assertIsNone(weights)

    def test_rejects_invalid_configuration(self):
        cases = (
            (
                MixerAttentionConfig(
                    embedding_dim=0,
                    sequence_length=3,
                    batch_first_flag=True,
                    causal_attention_mask_flag=False,
                    mixing_model_config=_linear_stack(3, 3),
                ),
                ValueError,
                "embedding_dim must be greater than 0",
            ),
            (
                MixerAttentionConfig(
                    embedding_dim=2,
                    sequence_length=0,
                    batch_first_flag=True,
                    causal_attention_mask_flag=False,
                    mixing_model_config=_linear_stack(3, 3),
                ),
                ValueError,
                "sequence_length must be greater than 0",
            ),
            (
                MixerAttentionConfig(
                    embedding_dim=2,
                    sequence_length=3,
                    batch_first_flag=None,
                    causal_attention_mask_flag=False,
                    mixing_model_config=_linear_stack(3, 3),
                ),
                ValueError,
                "batch_first_flag is required",
            ),
            (
                _mixer_config(causal_attention_mask_flag=True),
                ValueError,
                "causal_attention_mask_flag must be False",
            ),
            (
                MixerAttentionConfig(
                    embedding_dim=2,
                    sequence_length=3,
                    batch_first_flag=True,
                    causal_attention_mask_flag=False,
                    mixing_model_config=LinearLayerConfig(
                        input_dim=3,
                        output_dim=3,
                        bias_flag=True,
                    ),
                ),
                TypeError,
                "mixing_model_config must be a LayerStackConfig or "
                "RecurrentLayerConfig",
            ),
        )

        for config, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, message):
                    config.build()

    def test_rejects_wrong_runtime_shape_type_and_empty_dimensions(self):
        model = _mixer_config(embedding_dim=2).build()
        cases = (
            (torch.randn(2, 4, 2), RuntimeError, "sequence length"),
            (torch.randn(2, 3, 4), RuntimeError, "embedding width"),
            (torch.randn(3, 2), RuntimeError, "rank three"),
            (torch.ones(2, 3, 2, dtype=torch.int64), RuntimeError, "floating point"),
            (torch.empty(0, 3, 2), RuntimeError, "must be non-empty"),
            (object(), TypeError, "must be a Tensor"),
        )

        for values, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, message):
                    model(values, values, values)

    def test_rejects_non_self_masks_and_static_inputs(self):
        model = _mixer_config(embedding_dim=2).build()
        values = torch.randn(2, 3, 2)
        clone = values.clone()
        with self.assertRaisesRegex(RuntimeError, "same tensor object"):
            model(values, clone, values)

        unsupported_cases = (
            ("k_padding_mask", torch.zeros(2, 3, dtype=torch.bool), "padding"),
            ("attention_mask", torch.zeros(3, 3), "attention masks"),
            ("static_k", torch.zeros(2, 3, 2), "static key/value"),
            ("static_v", torch.zeros(2, 3, 2), "static key/value"),
        )
        for argument, unsupported_value, message in unsupported_cases:
            with self.subTest(argument=argument):
                with self.assertRaisesRegex(RuntimeError, message):
                    model(values, values, values, **{argument: unsupported_value})

    def test_rejects_non_shape_preserving_nested_output(self):
        model = _mixer_config(embedding_dim=2).build()
        model.mixing_model = _WrongShapeModel()
        values = torch.randn(2, 3, 2)

        with self.assertRaisesRegex(RuntimeError, "preserve flattened shape"):
            model(values, values, values)

    def test_rejects_invalid_nested_state_hidden_dtype_and_device(self):
        values = torch.randn(2, 3, 2)
        cases = (
            ("state", "must return a LayerState"),
            ("hidden", "LayerState.hidden must be a Tensor"),
            ("dtype", "must preserve dtype"),
            ("device", "must preserve device"),
        )

        for failure, message in cases:
            with self.subTest(failure=failure):
                model = _mixer_config(embedding_dim=2).build()
                model.mixing_model = _InvalidNestedResultModel(failure)
                with self.assertRaisesRegex(RuntimeError, message):
                    model(values, values, values)

    def test_attention_monitor_handles_no_projectors_or_weights(self):
        model = _mixer_config(embedding_dim=2).build()
        host = _MonitorHost(model)
        monitor = AttentionMonitorCallback(log_every_n_steps=1)
        values = torch.randn(2, 3, 2)

        monitor.on_fit_start(object(), host)
        model(values, values, values)

        self.assertEqual(monitor._tracker_manager.module_names, ("attention",))
        self.assertEqual(monitor._tracker_manager.replacement_count, 0)
        self.assertIn("attention/attention/output_norm", host.logged)
        self.assertNotIn("attention/attention/q_norm_mean", host.logged)
        self.assertNotIn("attention/attention/entropy_mean", host.logged)
        monitor.on_fit_end(object(), host)


class TestTransformerMixerAttention(unittest.TestCase):
    def test_encoder_and_decoder_self_processing_reject_cross_attention_config(self):
        independent_config = _cross_attention_config(
            embedding_dim=4,
            batch_size=2,
            target_sequence_length=3,
            source_sequence_length=3,
        )
        encoder_config = TransformerEncoderLayerConfig(
            embedding_dim=4,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            dropout_probability=0.0,
            residual_config=None,
            attention_config=independent_config,
            feed_forward_config=_feed_forward_config(4),
        )
        decoder_config = TransformerDecoderLayerConfig(
            embedding_dim=4,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            dropout_probability=0.0,
            residual_config=None,
            self_attention_config=independent_config,
            cross_attention_config=None,
            feed_forward_config=_feed_forward_config(4),
        )

        with self.assertRaisesRegex(TypeError, "attention_config must be"):
            TransformerEncoderLayer(encoder_config)
        with self.assertRaisesRegex(TypeError, "self_attention_config must be"):
            TransformerDecoderLayer(decoder_config)

    def test_encoder_uses_mixer_and_existing_feed_forward_branch(self):
        recurrent_config = RecurrentLayerConfig(
            input_dim=3,
            output_dim=3,
            max_steps=2,
            recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
            block_config=_LossBlockConfig(input_dim=3, output_dim=3),
            gate_config=None,
            residual_config=None,
            halting_config=None,
            memory_config=None,
        )
        config = TransformerEncoderLayerConfig(
            embedding_dim=4,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            dropout_probability=0.0,
            residual_config=None,
            attention_config=_mixer_config(
                embedding_dim=4,
                mixing_model_config=recurrent_config,
            ),
            feed_forward_config=_feed_forward_config(4),
        )
        model = TransformerEncoderLayer(config)
        values = torch.randn(2, 3, 4, requires_grad=True)

        output, loss = model(values)
        (output.square().mean() + loss).backward()

        self.assertIsInstance(model.self_attention_model, MixerAttention)
        self.assertIsInstance(model.feed_forward_model, FeedForward)
        self.assertEqual(
            type(model.self_attention_layer).__name__,
            "_EncoderSelfAttentionLayer",
        )
        self.assertEqual(output.shape, values.shape)
        torch.testing.assert_close(loss, torch.tensor(0.5))
        self.assertIsNotNone(values.grad)

    def test_non_causal_decoder_self_mixing_without_cross_attention(self):
        config = TransformerDecoderLayerConfig(
            embedding_dim=4,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            dropout_probability=0.0,
            residual_config=None,
            self_attention_config=_mixer_config(embedding_dim=4),
            cross_attention_config=None,
            feed_forward_config=_feed_forward_config(4),
        )
        model = TransformerDecoderLayer(config)
        target = torch.randn(2, 3, 4)

        output, loss = model(target)

        self.assertIsInstance(model.self_attention_model, MixerAttention)
        self.assertIsNone(model.cross_attention_model)
        self.assertEqual(
            type(model.self_attention_layer).__name__,
            "_DecoderSelfAttentionLayer",
        )
        self.assertEqual(output.shape, target.shape)
        self.assertEqual(loss.shape, ())

    def test_decoder_self_mixing_coexists_with_ordinary_cross_attention(self):
        config = TransformerDecoderLayerConfig(
            embedding_dim=4,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            dropout_probability=0.0,
            residual_config=None,
            self_attention_config=_mixer_config(embedding_dim=4),
            cross_attention_config=_cross_attention_config(
                embedding_dim=4,
                batch_size=2,
                target_sequence_length=3,
                source_sequence_length=5,
            ),
            feed_forward_config=_feed_forward_config(4),
        )
        model = TransformerDecoderLayer(config)
        target = torch.randn(2, 3, 4, requires_grad=True)
        encoder_output = torch.randn(2, 5, 4, requires_grad=True)

        output, loss = model(target, encoder_output=encoder_output)
        (output.square().mean() + loss).backward()

        self.assertIsInstance(model.self_attention_model, MixerAttention)
        self.assertIsNotNone(model.cross_attention_model)
        self.assertEqual(output.shape, target.shape)
        self.assertIsNotNone(target.grad)
        self.assertIsNotNone(encoder_output.grad)

    def test_mixer_is_rejected_as_decoder_cross_attention(self):
        config = TransformerDecoderLayerConfig(
            embedding_dim=4,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            dropout_probability=0.0,
            residual_config=None,
            self_attention_config=_mixer_config(embedding_dim=4),
            cross_attention_config=_mixer_config(embedding_dim=4),
            feed_forward_config=_feed_forward_config(4),
        )

        with self.assertRaisesRegex(
            TypeError,
            "MixerAttentionConfig is self-processing only",
        ):
            TransformerDecoderLayer(config)


if __name__ == "__main__":
    unittest.main()
