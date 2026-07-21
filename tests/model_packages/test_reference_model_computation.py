import math
import unittest

import torch
import torch.nn.functional as F

import models.bert.linear.config as bert_config
import models.gpt.linear.config as gpt_config
from models.bert.linear._builder_adapter import (
    linear_builder_kwargs_from_flat as bert_builder_kwargs,
)
from models.bert.linear.config_builder import BertLinearConfigBuilder
from models.bert.linear.model import Model as BertModel
from models.gpt.linear._builder_adapter import (
    linear_builder_kwargs_from_flat as gpt_builder_kwargs,
)
from models.gpt.linear.config_builder import GptLinearConfigBuilder
from models.gpt.linear.model import Model as GptModel
from models.transformer.linear.config_builder import TransformerLinearConfigBuilder
from models.transformer.linear.model import Model as TransformerModel


def _affine(
    parameters: dict[str, torch.Tensor],
    input_tensor: torch.Tensor,
    prefix: str,
) -> torch.Tensor:
    return (
        input_tensor @ parameters[f"{prefix}.weight_params"]
        + parameters[f"{prefix}.bias_params"]
    )


def _layer_norm(
    parameters: dict[str, torch.Tensor],
    input_tensor: torch.Tensor,
    prefix: str,
    hidden_dim: int,
) -> torch.Tensor:
    return F.layer_norm(
        input_tensor,
        (hidden_dim,),
        parameters[f"{prefix}.weight"],
        parameters[f"{prefix}.bias"],
        1e-5,
    )


def _attention(
    parameters: dict[str, torch.Tensor],
    query_input: torch.Tensor,
    key_input: torch.Tensor,
    value_input: torch.Tensor,
    prefix: str,
    hidden_dim: int,
    causal: bool = False,
) -> torch.Tensor:
    projection_prefix = f"{prefix}model.projector."
    query = _affine(
        parameters,
        query_input,
        f"{projection_prefix}query_model.layers.0.model",
    )
    key = _affine(
        parameters,
        key_input,
        f"{projection_prefix}key_model.layers.0.model",
    )
    value = _affine(
        parameters,
        value_input,
        f"{projection_prefix}value_model.layers.0.model",
    )
    scores = query @ key.transpose(-2, -1) / math.sqrt(hidden_dim)
    if causal:
        causal_mask = torch.triu(
            torch.ones(
                scores.shape[-2:],
                dtype=torch.bool,
                device=scores.device,
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
    weighted_values = torch.softmax(scores, dim=-1) @ value
    return _affine(
        parameters,
        weighted_values,
        f"{projection_prefix}output_model.layers.0.model",
    )


def _feed_forward(
    parameters: dict[str, torch.Tensor],
    input_tensor: torch.Tensor,
    prefix: str,
    activation,
) -> torch.Tensor:
    expanded = _affine(
        parameters,
        input_tensor,
        f"{prefix}model.model.layers.0.model",
    )
    activated = activation(expanded)
    return _affine(
        parameters,
        activated,
        f"{prefix}model.model.layers.1.model",
    )


def _sinusoidal_positions(token_ids: torch.Tensor, hidden_dim: int) -> torch.Tensor:
    non_padding = token_ids.ne(0)
    positions = torch.cumsum(non_padding.int(), dim=1) * non_padding.int()
    half_dim = hidden_dim // 2
    frequency_scale = math.log(10000) / (half_dim - 1)
    frequencies = torch.exp(
        torch.arange(half_dim, dtype=torch.float32) * -frequency_scale
    )
    scaled_positions = positions.float().unsqueeze(-1) * frequencies
    embeddings = torch.cat(
        (torch.sin(scaled_positions), torch.cos(scaled_positions)),
        dim=-1,
    )
    return embeddings.masked_fill(~non_padding.unsqueeze(-1), 0.0)


class TestReferenceModelComputation(unittest.TestCase):
    def assert_gradients_match(
        self,
        parameters: dict[str, torch.Tensor],
        actual: torch.Tensor,
        expected: torch.Tensor,
    ) -> None:
        parameter_values = tuple(parameters.values())
        actual_gradients = torch.autograd.grad(
            actual,
            parameter_values,
            retain_graph=True,
            allow_unused=True,
        )
        expected_gradients = torch.autograd.grad(
            expected,
            parameter_values,
            allow_unused=True,
        )
        for name, actual_gradient, expected_gradient in zip(
            parameters,
            actual_gradients,
            expected_gradients,
            strict=True,
        ):
            with self.subTest(parameter=name):
                self.assertEqual(
                    actual_gradient is None,
                    expected_gradient is None,
                )
                if actual_gradient is not None:
                    torch.testing.assert_close(
                        actual_gradient,
                        expected_gradient,
                        atol=3e-5,
                        rtol=3e-5,
                    )

    def test_bert_matches_explicit_bidirectional_reference(self):
        torch.manual_seed(41)
        hidden_dim = 4
        flat_options = {
            "batch_size": 1,
            "input_dim": 8,
            "output_dim": 8,
            "sequence_length": 3,
            "hidden_dim": hidden_dim,
            "stack_num_layers": 1,
            "stack_dropout_probability": 0.0,
            "embedding_dropout_probability": 0.0,
            "attn_num_heads": 1,
            "attn_num_layers": 1,
            "ff_num_layers": 1,
            "ff_stack_hidden_dim": 8,
        }
        config = BertLinearConfigBuilder(
            **bert_builder_kwargs(flat_options, bert_config)
        ).build()
        model = BertModel(config).eval()
        parameters = dict(model.named_parameters())
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.tensor([[0, 1, 0]])

        actual_mlm, actual_nsp, actual_auxiliary_loss = model(
            input_ids,
            attention_mask,
            token_type_ids,
        )

        positions = torch.tensor([[1, 2, 3]])
        hidden = (
            parameters["token_embedding.weight"][input_ids]
            + parameters["token_type_embedding.weight"][token_type_ids]
            + parameters["positional_embedding.embedding_model.weight"][positions]
        )
        hidden = _layer_norm(
            parameters,
            hidden,
            "embedding_layer_norm",
            hidden_dim,
        )
        block_prefix = "transformer.layers.0.model."
        attention_prefix = f"{block_prefix}self_attention_layer."
        attention_output = _attention(
            parameters,
            hidden,
            hidden,
            hidden,
            attention_prefix,
            hidden_dim,
        )
        hidden = _layer_norm(
            parameters,
            hidden + attention_output,
            f"{attention_prefix}layer_norm_module",
            hidden_dim,
        )
        feed_forward_prefix = f"{block_prefix}feed_forward_layer."
        feed_forward_output = _feed_forward(
            parameters,
            hidden,
            feed_forward_prefix,
            F.gelu,
        )
        hidden = _layer_norm(
            parameters,
            hidden + feed_forward_output,
            f"{feed_forward_prefix}layer_norm_module",
            hidden_dim,
        )
        expected_mlm = F.linear(
            hidden,
            parameters["mlm_dense.weight"],
            parameters["mlm_dense.bias"],
        )
        expected_mlm = F.gelu(expected_mlm)
        expected_mlm = _layer_norm(
            parameters,
            expected_mlm,
            "mlm_layer_norm",
            hidden_dim,
        )
        expected_mlm = F.linear(
            expected_mlm,
            parameters["token_embedding.weight"],
            parameters["mlm_decoder_bias"],
        )
        pooled = torch.tanh(
            F.linear(
                hidden[:, 0],
                parameters["pooler.weight"],
                parameters["pooler.bias"],
            )
        )
        expected_nsp = F.linear(
            pooled,
            parameters["nsp_head.weight"],
            parameters["nsp_head.bias"],
        )

        torch.testing.assert_close(actual_mlm, expected_mlm, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(actual_nsp, expected_nsp, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(
            actual_auxiliary_loss,
            actual_auxiliary_loss.new_zeros(()),
        )
        self.assert_gradients_match(
            parameters,
            actual_mlm.sum() + actual_nsp.sum(),
            expected_mlm.sum() + expected_nsp.sum(),
        )

    def test_gpt_matches_explicit_causal_reference(self):
        torch.manual_seed(43)
        hidden_dim = 4
        flat_options = {
            "batch_size": 1,
            "input_dim": 8,
            "output_dim": 8,
            "sequence_length": 3,
            "hidden_dim": hidden_dim,
            "stack_num_layers": 1,
            "stack_dropout_probability": 0.0,
            "embedding_dropout_probability": 0.0,
            "attn_num_heads": 1,
            "attn_num_layers": 1,
            "ff_num_layers": 1,
            "ff_stack_hidden_dim": 8,
        }
        config = GptLinearConfigBuilder(
            **gpt_builder_kwargs(flat_options, gpt_config)
        ).build()
        model = GptModel(config).eval()
        parameters = dict(model.named_parameters())
        input_ids = torch.tensor([[1, 2, 3]])

        actual_logits, actual_auxiliary_loss = model(input_ids)

        positions = torch.tensor([[1, 2, 3]])
        hidden = (
            parameters["token_embedding.weight"][input_ids]
            + parameters["positional_embedding.embedding_model.weight"][positions]
        )
        block_prefix = "transformer.layers.0.model."
        attention_prefix = f"{block_prefix}self_attention_layer."
        normalized = _layer_norm(
            parameters,
            hidden,
            f"{attention_prefix}layer_norm_module",
            hidden_dim,
        )
        hidden = hidden + _attention(
            parameters,
            normalized,
            normalized,
            normalized,
            attention_prefix,
            hidden_dim,
            causal=True,
        )
        feed_forward_prefix = f"{block_prefix}feed_forward_layer."
        normalized = _layer_norm(
            parameters,
            hidden,
            f"{feed_forward_prefix}layer_norm_module",
            hidden_dim,
        )
        hidden = hidden + _feed_forward(
            parameters,
            normalized,
            feed_forward_prefix,
            F.gelu,
        )
        hidden = _layer_norm(
            parameters,
            hidden,
            "decoder_layer_norm",
            hidden_dim,
        )
        expected_logits = F.linear(hidden, parameters["token_embedding.weight"])

        torch.testing.assert_close(actual_logits, expected_logits, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(
            actual_auxiliary_loss,
            actual_auxiliary_loss.new_zeros(()),
        )
        self.assert_gradients_match(
            parameters,
            actual_logits.sum(),
            expected_logits.sum(),
        )

    def test_transformer_matches_explicit_encoder_decoder_reference(self):
        torch.manual_seed(47)
        hidden_dim = 4
        config = TransformerLinearConfigBuilder(
            batch_size=1,
            vocab_size=8,
            model_dim=hidden_dim,
            source_sequence_length=3,
            target_sequence_length=3,
            encoder_num_layers=1,
            decoder_num_layers=1,
            attn_num_heads=1,
            ff_num_layers=1,
            ff_stack_hidden_dim=8,
            dropout_probability=0.0,
        ).build()
        model = TransformerModel(config).eval()
        parameters = dict(model.named_parameters())
        source_ids = torch.tensor([[1, 2, 3]])
        target_ids = torch.tensor([[1, 3, 2]])

        actual_logits, actual_auxiliary_loss = model(source_ids, target_ids)

        embedding_scale = math.sqrt(hidden_dim)
        encoder_hidden = parameters["shared_embedding.weight"][
            source_ids
        ] * embedding_scale + _sinusoidal_positions(source_ids, hidden_dim)
        encoder_prefix = "encoder.layers.0.model."
        encoder_attention_prefix = f"{encoder_prefix}self_attention_layer."
        normalized = _layer_norm(
            parameters,
            encoder_hidden,
            f"{encoder_attention_prefix}layer_norm_module",
            hidden_dim,
        )
        encoder_hidden = encoder_hidden + _attention(
            parameters,
            normalized,
            normalized,
            normalized,
            encoder_attention_prefix,
            hidden_dim,
        )
        encoder_feed_forward_prefix = f"{encoder_prefix}feed_forward_layer."
        normalized = _layer_norm(
            parameters,
            encoder_hidden,
            f"{encoder_feed_forward_prefix}layer_norm_module",
            hidden_dim,
        )
        encoder_hidden = encoder_hidden + _feed_forward(
            parameters,
            normalized,
            encoder_feed_forward_prefix,
            F.relu,
        )
        encoder_hidden = _layer_norm(
            parameters,
            encoder_hidden,
            "encoder_layer_norm",
            hidden_dim,
        )
        decoder_hidden = parameters["shared_embedding.weight"][
            target_ids
        ] * embedding_scale + _sinusoidal_positions(target_ids, hidden_dim)
        decoder_prefix = "decoder.layers.0.model."
        decoder_attention_prefix = f"{decoder_prefix}self_attention_layer."
        normalized = _layer_norm(
            parameters,
            decoder_hidden,
            f"{decoder_attention_prefix}layer_norm_module",
            hidden_dim,
        )
        decoder_hidden = decoder_hidden + _attention(
            parameters,
            normalized,
            normalized,
            normalized,
            decoder_attention_prefix,
            hidden_dim,
            causal=True,
        )
        cross_attention_prefix = f"{decoder_prefix}cross_attention_layer."
        normalized = _layer_norm(
            parameters,
            decoder_hidden,
            f"{cross_attention_prefix}layer_norm_module",
            hidden_dim,
        )
        decoder_hidden = decoder_hidden + _attention(
            parameters,
            normalized,
            encoder_hidden,
            encoder_hidden,
            cross_attention_prefix,
            hidden_dim,
        )
        decoder_feed_forward_prefix = f"{decoder_prefix}feed_forward_layer."
        normalized = _layer_norm(
            parameters,
            decoder_hidden,
            f"{decoder_feed_forward_prefix}layer_norm_module",
            hidden_dim,
        )
        decoder_hidden = decoder_hidden + _feed_forward(
            parameters,
            normalized,
            decoder_feed_forward_prefix,
            F.relu,
        )
        decoder_hidden = _layer_norm(
            parameters,
            decoder_hidden,
            "decoder_layer_norm",
            hidden_dim,
        )
        expected_logits = F.linear(
            decoder_hidden,
            parameters["shared_embedding.weight"],
        )

        torch.testing.assert_close(actual_logits, expected_logits, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(
            actual_auxiliary_loss,
            actual_auxiliary_loss.new_zeros(()),
        )
        self.assert_gradients_match(
            parameters,
            actual_logits.sum(),
            expected_logits.sum(),
        )


if __name__ == "__main__":
    unittest.main()
