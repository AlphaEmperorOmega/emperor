import unittest

import torch

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


def _optimize_fixed_batch(model, batch, steps: int) -> tuple[float, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    initial_loss = float(model._model_step(batch).detach())
    for _ in range(steps):
        optimizer.zero_grad()
        loss = model._model_step(batch)
        loss.backward()
        optimizer.step()
    final_loss = float(model._model_step(batch).detach())
    return initial_loss, final_loss


class TestModelLearningAcceptance(unittest.TestCase):
    def test_bert_overfits_a_fixed_pretraining_batch(self):
        torch.manual_seed(103)
        flat_options = {
            "batch_size": 2,
            "input_dim": 8,
            "output_dim": 8,
            "sequence_length": 3,
            "hidden_dim": 8,
            "stack_num_layers": 1,
            "stack_dropout_probability": 0.0,
            "embedding_dropout_probability": 0.0,
            "attn_num_heads": 2,
            "attn_num_layers": 1,
            "ff_num_layers": 1,
            "ff_stack_hidden_dim": 16,
        }
        config = BertLinearConfigBuilder(
            **bert_builder_kwargs(flat_options, bert_config)
        ).build()
        model = BertModel(config)
        input_ids = torch.tensor([[1, 6, 3], [1, 6, 2]])
        labels = torch.tensor([[-100, 2, -100], [-100, 3, -100]])
        batch = (
            input_ids,
            labels,
            torch.ones_like(input_ids),
            torch.tensor([[0, 0, 0], [1, 1, 1]]),
            torch.tensor([0, 1]),
        )

        initial_loss, final_loss = _optimize_fixed_batch(model, batch, 40)

        self.assertLess(final_loss, initial_loss * 0.2)

    def test_gpt_overfits_a_fixed_next_token_batch(self):
        torch.manual_seed(101)
        flat_options = {
            "batch_size": 2,
            "input_dim": 8,
            "output_dim": 8,
            "sequence_length": 3,
            "hidden_dim": 8,
            "stack_num_layers": 1,
            "stack_dropout_probability": 0.0,
            "embedding_dropout_probability": 0.0,
            "attn_num_heads": 2,
            "attn_num_layers": 1,
            "ff_num_layers": 1,
            "ff_stack_hidden_dim": 16,
        }
        config = GptLinearConfigBuilder(
            **gpt_builder_kwargs(flat_options, gpt_config)
        ).build()
        model = GptModel(config)
        batch = (
            torch.tensor([[1, 2, 3], [1, 3, 2]]),
            torch.tensor([[2, 3, 4], [3, 2, 4]]),
        )

        initial_loss, final_loss = _optimize_fixed_batch(model, batch, 40)

        self.assertLess(final_loss, initial_loss * 0.2)

    def test_transformer_overfits_a_fixed_copy_batch(self):
        torch.manual_seed(107)
        config = TransformerLinearConfigBuilder(
            batch_size=2,
            vocab_size=8,
            model_dim=8,
            source_sequence_length=4,
            target_sequence_length=4,
            encoder_num_layers=1,
            decoder_num_layers=1,
            attn_num_heads=2,
            ff_num_layers=1,
            ff_stack_hidden_dim=16,
            dropout_probability=0.0,
        ).build()
        model = TransformerModel(config)
        copy_sequences = torch.tensor([[2, 4, 3], [2, 5, 3]])
        batch = (copy_sequences, copy_sequences.clone())

        initial_loss, final_loss = _optimize_fixed_batch(model, batch, 50)

        self.assertLess(final_loss, initial_loss * 0.2)


if __name__ == "__main__":
    unittest.main()
