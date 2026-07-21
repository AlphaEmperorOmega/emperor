import unittest

import torch

import models.bert.linear.config as bert_config
import models.gpt.linear.config as gpt_config
import models.vit.linear.config as vit_config
from models.bert.linear.config_builder import BertLinearConfigBuilder
from models.bert.linear.model import Model as BertModel
from models.bert.linear.runtime_defaults import runtime_from_flat as bert_runtime
from models.gpt.linear import GptLinearConfigBuilder
from models.gpt.linear import Model as GptModel
from models.gpt.linear.runtime_defaults import runtime_from_flat as gpt_runtime
from models.vit.linear.config_builder import VitLinearConfigBuilder
from models.vit.linear.model import Model as VitModel
from models.vit.linear.runtime_defaults import runtime_from_flat as vit_runtime


class TestRealModelAttentionIntegration(unittest.TestCase):
    def test_bert_causal_batch_first_forward_causality_and_gradients(self):
        runtime = bert_runtime(
            {
                "batch_size": 2,
                "input_dim": 32,
                "output_dim": 32,
                "sequence_length": 5,
                "hidden_dim": 8,
                "stack_num_layers": 1,
                "stack_dropout_probability": 0.0,
                "attn_num_heads": 2,
                "causal_attention_mask_flag": True,
            },
            bert_config,
        )
        torch.manual_seed(5101)
        model = BertModel(BertLinearConfigBuilder(runtime=runtime).build()).eval()
        input_ids = torch.tensor(((2, 5, 7, 11, 13), (2, 17, 19, 23, 29)))
        attention_mask = torch.tensor(((1, 1, 1, 0, 1), (1, 1, 0, 1, 1)))
        token_type_ids = torch.zeros_like(input_ids)

        mlm_logits, nsp_logits, auxiliary_loss = model(
            input_ids,
            attention_mask,
            token_type_ids,
        )

        self.assertEqual(mlm_logits.shape, (2, 5, 32))
        self.assertEqual(nsp_logits.shape, (2, 2))
        self.assertEqual(auxiliary_loss.shape, ())
        self.assertTrue(torch.isfinite(mlm_logits).all())
        self.assertTrue(torch.isfinite(nsp_logits).all())
        self.assertTrue(torch.isfinite(auxiliary_loss))
        objective = (
            mlm_logits.square().mean() + nsp_logits.square().mean() + auxiliary_loss
        )
        objective.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue(
            any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
        )

        changed_future = input_ids.clone()
        changed_future[:, -1] = torch.tensor((31, 3))
        with torch.no_grad():
            baseline_mlm, _baseline_nsp, _ = model(
                input_ids,
                attention_mask,
                token_type_ids,
            )
            changed_mlm, _changed_nsp, _ = model(
                changed_future,
                attention_mask,
                token_type_ids,
            )
        torch.testing.assert_close(
            changed_mlm[:, :-1],
            baseline_mlm[:, :-1],
            rtol=0.0,
            atol=0.0,
        )
        self.assertFalse(torch.equal(changed_mlm[:, -1], baseline_mlm[:, -1]))

    def test_bert_equal_batch_and_sequence_lengths_preserve_sample_isolation(self):
        runtime = bert_runtime(
            {
                "batch_size": 2,
                "input_dim": 32,
                "output_dim": 32,
                "sequence_length": 2,
                "hidden_dim": 8,
                "stack_num_layers": 1,
                "stack_dropout_probability": 0.0,
                "attn_num_heads": 2,
            },
            bert_config,
        )
        torch.manual_seed(5102)
        model = BertModel(BertLinearConfigBuilder(runtime=runtime).build()).eval()
        original_ids = torch.tensor(((2, 5), (7, 11)))
        changed_ids = torch.tensor(((2, 5), (13, 17)))
        attention_mask = torch.ones_like(original_ids)
        token_type_ids = torch.zeros_like(original_ids)

        with torch.no_grad():
            original_mlm, original_nsp, _ = model(
                original_ids,
                attention_mask,
                token_type_ids,
            )
            changed_mlm, changed_nsp, _ = model(
                changed_ids,
                attention_mask,
                token_type_ids,
            )

        torch.testing.assert_close(
            changed_mlm[0],
            original_mlm[0],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            changed_nsp[0],
            original_nsp[0],
            rtol=0.0,
            atol=0.0,
        )
        self.assertGreater(
            torch.max(torch.abs(changed_mlm[1] - original_mlm[1])).item(),
            1e-6,
        )
        self.assertGreater(
            torch.max(torch.abs(changed_nsp[1] - original_nsp[1])).item(),
            1e-6,
        )

    def test_gpt_equal_batch_and_sequence_lengths_preserve_sample_isolation(self):
        runtime = gpt_runtime(
            {
                "batch_size": 2,
                "input_dim": 16,
                "output_dim": 16,
                "sequence_length": 2,
                "hidden_dim": 8,
                "stack_num_layers": 1,
                "stack_dropout_probability": 0.0,
                "attn_num_heads": 2,
            },
            gpt_config,
        )
        torch.manual_seed(5103)
        model = GptModel(GptLinearConfigBuilder(runtime=runtime).build()).eval()
        original_ids = torch.tensor(((1, 2), (3, 4)))
        changed_ids = torch.tensor(((5, 6), (3, 4)))

        with torch.no_grad():
            original_logits, _ = model(original_ids)
            changed_logits, _ = model(changed_ids)

        torch.testing.assert_close(
            changed_logits[1],
            original_logits[1],
            rtol=0.0,
            atol=0.0,
        )
        self.assertGreater(
            torch.max(torch.abs(changed_logits[0] - original_logits[0])).item(),
            1e-6,
        )

    def test_vit_equal_batch_and_sequence_lengths_preserve_sample_isolation(self):
        runtime = vit_runtime(
            {
                "batch_size": 5,
                "output_dim": 5,
                "image_patch_size": 4,
                "image_height": 8,
                "input_channels": 3,
                "hidden_dim": 16,
                "stack_num_layers": 1,
                "stack_dropout_probability": 0.0,
                "attn_num_heads": 4,
            },
            vit_config,
        )
        torch.manual_seed(5104)
        model = VitModel(VitLinearConfigBuilder(runtime=runtime).build()).eval()
        original_images = torch.randn(5, 3, 8, 8)
        changed_images = original_images.clone()
        changed_images[1] = torch.randn_like(changed_images[1]).mul_(4.0)

        with torch.no_grad():
            original_logits = model(original_images)
            changed_logits = model(changed_images)

        torch.testing.assert_close(
            changed_logits[0],
            original_logits[0],
            rtol=0.0,
            atol=0.0,
        )
        self.assertGreater(
            torch.max(torch.abs(changed_logits[1] - original_logits[1])).item(),
            1e-6,
        )


if __name__ == "__main__":
    unittest.main()
