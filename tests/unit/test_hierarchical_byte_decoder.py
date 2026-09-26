"""Byte prediction through the public hierarchical decoder interface."""

import copy
import io
import unittest

import torch

from tests.support.hierarchical_decoding import decoder_config


def constant_output(decoder, scores):
    # A real configured linear head with fixed logits makes decoding choices
    # deterministic while retaining all native encoder/attention execution.
    with torch.no_grad():
        for parameter in decoder.output_projection.parameters():
            parameter.zero_()
        decoder.output_projection.layers[0].model.bias_params.copy_(scores)


class HierarchicalByteDecoderTests(unittest.TestCase):
    def test_cpu_autocast_and_control_generation(self):
        decoder = decoder_config(max_bytes=4).build().eval()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = decoder(
                torch.zeros(1, 1, 6), torch.tensor([[[0, 97]]]), torch.tensor([[2]])
            )
        self.assertEqual(output.logits.dtype, torch.bfloat16)
        self.assertTrue(output.logits.isfinite().all())
        scores = torch.zeros(258)
        scores[256], scores[257], scores[97] = 30, 20, 10
        constant_output(decoder, scores)
        first = decoder.generate_token(torch.zeros(6))
        self.assertEqual(first.stop_reason, "end_of_document")
        self.assertEqual(first.text, "")
        partial = decoder.generate_token(torch.zeros(6), prefix="m")
        self.assertEqual(partial.stop_reason, "end_of_token")
        self.assertEqual(partial.text, "m")
        scores.zero_()
        scores[97] = 10
        constant_output(decoder, scores)
        chunk = decoder.generate_token(torch.zeros(6))
        self.assertEqual(chunk.text, "aaaa")
        self.assertEqual(chunk.stop_reason, "byte_limit")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_device_movement(self):
        decoder = decoder_config().build().cuda().eval()
        result = decoder(
            torch.zeros(1, 1, 6, device="cuda"),
            torch.tensor([[[97]]]),
            torch.tensor([[1]]),
        )
        self.assertEqual(result.logits.device.type, "cuda")
        self.assertEqual(result.token_offsets.device.type, "cuda")

    def test_dense_adaptive_and_mixture_auxiliary_losses(self):
        from dataclasses import fields

        from emperor.attention import MixtureOfAttentionHeadsConfig
        from tests.support.hierarchical_embedding import adaptive_stack, moe_config

        for adaptive, experts in ((True, False), (False, True), (True, True)):
            with self.subTest(adaptive=adaptive, experts=experts):
                cfg = decoder_config()
                layer = cfg.decoder_config.decoder_stack_config.layer_config.layer_model_config
                if experts:
                    source = layer.self_attention_config
                    values = {
                        field.name: getattr(source, field.name)
                        for field in fields(MixtureOfAttentionHeadsConfig)
                        if hasattr(source, field.name)
                    }
                    values["experts_config"] = moe_config(
                        adaptive=adaptive
                    ).stack_config.layer_config.layer_model_config
                    values["use_kv_expert_models_flag"] = False
                    layer.self_attention_config = MixtureOfAttentionHeadsConfig(
                        **values
                    )
                    layer.feed_forward_config.stack_config = moe_config(
                        adaptive=adaptive
                    )
                    cfg.output_projection_config = moe_config(8, 258, adaptive=adaptive)
                else:
                    layer.self_attention_config.projection_model_config = (
                        adaptive_stack()
                    )
                    layer.feed_forward_config.stack_config = adaptive_stack()
                    cfg.output_projection_config = adaptive_stack(8, 258)
                decoder = cfg.build().eval()
                losses = []
                hook = decoder.decoder.register_forward_hook(
                    lambda _m, _i, out, losses=losses: losses.append(
                        (out[1], out[0].shape[0] * out[0].shape[1])
                    )
                )
                projection_losses = []
                hooks = [
                    child.register_forward_hook(
                        lambda _m,
                        _i,
                        out,
                        projection_losses=projection_losses: projection_losses.append(
                            out.loss
                        )
                    )
                    for child in (
                        decoder.conditioning_projection,
                        decoder.output_projection,
                    )
                ]
                context = torch.randn(1, 3, 6, requires_grad=True)
                prefix = torch.tensor([[[97, 98, 99], [32, 0, 0], [0, 0, 0]]])
                lengths = torch.tensor([[3, 1, 0]])
                result = decoder(context, prefix, lengths)
                expected = sum(loss * count / 7 for loss, count in losses) + sum(
                    loss for loss in projection_losses if loss is not None
                )
                torch.testing.assert_close(result.loss, expected)
                if experts:
                    self.assertGreater(result.loss.item(), 0)
                hook.remove()
                for item in hooks:
                    item.remove()
                single = decoder(context[:, :1], prefix[:, :1], lengths[:, :1])
                torch.testing.assert_close(
                    result.logits[:4], single.logits, atol=1e-5, rtol=1e-5
                )
                companions = decoder(
                    torch.cat((context[:, :1], context[:, 1:2]), dim=1),
                    prefix[:, :1].expand(-1, 2, -1),
                    lengths[:, :1].expand(-1, 2),
                )
                torch.testing.assert_close(
                    single.logits, companions.logits[:4], atol=1e-5, rtol=1e-5
                )
                changed = prefix.clone()
                changed[0, 0, 1:] = torch.tensor([100, 101])
                future = decoder(context, changed, lengths)
                torch.testing.assert_close(result.logits[:2], future.logits[:2])
                (result.logits.square().mean() + result.loss).backward()
                self.assertTrue(torch.isfinite(context.grad).all())
                self.assertGreater(context.grad.abs().sum(), 0)

    def test_rejects_memory_cross_row_grouping_and_changing_recurrence(self):
        from emperor.augmentations.adaptive_parameters import MeanGroupingConfig
        from emperor.layers import RecurrentLayerConfig
        from emperor.memory import AttentionDynamicMemoryConfig
        from tests.support.hierarchical_embedding import adaptive_stack, linear_stack

        cfg = decoder_config()
        attention = cfg.decoder_config.decoder_stack_config.layer_config.layer_model_config.self_attention_config
        attention.memory_config = AttentionDynamicMemoryConfig()
        with self.assertRaisesRegex(ValueError, "memory_config"):
            cfg.build()
        cfg = decoder_config()
        cfg.output_projection_config = adaptive_stack(8, 258)
        cfg.output_projection_config.layer_config.layer_model_config.adaptive_augmentation_config.grouping_config = MeanGroupingConfig()
        with self.assertRaisesRegex(ValueError, "grouping_config"):
            cfg.build()
        cfg = decoder_config(conditioning_dim=8)
        cfg.conditioning_projection_config = RecurrentLayerConfig(
            input_dim=8,
            output_dim=8,
            max_steps=2,
            initial_iterations=1,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            block_config=linear_stack(8, 8),
        )
        with self.assertRaisesRegex(ValueError, "initial_iterations"):
            cfg.build()

    def test_rejects_incompatible_children_and_invalid_inputs(self):
        from emperor.augmentations.adaptive_parameters import WeightDecayScheduleOptions
        from emperor.decoding.hierarchical import HierarchicalByteDecoderConfig
        from tests.support.hierarchical_embedding import adaptive_stack, moe_config

        with self.assertRaisesRegex(ValueError, "conditioning_dim"):
            HierarchicalByteDecoderConfig().build()
        for attribute, value, message in (
            ("batch_first_flag", False, "batch_first"),
            ("causal_attention_mask_flag", False, "causal"),
            ("target_sequence_length", 2, "max_token_bytes"),
        ):
            cfg = decoder_config()
            setattr(
                cfg.decoder_config.decoder_stack_config.layer_config.layer_model_config.self_attention_config,
                attribute,
                value,
            )
            with self.assertRaisesRegex(ValueError, message):
                cfg.build()
        cfg = decoder_config()
        cfg.output_projection_config = moe_config(8, 258, capacity=1.0)
        with self.assertRaisesRegex(ValueError, "capacity_factor"):
            cfg.build()
        cfg.output_projection_config = adaptive_stack(8, 258)
        cfg.output_projection_config.layer_config.layer_model_config.adaptive_augmentation_config.bias_config.decay_schedule = WeightDecayScheduleOptions.EXPONENTIAL
        with self.assertRaisesRegex(ValueError, "decay_schedule"):
            cfg.build()
        decoder = decoder_config(max_batch=1).build()
        context = torch.zeros(1, 1, 6)
        with self.assertRaisesRegex(ValueError, "byte_lengths"):
            decoder(
                context, torch.zeros(1, 1, 2, dtype=torch.long), torch.tensor([[3]])
            )
        with self.assertRaisesRegex(ValueError, "byte values"):
            decoder(context, torch.tensor([[[256]]]), torch.tensor([[1]]))
        with self.assertRaisesRegex(TypeError, "bool"):
            decoder(
                context,
                torch.zeros(1, 1, 0, dtype=torch.long),
                torch.tensor([[0]]),
                torch.ones(1, 1),
            )
        with self.assertRaisesRegex(ValueError, "batch_size"):
            decoder(
                context.expand(1, 2, 6),
                torch.zeros(1, 2, 0, dtype=torch.long),
                torch.zeros(1, 2, dtype=torch.long),
            )

    def test_seeded_sampling_prefix_limits_and_control_symbols(self):
        from emperor.decoding.hierarchical import ByteGenerationOptions

        decoder = decoder_config(max_bytes=8).build()
        scores = torch.full((258,), -20.0)
        scores[97:100] = 0
        constant_output(decoder, scores)
        options = ByteGenerationOptions(
            do_sample=True, top_k=3, top_p=0.9, temperature=0.7, seed=123
        )
        first = decoder.generate_token(
            torch.zeros(6), prefix="é", max_new_bytes=3, options=options
        )
        second = decoder.generate_token(
            torch.zeros(6), prefix="é", max_new_bytes=3, options=options
        )
        self.assertEqual(first, second)
        self.assertEqual(first.stop_reason, "byte_budget")
        self.assertTrue(first.text.startswith("é"))
        self.assertEqual(first.new_bytes, 3)
        for prefix in ("\ud800", "toolongprefix"):
            with self.assertRaises(ValueError):
                decoder.generate_token(torch.zeros(6), prefix=prefix)
        with self.assertRaisesRegex(ValueError, "temperature"):
            decoder.generate_token(
                torch.zeros(6), options=ByteGenerationOptions(temperature=0)
            )

    def test_generation_constrains_utf8_and_restores_training_state(self):
        from emperor.decoding.hierarchical import ByteGenerationOptions

        decoder = decoder_config(max_bytes=5).build().train()
        decoder.byte_position.eval()
        scores = torch.full((258,), -100.0)
        scores[255] = 100  # Invalid UTF-8 must never win.
        scores[240] = 20
        scores[159] = 19
        scores[256] = 18
        constant_output(decoder, scores)
        result = decoder.generate_token(torch.zeros(6), options=ByteGenerationOptions())
        self.assertEqual(result.text.encode(), bytes([240, 159, 159, 159]))
        self.assertEqual(result.stop_reason, "end_of_token")
        self.assertEqual(result.new_bytes, 4)
        self.assertTrue(decoder.training)
        self.assertFalse(decoder.byte_position.training)

    def test_causality_masking_configuration_and_checkpoint(self):
        config = decoder_config()
        original = copy.deepcopy(config)
        decoder = config.build().double().eval()
        context = torch.randn(1, 2, 6, dtype=torch.float64)
        prefix = torch.tensor([[[97, 98, 99], [1, 2, 3]]])
        lengths = torch.tensor([[3, 3]])
        first = decoder(context, prefix, lengths)
        prefix[0, 0, 1:] = torch.tensor([100, 101])
        context[0, 1] += 10
        second = decoder(context, prefix, lengths)
        torch.testing.assert_close(first.logits[:2], second.logits[:2])
        self.assertFalse(torch.allclose(first.logits[2:4], second.logits[2:4]))
        self.assertEqual(config, original)
        empty = decoder(
            context,
            prefix.fill_(999),
            lengths.fill_(-1),
            torch.zeros(1, 2, dtype=torch.bool),
        )
        self.assertEqual(empty.logits.shape, (0, 258))
        self.assertEqual(empty.token_offsets.tolist(), [0, 0, 0])
        self.assertEqual(empty.loss.item(), 0)
        buffer = io.BytesIO()
        torch.save(decoder.state_dict(), buffer)
        buffer.seek(0)
        clone = config.build().double().eval()
        clone.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        prefix.zero_()
        lengths.zero_()
        torch.testing.assert_close(
            decoder(context, prefix, lengths).logits,
            clone(context, prefix, lengths).logits,
        )

    def test_packed_predictions_match_individual_tokens_and_backpropagate(self):
        torch.manual_seed(7)
        decoder = decoder_config().build().eval()
        conditioning = torch.randn(2, 2, 6, requires_grad=True)
        prefixes = torch.tensor([[[109, 97, 116], [0, 0, 0]], [[99, 0, 0], [0, 0, 0]]])
        lengths = torch.tensor([[3, 0], [1, 0]])
        mask = torch.tensor([[True, True], [True, False]])
        output = decoder(conditioning, prefixes, lengths, mask)
        self.assertEqual(tuple(output.logits.shape), (7, 258))
        self.assertEqual(output.token_offsets.tolist(), [0, 4, 5, 7, 7])
        self.assertEqual(output.loss.ndim, 0)
        for flat_index in range(3):
            row, column = divmod(flat_index, 2)
            single = decoder(
                conditioning[row : row + 1, column : column + 1],
                prefixes[row : row + 1, column : column + 1],
                lengths[row : row + 1, column : column + 1],
            )
            start, end = output.token_offsets[flat_index : flat_index + 2]
            torch.testing.assert_close(output.logits[start:end], single.logits)
        labels = torch.tensor([109, 97, 116, 256, 257, 99, 256])
        (
            torch.nn.functional.cross_entropy(output.logits, labels) + output.loss
        ).backward()
        self.assertTrue(torch.isfinite(conditioning.grad).all())
        self.assertEqual(conditioning.grad[1, 1].abs().sum(), 0)
        for child in (
            decoder.byte_embedding,
            decoder.byte_position,
            decoder.decoder,
            decoder.conditioning_projection,
            decoder.output_projection,
        ):
            gradients = [p.grad for p in child.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(
                all(torch.isfinite(gradient).all() for gradient in gradients)
            )
            self.assertTrue(any(gradient.abs().sum() > 0 for gradient in gradients))
