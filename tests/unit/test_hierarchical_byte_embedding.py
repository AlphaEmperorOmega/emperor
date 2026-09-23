import copy
import io
import unittest
from dataclasses import fields, replace

import torch

from emperor.attention import MixtureOfAttentionHeadsConfig
from emperor.augmentations.adaptive_parameters import (
    MeanGroupingConfig,
    WeightDecayScheduleOptions,
)
from emperor.embedding.absolute import TextSinusoidalPositionalEmbeddingConfig
from emperor.embedding.hierarchical import HierarchicalByteEmbeddingConfig
from emperor.layers import (
    InnerThinkingRecurrentConfig,
    LayerNormPositionOptions,
    LayerState,
    RecurrentLayerConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.memory import AttentionDynamicMemoryConfig
from tests.support.hierarchical_embedding import (
    adaptive_stack,
    embedding_config,
    encoder_layer_config,
    linear_stack,
    moe_config,
)


class TestHierarchicalByteEmbedding(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)

    def test_utf8_tokens_produce_independent_vectors_in_original_order(self):
        model = embedding_config().build().eval()
        tokens = [["novelword", "猫", "😀", ""], ["é", "\x00", " ", "novelword"]]

        state = model(tokens)

        self.assertIsInstance(state, LayerState)
        self.assertEqual(state.hidden.shape, (2, 4, 6))
        self.assertEqual(state.loss.shape, ())
        self.assertTrue(torch.isfinite(state.hidden).all())
        for batch_index, row in enumerate(tokens):
            for token_index, token in enumerate(row):
                torch.testing.assert_close(
                    state.hidden[batch_index, token_index],
                    model([[token]]).hidden[0, 0],
                    atol=1e-6,
                    rtol=1e-5,
                )

    def test_masked_tokens_never_enter_children_and_all_masked_is_zero(self):
        model = embedding_config().build().eval()
        observed_batches = []
        handles = [
            child.register_forward_pre_hook(
                lambda module, args: observed_batches.append(type(module).__name__)
            )
            for child in model.children()
        ]
        try:
            mask = torch.tensor([[True, False], [False, True]])
            state = model([["same", "ignored"], ["x", "same"]], mask)
            torch.testing.assert_close(state.hidden[~mask], torch.zeros(2, 6))
            torch.testing.assert_close(state.hidden[mask][0], state.hidden[mask][1])
            observed_batches.clear()
            empty = model([["a", "b"]], torch.zeros(1, 2, dtype=torch.bool))
            self.assertEqual(observed_batches, [])
            self.assertEqual(empty.hidden.count_nonzero(), 0)
            self.assertEqual(empty.loss.item(), 0)
        finally:
            for handle in handles:
                handle.remove()

    def test_rejects_invalid_text_and_masks_before_running_children(self):
        model = embedding_config(max_bytes=4).build()
        invalid = [
            ([], None, ValueError, "nonempty"),
            ([[]], None, ValueError, "nonempty"),
            ("hello", None, TypeError, "token_texts"),
            (["hello"], None, TypeError, "row"),
            ([["a"], ["b", "c"]], None, ValueError, "rectangular"),
            ([[1]], None, TypeError, "token_texts\\[0\\]\\[0\\]"),
            ([["\ud800"]], None, ValueError, "UTF-8"),
            ([["ééa"]], None, ValueError, "5 UTF-8 bytes"),
            ([["a"]], [[True]], TypeError, "attention_mask"),
            ([["a"]], torch.ones(1, 1), TypeError, "bool"),
            ([["a"]], torch.ones(1, 2, dtype=torch.bool), ValueError, "shape"),
        ]
        for tokens, mask, error, message in invalid:
            with self.subTest(tokens=repr(tokens), mask=mask):
                with self.assertRaisesRegex(error, message):
                    model(tokens, mask)
        self.assertEqual(model([["😀"]]).hidden.shape, (1, 1, 6))

    def test_configuration_requires_explicit_dimensions_and_encoder_contract(self):
        for field in fields(HierarchicalByteEmbeddingConfig):
            cfg = embedding_config()
            setattr(cfg, field.name, None)
            with self.subTest(missing=field.name):
                with self.assertRaisesRegex((TypeError, ValueError), field.name):
                    cfg.build()
        for name, value in (
            ("batch_first_flag", None),
            ("batch_first_flag", False),
            ("causal_attention_mask_flag", True),
            ("source_sequence_length", 3),
            ("embedding_dim", 10),
        ):
            cfg = embedding_config()
            attention = cfg.encoder_config.encoder_stack_config.layer_config.layer_model_config.attention_config
            setattr(attention, name, value)
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(ValueError, name):
                    cfg.build()
        cfg = embedding_config()
        cfg.encoder_config = cfg.projection_config
        with self.assertRaisesRegex(TypeError, "encoder-only"):
            cfg.build()
        model = embedding_config(max_batch=1).build()
        with self.assertRaisesRegex(ValueError, "batch_size"):
            model([["a", "b"]])

    def test_rejects_stateful_and_cross_token_adaptive_children(self):
        for setting in ("memory", "grouping", "decay"):
            cfg = embedding_config()
            cfg.projection_config = adaptive_stack(8, 6)
            augmentation = cfg.projection_config.layer_config.layer_model_config.adaptive_augmentation_config
            if setting == "memory":
                attention = cfg.encoder_config.encoder_stack_config.layer_config.layer_model_config.attention_config
                attention.memory_config = AttentionDynamicMemoryConfig()
                message = "memory_config"
            elif setting == "grouping":
                augmentation.grouping_config = MeanGroupingConfig()
                message = "grouping_config"
            else:
                augmentation.bias_config.decay_schedule = (
                    WeightDecayScheduleOptions.LINEAR
                )
                message = "decay_schedule"
            with self.subTest(setting=setting):
                with self.assertRaisesRegex(ValueError, message):
                    cfg.build()

    def test_rejects_changing_recurrent_schedule_and_attention_on_flat_rows(self):
        cfg = embedding_config(output_dim=8)
        cfg.projection_config = RecurrentLayerConfig(
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
        cfg = embedding_config()
        cfg.projection_config.layer_config.layer_model_config = (
            encoder_layer_config().attention_config
        )
        with self.assertRaisesRegex(ValueError, "attention.*byte encoder"):
            cfg.build()

    def test_invalid_masked_text_is_validated_without_child_execution(self):
        model = embedding_config(max_bytes=4).build()
        for token, message in (("\ud800", "UTF-8"), ("exceeds", "max_token_bytes")):
            with self.subTest(token=repr(token)):
                with self.assertRaisesRegex(ValueError, message):
                    model([[token]], torch.zeros(1, 1, dtype=torch.bool))

    def test_incompatible_position_projection_and_encoder_settings_fail_on_build(self):
        cfg = embedding_config()
        with self.assertRaisesRegex(TypeError, "cfg"):
            cfg.registry_owner()(object())
        with self.assertRaisesRegex(TypeError, "overrides"):
            cfg.build(overrides=object())
        for value in (True, 0, -1, 1.5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "byte_embedding_dim"):
                    replace(cfg, byte_embedding_dim=value).build()
        for positions in (
            linear_stack(),
            replace(cfg.byte_position_config, num_embeddings=1),
            replace(cfg.byte_position_config, padding_idx=0),
            replace(cfg.byte_position_config, embedding_dim=4),
        ):
            with self.subTest(positions=positions):
                with self.assertRaisesRegex(
                    (TypeError, ValueError), "byte_position_config"
                ):
                    replace(cfg, byte_position_config=positions).build()
        with self.assertRaisesRegex(ValueError, "encoder-only"):
            replace(
                cfg,
                encoder_config=replace(
                    cfg.encoder_config, decoder_stack_config=linear_stack()
                ),
            ).build()
        with self.assertRaisesRegex(ValueError, "byte-attention"):
            replace(
                cfg,
                encoder_config=replace(
                    cfg.encoder_config, encoder_stack_config=linear_stack(hidden_dim=8)
                ),
            ).build()
        with self.assertRaisesRegex(TypeError, "input_dim"):
            replace(
                cfg,
                encoder_config=replace(
                    cfg.encoder_config, encoder_stack_config=cfg.byte_position_config
                ),
            ).build()
        for projection in (
            LinearLayerConfig(bias_flag=True),
            moe_config().stack_config.layer_config,
        ):
            with self.subTest(projection=type(projection).__name__):
                with self.assertRaisesRegex(TypeError, "LayerState"):
                    replace(cfg, projection_config=projection).build()
        with self.assertRaisesRegex(ValueError, "output_dim"):
            replace(cfg, projection_config=linear_stack(8, 9)).build()

    def test_fixed_recurrent_encoder_is_independent_but_token_selection_is_rejected(
        self,
    ):
        cfg = embedding_config()
        recurrent = RecurrentLayerConfig(
            input_dim=8,
            output_dim=8,
            max_steps=2,
            initial_iterations=2,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
            block_config=cfg.encoder_config.encoder_stack_config,
        )
        cfg.encoder_config.encoder_stack_config = recurrent
        model = cfg.build().eval()
        original = model([["one"]]).hidden
        companions = model([["one", "new", "longer"]]).hidden
        torch.testing.assert_close(original[:, 0], companions[:, 0])
        recurrent_values = {
            field.name: getattr(recurrent, field.name) for field in fields(recurrent)
        }
        cfg.encoder_config.encoder_stack_config = InnerThinkingRecurrentConfig(
            **recurrent_values,
            sampler_config=moe_config().stack_config.layer_config.layer_model_config.sampler_config,
        )
        with self.assertRaisesRegex(ValueError, "sampler_config"):
            cfg.build()

    def test_adaptive_moe_gradients_and_token_weighted_auxiliary_loss(self):
        cfg = embedding_config()
        layer = cfg.encoder_config.encoder_stack_config.layer_config.layer_model_config
        layer.attention_config.projection_model_config = adaptive_stack()
        layer.feed_forward_config.stack_config = moe_config()
        cfg.projection_config = moe_config(8, 6)
        cfg.projection_config.stack_config.layer_config.layer_model_config.capacity_factor = 1.0
        with self.assertRaisesRegex(ValueError, "capacity_factor"):
            cfg.build()
        cfg.projection_config.stack_config.layer_config.layer_model_config.capacity_factor = 0.0
        model = cfg.build().eval()
        group_results, projection_losses = [], []
        handles = [
            model.encoder.register_forward_hook(
                lambda module, args, result: group_results.append(
                    (result[0].shape, result[1])
                )
            ),
            model.projection.register_forward_hook(
                lambda module, args, state: projection_losses.append(state.loss)
            ),
        ]
        try:
            result = model([["a", "b", "long", ""]])
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(
            [shape[:2] for shape, _ in group_results], [(2, 2), (1, 5), (1, 1)]
        )
        expected_loss = (
            sum(shape[0] * loss / 4 for shape, loss in group_results)
            + projection_losses[0]
        )
        torch.testing.assert_close(result.loss, expected_loss)
        self.assertGreater(result.loss.item(), 0)
        (result.hidden.square().mean() + result.loss).backward()
        for child in (
            model.byte_embedding,
            model.byte_position,
            model.encoder,
            model.projection,
        ):
            gradients = [
                parameter.grad
                for parameter in child.parameters()
                if parameter.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(
                all(torch.isfinite(gradient).all() for gradient in gradients)
            )
            self.assertTrue(any(gradient.abs().sum() > 0 for gradient in gradients))
        torch.testing.assert_close(
            result.hidden[0, 0], model([["a"]]).hidden[0, 0], atol=1e-6, rtol=1e-5
        )

    def test_native_mixture_of_attention_heads_preserves_token_independence(self):
        cfg = embedding_config()
        layer = cfg.encoder_config.encoder_stack_config.layer_config.layer_model_config
        attention_fields = {
            field.name: getattr(layer.attention_config, field.name)
            for field in fields(MixtureOfAttentionHeadsConfig)
            if hasattr(layer.attention_config, field.name)
        }
        layer.attention_config = MixtureOfAttentionHeadsConfig(
            **attention_fields,
            experts_config=moe_config().stack_config.layer_config.layer_model_config,
            use_kv_expert_models_flag=True,
        )
        model = cfg.build().eval()
        state = model([["one", "two", "longer"]])
        torch.testing.assert_close(state.hidden[0, 0], model([["one"]]).hidden[0, 0])
        self.assertGreater(state.loss.item(), 0)
        (state.hidden.square().mean() + state.loss).backward()
        self.assertTrue(torch.isfinite(model.byte_embedding.weight.grad).all())

    def test_cpu_autocast_preserves_projected_dtype_and_gradients(self):
        model = embedding_config().build()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            state = model([["one", "longer"]])
            objective = state.hidden.float().square().mean() + state.loss
        self.assertEqual(state.hidden.dtype, torch.bfloat16)
        objective.backward()
        self.assertTrue(torch.isfinite(model.byte_embedding.weight.grad).all())

    def test_complete_bytes_order_and_whitespace_affect_embeddings(self):
        model = embedding_config().build().eval()
        pairs = [
            ("abcdX", "abcdY"),
            ("ab", "ba"),
            ("", "\x00"),
            ("a", " a"),
            ("é", "e\u0301"),
        ]
        for first, second in pairs:
            state = model([[first, second]])
            with self.subTest(first=first, second=second):
                self.assertFalse(torch.allclose(state.hidden[0, 0], state.hidden[0, 1]))

    def test_configuration_overrides_and_dimensions_never_mutate_callers(self):
        cfg = embedding_config()
        cfg.byte_position_config.embedding_dim = None
        cfg.projection_config.input_dim = cfg.projection_config.output_dim = None
        cfg.encoder_config.encoder_stack_config.input_dim = None
        before = copy.deepcopy(cfg)
        overrides = HierarchicalByteEmbeddingConfig(output_dim=10)
        original_overrides = copy.deepcopy(overrides)
        model = cfg.build(overrides)
        self.assertEqual(cfg, before)
        self.assertEqual(overrides, original_overrides)
        self.assertEqual(model([["new"]]).hidden.shape, (1, 1, 10))
        model.cfg.projection_config.hidden_dim = 100
        self.assertEqual(cfg, before)

    def test_dtype_registration_gradients_and_strict_checkpoint_round_trip(self):
        for dtype in (torch.float32, torch.float64):
            for sinusoidal in (False, True):
                with self.subTest(dtype=dtype, sinusoidal=sinusoidal):
                    cfg = embedding_config()
                    if sinusoidal:
                        cfg.byte_position_config = (
                            TextSinusoidalPositionalEmbeddingConfig(
                                embedding_dim=8,
                                num_embeddings=33,
                                auto_expand_flag=False,
                            )
                        )
                    model = cfg.build().to(dtype=dtype).eval()
                    state = model([["abcdef", "猫", ""]])
                    self.assertEqual(state.hidden.dtype, dtype)
                    self.assertEqual(state.loss.dtype, dtype)
                    (state.hidden.square().mean() + state.loss).backward()
                    for name, child in model.named_children():
                        parameters = list(child.parameters())
                        if not parameters:
                            continue
                        self.assertTrue(
                            all(
                                p.grad is not None and torch.isfinite(p.grad).all()
                                for p in parameters
                            ),
                            name,
                        )
                    buffer = io.BytesIO()
                    torch.save(model.state_dict(), buffer)
                    buffer.seek(0)
                    restored = cfg.build().to(dtype=dtype).eval()
                    restored.load_state_dict(
                        torch.load(buffer, weights_only=True), strict=True
                    )
                    torch.testing.assert_close(
                        state.hidden, restored([["abcdef", "猫", ""]]).hidden
                    )
                    restored.to("meta")
                    self.assertTrue(
                        all(p.device.type == "meta" for p in restored.parameters())
                    )

    def test_no_word_vocabulary_or_input_dependent_checkpoint_growth(self):
        model = embedding_config().build().eval()
        parameter_count = sum(p.numel() for p in model.parameters())
        before = copy.deepcopy(model.state_dict())
        model([[f"never-seen-{i}" for i in range(30)]])
        model([["long-word", ""]])
        after = model.state_dict()
        self.assertEqual(sum(p.numel() for p in model.parameters()), parameter_count)
        self.assertEqual(before.keys(), after.keys())
        for name in before:
            torch.testing.assert_close(before[name], after[name], msg=name)

    def test_masking_and_batch_companions_preserve_hidden_and_loss(self):
        for adaptive in (False, True):
            cfg = embedding_config()
            if adaptive:
                cfg.projection_config = adaptive_stack(8, 6)
            model = cfg.build().eval()
            reference = model([["kept", "a"]])
            padded = model(
                [["kept", "ignore", "a"]], torch.tensor([[True, False, True]])
            )
            torch.testing.assert_close(padded.hidden[:, [0, 2]], reference.hidden)
            torch.testing.assert_close(padded.loss, reference.loss)
            companions = model([["other", "a"], ["kept", "future"]])
            torch.testing.assert_close(companions.hidden[1, 0], reference.hidden[0, 0])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_movement_and_backward(self):
        model = embedding_config().build().cuda()
        state = model(
            [["cuda", "🙂"]], torch.ones(1, 2, dtype=torch.bool, device="cuda")
        )
        self.assertEqual(state.hidden.device.type, "cuda")
        state.hidden.square().mean().backward()
        self.assertTrue(torch.isfinite(model.byte_embedding.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
