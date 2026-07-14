import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

import torch
from emperor.attention.core.monitor import AttentionMonitorCallback
from emperor.attention.core.runtime import QKV
from emperor.attention.core.variants.self_attention.config import SelfAttentionConfig

from support.attention import build_attention_config
from support.monitor import (
    CaptureExperiment,
    CaptureLightningModule,
    NoExperimentLightningModule,
    TrainerStub,
    same_bound_method,
)


class TestAttentionMonitorCallback(unittest.TestCase):
    def attention(self):
        cfg = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=2,
            num_heads=2,
            embedding_dim=4,
            target_sequence_length=3,
            source_sequence_length=3,
            return_attention_weights_flag=True,
        )
        return cfg.build()

    def qkv(self):
        tensor = torch.randn(3, 2, 4)
        return tensor, tensor, tensor

    def test_rejects_non_positive_cadence(self):
        for bad in (0, -1):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as raised:
                    AttentionMonitorCallback(log_every_n_steps=bad)
                self.assertEqual(
                    str(raised.exception),
                    "log_every_n_steps must be greater than 0.",
                )

    def test_rejects_non_positive_history_size(self):
        for bad in (0, -1):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as raised:
                    AttentionMonitorCallback(history_size=bad)
                self.assertEqual(
                    str(raised.exception),
                    "history_size must be greater than 0.",
                )

    def test_default_monitor_configuration_is_explicit(self):
        callback = AttentionMonitorCallback()

        self.assertEqual(callback.log_every_n_steps, 100)
        self.assertEqual(callback.history_size, 128)
        self.assertIs(callback.log_per_head_scalars, False)

    def test_approximate_boolean_mask_uses_true_as_masked(self):
        callback = AttentionMonitorCallback()
        query = torch.ones(1, 1, 1)
        key = torch.ones(1, 2, 1)
        mask = torch.tensor([[False, True]])

        weights = callback._AttentionMonitorCallback__compute_approximate_weights(
            query,
            key,
            mask,
        )

        expected = torch.tensor([[[1.0, 0.0]]])
        torch.testing.assert_close(weights, expected)

    def test_discovers_only_attention_modules(self):
        module = CaptureLightningModule(
            attn=self.attention(), other=torch.nn.Linear(4, 4)
        )
        callback = AttentionMonitorCallback(log_every_n_steps=1)

        callback.on_fit_start(TrainerStub(), module)

        self.assertEqual([name for name, _ in callback._attention_modules], ["attn"])
        callback.on_fit_end(TrainerStub(), module)

    def test_respects_global_step_cadence(self):
        attention = self.attention()
        module = CaptureLightningModule(attn=attention)
        callback = AttentionMonitorCallback(log_every_n_steps=2)
        callback.on_fit_start(TrainerStub(), module)

        module.global_step = 1
        attention(*self.qkv())
        self.assertEqual(module.logged, [])

        module.global_step = 2
        attention(*self.qkv())
        self.assertIn("attn/attention/q_norm_mean", module.logged_tags)
        callback.on_fit_end(TrainerStub(), module)

    def test_sampling_defaults_to_step_zero_when_global_step_is_absent(self):
        callback = AttentionMonitorCallback(log_every_n_steps=2)
        should_sample = callback._AttentionMonitorCallback__should_sample

        self.assertIs(should_sample(SimpleNamespace()), True)
        self.assertIs(should_sample(SimpleNamespace(global_step=1)), False)
        self.assertIs(should_sample(SimpleNamespace(global_step=2)), True)

    def test_logs_expected_finite_scalars(self):
        attention = self.attention()
        module = CaptureLightningModule(attn=attention)
        callback = AttentionMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        attention(*self.qkv())

        expected_tags = {
            "attn/attention/q_norm_mean",
            "attn/attention/k_norm_mean",
            "attn/attention/v_norm_mean",
            "attn/attention/output_norm",
            "attn/attention/entropy_mean",
            "attn/attention/max_probability_mean",
            "attn/attention/dead_head_fraction",
            "attn/attention/mask_coverage",
            "attn/attention/configured_dropout_probability",
            "attn/attention/dropout_zero_fraction",
        }
        self.assertTrue(expected_tags.issubset(set(module.logged_tags)))
        for tag in expected_tags:
            value = module.logged_value(tag)
            self.assertTrue(torch.isfinite(torch.as_tensor(value)).all(), tag)
        callback.on_fit_end(TrainerStub(), module)

    def test_captures_named_qkv_and_merged_attention_mask(self):
        attention = self.attention()
        module = CaptureLightningModule(attn=attention)
        callback = AttentionMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)
        attention_mask = torch.zeros(3, 3, dtype=torch.bool)

        attention(*self.qkv(), attention_mask=attention_mask)

        trace = callback._traces[id(attention)]
        query, key, value = trace["qkv"]
        processor_query, processor_key, merged_attention_mask = trace[
            "attention_inputs"
        ]
        for tensor in (query, key, value, processor_query, processor_key):
            self.assertIsInstance(tensor, torch.Tensor)
            self.assertFalse(tensor.requires_grad)
        self.assertIsInstance(merged_attention_mask, torch.Tensor)
        self.assertFalse(merged_attention_mask.requires_grad)
        callback.on_fit_end(TrainerStub(), module)

    def test_pre_hook_uses_the_registered_module_identity_and_name(self):
        callback = AttentionMonitorCallback()
        module = SimpleNamespace()
        hook = callback._AttentionMonitorCallback__make_forward_pre_hook(
            "attention",
            module,
        )

        hook(object(), ())

        self.assertEqual(
            callback._traces,
            {id(module): {"name": "attention"}},
        )

    def test_emits_histograms_and_images_when_experiment_supports_them(self):
        attention = self.attention()
        module = CaptureLightningModule(attn=attention)
        callback = AttentionMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        attention(*self.qkv())

        experiment = module.logger.experiment
        self.assertTrue(
            any(
                tag == "attn/attention/histogram/entropy_by_head"
                for tag, _, _ in experiment.histograms
            )
        )
        self.assertTrue(
            any(
                tag == "attn/attention/heatmap/entropy_by_head"
                for tag, _, _, _ in experiment.images
            )
        )
        callback.on_fit_end(TrainerStub(), module)

    def test_skips_visual_summaries_without_experiment(self):
        attention = self.attention()
        module = NoExperimentLightningModule(attn=attention)
        callback = AttentionMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        attention(*self.qkv())

        self.assertIn("attn/attention/entropy_mean", module.logged_tags)
        callback.on_fit_end(TrainerStub(), module)

    def test_restores_wrappers_and_clears_state_on_fit_end(self):
        attention = self.attention()
        original_projection = attention.projector.compute_qkv_projections
        original_processor = attention.processor.compute_attention
        module = CaptureLightningModule(attn=attention)
        callback = AttentionMonitorCallback(log_every_n_steps=1)

        callback.on_fit_start(TrainerStub(), module)
        self.assertIsNot(
            attention.projector.compute_qkv_projections, original_projection
        )
        self.assertIsNot(attention.processor.compute_attention, original_processor)

        callback.on_fit_end(TrainerStub(), module)

        self.assertTrue(
            same_bound_method(
                attention.projector.compute_qkv_projections,
                original_projection,
            )
        )
        self.assertTrue(
            same_bound_method(
                attention.processor.compute_attention,
                original_processor,
            )
        )
        self.assertEqual(callback._attention_modules, [])
        self.assertEqual(callback._wrapped_methods, [])
        self.assertEqual(callback._traces, {})

    def test_repeated_fit_start_does_not_accumulate_hooks_or_wrappers(self):
        attention = self.attention()
        module = CaptureLightningModule(attn=attention)
        callback = AttentionMonitorCallback(log_every_n_steps=1)

        callback.on_fit_start(TrainerStub(), module)
        first_hook_count = len(callback._hooks)
        first_wrapper_count = len(callback._wrapped_methods)
        callback.on_fit_start(TrainerStub(), module)

        self.assertEqual(len(callback._hooks), first_hook_count)
        self.assertEqual(len(callback._wrapped_methods), first_wrapper_count)
        self.assertEqual([name for name, _ in callback._attention_modules], ["attn"])
        callback.on_fit_end(TrainerStub(), module)

    def test_head_first_4d_weights_normalize_to_batch_head_layout(self):
        callback = AttentionMonitorCallback()
        module = torch.nn.Module()
        module.num_heads = 2
        head_first = torch.arange(24, dtype=torch.float32).view(2, 3, 2, 2)

        normalized = callback._AttentionMonitorCallback__reshape_weights_by_head(
            module,
            head_first,
        )

        expected = head_first.permute(1, 0, 2, 3)
        torch.testing.assert_close(normalized, expected)

    def test_wrapping_skips_missing_components_and_non_tensor_outputs(self):
        callback = AttentionMonitorCallback(log_every_n_steps=1)
        empty_module = SimpleNamespace()
        callback._AttentionMonitorCallback__wrap_projector(empty_module)
        callback._AttentionMonitorCallback__wrap_processor(
            empty_module,
            CaptureLightningModule(),
        )

        projector = SimpleNamespace(compute_qkv_projections=lambda: object())
        module = SimpleNamespace(projector=projector)
        callback._AttentionMonitorCallback__wrap_projector(module)

        self.assertNotIsInstance(projector.compute_qkv_projections(), tuple)
        self.assertNotIn("qkv", callback._traces[id(module)])
        callback._AttentionMonitorCallback__cleanup()

    def test_processor_wrapper_handles_kwargs_and_non_tensor_private_weights(self):
        callback = AttentionMonitorCallback(log_every_n_steps=1)
        processor = SimpleNamespace(
            compute_attention=lambda **kwargs: kwargs.get("qkv"),
        )
        private_method_name = (
            "_SelfAttentionProcessor__compute_masked_attention_weights"
        )
        setattr(processor, private_method_name, lambda: object())
        module = SimpleNamespace(processor=processor)
        lightning_module = CaptureLightningModule()
        callback._AttentionMonitorCallback__wrap_processor(
            module,
            lightning_module,
        )

        marker = object()
        self.assertIs(
            processor.compute_attention(
                qkv=marker,
                merged_attention_mask=None,
            ),
            marker,
        )
        self.assertNotIn("approximate_weights", callback._traces[id(module)])
        self.assertNotIsInstance(
            getattr(processor, private_method_name)(),
            torch.Tensor,
        )
        self.assertNotIn("exact_weights", callback._traces[id(module)])
        callback._AttentionMonitorCallback__cleanup()

    def test_projector_wrapper_forwards_kwargs_and_captures_exact_qkv(self):
        callback = AttentionMonitorCallback()
        query = torch.tensor([[[1.0, 2.0]]])
        key = torch.tensor([[[3.0, 4.0]]])
        value = torch.tensor([[[5.0, 6.0]]])
        qkv = QKV(query=query, key=key, value=value)

        def project(*, projected):
            return projected

        projector = SimpleNamespace(compute_qkv_projections=project)
        module = SimpleNamespace(projector=projector)
        callback._AttentionMonitorCallback__wrap_projector(module)

        self.assertIs(projector.compute_qkv_projections(projected=qkv), qkv)
        captured = callback._traces[id(module)]["qkv"]
        for actual, expected in zip(captured, (query, key, value), strict=True):
            torch.testing.assert_close(actual, expected)

        callback._AttentionMonitorCallback__cleanup()
        self.assertIs(projector.compute_qkv_projections, project)

    def test_wrappers_skip_present_components_without_required_methods(self):
        callback = AttentionMonitorCallback()

        callback._AttentionMonitorCallback__wrap_projector(
            SimpleNamespace(projector=object())
        )
        callback._AttentionMonitorCallback__wrap_processor(
            SimpleNamespace(processor=object()),
            CaptureLightningModule(),
        )

        self.assertEqual(callback._wrapped_methods, [])

    def test_processor_wrapper_preserves_positional_and_keyword_inputs(self):
        callback = AttentionMonitorCallback(log_every_n_steps=1)
        lightning_module = CaptureLightningModule()
        query = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
        key = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])
        value = torch.ones_like(key)
        qkv = QKV(query=query, key=key, value=value)
        mask = torch.tensor([[0.0, -2.0], [1.0, 0.0]])
        marker = object()

        def compute_attention(*args, **kwargs):
            return marker

        processor = SimpleNamespace(compute_attention=compute_attention)
        module = SimpleNamespace(processor=processor)
        callback._AttentionMonitorCallback__wrap_processor(
            module,
            lightning_module,
        )

        self.assertIs(
            processor.compute_attention(
                qkv=qkv,
                merged_attention_mask=mask,
            ),
            marker,
        )
        trace = callback._traces[id(module)]
        for actual, expected in zip(
            trace["attention_inputs"],
            (query, key, mask),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected)
        expected_weights = torch.softmax(
            torch.matmul(query * (2**-0.5), key.transpose(-2, -1)) + mask,
            dim=-1,
        )
        torch.testing.assert_close(trace["approximate_weights"], expected_weights)

        processor.compute_attention(qkv)
        positional_inputs = callback._traces[id(module)]["attention_inputs"]
        torch.testing.assert_close(positional_inputs[0], query)
        torch.testing.assert_close(positional_inputs[1], key)
        self.assertIsNone(positional_inputs[2])
        processor.compute_attention(qkv, mask)
        torch.testing.assert_close(
            callback._traces[id(module)]["attention_inputs"][2],
            mask,
        )
        callback._AttentionMonitorCallback__cleanup()
        self.assertIs(processor.compute_attention, compute_attention)

    def test_private_weight_wrappers_cover_both_processor_names_and_kwargs(self):
        callback = AttentionMonitorCallback()
        weights = torch.tensor([[[0.25, 0.75]]])
        method_names = (
            "_SelfAttentionProcessor__compute_masked_attention_weights",
            "_MixtureOfAttentionHeadsProcessor__compute_masked_attention_weights",
        )

        for method_name in method_names:
            with self.subTest(method_name=method_name):

                def compute_attention(*args, **kwargs):
                    return None

                def private_weights(*, scale):
                    return weights * scale

                processor = SimpleNamespace(compute_attention=compute_attention)
                setattr(processor, method_name, private_weights)
                module = SimpleNamespace(processor=processor)
                callback._AttentionMonitorCallback__wrap_processor(
                    module,
                    CaptureLightningModule(),
                )

                actual = getattr(processor, method_name)(scale=2.0)
                torch.testing.assert_close(actual, weights * 2.0)
                torch.testing.assert_close(
                    callback._traces[id(module)]["exact_weights"],
                    weights * 2.0,
                )
                callback._AttentionMonitorCallback__cleanup()
                self.assertIs(getattr(processor, method_name), private_weights)

    def test_approximate_weights_reject_invalid_inputs_and_mask_broadcasts(self):
        callback = AttentionMonitorCallback()
        compute = callback._AttentionMonitorCallback__compute_approximate_weights
        query = torch.ones(1, 1, 2)
        key = torch.ones(1, 1, 2)

        self.assertIsNone(compute(object(), key, None))
        self.assertIsNone(compute(query.squeeze(0), key, None))
        self.assertIsNone(
            compute(
                torch.ones(1, 2, 2),
                torch.ones(1, 3, 2),
                torch.zeros(4, 5),
            )
        )

        four_dimensional = torch.ones(1, 2, 3, 4)
        self.assertIsNotNone(compute(four_dimensional, four_dimensional, None))
        self.assertIsNone(compute(torch.ones(1, 1, 2, 3, 4), four_dimensional, None))
        self.assertIsNone(compute(four_dimensional, torch.ones(1, 1, 2, 3, 4), None))

    def test_approximate_weights_match_scaled_dot_product_with_additive_mask(self):
        callback = AttentionMonitorCallback()
        compute = callback._AttentionMonitorCallback__compute_approximate_weights
        query = torch.arange(24, dtype=torch.float32).view(2, 3, 4) / 10
        key = torch.arange(40, dtype=torch.float32).view(2, 5, 4) / 7
        mask = torch.tensor(
            [
                [0.0, -1.0, 0.5, 0.0, -0.5],
                [1.0, 0.0, -2.0, 0.5, 0.0],
                [0.0, 0.25, 0.0, -0.75, 1.0],
            ]
        )

        actual = compute(query, key, mask)
        expected = torch.softmax(
            torch.matmul(query * (4**-0.5), key.transpose(-2, -1)) + mask,
            dim=-1,
        )

        torch.testing.assert_close(actual, expected)

    def test_forward_hook_uses_returned_weights_and_auxiliary_loss_fallback(self):
        callback = AttentionMonitorCallback(log_every_n_steps=1)
        module = SimpleNamespace(num_heads=1, dropout_probability=0.0)
        lightning_module = CaptureLightningModule()
        callback._entropy_history["attention"] = []
        callback._max_probability_history["attention"] = []
        hook = callback._AttentionMonitorCallback__make_forward_hook(
            "attention",
            module,
            lightning_module,
        )
        weights = torch.ones(1, 1, 1)
        auxiliary_loss = torch.tensor(2.0)

        hook(None, (), (None, weights, auxiliary_loss))

        trace = callback._traces[id(module)]
        torch.testing.assert_close(trace["exact_weights"], weights)
        torch.testing.assert_close(trace["auxiliary_loss"], auxiliary_loss)
        self.assertEqual(trace["exact_weights"].data_ptr(), weights.data_ptr())
        self.assertEqual(trace["auxiliary_loss"].data_ptr(), auxiliary_loss.data_ptr())
        self.assertNotIn("output", trace)
        self.assertIn(
            "attention/attention/auxiliary_loss",
            lightning_module.logged_tags,
        )

    def test_parse_forward_output_supports_non_tuple_and_non_tensor_values(self):
        callback = AttentionMonitorCallback()
        parse = callback._AttentionMonitorCallback__parse_forward_output
        tensor = torch.ones(1, requires_grad=True)

        output, weights, loss = parse(tensor)
        self.assertFalse(output.requires_grad)
        self.assertIsNone(weights)
        self.assertIsNone(loss)
        self.assertEqual(parse(object()), (None, None, None))
        self.assertEqual(parse(()), (None, None, None))

        weights_tensor = torch.full((1,), 2.0, requires_grad=True)
        loss_tensor = torch.full((1,), 3.0, requires_grad=True)
        parsed_one = parse((tensor,))
        parsed_two = parse((tensor, weights_tensor))
        parsed_three = parse((tensor, weights_tensor, loss_tensor))
        self.assertEqual(parsed_one, (tensor.detach(), None, None))
        self.assertEqual(parsed_two, (tensor.detach(), weights_tensor.detach(), None))
        self.assertEqual(
            parsed_three,
            (tensor.detach(), weights_tensor.detach(), loss_tensor.detach()),
        )

    def test_trace_logging_uses_approximate_fallback_and_boolean_coverage(self):
        callback = AttentionMonitorCallback(log_every_n_steps=1)
        module = SimpleNamespace(num_heads=1, dropout_probability=0.0)
        lightning_module = CaptureLightningModule()
        weights = torch.tensor([[[0.25, 0.75]]])
        boolean_mask = torch.tensor([False, True])
        trace = {
            "attention_inputs": (None, None, boolean_mask),
            "approximate_weights": weights,
        }

        callback._AttentionMonitorCallback__log_trace(
            lightning_module,
            "attention",
            module,
            trace,
        )

        self.assertEqual(
            lightning_module.logged_value("attention/attention/mask_coverage"),
            torch.tensor(0.5),
        )
        self.assertIn(
            "attention/attention/approximate_entropy_mean",
            lightning_module.logged_tags,
        )
        self.assertNotIn("attention", callback._entropy_history)

    def test_mask_coverage_handles_singletons_and_additive_values_exactly(self):
        callback = AttentionMonitorCallback()
        coverage = callback._AttentionMonitorCallback__mask_coverage

        torch.testing.assert_close(coverage(None), torch.tensor(0.0))
        torch.testing.assert_close(coverage(torch.empty(0)), torch.tensor(0.0))
        torch.testing.assert_close(
            coverage(torch.tensor([True])),
            torch.tensor(1.0),
        )
        torch.testing.assert_close(
            coverage(torch.tensor([2.0])),
            torch.tensor(1.0),
        )
        torch.testing.assert_close(
            coverage(torch.tensor([0.0, 1.0, 1.0])),
            torch.tensor(2 / 3),
        )

    def test_tensor_norm_logging_reduces_only_the_embedding_axis(self):
        callback = AttentionMonitorCallback()
        module = CaptureLightningModule()
        values = torch.arange(24, dtype=torch.float32).view(2, 3, 4)

        callback._AttentionMonitorCallback__log_tensor_norm_mean(
            module,
            "attention",
            "q",
            values,
        )

        self.assertEqual(module.logged_tags, ["attention/q_norm_mean"])
        torch.testing.assert_close(
            module.logged_value("attention/q_norm_mean"),
            values.norm(dim=-1).mean(),
        )

    def test_trace_logging_keeps_query_key_and_value_metrics_distinct(self):
        callback = AttentionMonitorCallback()
        module = SimpleNamespace(num_heads=1, dropout_probability=0.0)
        lightning_module = CaptureLightningModule()
        query = torch.tensor([[[3.0, 4.0]]])
        key = torch.tensor([[[0.0, 12.0]]])
        value = torch.tensor([[[8.0, 15.0]]])

        callback._AttentionMonitorCallback__log_trace(
            lightning_module,
            "attention",
            module,
            {"qkv": (query, key, value)},
        )

        prefix = "attention/attention"
        torch.testing.assert_close(
            lightning_module.logged_value(f"{prefix}/q_norm_mean"),
            torch.tensor(5.0),
        )
        torch.testing.assert_close(
            lightning_module.logged_value(f"{prefix}/k_norm_mean"),
            torch.tensor(12.0),
        )
        torch.testing.assert_close(
            lightning_module.logged_value(f"{prefix}/v_norm_mean"),
            torch.tensor(17.0),
        )

    def test_empty_trace_logs_baseline_metrics_without_optional_values(self):
        callback = AttentionMonitorCallback()
        module = SimpleNamespace(num_heads=1, dropout_probability=0.25)
        lightning_module = CaptureLightningModule()

        callback._AttentionMonitorCallback__log_trace(
            lightning_module,
            "attention",
            module,
            {},
        )

        self.assertEqual(
            lightning_module.logged_tags,
            [
                "attention/attention/configured_dropout_probability",
                "attention/attention/mask_coverage",
            ],
        )
        torch.testing.assert_close(
            lightning_module.logged_value(
                "attention/attention/configured_dropout_probability"
            ),
            torch.tensor(0.25),
        )

    def test_trace_logging_records_exact_optional_values_and_zero_fraction(self):
        callback = AttentionMonitorCallback()
        module = SimpleNamespace(num_heads=1, dropout_probability=0.25)
        lightning_module = CaptureLightningModule()
        callback._entropy_history["attention"] = []
        callback._max_probability_history["attention"] = []
        weights = torch.tensor([[[0.0, 2.0, 3.0]]])
        trace = {
            "auxiliary_loss": torch.tensor([2.0, 4.0]),
            "exact_weights": weights,
        }

        callback._AttentionMonitorCallback__log_trace(
            lightning_module,
            "attention",
            module,
            trace,
        )

        torch.testing.assert_close(
            lightning_module.logged_value("attention/attention/auxiliary_loss"),
            torch.tensor(3.0),
        )
        torch.testing.assert_close(
            lightning_module.logged_value(
                "attention/attention/configured_dropout_probability"
            ),
            torch.tensor(0.25),
        )
        torch.testing.assert_close(
            lightning_module.logged_value("attention/attention/dropout_zero_fraction"),
            torch.tensor(1 / 3),
        )

    def test_trace_logging_defaults_missing_dropout_probability_to_zero(self):
        callback = AttentionMonitorCallback()
        lightning_module = CaptureLightningModule()

        callback._AttentionMonitorCallback__log_trace(
            lightning_module,
            "attention",
            SimpleNamespace(num_heads=1),
            {},
        )

        torch.testing.assert_close(
            lightning_module.logged_value(
                "attention/attention/configured_dropout_probability"
            ),
            torch.tensor(0.0),
        )

    def test_weight_stats_cover_per_head_scalars_history_and_layouts(self):
        callback = AttentionMonitorCallback(
            history_size=1,
            log_per_head_scalars=True,
        )
        module = SimpleNamespace(num_heads=2)
        lightning_module = CaptureLightningModule()
        callback._entropy_history["attention"] = []
        callback._max_probability_history["attention"] = []
        batch_head_weights = torch.tensor(
            [[[[0.25, 0.75]], [[0.5, 0.5]]]],
        )

        callback._AttentionMonitorCallback__log_weight_stats(
            lightning_module,
            "attention",
            module,
            batch_head_weights,
            approximate=False,
        )
        callback._AttentionMonitorCallback__log_weight_stats(
            lightning_module,
            "attention",
            module,
            batch_head_weights.flip(-1),
            approximate=False,
        )

        self.assertIn(
            "attention/attention/head_0/entropy",
            lightning_module.logged_tags,
        )
        self.assertIn(
            "attention/attention/head_1/max_probability",
            lightning_module.logged_tags,
        )
        self.assertEqual(len(callback._entropy_history["attention"]), 1)
        self.assertEqual(len(callback._max_probability_history["attention"]), 1)

        reshape = callback._AttentionMonitorCallback__reshape_weights_by_head
        torch.testing.assert_close(
            reshape(module, batch_head_weights),
            batch_head_weights,
        )
        expert_weights = torch.ones(1, 3, 2, 1, 2)
        self.assertEqual(reshape(module, expert_weights).shape, (3, 2, 1, 2))
        self.assertIsNone(reshape(module, torch.ones(3, 4, 1, 2)))
        self.assertIsNone(reshape(SimpleNamespace(num_heads=0), torch.ones(1, 1, 1)))

        flattened = torch.arange(60, dtype=torch.float32).view(4, 3, 5)
        torch.testing.assert_close(
            reshape(module, flattened),
            flattened.view(2, 2, 3, 5),
        )
        self.assertIsNone(reshape(module, torch.ones(4, 3, 2, 2)))

    def test_per_head_statistics_match_the_manual_probability_equations(self):
        callback = AttentionMonitorCallback()
        module = SimpleNamespace(num_heads=2)
        weights = torch.tensor(
            [
                [
                    [[0.1, 0.2, 0.3], [0.2, 0.1, 0.1]],
                    [[0.05, 0.15, 0.2], [0.3, 0.2, 0.1]],
                ]
            ]
        )

        entropy, maximum = callback._AttentionMonitorCallback__per_head_stats(
            module,
            weights,
        )

        normalized = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        expected_entropy = (
            -(normalized.clamp_min(1e-12).log() * normalized)
            .sum(dim=-1)
            .mean(dim=(0, 2))
        )
        expected_maximum = normalized.max(dim=-1).values.mean(dim=(0, 2))
        torch.testing.assert_close(entropy, expected_entropy)
        torch.testing.assert_close(maximum, expected_maximum)

    def test_weight_logging_records_exact_head_values_and_dead_head_boundary(self):
        callback = AttentionMonitorCallback(log_per_head_scalars=True)
        module = SimpleNamespace(num_heads=2)
        lightning_module = CaptureLightningModule()
        callback._entropy_history["attention"] = []
        callback._max_probability_history["attention"] = []
        entropy = torch.tensor([callback.DEAD_HEAD_ENTROPY_FLOOR, 0.5])
        maximum = torch.tensor([0.9, 0.75])

        with (
            patch.object(
                callback,
                "_AttentionMonitorCallback__per_head_stats",
                return_value=(entropy, maximum),
            ),
            patch.object(
                callback,
                "_AttentionMonitorCallback__log_visual_summaries",
            ),
        ):
            callback._AttentionMonitorCallback__log_weight_stats(
                lightning_module,
                "attention",
                module,
                torch.ones(1, 2, 1, 1),
                approximate=False,
            )

        expected = {
            "attention/attention/entropy_mean": entropy.mean(),
            "attention/attention/max_probability_mean": maximum.mean(),
            "attention/attention/dead_head_fraction": torch.tensor(0.5),
            "attention/attention/head_0/entropy": entropy[0],
            "attention/attention/head_1/entropy": entropy[1],
            "attention/attention/head_0/max_probability": maximum[0],
            "attention/attention/head_1/max_probability": maximum[1],
        }
        self.assertEqual(set(lightning_module.logged_tags), set(expected))
        for tag, value in expected.items():
            torch.testing.assert_close(lightning_module.logged_value(tag), value)

    def test_weight_logging_requires_a_strict_boolean_mode(self):
        callback = AttentionMonitorCallback()

        with self.assertRaises(TypeError) as raised:
            callback._AttentionMonitorCallback__log_weight_stats(
                CaptureLightningModule(),
                "attention",
                SimpleNamespace(num_heads=1),
                torch.ones(1, 1, 1),
                approximate=None,
            )

        self.assertEqual(str(raised.exception), "approximate must be a bool.")

    def test_empty_or_invalid_weight_layouts_produce_no_metrics(self):
        callback = AttentionMonitorCallback()
        lightning_module = CaptureLightningModule()
        invalid_module = SimpleNamespace(num_heads=0)

        callback._AttentionMonitorCallback__log_weight_stats(
            lightning_module,
            "attention",
            invalid_module,
            torch.ones(1, 1, 1),
            approximate=False,
        )
        empty_entropy, empty_probability = (
            callback._AttentionMonitorCallback__per_head_stats(
                SimpleNamespace(num_heads=2),
                torch.empty(0, 2, 1, 1),
            )
        )

        self.assertEqual(lightning_module.logged, [])
        self.assertEqual(empty_entropy.numel(), 0)
        self.assertEqual(empty_probability.numel(), 0)

    def test_visual_fallbacks_tolerate_missing_capabilities_and_empty_history(self):
        callback = AttentionMonitorCallback()
        log_heatmap = callback._AttentionMonitorCallback__log_heatmap
        add_image_experiment = CaptureLightningModule().logger.experiment

        log_heatmap(object(), "tag", [torch.ones(1)], 0)
        log_heatmap(add_image_experiment, "tag", [], 0)
        log_heatmap(add_image_experiment, "tag", [torch.empty(0)], 0)

        module = CaptureLightningModule()
        module.logger = SimpleNamespace(experiment=object())
        callback._entropy_history["attention"] = []
        callback._max_probability_history["attention"] = []
        callback._AttentionMonitorCallback__log_weight_stats(
            module,
            "attention",
            SimpleNamespace(num_heads=1),
            torch.ones(1, 1, 1),
            approximate=False,
        )
        self.assertIn("attention/attention/entropy_mean", module.logged_tags)

    def test_heatmap_padding_orientation_normalization_and_metadata_are_exact(self):
        callback = AttentionMonitorCallback()
        experiment = CaptureExperiment()
        history = [torch.tensor([0.1, 0.2]), torch.tensor([0.3])]

        callback._AttentionMonitorCallback__log_heatmap(
            experiment,
            "attention/heatmap",
            history,
            7,
        )

        self.assertEqual(len(experiment.images), 1)
        tag, image, step, dataformats = experiment.images[0]
        self.assertEqual(tag, "attention/heatmap")
        self.assertEqual(step, 7)
        self.assertEqual(dataformats, "CHW")
        expected = torch.tensor([[[1 / 3, 1.0], [2 / 3, 0.0]]])
        torch.testing.assert_close(image, expected)

    def test_visual_summary_dispatch_forwards_exact_tags_values_and_step(self):
        callback = AttentionMonitorCallback()
        module = CaptureLightningModule()
        module.global_step = 7
        experiment = module.logger.experiment
        entropy = torch.tensor([0.1, 0.2])
        maximum = torch.tensor([0.7, 0.8])
        callback._entropy_history["attention"] = []
        callback._max_probability_history["attention"] = []

        with (
            patch.object(
                callback,
                "_AttentionMonitorCallback__log_histogram",
            ) as histogram,
            patch.object(
                callback,
                "_AttentionMonitorCallback__log_heatmap",
            ) as heatmap,
        ):
            callback._AttentionMonitorCallback__log_visual_summaries(
                module,
                "attention",
                entropy,
                maximum,
            )

        self.assertEqual(
            histogram.call_args_list,
            [
                call(
                    experiment,
                    "attention/attention/histogram/entropy_by_head",
                    entropy,
                    7,
                ),
                call(
                    experiment,
                    "attention/attention/histogram/max_probability_by_head",
                    maximum,
                    7,
                ),
            ],
        )
        self.assertEqual(
            heatmap.call_args_list,
            [
                call(
                    experiment,
                    "attention/attention/heatmap/entropy_by_head",
                    callback._entropy_history["attention"],
                    7,
                ),
                call(
                    experiment,
                    "attention/attention/heatmap/max_probability_by_head",
                    callback._max_probability_history["attention"],
                    7,
                ),
            ],
        )

    def test_visual_summary_fallbacks_cover_missing_logger_and_step(self):
        callback = AttentionMonitorCallback()
        entropy = torch.tensor([0.1])
        maximum = torch.tensor([0.9])
        callback._entropy_history["attention"] = []
        callback._max_probability_history["attention"] = []

        self.assertIsNone(
            callback._AttentionMonitorCallback__log_visual_summaries(
                SimpleNamespace(),
                "attention",
                entropy,
                maximum,
            )
        )

        experiment = CaptureExperiment()
        module = SimpleNamespace(logger=SimpleNamespace(experiment=experiment))
        with (
            patch.object(
                callback,
                "_AttentionMonitorCallback__log_histogram",
            ) as histogram,
            patch.object(
                callback,
                "_AttentionMonitorCallback__log_heatmap",
            ) as heatmap,
        ):
            callback._AttentionMonitorCallback__log_visual_summaries(
                module,
                "attention",
                entropy,
                maximum,
            )

        for logged_call in histogram.call_args_list + heatmap.call_args_list:
            self.assertEqual(logged_call.args[-1], 0)

    def test_forward_hook_preserves_private_weights_and_exact_trace_schema(self):
        callback = AttentionMonitorCallback(log_every_n_steps=1)
        module = SimpleNamespace(num_heads=1, dropout_probability=0.0)
        lightning_module = CaptureLightningModule()
        output = torch.tensor([1.0])
        returned_weights = torch.tensor([[[0.25, 0.75]]])
        private_weights = torch.tensor([[[0.6, 0.4]]])
        hook = callback._AttentionMonitorCallback__make_forward_hook(
            "attention",
            module,
            lightning_module,
        )

        with patch.object(
            callback,
            "_AttentionMonitorCallback__log_trace",
        ):
            hook(None, (), output)
            first_trace = callback._traces[id(module)]
            self.assertEqual(set(first_trace), {"name", "output"})
            self.assertEqual(first_trace["name"], "attention")

            callback._traces[id(module)] = {"exact_weights": private_weights}
            hook(None, (), (output, returned_weights, None))

        trace = callback._traces[id(module)]
        self.assertIs(trace["exact_weights"], private_weights)

    def test_fit_start_passes_name_module_and_training_step_to_wrappers(self):
        attention = self.attention()
        module = CaptureLightningModule(attn=attention)
        module.global_step = 1
        callback = AttentionMonitorCallback(log_every_n_steps=2)
        callback.on_fit_start(TrainerStub(), module)

        attention(*self.qkv())

        trace = callback._traces[id(attention)]
        self.assertEqual(trace["name"], "attn")
        self.assertNotIn("approximate_weights", trace)
        self.assertNotIn(id(None), callback._traces)
        callback.on_fit_end(TrainerStub(), module)


if __name__ == "__main__":
    unittest.main()
