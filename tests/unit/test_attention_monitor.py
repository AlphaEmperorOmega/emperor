import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from emperor.attention import (
    AttentionMonitorCallback,
    IndependentAttentionConfig,
    SelfAttentionConfig,
)
from emperor.attention._monitoring.callback import (
    _AttentionDiagnosticsTracker,
    _AttentionDiagnosticsTrackerManager,
    _AttentionMethodInstrumentation,
    _AttentionMethodObserver,
    _AttentionTrackingContext,
)
from emperor.attention._monitoring.diagnostics import (
    _AttentionDiagnosticMetrics,
    _AttentionDiagnostics,
    _AttentionMonitorAdapter,
    _AttentionObservation,
    _resolve_attention_monitor_adapter,
    _SelfAttentionMonitorAdapter,
)
from emperor.attention._runtime import MultiHeadAttentionInputs
from emperor.attention._variants.mixture.monitoring import (
    _MixtureOfAttentionHeadsMonitorAdapter,
)
from support.attention import build_attention_config
from support.monitor import (
    CaptureLightningModule,
    NoExperimentLightningModule,
    TrainerStub,
    orchestration_calls,
    same_bound_method,
)


class InstrumentedAttention(torch.nn.Module):
    def __init__(
        self,
        *,
        private_method_name: str | None = None,
        private_weights: torch.Tensor | None = None,
        returned_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.projector = SimpleNamespace(
            compute_qkv_projections=lambda *, attention_inputs: attention_inputs
        )
        self.processor = SimpleNamespace(
            compute_attention=lambda *, attention_inputs: attention_inputs.value,
            _compute_raw_masked_attention_logits=(
                lambda *, scale: private_weights * scale
            ),
            _compute_normalized_attention_weights=(
                lambda *, scale: private_weights * scale
            ),
        )
        self.private_method_name = private_method_name
        self.private_weights = private_weights
        self.returned_weights = returned_weights
        if private_method_name is not None:
            setattr(
                self.processor,
                private_method_name,
                lambda *, scale: private_weights * scale,
            )

    def forward(
        self,
        attention_inputs: MultiHeadAttentionInputs,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        projected_inputs = self.projector.compute_qkv_projections(
            attention_inputs=attention_inputs
        )
        if attention_mask is not None:
            projected_inputs = replace(
                projected_inputs,
                merged_attention_mask=attention_mask,
            )
        output = self.processor.compute_attention(
            attention_inputs=projected_inputs,
        )
        if self.private_method_name is not None:
            self.processor._compute_raw_masked_attention_logits(scale=0.5)
            self.processor._compute_normalized_attention_weights(scale=1.0)
            getattr(self.processor, self.private_method_name)(scale=2.0)
        return output, self.returned_weights, torch.tensor(3.0, requires_grad=True)


class ProcessorOnlyAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.processor = SimpleNamespace(
            compute_attention=lambda *, attention_inputs: attention_inputs.value
        )

    def forward(
        self,
        attention_inputs: MultiHeadAttentionInputs,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            attention_inputs = replace(
                attention_inputs,
                merged_attention_mask=attention_mask,
            )
        return self.processor.compute_attention(
            attention_inputs=attention_inputs,
        )


class PositionalProcessorAttention(torch.nn.Module):
    def __init__(self, *, include_mask: bool) -> None:
        super().__init__()
        self.include_mask = include_mask
        self.processor = SimpleNamespace(
            compute_attention=lambda attention_inputs: attention_inputs.value
        )

    def forward(
        self,
        attention_inputs: MultiHeadAttentionInputs,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.include_mask and attention_mask is not None:
            attention_inputs = replace(
                attention_inputs,
                merged_attention_mask=attention_mask,
            )
        return self.processor.compute_attention(attention_inputs)


class ExactWeightMonitorAdapter(_AttentionMonitorAdapter):
    @property
    def exact_weight_method_name(self) -> str:
        return "_compute_masked_attention_weights"


class MethodOwner:
    def compute(self, value, *, scale=1):
        return value * scale


class TestAttentionMethodInstrumentation(unittest.TestCase):
    def observer(self, label, records):
        return _AttentionMethodObserver(
            before_call=lambda args, kwargs: (label, args, kwargs),
            after_call=lambda call, result: records.append((*call, result)),
        )

    def test_last_subscription_restores_exact_attribute_provenance(self):
        instrumentation = _AttentionMethodInstrumentation()
        owner = MethodOwner()
        original_method = owner.compute
        records = []

        remove = instrumentation.subscribe(
            owner,
            "compute",
            self.observer("first", records),
        )

        self.assertIn("compute", vars(owner))
        self.assertEqual(owner.compute(3, scale=2), 6)
        remove()

        self.assertNotIn("compute", vars(owner))
        self.assertTrue(same_bound_method(owner.compute, original_method))
        self.assertEqual(records, [("first", (3,), {"scale": 2}, 6)])
        self.assertEqual(instrumentation.probe_count, 0)

    def test_subscriptions_can_be_removed_repeatedly_and_in_either_order(self):
        for removal_order in ((0, 1), (1, 0)):
            with self.subTest(removal_order=removal_order):
                instrumentation = _AttentionMethodInstrumentation()
                owner = MethodOwner()
                original_method = owner.compute
                records = []
                removers = [
                    instrumentation.subscribe(
                        owner,
                        "compute",
                        self.observer(label, records),
                    )
                    for label in ("first", "second")
                ]

                self.assertEqual(owner.compute(2), 2)
                removers[removal_order[0]]()
                removers[removal_order[0]]()
                self.assertEqual(owner.compute(3), 3)
                removers[removal_order[1]]()
                removers[removal_order[1]]()

                first_label = ("first", "second")[removal_order[0]]
                surviving_label = ("first", "second")[removal_order[1]]
                self.assertEqual(
                    [record[0] for record in records],
                    ["first", "second", surviving_label],
                )
                self.assertNotEqual(first_label, surviving_label)
                self.assertNotIn("compute", vars(owner))
                self.assertTrue(same_bound_method(owner.compute, original_method))
                self.assertEqual(instrumentation.probe_count, 0)

    def test_instrumentation_keeps_distinct_owners_in_distinct_probes(self):
        instrumentation = _AttentionMethodInstrumentation()
        first_owner = MethodOwner()
        second_owner = MethodOwner()
        records = []
        removers = [
            instrumentation.subscribe(
                owner,
                "compute",
                self.observer(label, records),
            )
            for owner, label in (
                (first_owner, "first"),
                (second_owner, "second"),
            )
        ]

        self.assertEqual(instrumentation.probe_count, 2)
        self.assertEqual(first_owner.compute(1), 1)
        self.assertEqual(second_owner.compute(2), 2)
        self.assertEqual([record[0] for record in records], ["first", "second"])

        for remove in removers:
            remove()
        self.assertEqual(instrumentation.probe_count, 0)

    def test_probe_preserves_result_identity_and_original_exception(self):
        class IdentityOwner:
            @staticmethod
            def compute(value):
                if isinstance(value, BaseException):
                    raise value
                return value

        instrumentation = _AttentionMethodInstrumentation()
        owner = IdentityOwner()
        result = object()
        failure = RuntimeError("deliberate method failure")
        observed_results = []
        remove = instrumentation.subscribe(
            owner,
            "compute",
            _AttentionMethodObserver(
                before_call=lambda _args, _kwargs: None,
                after_call=lambda _call, observed: observed_results.append(observed),
            ),
        )

        self.assertIs(owner.compute(result), result)
        with self.assertRaises(RuntimeError) as raised:
            owner.compute(failure)

        self.assertIs(raised.exception, failure)
        self.assertEqual(observed_results, [result])
        remove()


def diagnostic_metrics(
    *,
    auxiliary_loss: torch.Tensor | None = None,
    per_head_entropy: torch.Tensor | None = None,
    per_head_max_probability: torch.Tensor | None = None,
    weight_source: str | None = None,
) -> _AttentionDiagnosticMetrics:
    return _AttentionDiagnosticMetrics(
        query_norm_mean=None,
        key_norm_mean=None,
        value_norm_mean=None,
        output_norm=None,
        auxiliary_loss=auxiliary_loss,
        configured_dropout_probability=torch.tensor(0.0),
        mask_coverage=torch.tensor(0.0),
        per_head_entropy=per_head_entropy,
        per_head_max_probability=per_head_max_probability,
        weight_source=weight_source,
        dropout_zero_fraction=None,
    )


class TestAttentionObservationAndTracker(unittest.TestCase):
    def attention_inputs(
        self,
        *,
        requires_grad: bool = False,
        merged_attention_mask: torch.Tensor | None = None,
    ) -> MultiHeadAttentionInputs:
        query = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=requires_grad)
        key = torch.tensor([[[2.0, 1.0], [4.0, 3.0]]], requires_grad=requires_grad)
        value = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]], requires_grad=requires_grad)
        return MultiHeadAttentionInputs(
            query=query,
            key=key,
            value=value,
            merged_attention_mask=merged_attention_mask,
        )

    def test_tracker_records_detached_typed_values(self):
        tracker = _AttentionDiagnosticsTracker("attention")
        mask = torch.tensor([[0.0, -1.0]], requires_grad=True)
        projected_inputs = self.attention_inputs(requires_grad=True)
        processor_inputs = self.attention_inputs(
            requires_grad=True,
            merged_attention_mask=mask,
        )
        private_weights = torch.tensor([[[0.2, 0.8]]], requires_grad=True)
        raw_attention_logits = torch.tensor(
            [[[1.0, -torch.inf]]],
            requires_grad=True,
        )
        normalized_attention_weights = torch.tensor(
            [[[1.0, 0.0]]],
            requires_grad=True,
        )
        returned_weights = torch.tensor([[[0.7, 0.3]]], requires_grad=True)
        output = torch.tensor([3.0, 4.0], requires_grad=True)
        auxiliary_loss = torch.tensor([2.0, 4.0], requires_grad=True)

        tracker.begin_observation()
        tracker.record_projected_inputs(projected_inputs)
        tracker.record_processor_inputs(processor_inputs)
        tracker.record_raw_attention_logits(raw_attention_logits)
        tracker.record_normalized_attention_weights(normalized_attention_weights)
        tracker.record_exact_attention_weights(private_weights)
        tracker.record_forward_output((output, returned_weights, auxiliary_loss))

        observation = tracker.latest_observation
        self.assertIsInstance(observation, _AttentionObservation)
        self.assertIsNotNone(observation.projected_inputs)
        self.assertIsNotNone(observation.processor_inputs)
        captured_tensors = (
            observation.projected_inputs.query,
            observation.projected_inputs.key,
            observation.projected_inputs.value,
            observation.processor_inputs.query,
            observation.processor_inputs.key,
            observation.processor_inputs.value,
            observation.processor_inputs.merged_attention_mask,
            observation.raw_attention_logits,
            observation.normalized_attention_weights,
            observation.exact_attention_weights,
            observation.restored_output,
            observation.auxiliary_loss,
        )
        for captured_tensor in captured_tensors:
            self.assertIsInstance(captured_tensor, torch.Tensor)
            self.assertFalse(captured_tensor.requires_grad)
        self.assertEqual(
            observation.raw_attention_logits.data_ptr(),
            raw_attention_logits.data_ptr(),
        )
        self.assertEqual(
            observation.normalized_attention_weights.data_ptr(),
            normalized_attention_weights.data_ptr(),
        )
        self.assertEqual(
            observation.exact_attention_weights.data_ptr(),
            private_weights.data_ptr(),
        )
        self.assertEqual(observation.restored_output.data_ptr(), output.data_ptr())

    def test_private_exact_weights_take_priority_over_returned_weights(self):
        tracker = _AttentionDiagnosticsTracker("attention")
        private_weights = torch.tensor([[[0.2, 0.8]]])
        returned_weights = torch.tensor([[[0.7, 0.3]]])

        tracker.record_exact_attention_weights(private_weights)
        tracker.record_forward_output((torch.ones(1), returned_weights, None))

        self.assertEqual(
            tracker.latest_observation.exact_attention_weights.data_ptr(),
            private_weights.data_ptr(),
        )

    def test_returned_weights_are_used_when_private_weights_are_absent(self):
        tracker = _AttentionDiagnosticsTracker("attention")
        returned_weights = torch.tensor([[[0.7, 0.3]]], requires_grad=True)

        tracker.record_forward_output((object(), returned_weights, object()))

        observation = tracker.latest_observation
        self.assertIsNone(observation.restored_output)
        self.assertIsNone(observation.auxiliary_loss)
        self.assertFalse(observation.exact_attention_weights.requires_grad)
        self.assertEqual(
            observation.exact_attention_weights.data_ptr(),
            returned_weights.data_ptr(),
        )

    def test_non_tuple_outputs_and_new_observations_are_handled_explicitly(self):
        tracker = _AttentionDiagnosticsTracker("attention")
        output = torch.ones(1, requires_grad=True)

        tracker.record_forward_output(output)
        self.assertFalse(tracker.latest_observation.restored_output.requires_grad)
        tracker.begin_observation()

        self.assertEqual(tracker.latest_observation, _AttentionObservation())

    def test_tuple_outputs_are_parsed_only_at_positions_that_exist(self):
        output = torch.tensor([1.0], requires_grad=True)
        returned_weights = torch.tensor([[[0.25, 0.75]]], requires_grad=True)
        cases = (
            ((), None, None),
            ((output,), output, None),
            ((output, returned_weights), output, returned_weights),
        )

        for forward_output, expected_output, expected_weights in cases:
            with self.subTest(tuple_length=len(forward_output)):
                tracker = _AttentionDiagnosticsTracker("attention")

                tracker.record_forward_output(forward_output)

                observation = tracker.latest_observation
                if expected_output is None:
                    self.assertIsNone(observation.restored_output)
                else:
                    self.assertEqual(
                        observation.restored_output.data_ptr(),
                        expected_output.data_ptr(),
                    )
                if expected_weights is None:
                    self.assertIsNone(observation.exact_attention_weights)
                else:
                    self.assertEqual(
                        observation.exact_attention_weights.data_ptr(),
                        expected_weights.data_ptr(),
                    )
                self.assertIsNone(observation.auxiliary_loss)

    def test_invalid_diagnostic_values_are_ignored_without_overwriting_state(self):
        tracker = _AttentionDiagnosticsTracker("attention")
        raw_logits = torch.tensor([[[1.0, -1.0]]])
        normalized_weights = torch.tensor([[[0.75, 0.25]]])

        tracker.record_raw_attention_logits(raw_logits)
        tracker.record_normalized_attention_weights(normalized_weights)
        recorded_raw_logits = tracker.latest_observation.raw_attention_logits
        recorded_normalized_weights = (
            tracker.latest_observation.normalized_attention_weights
        )

        tracker.record_projected_inputs(object())
        tracker.record_raw_attention_logits(object())
        tracker.record_normalized_attention_weights(None)
        tracker.record_exact_attention_weights(object())

        self.assertIsNone(tracker.latest_observation.projected_inputs)
        self.assertIs(
            tracker.latest_observation.raw_attention_logits,
            recorded_raw_logits,
        )
        self.assertIs(
            tracker.latest_observation.normalized_attention_weights,
            recorded_normalized_weights,
        )
        self.assertEqual(recorded_raw_logits.data_ptr(), raw_logits.data_ptr())
        self.assertEqual(
            recorded_normalized_weights.data_ptr(),
            normalized_weights.data_ptr(),
        )
        self.assertIsNone(tracker.latest_observation.exact_attention_weights)
        self.assertIsNone(tracker.latest_observation.exact_attention_weights)


class TestAttentionDiagnostics(unittest.TestCase):
    def diagnostics(self) -> _AttentionDiagnostics:
        return _AttentionDiagnostics()

    def attention_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> MultiHeadAttentionInputs:
        return MultiHeadAttentionInputs(
            query=query,
            key=key,
            value=torch.ones_like(key),
            merged_attention_mask=attention_mask,
        )

    def test_boolean_mask_uses_true_as_masked(self):
        query = torch.ones(1, 1, 1)
        key = torch.ones(1, 2, 1)
        mask = torch.tensor([[False, True]])

        weights = self.diagnostics().approximate_attention_weights(
            self.attention_inputs(query, key, mask),
        )

        torch.testing.assert_close(weights, torch.tensor([[[1.0, 0.0]]]))

    def test_fully_masked_approximation_rows_are_exact_zero_and_finite(self):
        query = torch.tensor([[[1.0], [2.0]]])
        key = torch.tensor([[[1.0], [2.0], [3.0]]])
        boolean_mask = torch.tensor(
            [
                [True, True, True],
                [False, True, False],
            ]
        )
        additive_mask = torch.zeros_like(boolean_mask, dtype=torch.float32).masked_fill(
            boolean_mask,
            -torch.inf,
        )
        selected_probabilities = torch.softmax(torch.tensor([2.0, 6.0]), dim=0)
        expected_weights = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [
                        selected_probabilities[0],
                        0.0,
                        selected_probabilities[1],
                    ],
                ]
            ]
        )
        expected_entropy = (
            -(selected_probabilities * selected_probabilities.log()).sum() / 2
        )
        expected_maximum = selected_probabilities.max() / 2

        for attention_mask in (boolean_mask, additive_mask):
            with self.subTest(mask_dtype=attention_mask.dtype):
                diagnostics = self.diagnostics()
                processor_inputs = self.attention_inputs(
                    query,
                    key,
                    attention_mask,
                )

                weights = diagnostics.approximate_attention_weights(
                    processor_inputs,
                )
                metrics = diagnostics.calculate(
                    _AttentionObservation(
                        processor_inputs=processor_inputs,
                    ),
                    num_heads=1,
                    configured_dropout_probability=0.0,
                )

                self.assertTrue(torch.isfinite(weights).all())
                torch.testing.assert_close(weights, expected_weights)
                self.assertTrue(torch.isfinite(metrics.per_head_entropy).all())
                self.assertTrue(torch.isfinite(metrics.per_head_max_probability).all())
                torch.testing.assert_close(
                    metrics.per_head_entropy,
                    expected_entropy.unsqueeze(0),
                )
                torch.testing.assert_close(
                    metrics.per_head_max_probability,
                    expected_maximum.unsqueeze(0),
                )

    def test_scaled_dot_product_approximation_matches_manual_equation(self):
        query = torch.arange(24, dtype=torch.float32).view(2, 3, 4) / 10
        key = torch.arange(40, dtype=torch.float32).view(2, 5, 4) / 7
        mask = torch.tensor(
            [
                [0.0, -1.0, 0.5, 0.0, -0.5],
                [1.0, 0.0, -2.0, 0.5, 0.0],
                [0.0, 0.25, 0.0, -0.75, 1.0],
            ]
        )

        actual = self.diagnostics().approximate_attention_weights(
            self.attention_inputs(query, key, mask),
        )
        expected = torch.softmax(
            torch.matmul(query * (4**-0.5), key.transpose(-2, -1)) + mask,
            dim=-1,
        )

        torch.testing.assert_close(actual, expected)

    def test_approximation_rejects_invalid_ranks_and_mask_broadcasts(self):
        diagnostics = self.diagnostics()
        valid_query = torch.ones(1, 2, 2)
        valid_key = torch.ones(1, 3, 2)
        invalid_mask = torch.zeros(4, 5)
        cases = (
            self.attention_inputs(valid_query.squeeze(0), valid_key),
            self.attention_inputs(torch.ones(1, 1, 2, 3, 4), valid_key),
            self.attention_inputs(valid_query, torch.ones(1, 1, 2, 3, 4)),
        )

        for processor_inputs in cases:
            with self.subTest(query_shape=processor_inputs.query.shape):
                self.assertIsNone(
                    diagnostics.approximate_attention_weights(processor_inputs)
                )
        self.assertIsNone(
            diagnostics.approximate_attention_weights(
                self.attention_inputs(valid_query, valid_key, invalid_mask),
            )
        )

    def test_mask_coverage_handles_empty_boolean_and_additive_masks(self):
        diagnostics = self.diagnostics()
        cases = (
            (None, torch.tensor(0.0)),
            (torch.empty(0), torch.tensor(0.0)),
            (torch.tensor([True]), torch.tensor(1.0)),
            (torch.tensor([2.0]), torch.tensor(1.0)),
            (torch.tensor([0.0, 1.0, 1.0]), torch.tensor(2 / 3)),
        )

        for attention_mask, expected in cases:
            with self.subTest(attention_mask=attention_mask):
                torch.testing.assert_close(
                    diagnostics.mask_coverage(attention_mask),
                    expected,
                )

    def test_standard_monitor_adapter_supports_rank_three_and_four_only(self):
        adapter = _AttentionMonitorAdapter()
        batch_head_weights = torch.arange(24, dtype=torch.float32).view(3, 2, 2, 2)
        head_first_weights = batch_head_weights.permute(1, 0, 2, 3)
        expert_weights = torch.ones(1, 3, 2, 1, 2)
        flattened_weights = torch.arange(60, dtype=torch.float32).view(4, 3, 5)

        torch.testing.assert_close(
            adapter.canonicalize(batch_head_weights, 2),
            batch_head_weights,
        )
        torch.testing.assert_close(
            adapter.canonicalize(head_first_weights, 2),
            batch_head_weights,
        )
        self.assertIsNone(adapter.canonicalize(expert_weights, 2))
        torch.testing.assert_close(
            adapter.canonicalize(flattened_weights, 2),
            flattened_weights.view(2, 2, 3, 5),
        )
        self.assertIsNone(adapter.canonicalize(torch.ones(3, 4, 1, 2), 2))
        self.assertIsNone(adapter.canonicalize(torch.ones(1, 1, 1), 0))

    def test_mixture_monitor_adapter_owns_rank_five_canonicalization(self):
        adapter = _MixtureOfAttentionHeadsMonitorAdapter()
        weights = torch.arange(96, dtype=torch.float32).view(2, 3, 2, 2, 4)

        canonical = adapter.canonicalize(weights, 2)

        self.assertEqual(canonical.shape, (6, 2, 2, 4))
        torch.testing.assert_close(canonical, weights.reshape(6, 2, 2, 4))
        self.assertIsNone(adapter.canonicalize(weights, 0))

        flattened = torch.arange(24, dtype=torch.float32).view(4, 2, 3)
        torch.testing.assert_close(
            adapter.canonicalize(flattened, 2),
            flattened.view(2, 2, 2, 3),
        )

    def test_monitor_adapter_resolution_is_owned_by_monitoring(self):
        attention = InstrumentedAttention()
        self_attention = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=1,
            source_sequence_length=1,
        ).build()

        default_adapter = _resolve_attention_monitor_adapter(attention)
        self_adapter = _resolve_attention_monitor_adapter(self_attention)

        self.assertIs(type(default_adapter), _AttentionMonitorAdapter)
        self.assertIsInstance(self_adapter, _SelfAttentionMonitorAdapter)
        self.assertFalse(hasattr(self_attention, "_MONITOR_ADAPTER"))

    def test_diagnostics_use_the_selected_mixture_adapter(self):
        weights = torch.tensor([0.25, 0.75, 0.5, 0.5, 0.1, 0.9, 0.8, 0.2]).view(
            1, 2, 2, 1, 2
        )

        metrics = self.diagnostics().calculate(
            _AttentionObservation(exact_attention_weights=weights),
            num_heads=2,
            configured_dropout_probability=0.0,
            monitor_adapter=_MixtureOfAttentionHeadsMonitorAdapter(),
        )

        self.assertEqual(metrics.weight_source, "exact")
        self.assertEqual(metrics.per_head_entropy.shape, (2,))
        self.assertEqual(metrics.per_head_max_probability.shape, (2,))

    def test_per_head_statistics_match_manual_probability_equations(self):
        weights = torch.tensor(
            [
                [
                    [[0.1, 0.2, 0.3], [0.2, 0.1, 0.1]],
                    [[0.05, 0.15, 0.2], [0.3, 0.2, 0.1]],
                ]
            ]
        )

        entropy, maximum = self.diagnostics().per_head_statistics(weights, 2)

        normalized = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        expected_entropy = (
            -(normalized.clamp_min(1e-12).log() * normalized)
            .sum(dim=-1)
            .mean(dim=(0, 2))
        )
        expected_maximum = normalized.max(dim=-1).values.mean(dim=(0, 2))
        torch.testing.assert_close(entropy, expected_entropy)
        torch.testing.assert_close(maximum, expected_maximum)

    def test_calculator_returns_exact_projection_output_and_mask_metrics(self):
        projected_inputs = MultiHeadAttentionInputs(
            query=torch.tensor([[[3.0, 4.0]]]),
            key=torch.tensor([[[0.0, 12.0]]]),
            value=torch.tensor([[[8.0, 15.0]]]),
        )
        processor_inputs = replace(
            projected_inputs,
            merged_attention_mask=torch.tensor([False, True]),
        )
        exact_weights = torch.tensor([[[0.0, 2.0, 3.0]]])
        raw_attention_logits = torch.tensor([[[1.0, -torch.inf, 3.0]]])
        normalized_attention_weights = torch.tensor([[[0.2, 0.3, 0.5]]])
        observation = _AttentionObservation(
            projected_inputs=projected_inputs,
            processor_inputs=processor_inputs,
            raw_attention_logits=raw_attention_logits,
            normalized_attention_weights=normalized_attention_weights,
            exact_attention_weights=exact_weights,
            restored_output=torch.tensor([3.0, 4.0]),
            auxiliary_loss=torch.tensor([2.0, 4.0]),
        )

        metrics = self.diagnostics().calculate(
            observation,
            num_heads=1,
            configured_dropout_probability=0.25,
        )

        self.assertIsInstance(metrics, _AttentionDiagnosticMetrics)
        torch.testing.assert_close(metrics.query_norm_mean, torch.tensor(5.0))
        torch.testing.assert_close(metrics.key_norm_mean, torch.tensor(12.0))
        torch.testing.assert_close(metrics.value_norm_mean, torch.tensor(17.0))
        torch.testing.assert_close(metrics.output_norm, torch.tensor(5.0))
        torch.testing.assert_close(metrics.auxiliary_loss, torch.tensor(3.0))
        torch.testing.assert_close(
            metrics.configured_dropout_probability,
            torch.tensor(0.25),
        )
        torch.testing.assert_close(metrics.mask_coverage, torch.tensor(0.5))
        torch.testing.assert_close(
            metrics.dropout_zero_fraction,
            torch.tensor(1 / 3),
        )
        torch.testing.assert_close(
            metrics.finite_raw_attention_logit_mean,
            torch.tensor(2.0),
        )
        torch.testing.assert_close(
            metrics.finite_raw_attention_logit_std,
            torch.tensor(1.0),
        )
        expected_pre_dropout_entropy, expected_pre_dropout_maximum = (
            self.diagnostics().per_head_statistics(
                normalized_attention_weights,
                1,
            )
        )
        torch.testing.assert_close(
            metrics.pre_dropout_per_head_entropy,
            expected_pre_dropout_entropy,
        )
        torch.testing.assert_close(
            metrics.pre_dropout_per_head_max_probability,
            expected_pre_dropout_maximum,
        )
        self.assertEqual(metrics.weight_source, "exact")

    def test_raw_logit_statistics_ignore_non_finite_values(self):
        diagnostics = self.diagnostics()

        mean, standard_deviation = diagnostics.finite_tensor_statistics(
            torch.tensor([-torch.inf, 1.0, 3.0, torch.inf, torch.nan])
        )

        torch.testing.assert_close(mean, torch.tensor(2.0))
        torch.testing.assert_close(standard_deviation, torch.tensor(1.0))
        self.assertEqual(diagnostics.finite_tensor_statistics(None), (None, None))
        self.assertEqual(
            diagnostics.finite_tensor_statistics(torch.tensor([-torch.inf])),
            (None, None),
        )

    def test_exact_weights_take_priority_over_approximation(self):
        processor_inputs = self.attention_inputs(
            torch.tensor([[[1.0], [2.0]]]),
            torch.tensor([[[1.0], [3.0]]]),
        )
        exact_weights = torch.tensor([[[0.25, 0.75], [0.5, 0.5]]])
        observation = _AttentionObservation(
            processor_inputs=processor_inputs,
            exact_attention_weights=exact_weights,
        )

        metrics = self.diagnostics().calculate(
            observation,
            num_heads=1,
            configured_dropout_probability=0.0,
        )

        expected_entropy, expected_maximum = self.diagnostics().per_head_statistics(
            exact_weights,
            1,
        )
        self.assertEqual(metrics.weight_source, "exact")
        torch.testing.assert_close(metrics.per_head_entropy, expected_entropy)
        torch.testing.assert_close(
            metrics.per_head_max_probability,
            expected_maximum,
        )

    def test_missing_exact_weights_use_approximation_without_dropout_metric(self):
        processor_inputs = self.attention_inputs(
            torch.tensor([[[1.0], [2.0]]]),
            torch.tensor([[[1.0], [3.0]]]),
        )

        metrics = self.diagnostics().calculate(
            _AttentionObservation(processor_inputs=processor_inputs),
            num_heads=1,
            configured_dropout_probability=0.0,
        )

        self.assertEqual(metrics.weight_source, "approximate")
        self.assertIsNotNone(metrics.per_head_entropy)
        self.assertIsNotNone(metrics.per_head_max_probability)
        self.assertIsNone(metrics.dropout_zero_fraction)

    def test_invalid_weights_produce_no_per_head_metrics(self):
        metrics = self.diagnostics().calculate(
            _AttentionObservation(exact_attention_weights=torch.ones(1, 1, 1)),
            num_heads=0,
            configured_dropout_probability=0.0,
        )

        self.assertEqual(metrics.weight_source, "exact")
        self.assertIsNone(metrics.per_head_entropy)
        self.assertIsNone(metrics.per_head_max_probability)
        torch.testing.assert_close(metrics.dropout_zero_fraction, torch.tensor(0.0))

    def test_missing_processor_and_weights_produce_no_weight_metrics(self):
        diagnostics = self.diagnostics()

        self.assertIsNone(diagnostics.approximate_attention_weights(None))
        self.assertEqual(
            diagnostics.per_head_statistics(None, num_heads=2),
            (None, None),
        )

        metrics = diagnostics.calculate(
            _AttentionObservation(),
            num_heads=2,
            configured_dropout_probability=0.0,
        )

        self.assertIsNone(metrics.weight_source)
        self.assertIsNone(metrics.per_head_entropy)
        self.assertIsNone(metrics.per_head_max_probability)
        self.assertIsNone(metrics.dropout_zero_fraction)


class TestAttentionDiagnosticsTrackerManager(unittest.TestCase):
    def attention_inputs(self) -> MultiHeadAttentionInputs:
        return MultiHeadAttentionInputs(
            query=torch.tensor([[[1.0, 0.0], [0.0, 2.0]]]),
            key=torch.tensor([[[1.0, 0.0], [1.0, 1.0]]]),
            value=torch.ones(1, 2, 2),
        )

    def test_manager_captures_kwargs_and_restores_methods(self):
        returned_weights = torch.tensor([[[0.25, 0.75]]])
        attention = InstrumentedAttention(returned_weights=returned_weights)
        original_projection = attention.projector.compute_qkv_projections
        original_attention = attention.processor.compute_attention
        observations = []
        manager = _AttentionDiagnosticsTrackerManager()
        attention_mask = torch.tensor([[0.0, -2.0], [1.0, 0.0]])

        manager.attach(
            "attention",
            attention,
            lambda: True,
            lambda name, module, observation: observations.append(
                (name, module, observation)
            ),
        )
        attention(self.attention_inputs(), attention_mask)

        self.assertEqual(manager.module_names, ("attention",))
        self.assertEqual(manager.hook_count, 1)
        self.assertEqual(len(observations), 1)
        module_name, observed_module, observation = observations[0]
        self.assertEqual(module_name, "attention")
        self.assertIs(observed_module, attention)
        torch.testing.assert_close(
            observation.processor_inputs.query,
            self.attention_inputs().query,
        )
        torch.testing.assert_close(
            observation.processor_inputs.merged_attention_mask,
            attention_mask,
        )
        torch.testing.assert_close(
            observation.exact_attention_weights,
            returned_weights,
        )

        manager.detach()

        self.assertIs(
            attention.projector.compute_qkv_projections,
            original_projection,
        )
        self.assertIs(attention.processor.compute_attention, original_attention)
        self.assertEqual(manager.module_names, ())
        self.assertEqual(manager.hook_count, 0)
        self.assertEqual(manager.subscription_count, 0)

    def test_manager_captures_exact_weights_through_selected_variant_adapter(self):
        method_name = "_compute_masked_attention_weights"
        monitor_adapters = (
            _SelfAttentionMonitorAdapter(),
            _MixtureOfAttentionHeadsMonitorAdapter(),
        )
        private_weights = torch.tensor([[[0.2, 0.8]]])

        for monitor_adapter in monitor_adapters:
            with self.subTest(
                monitor_adapter=type(monitor_adapter).__name__,
            ):
                attention = InstrumentedAttention(
                    private_method_name=method_name,
                    private_weights=private_weights,
                    returned_weights=torch.tensor([[[0.9, 0.1]]]),
                )
                original_private_method = getattr(attention.processor, method_name)
                observations = []
                manager = _AttentionDiagnosticsTrackerManager()

                manager.attach(
                    "attention",
                    attention,
                    lambda: True,
                    lambda name, module, observation, records=observations: (
                        records.append((name, module, observation))
                    ),
                    monitor_adapter=monitor_adapter,
                )
                attention(self.attention_inputs())

                torch.testing.assert_close(
                    observations[0][2].raw_attention_logits,
                    private_weights * 0.5,
                )
                torch.testing.assert_close(
                    observations[0][2].normalized_attention_weights,
                    private_weights,
                )
                torch.testing.assert_close(
                    observations[0][2].exact_attention_weights,
                    private_weights * 2.0,
                )
                manager.detach()
                self.assertIs(
                    getattr(attention.processor, method_name),
                    original_private_method,
                )

    def test_default_adapter_does_not_claim_an_exact_weight_seam(self):
        method_name = "_compute_masked_attention_weights"
        private_weights = torch.tensor([[[0.2, 0.8]]])
        returned_weights = torch.tensor([[[0.9, 0.1]]])
        attention = InstrumentedAttention(
            private_method_name=method_name,
            private_weights=private_weights,
            returned_weights=returned_weights,
        )
        observations = []
        manager = _AttentionDiagnosticsTrackerManager()

        manager.attach(
            "attention",
            attention,
            lambda: True,
            lambda name, module, observation: observations.append(observation),
        )
        attention(self.attention_inputs())

        torch.testing.assert_close(
            observations[0].exact_attention_weights,
            returned_weights,
        )
        manager.detach()

    def test_manager_skips_capture_outside_cadence(self):
        attention = InstrumentedAttention(returned_weights=torch.ones(1, 1, 1))
        observations = []
        manager = _AttentionDiagnosticsTrackerManager()

        manager.attach(
            "attention",
            attention,
            lambda: False,
            lambda name, module, observation: observations.append(
                (name, module, observation)
            ),
        )
        attention(self.attention_inputs())

        self.assertEqual(observations, [])
        self.assertEqual(
            manager.tracker_for(attention).latest_observation,
            _AttentionObservation(),
        )
        manager.detach()

    def test_manager_supports_modules_without_projector_or_processor(self):
        attention = torch.nn.Identity()
        observations = []
        manager = _AttentionDiagnosticsTrackerManager()
        output = torch.ones(2, requires_grad=True)

        manager.attach(
            "attention",
            attention,
            lambda: True,
            lambda name, module, observation: observations.append(
                (name, module, observation)
            ),
        )
        manager.tracker_for(attention).record_exact_attention_weights(
            torch.ones(1, 1, 1)
        )
        attention(output)

        self.assertEqual(manager.hook_count, 1)
        self.assertEqual(manager.subscription_count, 0)
        self.assertFalse(observations[0][2].restored_output.requires_grad)
        self.assertIsNone(observations[0][2].exact_attention_weights)
        manager.detach()

    def test_processor_capture_distinguishes_one_and_two_positional_arguments(self):
        attention_mask = torch.tensor([[0.0, -2.0], [1.0, 0.0]])
        cases = (
            (False, None),
            (True, attention_mask),
        )

        for include_mask, expected_mask in cases:
            with self.subTest(include_mask=include_mask):
                attention = PositionalProcessorAttention(
                    include_mask=include_mask,
                )
                observations = []
                manager = _AttentionDiagnosticsTrackerManager()

                def record_observation(
                    _name,
                    _module,
                    observation,
                    records=observations,
                ):
                    records.append(observation)

                manager.attach(
                    "attention",
                    attention,
                    lambda: True,
                    record_observation,
                )

                attention(self.attention_inputs(), attention_mask)

                self.assertEqual(len(observations), 1)
                observation = observations[0]
                torch.testing.assert_close(
                    observation.processor_inputs.query,
                    self.attention_inputs().query,
                )
                if expected_mask is None:
                    self.assertIsNone(
                        observation.processor_inputs.merged_attention_mask
                    )
                else:
                    torch.testing.assert_close(
                        observation.processor_inputs.merged_attention_mask,
                        expected_mask,
                    )
                manager.detach()

    def test_missing_declared_exact_weight_seam_rolls_back_attachment(self):
        attention = InstrumentedAttention(returned_weights=None)
        original_projection = attention.projector.compute_qkv_projections
        original_attention = attention.processor.compute_attention
        manager = _AttentionDiagnosticsTrackerManager()

        with self.assertRaisesRegex(
            AttributeError,
            "_compute_masked_attention_weights must be callable",
        ):
            manager.attach(
                "attention",
                attention,
                lambda: True,
                lambda _name, _module, _observation: None,
                monitor_adapter=ExactWeightMonitorAdapter(),
            )

        self.assertEqual(manager.module_names, ())
        self.assertEqual(manager.hook_count, 0)
        self.assertEqual(manager.subscription_count, 0)
        self.assertIs(
            attention.projector.compute_qkv_projections,
            original_projection,
        )
        self.assertIs(attention.processor.compute_attention, original_attention)

    def test_failed_attachment_preserves_preexisting_manager_state(self):
        first_attention = InstrumentedAttention(returned_weights=torch.ones(1, 1, 1))
        failing_attention = InstrumentedAttention(returned_weights=None)
        failing_original_projection = (
            failing_attention.projector.compute_qkv_projections
        )
        failing_original_attention = failing_attention.processor.compute_attention
        observations = []
        manager = _AttentionDiagnosticsTrackerManager()
        manager.attach(
            "first",
            first_attention,
            lambda: True,
            lambda name, _module, _observation: observations.append(name),
        )
        initial_hook_count = manager.hook_count
        initial_subscription_count = manager.subscription_count

        with self.assertRaisesRegex(
            AttributeError,
            "_compute_masked_attention_weights must be callable",
        ):
            manager.attach(
                "failing",
                failing_attention,
                lambda: True,
                lambda _name, _module, _observation: None,
                monitor_adapter=ExactWeightMonitorAdapter(),
            )

        self.assertEqual(manager.module_names, ("first",))
        self.assertEqual(manager.hook_count, initial_hook_count)
        self.assertEqual(manager.subscription_count, initial_subscription_count)
        self.assertIs(
            failing_attention.projector.compute_qkv_projections,
            failing_original_projection,
        )
        self.assertIs(
            failing_attention.processor.compute_attention,
            failing_original_attention,
        )

        first_attention(self.attention_inputs())

        self.assertEqual(observations, ["first"])
        manager.detach()

    def test_processor_starts_observation_when_projector_is_absent(self):
        attention = ProcessorOnlyAttention()
        observations = []
        manager = _AttentionDiagnosticsTrackerManager()
        manager.attach(
            "attention",
            attention,
            lambda: True,
            lambda name, module, observation: observations.append(observation),
        )
        tracker = manager.tracker_for(attention)
        tracker.record_exact_attention_weights(torch.ones(1, 1, 1))

        attention(self.attention_inputs())

        self.assertEqual(len(observations), 1)
        self.assertIsNotNone(observations[0].processor_inputs)
        self.assertIsNone(observations[0].projected_inputs)
        self.assertIsNone(observations[0].exact_attention_weights)
        manager.detach()


class TestAttentionMonitorCallback(unittest.TestCase):
    def test_tracking_orchestration_lists_each_tracked_fact(self):
        cls = AttentionMonitorCallback
        orchestration = cls._AttentionMonitorCallback__track_attention_observation

        self.assertEqual(
            orchestration_calls(orchestration),
            (
                "__track_query_norm_mean",
                "__track_key_norm_mean",
                "__track_value_norm_mean",
                "__track_output_norm",
                "__track_auxiliary_loss",
                "__track_configured_dropout_probability",
                "__track_mask_coverage",
                "__track_finite_raw_attention_logit_mean",
                "__track_finite_raw_attention_logit_std",
                "__track_pre_dropout_entropy_mean",
                "__track_pre_dropout_max_probability_mean",
                "__track_entropy_mean",
                "__track_max_probability_mean",
                "__track_dead_head_fraction",
                "__track_per_head_entropy",
                "__track_per_head_max_probability",
                "__track_entropy_history",
                "__track_max_probability_history",
                "__track_entropy_histogram",
                "__track_entropy_heatmap",
                "__track_max_probability_histogram",
                "__track_max_probability_heatmap",
                "__track_dropout_zero_fraction",
            ),
        )

    def test_max_probability_visual_emission_uses_exact_arguments(self):
        callback = AttentionMonitorCallback()
        entropy_history = object()
        maximum_history = object()
        callback._entropy_history["attn"] = entropy_history
        callback._max_probability_history["attn"] = maximum_history
        calls = []
        callback._emission_policy = SimpleNamespace(
            emit_histogram=lambda *args: calls.append(("histogram", args)),
            emit_history_heatmap=lambda *args: calls.append(("heatmap", args)),
        )
        experiment = object()
        maximum = torch.tensor([0.8, 0.6])
        context = _AttentionTrackingContext(
            pl_module=CaptureLightningModule(),
            module_name="attn",
            metric_prefix="attn/attention",
            metrics=diagnostic_metrics(
                per_head_entropy=torch.tensor([0.2, 0.4]),
                per_head_max_probability=maximum,
                weight_source="exact",
            ),
            experiment=experiment,
            global_step=7,
        )

        callback._AttentionMonitorCallback__track_max_probability_histogram(context)
        callback._AttentionMonitorCallback__track_max_probability_heatmap(context)

        self.assertEqual([kind for kind, _args in calls], ["histogram", "heatmap"])
        histogram_args = calls[0][1]
        self.assertIs(histogram_args[0], experiment)
        self.assertEqual(
            histogram_args[1],
            "attn/attention/histogram/max_probability_by_head",
        )
        self.assertIs(histogram_args[2], maximum)
        self.assertEqual(histogram_args[3], 7)
        heatmap_args = calls[1][1]
        self.assertIs(heatmap_args[0], experiment)
        self.assertEqual(
            heatmap_args[1],
            "attn/attention/heatmap/max_probability_by_head",
        )
        self.assertIs(heatmap_args[2], maximum_history)
        self.assertEqual(heatmap_args[3], 7)

    def attention(
        self,
        *,
        config_class=SelfAttentionConfig,
        return_attention_weights_flag: bool = True,
    ):
        config = build_attention_config(
            config_class=config_class,
            batch_size=2,
            num_heads=2,
            embedding_dim=4,
            target_sequence_length=3,
            source_sequence_length=3,
            return_attention_weights_flag=return_attention_weights_flag,
        )
        return config.build()

    def qkv(self):
        values = torch.arange(24, dtype=torch.float32).view(3, 2, 4) / 10
        return values, values, values

    def test_rejects_non_positive_configuration(self):
        cases = (
            ("log_every_n_steps", 0, "log_every_n_steps must be greater than 0."),
            ("log_every_n_steps", -1, "log_every_n_steps must be greater than 0."),
            ("history_size", 0, "history_size must be greater than 0."),
            ("history_size", -1, "history_size must be greater than 0."),
        )

        for option_name, value, expected_message in cases:
            with self.subTest(option_name=option_name, value=value):
                with self.assertRaises(ValueError) as raised:
                    AttentionMonitorCallback(**{option_name: value})
                self.assertEqual(str(raised.exception), expected_message)

    def test_default_configuration_is_explicit(self):
        callback = AttentionMonitorCallback()

        self.assertEqual(callback.log_every_n_steps, 100)
        self.assertEqual(callback.history_size, 128)
        self.assertIs(callback.log_per_head_scalars, False)

    def test_missing_global_step_samples_at_step_zero(self):
        callback = AttentionMonitorCallback(log_every_n_steps=2)

        should_sample = callback._AttentionMonitorCallback__should_sample(
            SimpleNamespace()
        )

        self.assertIs(should_sample, True)

    def test_discovers_only_attention_modules(self):
        module = CaptureLightningModule(
            attn=self.attention(),
            other=torch.nn.Linear(4, 4),
        )
        callback = AttentionMonitorCallback(log_every_n_steps=1)

        callback.on_fit_start(TrainerStub(), module)

        self.assertEqual(callback._tracker_manager.module_names, ("attn",))
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

    def test_logs_expected_finite_exact_metrics(self):
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
            "attn/attention/finite_raw_logit_mean",
            "attn/attention/finite_raw_logit_std",
            "attn/attention/pre_dropout_entropy_mean",
            "attn/attention/pre_dropout_max_probability_mean",
        }
        self.assertTrue(expected_tags.issubset(set(module.logged_tags)))
        for tag in expected_tags:
            self.assertTrue(
                torch.isfinite(torch.as_tensor(module.logged_value(tag))).all(),
                tag,
            )
        self.assertNotIn("attn/attention/approximate_entropy_mean", module.logged_tags)
        callback.on_fit_end(TrainerStub(), module)

    def test_independent_attention_uses_approximate_metric_prefixes(self):
        attention = self.attention(
            config_class=IndependentAttentionConfig,
            return_attention_weights_flag=False,
        )
        module = CaptureLightningModule(attn=attention)
        callback = AttentionMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        attention(*self.qkv())

        self.assertIn(
            "attn/attention/approximate_entropy_mean",
            module.logged_tags,
        )
        self.assertIn(
            "attn/attention/approximate_max_probability_mean",
            module.logged_tags,
        )
        self.assertNotIn("attn/attention/dropout_zero_fraction", module.logged_tags)
        self.assertNotIn("attn/attention/finite_raw_logit_mean", module.logged_tags)
        self.assertNotIn(
            "attn/attention/pre_dropout_entropy_mean",
            module.logged_tags,
        )
        callback.on_fit_end(TrainerStub(), module)

    def test_per_head_scalars_preserve_exact_tags(self):
        attention = self.attention()
        module = CaptureLightningModule(attn=attention)
        callback = AttentionMonitorCallback(
            log_every_n_steps=1,
            log_per_head_scalars=True,
        )
        callback.on_fit_start(TrainerStub(), module)

        attention(*self.qkv())

        expected_tags = {
            "attn/attention/head_0/entropy",
            "attn/attention/head_1/entropy",
            "attn/attention/head_0/max_probability",
            "attn/attention/head_1/max_probability",
        }
        self.assertTrue(expected_tags.issubset(set(module.logged_tags)))
        callback.on_fit_end(TrainerStub(), module)

    def test_mask_coverage_uses_merged_boolean_mask(self):
        attention = self.attention()
        module = CaptureLightningModule(attn=attention)
        callback = AttentionMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)
        attention_mask = torch.tensor(
            [
                [False, True, False],
                [True, False, False],
                [False, False, True],
            ]
        )

        attention(*self.qkv(), attention_mask=attention_mask)

        torch.testing.assert_close(
            module.logged_value("attn/attention/mask_coverage"),
            torch.tensor(1 / 3),
        )
        callback.on_fit_end(TrainerStub(), module)

    def test_histories_are_bounded_detached_and_cpu_resident(self):
        attention = self.attention()
        module = NoExperimentLightningModule(attn=attention)
        callback = AttentionMonitorCallback(
            log_every_n_steps=1,
            history_size=1,
        )
        callback.on_fit_start(TrainerStub(), module)

        attention(*self.qkv())
        attention(*self.qkv())

        entropy_history = callback._entropy_history["attn"]
        maximum_history = callback._max_probability_history["attn"]
        self.assertEqual(len(entropy_history), 1)
        self.assertEqual(len(maximum_history), 1)
        for history in (entropy_history, maximum_history):
            for tensor in history.tensors:
                self.assertEqual(tensor.device.type, "cpu")
                self.assertFalse(tensor.requires_grad)
        callback.on_fit_end(TrainerStub(), module)

    def test_visual_summaries_preserve_tags_step_and_chw_layout(self):
        attention = self.attention()
        module = CaptureLightningModule(attn=attention)
        module.global_step = 7
        callback = AttentionMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        attention(*self.qkv())

        experiment = module.logger.experiment
        histogram_tags = {tag for tag, _, _ in experiment.histograms}
        histogram_records = {
            tag: (values, step) for tag, values, step in experiment.histograms
        }
        image_records = {
            tag: (image, step, formats)
            for tag, image, step, formats in experiment.images
        }
        self.assertIn(
            "attn/attention/histogram/entropy_by_head",
            histogram_tags,
        )
        maximum_histogram, maximum_histogram_step = histogram_records[
            "attn/attention/histogram/max_probability_by_head"
        ]
        torch.testing.assert_close(
            maximum_histogram,
            callback._max_probability_history["attn"].tensors[-1],
        )
        self.assertEqual(maximum_histogram_step, 7)
        image, step, dataformats = image_records[
            "attn/attention/heatmap/entropy_by_head"
        ]
        self.assertEqual(step, 7)
        self.assertEqual(dataformats, "CHW")
        self.assertEqual(image.dim(), 3)
        maximum_image, maximum_step, maximum_formats = image_records[
            "attn/attention/heatmap/max_probability_by_head"
        ]
        self.assertEqual(maximum_step, 7)
        self.assertEqual(maximum_formats, "CHW")
        self.assertEqual(maximum_image.dim(), 3)
        callback.on_fit_end(TrainerStub(), module)

    def test_missing_experiment_does_not_suppress_scalar_metrics(self):
        attention = self.attention()
        module = NoExperimentLightningModule(attn=attention)
        callback = AttentionMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        attention(*self.qkv())

        self.assertIn("attn/attention/entropy_mean", module.logged_tags)
        callback.on_fit_end(TrainerStub(), module)

    def test_optional_metrics_are_logged_only_when_present(self):
        callback = AttentionMonitorCallback()
        module = CaptureLightningModule()
        present_metrics = _AttentionDiagnosticMetrics(
            query_norm_mean=torch.tensor(1.0),
            key_norm_mean=torch.tensor(2.0),
            value_norm_mean=torch.tensor(3.0),
            output_norm=torch.tensor(4.0),
            auxiliary_loss=torch.tensor(5.0),
            configured_dropout_probability=torch.tensor(0.1),
            mask_coverage=torch.tensor(0.2),
            per_head_entropy=torch.tensor([0.3, 0.4]),
            per_head_max_probability=torch.tensor([0.7, 0.6]),
            weight_source="approximate",
            dropout_zero_fraction=torch.tensor(0.5),
            finite_raw_attention_logit_mean=torch.tensor(0.6),
            finite_raw_attention_logit_std=torch.tensor(0.7),
            pre_dropout_per_head_entropy=torch.tensor([0.2, 0.3]),
            pre_dropout_per_head_max_probability=torch.tensor([0.8, 0.7]),
        )
        present_context = _AttentionTrackingContext(
            pl_module=module,
            module_name="attn",
            metric_prefix="attn/attention",
            metrics=present_metrics,
            experiment=None,
            global_step=0,
        )

        callback._AttentionMonitorCallback__track_attention_observation(present_context)

        expected_optional_tags = {
            "attn/attention/q_norm_mean",
            "attn/attention/k_norm_mean",
            "attn/attention/v_norm_mean",
            "attn/attention/output_norm",
            "attn/attention/auxiliary_loss",
            "attn/attention/approximate_entropy_mean",
            "attn/attention/approximate_max_probability_mean",
            "attn/attention/approximate_dead_head_fraction",
            "attn/attention/dropout_zero_fraction",
            "attn/attention/finite_raw_logit_mean",
            "attn/attention/finite_raw_logit_std",
            "attn/attention/pre_dropout_entropy_mean",
            "attn/attention/pre_dropout_max_probability_mean",
        }
        self.assertTrue(expected_optional_tags.issubset(set(module.logged_tags)))
        torch.testing.assert_close(
            module.logged_value("attn/attention/auxiliary_loss"),
            torch.tensor(5.0),
        )

        module.logged.clear()
        absent_metrics = _AttentionDiagnosticMetrics(
            query_norm_mean=None,
            key_norm_mean=None,
            value_norm_mean=None,
            output_norm=None,
            auxiliary_loss=None,
            configured_dropout_probability=torch.tensor(0.1),
            mask_coverage=torch.tensor(0.0),
            per_head_entropy=None,
            per_head_max_probability=None,
            weight_source=None,
            dropout_zero_fraction=None,
        )
        absent_context = _AttentionTrackingContext(
            pl_module=module,
            module_name="attn",
            metric_prefix="attn/attention",
            metrics=absent_metrics,
            experiment=None,
            global_step=0,
        )

        callback._AttentionMonitorCallback__track_attention_observation(absent_context)

        self.assertEqual(
            set(module.logged_tags),
            {
                "attn/attention/configured_dropout_probability",
                "attn/attention/mask_coverage",
            },
        )

    def test_per_head_scalar_gate_and_values_are_exact(self):
        entropy = torch.tensor([0.125, 0.5])
        maximum = torch.tensor([0.8, 0.6])
        metrics = diagnostic_metrics(
            per_head_entropy=entropy,
            per_head_max_probability=maximum,
            weight_source="exact",
        )

        disabled_module = CaptureLightningModule()
        disabled_context = _AttentionTrackingContext(
            pl_module=disabled_module,
            module_name="attn",
            metric_prefix="attn/attention",
            metrics=metrics,
            experiment=None,
            global_step=0,
        )
        disabled_callback = AttentionMonitorCallback(
            log_per_head_scalars=False,
        )
        disabled_callback._AttentionMonitorCallback__track_per_head_entropy(
            disabled_context
        )
        disabled_callback._AttentionMonitorCallback__track_per_head_max_probability(
            disabled_context
        )
        self.assertEqual(disabled_module.logged, [])

        enabled_module = CaptureLightningModule()
        enabled_context = _AttentionTrackingContext(
            pl_module=enabled_module,
            module_name="attn",
            metric_prefix="attn/attention",
            metrics=metrics,
            experiment=None,
            global_step=0,
        )
        enabled_callback = AttentionMonitorCallback(
            log_per_head_scalars=True,
        )
        enabled_callback._AttentionMonitorCallback__track_per_head_entropy(
            enabled_context
        )
        enabled_callback._AttentionMonitorCallback__track_per_head_max_probability(
            enabled_context
        )

        torch.testing.assert_close(
            enabled_module.logged_value("attn/attention/head_0/entropy"),
            entropy[0],
        )
        torch.testing.assert_close(
            enabled_module.logged_value("attn/attention/head_1/entropy"),
            entropy[1],
        )
        torch.testing.assert_close(
            enabled_module.logged_value("attn/attention/head_0/max_probability"),
            maximum[0],
        )
        torch.testing.assert_close(
            enabled_module.logged_value("attn/attention/head_1/max_probability"),
            maximum[1],
        )

    def test_dead_head_threshold_is_inclusive(self):
        callback = AttentionMonitorCallback()
        module = CaptureLightningModule()
        metrics = diagnostic_metrics(
            per_head_entropy=torch.tensor(
                [
                    callback.DEAD_HEAD_ENTROPY_FLOOR,
                    callback.DEAD_HEAD_ENTROPY_FLOOR * 2,
                ]
            ),
            per_head_max_probability=torch.tensor([1.0, 0.5]),
            weight_source="exact",
        )
        context = _AttentionTrackingContext(
            pl_module=module,
            module_name="attn",
            metric_prefix="attn/attention",
            metrics=metrics,
            experiment=None,
            global_step=0,
        )

        callback._AttentionMonitorCallback__track_dead_head_fraction(context)

        torch.testing.assert_close(
            module.logged_value("attn/attention/dead_head_fraction"),
            torch.tensor(0.5),
        )

    def test_exact_weight_history_requires_every_capability(self):
        entropy = torch.tensor([0.2, 0.3])
        maximum = torch.tensor([0.8, 0.7])
        cases = (
            ("all", "exact", entropy, maximum, True, True, True),
            ("no_histories", "exact", entropy, maximum, False, False, False),
            ("entropy_history_only", "exact", entropy, maximum, True, False, False),
            ("missing_maximum", "exact", entropy, None, True, True, False),
            ("approximate", "approximate", entropy, maximum, True, True, False),
        )

        for (
            case_name,
            weight_source,
            per_head_entropy,
            per_head_maximum,
            has_entropy_history,
            has_maximum_history,
            expected,
        ) in cases:
            with self.subTest(case_name=case_name):
                callback = AttentionMonitorCallback()
                if has_entropy_history:
                    callback._entropy_history["attn"] = object()
                if has_maximum_history:
                    callback._max_probability_history["attn"] = object()
                context = _AttentionTrackingContext(
                    pl_module=CaptureLightningModule(),
                    module_name="attn",
                    metric_prefix="attn/attention",
                    metrics=diagnostic_metrics(
                        per_head_entropy=per_head_entropy,
                        per_head_max_probability=per_head_maximum,
                        weight_source=weight_source,
                    ),
                    experiment=None,
                    global_step=0,
                )

                can_track = AttentionMonitorCallback._AttentionMonitorCallback__can_track_exact_weight_history
                actual = can_track(callback, context)

                self.assertIs(actual, expected)

    def test_exact_weight_visual_requires_experiment_and_history_capability(self):
        callback = AttentionMonitorCallback()
        callback._entropy_history["attn"] = object()
        callback._max_probability_history["attn"] = object()
        valid_metrics = diagnostic_metrics(
            per_head_entropy=torch.tensor([0.2]),
            per_head_max_probability=torch.tensor([0.8]),
            weight_source="exact",
        )
        invalid_metrics = diagnostic_metrics(
            per_head_entropy=torch.tensor([0.2]),
            per_head_max_probability=torch.tensor([0.8]),
            weight_source="approximate",
        )
        cases = (
            (None, valid_metrics, False),
            (object(), invalid_metrics, False),
            (object(), valid_metrics, True),
        )

        for experiment, metrics, expected in cases:
            with self.subTest(
                has_experiment=experiment is not None,
                weight_source=metrics.weight_source,
            ):
                context = _AttentionTrackingContext(
                    pl_module=CaptureLightningModule(),
                    module_name="attn",
                    metric_prefix="attn/attention",
                    metrics=metrics,
                    experiment=experiment,
                    global_step=0,
                )

                can_emit = AttentionMonitorCallback._AttentionMonitorCallback__can_emit_exact_weight_visual
                actual = can_emit(callback, context)

                self.assertIs(actual, expected)

    def test_fit_end_restores_methods_and_clears_monitor_state(self):
        attention = self.attention()
        original_projection = attention.projector.compute_qkv_projections
        original_attention = attention.processor.compute_attention
        original_raw_logits = attention.processor._compute_raw_masked_attention_logits
        original_normalized_weights = (
            attention.processor._compute_normalized_attention_weights
        )
        original_exact_weights = attention.processor._compute_masked_attention_weights
        module = CaptureLightningModule(attn=attention)
        callback = AttentionMonitorCallback(log_every_n_steps=1)

        callback.on_fit_start(TrainerStub(), module)
        self.assertIn("compute_qkv_projections", vars(attention.projector))
        self.assertIn("compute_attention", vars(attention.processor))
        self.assertIn(
            "_compute_raw_masked_attention_logits",
            vars(attention.processor),
        )
        self.assertIn(
            "_compute_normalized_attention_weights",
            vars(attention.processor),
        )
        self.assertIn("_compute_masked_attention_weights", vars(attention.processor))
        self.assertFalse(
            same_bound_method(
                attention.projector.compute_qkv_projections,
                original_projection,
            )
        )
        self.assertFalse(
            same_bound_method(
                attention.processor.compute_attention,
                original_attention,
            )
        )

        callback.on_fit_end(TrainerStub(), module)

        self.assertTrue(
            same_bound_method(
                attention.processor._compute_raw_masked_attention_logits,
                original_raw_logits,
            )
        )
        self.assertTrue(
            same_bound_method(
                attention.processor._compute_normalized_attention_weights,
                original_normalized_weights,
            )
        )
        self.assertTrue(
            same_bound_method(
                attention.projector.compute_qkv_projections,
                original_projection,
            )
        )
        self.assertTrue(
            same_bound_method(
                attention.processor.compute_attention,
                original_attention,
            )
        )
        self.assertTrue(
            same_bound_method(
                attention.processor._compute_masked_attention_weights,
                original_exact_weights,
            )
        )
        self.assertNotIn("compute_qkv_projections", vars(attention.projector))
        self.assertNotIn("compute_attention", vars(attention.processor))
        self.assertNotIn(
            "_compute_raw_masked_attention_logits",
            vars(attention.processor),
        )
        self.assertNotIn(
            "_compute_normalized_attention_weights",
            vars(attention.processor),
        )
        self.assertNotIn("_compute_masked_attention_weights", vars(attention.processor))
        self.assertEqual(callback._tracker_manager.module_names, ())
        self.assertEqual(callback._entropy_history, {})
        self.assertEqual(callback._max_probability_history, {})

    def test_duplicate_callbacks_can_cleanup_in_either_order(self):
        for cleanup_order in ((0, 1), (1, 0)):
            with self.subTest(cleanup_order=cleanup_order):
                attention = self.attention()
                module = CaptureLightningModule(attn=attention)
                callbacks = (
                    AttentionMonitorCallback(log_every_n_steps=1),
                    AttentionMonitorCallback(log_every_n_steps=1),
                )
                original_projection = attention.projector.compute_qkv_projections
                original_attention = attention.processor.compute_attention
                original_raw_logits = (
                    attention.processor._compute_raw_masked_attention_logits
                )
                original_normalized_weights = (
                    attention.processor._compute_normalized_attention_weights
                )
                original_exact_weights = (
                    attention.processor._compute_masked_attention_weights
                )

                for callback in callbacks:
                    callback.on_fit_start(TrainerStub(), module)
                attention(*self.qkv())

                self.assertEqual(
                    module.logged_tags.count("attn/attention/q_norm_mean"),
                    2,
                )
                callbacks[cleanup_order[0]].on_fit_end(TrainerStub(), module)
                module.logged.clear()
                attention(*self.qkv())

                self.assertEqual(
                    module.logged_tags.count("attn/attention/q_norm_mean"),
                    1,
                )
                callbacks[cleanup_order[1]].on_fit_end(TrainerStub(), module)

                self.assertTrue(
                    same_bound_method(
                        attention.processor._compute_raw_masked_attention_logits,
                        original_raw_logits,
                    )
                )
                self.assertTrue(
                    same_bound_method(
                        attention.processor._compute_normalized_attention_weights,
                        original_normalized_weights,
                    )
                )
                self.assertTrue(
                    same_bound_method(
                        attention.projector.compute_qkv_projections,
                        original_projection,
                    )
                )
                self.assertTrue(
                    same_bound_method(
                        attention.processor.compute_attention,
                        original_attention,
                    )
                )
                self.assertTrue(
                    same_bound_method(
                        attention.processor._compute_masked_attention_weights,
                        original_exact_weights,
                    )
                )
                self.assertNotIn(
                    "compute_qkv_projections",
                    vars(attention.projector),
                )
                self.assertNotIn("compute_attention", vars(attention.processor))
                self.assertNotIn(
                    "_compute_raw_masked_attention_logits",
                    vars(attention.processor),
                )
                self.assertNotIn(
                    "_compute_normalized_attention_weights",
                    vars(attention.processor),
                )
                self.assertNotIn(
                    "_compute_masked_attention_weights",
                    vars(attention.processor),
                )

    def test_fit_start_rolls_back_every_module_when_later_attachment_fails(self):
        first_attention = self.attention()
        second_attention = self.attention()
        module = CaptureLightningModule(
            first=first_attention,
            second=second_attention,
        )
        callback = AttentionMonitorCallback(log_every_n_steps=1)
        original_methods = {
            attention: (
                attention.projector.compute_qkv_projections,
                attention.processor.compute_attention,
                attention.processor._compute_raw_masked_attention_logits,
                attention.processor._compute_normalized_attention_weights,
                attention.processor._compute_masked_attention_weights,
            )
            for attention in (first_attention, second_attention)
        }
        original_register_forward_hook = torch.nn.Module.register_forward_hook

        def fail_second_registration(attention_module, hook, *args, **kwargs):
            if attention_module is second_attention:
                raise RuntimeError("deliberate second attachment failure")
            return original_register_forward_hook(
                attention_module,
                hook,
                *args,
                **kwargs,
            )

        with patch.object(
            torch.nn.Module,
            "register_forward_hook",
            new=fail_second_registration,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "deliberate second attachment failure",
            ):
                callback.on_fit_start(TrainerStub(), module)

        self.assertEqual(callback._tracker_manager.module_names, ())
        self.assertEqual(callback._tracker_manager.hook_count, 0)
        self.assertEqual(callback._tracker_manager.subscription_count, 0)
        self.assertEqual(callback._entropy_history, {})
        self.assertEqual(callback._max_probability_history, {})
        for attention, expected_methods in original_methods.items():
            self.assertTrue(
                same_bound_method(
                    attention.projector.compute_qkv_projections,
                    expected_methods[0],
                )
            )
            self.assertTrue(
                same_bound_method(
                    attention.processor.compute_attention,
                    expected_methods[1],
                )
            )
            self.assertTrue(
                same_bound_method(
                    attention.processor._compute_raw_masked_attention_logits,
                    expected_methods[2],
                )
            )
            self.assertTrue(
                same_bound_method(
                    attention.processor._compute_normalized_attention_weights,
                    expected_methods[3],
                )
            )
            self.assertTrue(
                same_bound_method(
                    attention.processor._compute_masked_attention_weights,
                    expected_methods[4],
                )
            )
            self.assertNotIn("compute_qkv_projections", vars(attention.projector))
            self.assertNotIn("compute_attention", vars(attention.processor))
            self.assertNotIn(
                "_compute_raw_masked_attention_logits",
                vars(attention.processor),
            )
            self.assertNotIn(
                "_compute_normalized_attention_weights",
                vars(attention.processor),
            )
            self.assertNotIn(
                "_compute_masked_attention_weights",
                vars(attention.processor),
            )

        callback.on_fit_start(TrainerStub(), module)
        self.assertEqual(callback._tracker_manager.module_names, ("first", "second"))
        callback.on_fit_end(TrainerStub(), module)

    def test_repeated_fit_start_does_not_accumulate_instrumentation(self):
        attention = self.attention()
        module = CaptureLightningModule(attn=attention)
        callback = AttentionMonitorCallback(log_every_n_steps=1)

        callback.on_fit_start(TrainerStub(), module)
        first_hook_count = callback._tracker_manager.hook_count
        first_subscription_count = callback._tracker_manager.subscription_count
        callback.on_fit_start(TrainerStub(), module)

        self.assertEqual(callback._tracker_manager.hook_count, first_hook_count)
        self.assertEqual(
            callback._tracker_manager.subscription_count,
            first_subscription_count,
        )
        self.assertEqual(callback._tracker_manager.module_names, ("attn",))
        callback.on_fit_end(TrainerStub(), module)

    def test_exception_cleanup_restores_instrumentation(self):
        attention = self.attention()
        original_projection = attention.projector.compute_qkv_projections
        module = CaptureLightningModule(attn=attention)
        callback = AttentionMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        callback.on_exception(TrainerStub(), module, RuntimeError("failure"))

        self.assertTrue(
            same_bound_method(
                attention.projector.compute_qkv_projections,
                original_projection,
            )
        )
        self.assertEqual(callback._tracker_manager.module_names, ())


if __name__ == "__main__":
    unittest.main()
