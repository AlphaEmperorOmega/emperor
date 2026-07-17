import unittest
from types import SimpleNamespace

import torch

from emperor.attention._monitoring.diagnostics import (
    _AttentionDiagnostics,
    _AttentionMonitorAdapter,
    _AttentionObservation,
)
from emperor.attention._runtime import QKV


class TestAttentionDiagnosticsFastContracts(unittest.TestCase):
    def diagnostics(self) -> _AttentionDiagnostics:
        return _AttentionDiagnostics()

    def test_rank_four_weights_never_use_the_flattened_rank_three_path(self):
        adapter = _AttentionMonitorAdapter()

        canonical = adapter.canonicalize(
            torch.ones(4, 3, 1, 2),
            num_heads=2,
        )

        self.assertIsNone(canonical)

    def test_approximation_receives_the_merged_attention_mask(self):
        processor_qkv = object()
        merged_attention_mask = object()
        approximation_calls = []
        diagnostics = self.diagnostics()
        diagnostics.approximate_attention_weights = lambda qkv, mask: (
            approximation_calls.append((qkv, mask))
        )
        diagnostics.mask_coverage = lambda _mask: torch.zeros(())

        metrics = diagnostics.calculate(
            _AttentionObservation(
                processor_qkv=processor_qkv,
                merged_attention_mask=merged_attention_mask,
            ),
            num_heads=1,
            configured_dropout_probability=0.0,
        )

        self.assertEqual(
            approximation_calls,
            [(processor_qkv, merged_attention_mask)],
        )
        self.assertIsNone(metrics.weight_source)

    def test_single_weight_continues_past_the_empty_weight_guard(self):
        class SingleWeightBranchReached(Exception):
            pass

        class SingleWeightProbe:
            @staticmethod
            def numel():
                return 1

            @staticmethod
            def sum(*, dim, keepdim):
                raise SingleWeightBranchReached((dim, keepdim))

        monitor_adapter = SimpleNamespace(
            canonicalize=lambda _weights, _heads: SingleWeightProbe()
        )

        with self.assertRaises(SingleWeightBranchReached):
            self.diagnostics().per_head_statistics(
                object(),
                num_heads=1,
                monitor_adapter=monitor_adapter,
            )

    def test_projection_norm_uses_the_last_dimension(self):
        expected = object()

        class ProjectionProbe:
            def detach(self):
                return self

            def float(self):
                return self

            def norm(self, *, dim):
                if dim != -1:
                    raise AssertionError(f"expected dim=-1, received {dim!r}")
                return SimpleNamespace(mean=lambda: expected)

        projected_qkv = QKV(
            query=ProjectionProbe(),
            key=object(),
            value=object(),
        )
        projection_norm = _AttentionDiagnostics._AttentionDiagnostics__projection_norm

        actual = projection_norm(projected_qkv, "query")

        self.assertIs(actual, expected)


if __name__ == "__main__":
    unittest.main()
