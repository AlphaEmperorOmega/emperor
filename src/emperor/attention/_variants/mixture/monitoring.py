"""Private mixture-of-attention-heads monitoring implementation."""

from torch import Tensor

from emperor.attention._monitoring.diagnostics import _AttentionMonitorAdapter


class _MixtureOfAttentionHeadsMonitorAdapter(_AttentionMonitorAdapter):
    """Capture and canonicalize mixture attention weights."""

    @property
    def raw_attention_logit_method_name(self) -> str:
        return "_compute_raw_masked_attention_logits"

    @property
    def normalized_attention_weight_method_name(self) -> str:
        return "_compute_normalized_attention_weights"

    @property
    def exact_weight_method_name(self) -> str:
        return "_compute_masked_attention_weights"

    @staticmethod
    def canonicalize(attention_weights: Tensor, num_heads: int) -> Tensor | None:
        if num_heads <= 0:
            return None
        detached_weights = attention_weights.detach().float()
        if detached_weights.dim() == 5 and detached_weights.size(2) == num_heads:
            (
                batch_size,
                selected_expert_count,
                head_count,
                target_length,
                source_length,
            ) = detached_weights.shape
            return detached_weights.reshape(
                batch_size * selected_expert_count,
                head_count,
                target_length,
                source_length,
            )
        return _AttentionMonitorAdapter.canonicalize(
            attention_weights,
            num_heads,
        )
