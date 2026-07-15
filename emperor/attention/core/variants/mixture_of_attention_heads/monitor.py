from torch import Tensor

from emperor.attention.core.monitor import _AttentionMonitorAdapter


class _MixtureOfAttentionHeadsMonitorAdapter(_AttentionMonitorAdapter):
    """Capture and canonicalize mixture attention weights."""

    @property
    def exact_weight_method_names(self) -> tuple[str, ...]:
        return (
            "_MixtureOfAttentionHeadsProcessor__compute_masked_attention_weights",
        )

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
