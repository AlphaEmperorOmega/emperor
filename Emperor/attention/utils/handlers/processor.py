import torch
import torch.nn.functional as F

from torch import Tensor
from Emperor.attention.utils.handlers.reshaper import ReshaperBase, ReshaperBuilder
from Emperor.attention.utils.handlers.validators._processor import ProcessorValidator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.attention.utils.layer import MultiHeadAttentionConfig
    from Emperor.attention.utils.handlers.projector import ProjectorBase


class ProcessorBuilder:
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        projector: "ProjectorBase",
    ):
        self.cfg = cfg
        self.projector = projector
        self.attention_option = self.cfg.attention_option
        self.return_attention_weights_flag = self.cfg.return_attention_weights_flag
        self.use_kv_expert_models_flag = self.cfg.use_kv_expert_models_flag

    def build(self) -> "ProcessorBase":
        from Emperor.attention.utils.enums import AttentionOptions

        inputs = (self.cfg, self.projector)
        match self.attention_option:
            case AttentionOptions.SELF_ATTENTION:
                return SelfAttentionProcessor(*inputs)
            case AttentionOptions.INDEPENDENT:
                return IndependentProcessor(*inputs)
            case AttentionOptions.MIXTURE_OF_ATTENTION_HEADS:
                return MixtureOfAttentionHeadsProcessor(*inputs)
            case _:
                raise ValueError(
                    f"Attention option not supported or unknown option given: {self.attention_option}"
                )


class ProcessorBase:
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        projector: "ProjectorBase",
    ):
        self.cfg = cfg
        self.projector = projector
        self.num_heads: int = self.cfg.num_heads
        self.batch_size: int = self.cfg.batch_size
        self.embedding_dim: int = self.cfg.embedding_dim
        self.dropout_probability: float = self.cfg.dropout_probability
        self.target_sequence_length: int = self.cfg.target_sequence_length
        self.source_sequence_length: int = self.cfg.source_sequence_length
        self.query_key_projection_dim: int = self.cfg.query_key_projection_dim
        self.value_projection_dim: int = self.cfg.value_projection_dim
        self.use_kv_expert_models_flag: bool = self.cfg.use_kv_expert_models_flag
        self.causal_attention_mask_flag: bool = self.cfg.causal_attention_mask_flag
        self.average_attention_weights_flag: bool = (
            self.cfg.average_attention_weights_flag
        )
        self.zero_attention_flag: bool = self.cfg.zero_attention_flag
        self.add_key_value_bias_flag: bool = self.cfg.add_key_value_bias_flag
        self.head_dim: int = self.embedding_dim // self.num_heads
        is_qk_dim = (
            self.query_key_projection_dim is not None
            and self.query_key_projection_dim != 0
        )
        is_v_dim = (
            self.value_projection_dim is not None and self.value_projection_dim != 0
        )
        self.qk_head_dim = (
            self.query_key_projection_dim // self.num_heads
            if is_qk_dim
            else self.head_dim
        )
        self.v_head_dim = (
            self.value_projection_dim // self.num_heads if is_v_dim else self.head_dim
        )
        self.reshaper = ReshaperBuilder(self.cfg).build()
        self.validator = ProcessorValidator(self)

    def _compute_attention_output(self, weighted_values: Tensor) -> Tensor:
        attention_output = self.projector.compute_output_projection(weighted_values)
        if isinstance(attention_output, tuple):
            # TODO: At the moment the attention mechanism does not handle a tuple output.
            # This needs to be fixed in the future.
            attention_output = attention_output[0]
        embedding_dim = attention_output.size(-1)
        return attention_output.view(
            self.target_sequence_length, self.batch_size, embedding_dim
        )

    def compute_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        raise NotImplementedError(
            "`compute_attention` method must be implemented by subclass"
        )


class SelfAttentionProcessor(ProcessorBase):
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        projector: "ProjectorBase",
    ):
        super().__init__(cfg, projector)

    def compute_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        weights = self.__compute_masked_attention_weights(query, key, attention_mask)
        weighted_value = self.__compute_weighted_values(weights, value)
        output = self._compute_attention_output(weighted_value)
        output, weights = self.__ensure_correct_shape_output(output, weights)

        return output, weights

    def __compute_masked_attention_weights(
        self,
        query: Tensor,
        key: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        scaled_query = self.__scale_query(query)
        raw_weights = self.__compute_raw_masked_attention_weights(
            scaled_query, key, attention_mask
        )
        weights = F.softmax(raw_weights, dim=-1)
        if self.dropout_probability > 0.0:
            weights = F.dropout(weights, p=self.dropout_probability)
        return weights

    def __scale_query(self, query: Tensor) -> Tensor:
        head_dim = query.size(-1)
        return query * head_dim**-0.5

    def __compute_raw_masked_attention_weights(
        self,
        query: Tensor,
        key: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        key = key.transpose(-2, -1)
        if attention_mask is not None:
            return torch.baddbmm(attention_mask, query, key)
        return torch.bmm(query, key)

    def __compute_weighted_values(
        self,
        attention_weights: Tensor,
        values: Tensor,
    ) -> Tensor:
        assert self.target_sequence_length == self.source_sequence_length, (
            f"Self-attention requires that `target_sequence_length`: {self.target_sequence_length} is equal to `source_sequence_length`:{self.source_sequence_length}."
        )
        weighted_values = torch.bmm(attention_weights, values)
        values = weighted_values.transpose(0, 1)
        values = values.contiguous()
        return values.view(
            self.target_sequence_length * self.batch_size,
            self.embedding_dim,
        )

    def __ensure_correct_shape_output(
        self,
        attention_output: Tensor,
        attention_weights: Tensor,
    ) -> tuple[Tensor, Tensor]:
        source_sequence_length = attention_weights.size(-1)
        attention_weights_shape = (
            self.batch_size,
            self.num_heads,
            self.target_sequence_length,
            source_sequence_length,
        )
        attention_weights = attention_weights.view(attention_weights_shape)
        attention_weights = self.__maybe_average_attention_weights(attention_weights)

        return self.__handle_batched_input(attention_output, attention_weights)

    def __maybe_average_attention_weights(self, attention_weights: Tensor) -> Tensor:
        if self.average_attention_weights_flag:
            return attention_weights.mean(dim=1)
        return attention_weights

    def __handle_batched_input(
        self, attention_output: Tensor, attention_weights: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self.validator.is_input_tensor_single_batch(attention_output):
            return attention_output, attention_weights
        output_with_removed_batch_dim = attention_output.squeeze(1)
        weights_with_removed_batch_dim = attention_weights.squeeze(0)
        return output_with_removed_batch_dim, weights_with_removed_batch_dim


class IndependentProcessor(ProcessorBase):
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        projector: "ProjectorBase",
    ):
        super().__init__(cfg, projector)

    def compute_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        attention_mask = self.__prepare_attnetion_mask(attention_mask)
        query, key, value = self.reshaper.reshape_before_attention(query, key, value)
        weighted_values = self.__compute_weighted_values(
            query, key, value, attention_mask
        )
        attention_output = self._compute_attention_output(weighted_values)
        if self.validator.is_input_tensor_single_batch(attention_output):
            attention_output = attention_output.squeeze(1)
        return attention_output, None

    def __prepare_attnetion_mask(
        self, attention_mask: Tensor | None = None
    ) -> Tensor | None:
        if attention_mask is None:
            return None
        is_mask_single_batch = attention_mask.size(0) == 1
        is_mask_batched = attention_mask.dim() == 3
        if is_mask_single_batch and is_mask_batched:
            return attention_mask.unsqueeze(0)
        return attention_mask.view(
            self.batch_size,
            self.num_heads,
            self.target_sequence_length,
            self.source_sequence_length,
        )

    def __compute_weighted_values(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        weighted_values = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attention_mask,
            self.dropout_probability,
            self.causal_attention_mask_flag,
        )

        weighted_values = weighted_values.permute(2, 0, 1, 3)
        weighted_values = weighted_values.contiguous()
        return weighted_values.view(
            self.batch_size * self.target_sequence_length,
            self.embedding_dim,
        )


class MixtureOfAttentionHeadsProcessor(ProcessorBase):
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        projector: "ProjectorBase",
    ):
        super().__init__(cfg, projector)
        self.top_k: int = self.projector.top_k

    def compute_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        query, key, value = self.reshaper.reshape_before_attention(query, key, value)
        weights = self.__compute_masked_attention_weights(query, key, attention_mask)
        weighted_value = self.__compute_weighted_values(weights, value)
        output = self._compute_attention_output(weighted_value)

        return output, None

    def __compute_masked_attention_weights(
        self,
        query: Tensor,
        key: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        scaled_query = self.__scale_query(query)
        raw_weights = self.__compute_raw_masked_attention_weights(
            scaled_query, key, attention_mask
        )
        weights = F.softmax(raw_weights, dim=-1)
        if self.dropout_probability > 0.0:
            weights = F.dropout(weights, p=self.dropout_probability)
        return weights

    def __scale_query(self, query: Tensor) -> Tensor:
        head_dim = query.size(-1)
        return query * head_dim**-0.5

    def __compute_raw_masked_attention_weights(
        self,
        query: Tensor,
        key: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        total_batch_size = self.batch_size * self.num_heads * self.top_k

        key = key.transpose(-2, -1)
        einsum_equation = "bkhie,bhej->bkhij"
        if self.use_kv_expert_models_flag:
            einsum_equation = "bkhie,bkhej->bkhij"
        raw_weights = torch.einsum(einsum_equation, query, key)
        raw_weights = raw_weights.contiguous().view(
            total_batch_size, self.source_sequence_length, self.target_sequence_length
        )

        # TODO: Add relative positional encoding support here
        if attention_mask is not None:
            return raw_weights + attention_mask
        return raw_weights

    def __compute_weighted_values(
        self,
        attention_weights: Tensor,
        values: Tensor,
    ) -> Tensor:
        einsum_equation = "bkhie,bhej->bkhij"
        # einsum_equation = "bkhij,bhje->bkhie"
        if self.use_kv_expert_models_flag:
            einsum_equation = "bkhie,bkhej->bkhij"

        attention_weights = attention_weights.contiguous().view(
            self.batch_size,
            self.top_k,
            self.num_heads,
            self.target_sequence_length,
            self.source_sequence_length,
        )
        weighted_values = torch.einsum(einsum_equation, attention_weights, values)
        values = weighted_values.permute(3, 0, 1, 2, 4)
        values = values.contiguous()
        return values.view(
            self.target_sequence_length,
            self.batch_size,
            self.top_k,
            self.embedding_dim,
        )
