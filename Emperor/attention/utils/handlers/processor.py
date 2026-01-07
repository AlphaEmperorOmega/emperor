import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.attention.utils.layer import MultiHeadAttentionConfig
    from Emperor.attention.utils.handlers.projector import ProjectorBase
    from Emperor.attention.utils._validator import MultiHeadAttentionConfigValidator


class ProcessorBase:
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        validator: "MultiHeadAttentionConfigValidator",
        projector: "ProjectorBase",
    ):
        self.cfg = cfg
        self.validator = validator
        self.projector = projector
        self.num_heads = self.cfg.num_heads
        self.batch_size = self.cfg.batch_size
        self.embedding_dim = self.cfg.embedding_dim
        self.dropout_probability = self.cfg.dropout_probability
        self.target_sequence_length = self.cfg.target_sequence_length
        self.source_sequence_length = self.cfg.source_sequence_length
        self.query_key_projection_dim = self.cfg.query_key_projection_dim
        self.value_projection_dim = self.cfg.value_projection_dim
        self.causal_attention_mask_flag = self.cfg.causal_attention_mask_flag
        self.average_attention_weights_flag = self.cfg.average_attention_weights_flag
        self.zero_attention_flag = self.cfg.zero_attention_flag
        self.add_key_value_bias_flag = self.cfg.add_key_value_bias_flag
        self.head_dim = self.embedding_dim // self.num_heads
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

    def _compute_attention_output(self, weighted_values: Tensor) -> Tensor:
        attention_output = self.projector.compute_output_projection(weighted_values)
        if isinstance(attention_output, tuple):
            # TODO: At the moment the attention mechanism does not handle a tuple output.
            # This needs to be fixed in the future.
            attention_output = attention_output[0]
        embedding_dim = attention_output.size(1)
        return attention_output.view(
            self.target_sequence_length, self.batch_size, embedding_dim
        )


class ProcessorWithReturnedWeights(ProcessorBase):
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        validator: "MultiHeadAttentionConfigValidator",
        output_model: nn.Module,
    ):
        super().__init__(cfg, validator, output_model)

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
        query_scalar = math.sqrt(1.0 / float(head_dim))
        return query * query_scalar

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
            f"At the moment different source and target sequence lengths are not supported so `target_sequence_length`: {self.target_sequence_length} must be equal to `source_sequence_length`:{self.source_sequence_length}. See if this can be fixed in the future."
        )
        # The reason this does not work is because `weighted_values` after torch.bmm(attention_weights, values)
        # need to be reshaped into `weighted_values_output_shape`, if `source_sequence_length` and
        # `target_sequence_length` are different then the reshaping will not work.
        weighted_values_output_shape = (
            self.target_sequence_length * self.batch_size,
            self.embedding_dim,
        )
        weighted_values = torch.bmm(attention_weights, values)
        weighted_values = weighted_values.transpose(0, 1)
        weighted_values = weighted_values.contiguous()
        return weighted_values.view(weighted_values_output_shape)

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
        if not self.validator.get_batched_input_flag():
            output_with_removed_batch_dim = attention_output.squeeze(1)
            weights_with_removed_batch_dim = attention_weights.squeeze(0)
            return output_with_removed_batch_dim, weights_with_removed_batch_dim
        return attention_output, attention_weights


class ProcessorDefault(ProcessorBase):
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        validator: "MultiHeadAttentionConfigValidator",
        output_model: nn.Module,
    ):
        super().__init__(cfg, validator, output_model)

    def compute_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, None]:
        attention_mask = self.__prepare_attnetion_mask(attention_mask)
        query, key, value = self.__reshape_qkv_for_attention(query, key, value)
        weighted_values = self.__compute_weighted_values(
            query, key, value, attention_mask
        )
        attention_output = self._compute_attention_output(weighted_values)
        if not self.validator.get_batched_input_flag():
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
            -1,
        )

    def __reshape_qkv_for_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ):
        source_sequence_length = key.size(1)
        q_shape = (
            self.batch_size,
            self.num_heads,
            self.target_sequence_length,
            self.qk_head_dim,
        )
        k_shape = (
            self.batch_size,
            self.num_heads,
            source_sequence_length,
            self.qk_head_dim,
        )
        v_shape = (
            self.batch_size,
            self.num_heads,
            source_sequence_length,
            self.v_head_dim,
        )
        query = query.view(q_shape)
        key = key.view(k_shape)
        value = value.view(v_shape)
        return query, key, value

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


class Processor:
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        validator: "MultiHeadAttentionConfigValidator",
        output_model: nn.Module,
    ):
        self.cfg = cfg
        self.validator = validator
        self.output_model = output_model
        self.return_attention_weights_flag = self.cfg.return_attention_weights_flag
        self.processor = self.__create_processor()

    def __create_processor(
        self,
    ) -> ProcessorDefault | ProcessorWithReturnedWeights:
        processor_type = ProcessorDefault
        if self.return_attention_weights_flag:
            processor_type = ProcessorWithReturnedWeights
        return processor_type(self.cfg, self.validator, self.output_model)

    def compute_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        return self.processor.compute_attention(query, key, value, attention_mask)
