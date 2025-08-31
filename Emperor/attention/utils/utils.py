import torch
import math
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import TYPE_CHECKING

from Emperor.base.utils import Module
from Emperor.layers.utils.base import LayerBlock
from Emperor.layers.utils.linears import LinearLayer

if TYPE_CHECKING:
    from Emperor.config import ModelConfig
    from Emperor.attention.attention import MultiHeadAttentionConfig


class BatchDimensionManager:
    def __init__(self, cfg: "MultiHeadAttentionConfig"):
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.should_transpose_first_two_dims = False

    def enforce_batch_as_second_dim(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        is_input_batched = query.dim() == 3
        _, batch_size, _ = query.size()
        is_expected_batch_size = batch_size != self.batch_size

        self.should_transpose_first_two_dims = (
            is_expected_batch_size and is_input_batched
        )
        if not self.should_transpose_first_two_dims:
            return query, key, value
        if key is value:
            return self.__transpose_shared_tensors(query, key)
        return (tensor.transpose(0, 1) for tensor in (query, key, value))

    def __transpose_shared_tensors(
        self, query: Tensor, key: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if query is key:
            query = key = value = query.transpose(0, 1)
            return query, key, value
        query, key = (tensor.transpose(0, 1) for tensor in (query, key))
        value = key
        return query, key, value

    def reverse_enforced_batch_as_second_dim(self, attention_output: Tensor) -> Tensor:
        if not self.should_transpose_first_two_dims:
            return attention_output
        return attention_output.transpose(1, 0)


class KeyValueBias(Module):
    def __init__(self, cfg: "MultiHeadAttentionConfig"):
        super().__init__()
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.embedding_dim = self.cfg.embedding_dim
        self.add_key_value_bias_flag = self.cfg.add_key_value_bias_flag
        self.key_bias_vector, self.value_bias_vector = self.__build_kv_bias_vectors()

    def __build_kv_bias_vectors(self):
        if not self.add_key_value_bias_flag:
            return None, None
        bias_k = self._init_parameter_bank((1, 1, self.embedding_dim))
        bias_v = self._init_parameter_bank((1, 1, self.embedding_dim))
        return bias_k, bias_v

    def add_kv_learnable_bias_vectors(
        self,
        key_projections: Tensor,
        value_projections: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        if not self.add_key_value_bias_flag:
            return (
                key_projections,
                value_projections,
                key_padding_mask,
                attention_mask,
            )
        repeated_key_bias = self.key_bias_vector.repeat(1, self.batch_size, 1)
        key_projections_with_bias_vector = torch.cat(
            [key_projections, repeated_key_bias]
        )
        repeated_value_bias = self.value_bias_vector.repeat(1, self.batch_size, 1)
        value_projections_with_bias_vector = torch.cat(
            [value_projections, repeated_value_bias]
        )
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
        if attention_mask is not None:
            attention_mask = F.pad(attention_mask, (0, 1))

        return (
            key_projections_with_bias_vector,
            value_projections_with_bias_vector,
            key_padding_mask,
            attention_mask,
        )


class Utils:
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        validator: "Validator",
    ):
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.validator = validator
        self.num_heads = self.cfg.num_heads
        self.embedding_dim = self.cfg.embedding_dim
        self.zero_attention_flag = self.cfg.zero_attention_flag
        self.query_key_projection_dim = self.cfg.query_key_projection_dim
        self.value_projection_dim = self.cfg.value_projection_dim
        self.source_sequence_length = self.cfg.source_sequence_length
        self.target_sequence_length = self.cfg.target_sequence_length
        self.head_dim = self.embedding_dim // self.num_heads
        self.qk_head_dim, self.v_head_dim = self.__resolve_qkv_head_dim()

    def __resolve_qkv_head_dim(self):
        qk_head_dim = (
            self.query_key_projection_dim // self.num_heads
            if self.query_key_projection_dim != 0
            else self.head_dim
        )
        v_head_dim = (
            self.value_projection_dim // self.num_heads
            if self.value_projection_dim != 0
            else self.head_dim
        )
        return qk_head_dim, v_head_dim

    def add_batch_dimension_if_missing(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        if self.validator.is_tensor_batched(query):
            return query, key, value, key_padding_mask, attention_mask
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(0)
        return query, key, value, key_padding_mask, attention_mask

    def reshape_qkv_for_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        static_keys: Tensor | None = None,
        static_values: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self.validator.check_static_projection_shapes(static_keys, static_values)

        query = self.__reshape_projection_tesnor(query, None, self.qk_head_dim)
        key = self.__reshape_projection_tesnor(key, static_keys, self.qk_head_dim)
        value = self.__reshape_projection_tesnor(value, static_values, self.v_head_dim)

        return query, key, value

    def __reshape_projection_tesnor(
        self,
        tensor: Tensor,
        static_tensor: Tensor | None = None,
        head_dim: int | None = None,
    ) -> Tensor:
        if static_tensor is not None:
            return static_tensor

        sequence_length = tensor.shape[0]
        # shape = (sequence_length, self.batch_size * self.num_heads, self.head_dim)
        head_dim = head_dim or self.head_dim
        shape = (sequence_length, self.batch_size * self.num_heads, head_dim)
        reshaped_tensor = tensor.view(shape)
        return reshaped_tensor.transpose(0, 1)

    def add_zero_attention(
        self,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        if not self.zero_attention_flag:
            return key, value, key_padding_mask, attention_mask

        padded_key = self.__concatenate_zeros_tensor(key)
        padded_value = self.__concatenate_zeros_tensor(value)
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
        if attention_mask is not None:
            attention_mask = F.pad(attention_mask, (0, 1))

        return padded_key, padded_value, key_padding_mask, attention_mask

    def __concatenate_zeros_tensor(self, tensor: Tensor) -> Tensor:
        zero_attetion_shape = (self.batch_size * self.num_heads, 1, self.head_dim)
        zeros_tensor = torch.zeros(
            zero_attetion_shape, dtype=tensor.dtype, device=tensor.device
        )
        return torch.cat([tensor, zeros_tensor], dim=1)

    def merge_padding_and_attention_mask(
        self,
        key: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor | None:
        if key_padding_mask is None:
            return attention_mask

        source_sequence_length = key.size(1)

        shape_view = (self.batch_size, 1, 1, source_sequence_length)
        key_padding_mask = key_padding_mask.view(shape_view)

        shape_expand = (-1, self.num_heads, -1, -1)
        key_padding_mask = key_padding_mask.expand(shape_expand)

        batch_size = self.batch_size * self.num_heads
        shape_reshape = (batch_size, 1, source_sequence_length)
        key_padding_mask = key_padding_mask.reshape(shape_reshape)

        attention_mask = self.__merge_attention_and_padding_mask(
            key_padding_mask, attention_mask
        )
        return attention_mask

    def __merge_attention_and_padding_mask(
        self,
        key_padding_mask: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor | None:
        if attention_mask is None:
            return key_padding_mask
        return attention_mask + key_padding_mask


class ProcessorBase:
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        validator: "Validator",
        projector: "Projector",
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
        self.qk_head_dim = (
            self.query_key_projection_dim // self.num_heads
            if self.query_key_projection_dim is not None
            and self.query_key_projection_dim != 0
            else self.head_dim
        )
        self.v_head_dim = (
            self.value_projection_dim // self.num_heads
            if self.value_projection_dim is not None and self.value_projection_dim != 0
            else self.head_dim
        )

    def _compute_attention_output(self, weighted_values: Tensor) -> Tensor:
        attention_output = self.projector.compute_output_projection(weighted_values)
        embedding_dim = attention_output.size(1)
        return attention_output.view(
            self.target_sequence_length,
            self.batch_size,
            embedding_dim,
        )


class ProcessorWithReturnedWeights(ProcessorBase):
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        validator: "Validator",
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
        validator: "Validator",
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
            query,
            key,
            value,
            attention_mask,
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
        validator: "Validator",
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


class ProjectorBase(Module):
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        main_cfg: "ModelConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.main_cfg = main_cfg
        self.model_type = self.cfg.model_type
        self.embedding_dim = self.cfg.embedding_dim
        self.value_projection_dim = self.cfg.value_projection_dim
        self.query_key_projection_dim = self.cfg.query_key_projection_dim
        self.__resolve_kv_dimensions()

    def __resolve_kv_dimensions(self):
        self.query_key_projection_dim = (
            self.embedding_dim
            if self.query_key_projection_dim == 0
            else self.query_key_projection_dim
        )
        self.value_projection_dim = (
            self.embedding_dim
            if self.value_projection_dim == 0
            else self.value_projection_dim
        )

    def _create_model(self, input_dim: int, output_dim: int) -> LayerBlock:
        config = self.__resolve_model_type_overrides(
            self.main_cfg, input_dim, output_dim
        )
        output_model = self.model_type.value(config)
        return LayerBlock(model=output_model)

    def __resolve_model_type_overrides(
        self,
        cfg: "ModelConfig",
        input_dim: int,
        output_dim: int,
    ):
        c = copy.deepcopy(cfg)
        if issubclass(self.model_type.value, LinearLayer):
            c.linear_layer_model_config.input_dim = input_dim
            c.linear_layer_model_config.output_dim = output_dim
            return c
        c.mixture_model_config.input_dim = input_dim
        c.mixture_model_config.output_dim = output_dim
        return c


class Projector(ProjectorBase):
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        main_cfg: "ModelConfig",
    ):
        super().__init__(cfg, main_cfg)
        self.use_separate_projection_weight_flag = (
            self.cfg.use_separate_projection_weight_flag
        )
        self.return_attention_weights_flag = self.cfg.return_attention_weights_flag

        m = self.__build_projection_models()
        if isinstance(m, tuple):
            self.query_model, self.key_model, self.value_model = m
        else:
            self.qkv_model = m

        self.output_model = self._create_model(
            self.value_projection_dim, self.embedding_dim
        )

    def __build_projection_models(self) -> tuple:
        if (
            not self.use_separate_projection_weight_flag
            and self.__are_qkv_dimensions_equal()
        ):
            return self.__build_shared_projection_models()
        return self.__build_separate_projection_models()

    def __build_separate_projection_models(self) -> tuple:
        query_model = self._create_model(
            self.embedding_dim, self.query_key_projection_dim
        )
        key_model = self._create_model(
            self.embedding_dim, self.query_key_projection_dim
        )
        value_model = self._create_model(self.embedding_dim, self.value_projection_dim)
        self.register_parameter("qkv_model", None)
        return query_model, key_model, value_model

    def __build_shared_projection_models(self) -> LayerBlock:
        self.register_parameter("query_model", None)
        self.register_parameter("key_model", None)
        self.register_parameter("value_model", None)
        qkv_model = self._create_model(self.embedding_dim, self.embedding_dim * 3)
        return qkv_model

    def __are_qkv_dimensions_equal(self) -> bool:
        are_qk_dims_same = self.embedding_dim == self.query_key_projection_dim
        are_qv_dims_same = self.embedding_dim == self.value_projection_dim
        return are_qk_dims_same and are_qv_dims_same

    def compute_qkv_projections(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        are_qkv_same = key is value and query is key
        should_perform_self_attention = (
            are_qkv_same and not self.use_separate_projection_weight_flag
        )
        if should_perform_self_attention:
            self.__check_self_attention_projection_inputs(key, value)
            return self.__compute_self_attention_projections(query)

        assert not self.return_attention_weights_flag, (
            "`attention_weights` can be returned only when self attention is performed, ensure that `use_separate_projection_weight_flag` is set to `False` and the `query`, `key` and `value` tensors are the same tensor."
        )
        self.__validate_attention_weights_flag_with_projection_type()
        self.__are_separate_projection_models_initialized()
        self.__check_indepentent_projections_inputs(key, value)
        return self.__compute_indepentet_projections(query, key, value)

    def __compute_self_attention_projections(
        self, query: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        qkv_projection = self.__compute_projection(query, self.qkv_model)
        query_projections, key_projections, value_projections = (
            self.__split_self_attention_projection(qkv_projection)
        )
        return query_projections, key_projections, value_projections

    def __split_self_attention_projection(
        self, qkv_projections: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        projections = qkv_projections.unflatten(-1, (3, -1))
        projections = projections.unsqueeze(0)
        projections = projections.transpose(0, -2)
        projections = projections.squeeze(-2)
        projections = projections.contiguous()
        query_projections = projections[0]
        key_projections = projections[1]
        value_projections = projections[2]

        return query_projections, key_projections, value_projections

    def __compute_indepentet_projections(
        self, query: Tensor, key: Tensor, value: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        query_projections = self.__compute_projection(query, self.query_model)
        key_projections = self.__compute_projection(key, self.key_model)
        value_projections = self.__compute_projection(value, self.value_model)

        return query_projections, key_projections, value_projections

    def __compute_projection(self, tensor: Tensor, model: nn.Module) -> Tensor:
        sequence_length, batch_size, embedding_dim = tensor.shape
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        tensor_reshaped = tensor.view(-1, embedding_dim)
        projections = model(tensor_reshaped)
        if isinstance(projections, tuple):
            projections, _ = projections
        return projections.view(sequence_length, batch_size, -1)

    def compute_output_projection(self, weighted_values: Tensor) -> Tensor:
        output = self.output_model(weighted_values)
        if isinstance(output, tuple):
            output, _ = output
        return output

    def __validate_attention_weights_flag_with_projection_type(self):
        assert not self.return_attention_weights_flag, (
            "`attention_weights` can be returned only when self attention is performed, ensure that `use_separate_projection_weight_flag` is set to `False` and the `query`, `key` and `value` tensors are the same tensor."
        )

    def __are_separate_projection_models_initialized(self) -> None:
        ensure_qkv_models_exist = (
            self.query_model is not None
            and self.key_model is not None
            and self.value_model is not None
        )
        assert ensure_qkv_models_exist, (
            "When query, key, and value are not the same and self attention is not performed, ensure `use_separate_projection_weight_flag` is `True`"
        )

    def __check_indepentent_projections_inputs(
        self, key: Tensor, value: Tensor
    ) -> None:
        k_sequence_length, k_batch_size, _ = key.shape
        v_sequence_length, v_batch_size, _ = value.shape
        is_kv_sequence_length_same = k_sequence_length == v_sequence_length
        is_kv_batch_size_same = k_batch_size == v_batch_size
        if not (is_kv_sequence_length_same and is_kv_batch_size_same):
            raise RuntimeError(
                f"key shape {key.shape} does not match value shape {value.shape}"
            )

    def __check_self_attention_projection_inputs(
        self, key: Tensor, value: Tensor
    ) -> None:
        are_kv_shapes_same = key.shape == value.shape
        if not are_kv_shapes_same:
            raise RuntimeError(
                f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
            )


class Mask:
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        validator: "Validator",
    ):
        self.cfg = cfg
        self.validator = validator
        self.target_dtype = self.cfg.target_dtype
        self.causal_attention_mask_flag = self.cfg.causal_attention_mask_flag
        self.return_attention_weights_flag = self.cfg.return_attention_weights_flag

    def check_padding_and_attention_masks(
        self,
        key_padding_mask: Tensor | None,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        key_padding_mask = self.__canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attention_mask),
            other_name="attention_mask",
            target_type=self.target_dtype,
        )
        attention_mask = self.__validate_attention_mask(
            key_padding_mask,
            attention_mask,
        )

        return key_padding_mask, attention_mask

    def __validate_attention_mask(
        self,
        key_padding_mask: Tensor | None,
        attention_mask: Tensor | None,
    ) -> Tensor | None:
        if (
            self.causal_attention_mask_flag
            and key_padding_mask is None
            and not self.return_attention_weights_flag
        ):
            return

        if key_padding_mask is not None:
            self.causal_attention_mask_flag = False

        attention_mask = self.__canonical_mask(
            mask=attention_mask,
            mask_name="attention_mask",
            other_type=None,
            other_name="",
            target_type=self.target_dtype,
            check_other=False,
        )

        return attention_mask

    def __canonical_mask(
        self,
        mask: Tensor | None,
        mask_name: str,
        other_type: "DType | None",
        other_name: str,
        target_type: "DType",
        check_other: bool = True,
    ) -> Tensor | None:
        if mask is None:
            return mask

        self.validator.is_mask_float_or_bool(mask, mask_name)
        self.validator.is_mask_correct_dtype(
            mask, mask_name, other_type, other_name, check_other
        )

        if not torch.is_floating_point(mask):
            mask_placeholder = torch.zeros_like(mask, dtype=target_type)
            mask = mask_placeholder.masked_fill_(mask, float("-inf"))
        return mask


class Validator:
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
    ):
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.num_heads = self.cfg.num_heads
        self.embeding_dim = self.cfg.embedding_dim
        self.causal_attention_mask_flag = self.cfg.causal_attention_mask_flag
        self.head_dim = self.embeding_dim // self.num_heads
        self.batched_input_flag = None
        self.source_sequence_length = self.cfg.source_sequence_length
        self.target_sequence_length = self.cfg.target_sequence_length
        self.return_attention_weights_flag = self.cfg.return_attention_weights_flag

    def assert_correct_head_dim(self, head_dim: int) -> None:
        assert (head_dim * self.num_heads) == self.embeding_dim, (
            "`embedding_dim` must be perfectly divisible by `number_of_heads`."
        )

    def is_mask_float_or_bool(
        self,
        mask: Tensor,
        mask_name: str,
    ) -> None:
        is_float_float = torch.is_floating_point(mask)
        is_mask_bool = mask.dtype == torch.bool
        is_mask_float_or_bool = not is_mask_bool and not is_float_float
        if is_mask_float_or_bool:
            raise RuntimeError(
                f"only bool and floating types of {mask_name} are supported"
            )

    def is_mask_correct_dtype(
        self,
        mask: Tensor,
        mask_name: str,
        other_type: "DType | None",
        other_name: str,
        check_other: bool = True,
    ):
        mask_dtype = mask.dtype
        should_check_other_dtype = check_other and other_type is not None
        if should_check_other_dtype:
            does_dtype_match = mask_dtype == other_type
            if not does_dtype_match:
                raise RuntimeError(
                    f"Support for mismatched {mask_name} and {other_name} "
                    "is deprecated. Use same type for both instead."
                )

    def check_attention_input_shapes(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> bool:
        self.batched_input_flag = self.is_tensor_batched(query)

        self.__check_query_dims(query)
        self.__check_query_key_value_dimension_count(key, value)
        self.__check_key_padding_mask_dimension_count(key_padding_mask)
        self.__check_attention_mask_dim_count_and_shape(attention_mask)
        self.__ensure_attention_mask_if_causal(attention_mask)

        return self.batched_input_flag

    def __check_query_dims(self, query: Tensor) -> None:
        if query.dim() not in (2, 3):
            raise RuntimeError(
                f"Query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D tensor"
            )

    def __check_query_key_value_dimension_count(
        self, key: Tensor, value: Tensor
    ) -> None:
        expected_dims = 3 if self.batched_input_flag else 2
        is_qk_dim_count_same = key.dim() == expected_dims
        is_qv_dim_count_same = value.dim() == expected_dims
        are_qkv_dim_counts_same = is_qk_dim_count_same and is_qv_dim_count_same
        if not are_qkv_dim_counts_same:
            raise RuntimeError(
                f"For {self.__format_dimension_context()} query, expected key and value to be {expected_dims}-D "
                f"but found {key.dim()}-D and {value.dim()}-D tensors respectively"
            )

    def __check_key_padding_mask_dimension_count(
        self,
        key_padding_mask: Tensor | None = None,
    ) -> None:
        if key_padding_mask is None:
            return

        expected_dim = 2 if self.batched_input_flag else 1
        if key_padding_mask.dim() != expected_dim:
            raise RuntimeError(
                f"For {self.__format_dimension_context()} query, expected `key_padding_mask` to be None or {expected_dim}-D "
                f"but found {key_padding_mask.dim()}-D tensor instead"
            )

    def __check_attention_mask_dim_count_and_shape(
        self,
        attention_mask: Tensor | None = None,
    ) -> None:
        if attention_mask is None:
            return

        if attention_mask.dim() not in (2, 3):
            raise RuntimeError(
                f"For {self.__format_dimension_context()} query, expected attention_mask to be None, 2-D, or 3-D tensor "
                f"but found {attention_mask.dim()}-D tensor instead"
            )

        expected_shape = self.__resolve_attention_mask_shape(attention_mask)
        if attention_mask.shape != expected_shape:
            raise RuntimeError(
                f"Expected `attention_mask` shape to be {expected_shape} but got {attention_mask.shape}"
            )

    def __format_dimension_context(self) -> str:
        return "batched (3-D)" if self.batched_input_flag else "unbatched (2-D)"

    def __resolve_attention_mask_shape(
        self, attention_mask: Tensor
    ) -> tuple[int, int, int] | tuple[int, int]:
        if attention_mask.dim() == 3:
            return (
                self.batch_size * self.num_heads,
                self.target_sequence_length,
                self.source_sequence_length,
            )
        return (self.target_sequence_length, self.source_sequence_length)

    def __ensure_attention_mask_if_causal(
        self,
        attention_mask: Tensor | None = None,
    ) -> None:
        if self.causal_attention_mask_flag and attention_mask is None:
            raise RuntimeError(
                "Need `attention_mask` if specifying the `causal_attention_mask_flag` hint. "
                "You may use the Transformer module method "
                "`generate_square_subsequent_mask` to create this mask."
            )

    def is_tensor_batched(self, tensor: Tensor) -> bool:
        self.batched_input_flag = tensor.dim() == 3
        return self.batched_input_flag

    def get_batched_input_flag(self) -> bool:
        return self.batched_input_flag

    def is_input_batched(self, tensor: Tensor | None = None) -> bool:
        if self.batched_input_flag is None:
            assert tensor is not None, (
                "Tensor must be provided to check batch dimension."
            )
            self.batched_input_flag = tensor.dim() == 3
            return self.batched_input_flag
        return self.batched_input_flag

    def assert_correct_embedding_dim(self, expected_embedding_dim: int):
        assert self.embedding_dim == expected_embedding_dim, (
            f"Was expecting embedding dimension of {expected_embedding_dim}, but got {self.embedding_dim}"
        )

    def are_separate_projection_models_initialized(self) -> None:
        ensure_qkv_models_exist = (
            self.query_model is not None
            and self.key_model is not None
            and self.value_model is not None
        )
        assert ensure_qkv_models_exist, (
            "When query, key, and value are not the same and self attention is not performed, ensure `use_separate_projection_weight_flag` is `True`"
        )

    def validate_attention_weights_flag_with_projection_type(self):
        assert not self.return_attention_weights_flag, (
            "`attention_weights` can be returned only when self attention is performed, ensure that `use_separate_projection_weight_flag` is set to `False` and the `query`, `key` and `value` tensors are the same tensor."
        )

    def check_indepentent_projections_inputs(self, key: Tensor, value: Tensor) -> None:
        k_sequence_length, k_batch_size, _ = key.shape
        v_sequence_length, v_batch_size, _ = value.shape
        is_kv_sequence_length_same = k_sequence_length == v_sequence_length
        is_kv_batch_size_same = k_batch_size == v_batch_size
        if not (is_kv_sequence_length_same and is_kv_batch_size_same):
            raise RuntimeError(
                f"key shape {key.shape} does not match value shape {value.shape}"
            )

    def check_self_attention_projection_inputs(
        self, key: Tensor, value: Tensor
    ) -> None:
        are_kv_shapes_same = key.shape == value.shape
        if not are_kv_shapes_same:
            raise RuntimeError(
                f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
            )

    def check_static_projection_shapes(
        self,
        static_keys: Tensor | None = None,
        static_values: Tensor | None = None,
    ):
        self.__resolve_static_projection_shape(static_keys)
        self.__resolve_static_projection_shape(static_values, True)

    def __resolve_static_projection_shape(
        self,
        static_tensor: Tensor | None = None,
        value_tensor_flag: bool = False,
    ) -> None:
        if static_tensor is not None:
            tensor_type = self.__resolve_static_projection_type(value_tensor_flag)
            expected_first_dim = self.batch_size * self.num_heads
            assert static_tensor.size(0) == expected_first_dim, (
                f"expecting {tensor_type}.size(0) of {expected_first_dim}, but got {static_tensor.size(0)}"
            )
            assert static_tensor.size(2) == self.head_dim, (
                f"expecting {tensor_type}.size(2) of {self.head_dim}, but got {static_tensor.size(2)}"
            )

    def __resolve_static_projection_type(self, value_tensor_flag: bool = False) -> str:
        return "static_values" if value_tensor_flag else "static_keys"
