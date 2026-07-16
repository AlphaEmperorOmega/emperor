"""Private shared attention validation."""

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor._validation import ValidatorBase

if TYPE_CHECKING:
    from emperor.attention._base import MultiHeadAttentionAbstract
    from emperor.attention._runtime import (
        QKV,
        AttentionMasks,
        AttentionRuntimeShape,
    )


class AttentionValidatorBase:
    @staticmethod
    def validate_standard_relative_position_query_shape(
        query: Tensor,
        num_heads: int,
    ) -> None:
        if query.dim() not in (3, 4):
            raise RuntimeError(
                f"relative-position query must be rank 3 or 4, got rank {query.dim()}."
            )
        if query.dim() == 4 and query.size(1) != num_heads:
            raise RuntimeError(
                "relative-position rank-4 query head dimension must equal "
                f"num_heads ({num_heads}), got {query.size(1)}."
            )

    @staticmethod
    def validate_relative_position_query_branch_count(
        branch_count: int,
        expected_branch_count: int,
    ) -> None:
        if branch_count != expected_branch_count:
            raise RuntimeError(
                "relative-position rank-3 query leading dimension must equal "
                "batch_size * num_heads "
                f"({expected_branch_count}), got {branch_count}."
            )

    @staticmethod
    def validate_head_divisibility(model: "MultiHeadAttentionAbstract") -> None:
        if model.embedding_dim % model.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({model.embedding_dim}) must be perfectly divisible "
                f"by num_heads ({model.num_heads})."
            )
        if (
            model.query_key_projection_dim
            and model.query_key_projection_dim % model.num_heads != 0
        ):
            raise ValueError(
                f"query_key_projection_dim ({model.query_key_projection_dim}) must be "
                f"perfectly divisible by num_heads ({model.num_heads})."
            )
        if (
            model.value_projection_dim
            and model.value_projection_dim % model.num_heads != 0
        ):
            raise ValueError(
                f"value_projection_dim ({model.value_projection_dim}) must be "
                f"perfectly divisible by num_heads ({model.num_heads})."
            )

    @staticmethod
    def validate_attention_weights_returned_for_self_attention_only(
        model: "MultiHeadAttentionAbstract",
    ) -> None:
        if model.return_attention_weights_flag:
            raise RuntimeError(
                "attention_weights can be returned only when self attention is "
                "computed; ensure the query, key, and value tensors are the same "
                "tensor."
            )

    @staticmethod
    def validate_input_shapes(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> None:
        if query.dim() not in (2, 3):
            raise RuntimeError(
                f"Query should be an unbatched 2D or batched 3D tensor but received "
                f"a {query.dim()}-D tensor."
            )
        batched = query.dim() == 3
        expected_dims = 3 if batched else 2
        if key.dim() != expected_dims or value.dim() != expected_dims:
            raise RuntimeError(
                f"For a {'batched (3-D)' if batched else 'unbatched (2-D)'} query, "
                f"expected key and value to be {expected_dims}-D but found "
                f"{key.dim()}-D and {value.dim()}-D tensors respectively."
            )
        if key_padding_mask is not None:
            expected_mask_dim = 2 if batched else 1
            if key_padding_mask.dim() != expected_mask_dim:
                raise RuntimeError(
                    f"For a {'batched (3-D)' if batched else 'unbatched (2-D)'} query, "
                    "expected key_padding_mask to be None or "
                    f"{expected_mask_dim}-D but "
                    f"found a {key_padding_mask.dim()}-D tensor instead."
                )
        if attention_mask is not None and attention_mask.dim() not in (2, 3):
            raise RuntimeError(
                f"Expected attention_mask to be None, 2-D, or 3-D but found a "
                f"{attention_mask.dim()}-D tensor instead."
            )

    @staticmethod
    def validate_key_value_projection_shapes(key: Tensor, value: Tensor) -> None:
        if key.shape[:-1] != value.shape[:-1]:
            raise RuntimeError(
                f"key shape {tuple(key.shape)} does not match value shape "
                f"{tuple(value.shape)} on the sequence and batch dimensions."
            )

    @staticmethod
    def validate_attention_ready_projection_branch_count(
        branch_count: int,
        expected_branch_count: int,
    ) -> None:
        if branch_count != expected_branch_count:
            raise RuntimeError(
                "Attention-ready key/value projections must have a leading "
                "dimension equal to batch_size * num_heads "
                f"({expected_branch_count}), got {branch_count}."
            )

    @classmethod
    def validate_static_projection_shapes(
        cls,
        model: "MultiHeadAttentionAbstract",
        static_keys: Tensor | None = None,
        static_values: Tensor | None = None,
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> None:
        cls._validate_static_projection_shape(
            model, static_keys, "static_keys", runtime_shape
        )
        cls._validate_static_projection_shape(
            model, static_values, "static_values", runtime_shape
        )

    @staticmethod
    def _validate_static_projection_shape(
        model: "MultiHeadAttentionAbstract",
        static_tensor: Tensor | None,
        tensor_name: str,
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> None:
        if static_tensor is None:
            return
        if static_tensor.dim() != 3:
            raise RuntimeError(
                f"{tensor_name} must be rank 3 with shape "
                "[batch * heads, source, head_width], got "
                f"rank {static_tensor.dim()}."
            )
        batch_size = (
            runtime_shape.batch_size if runtime_shape is not None else model.batch_size
        )
        expected_first_dim = batch_size * model.num_heads
        if static_tensor.size(0) != expected_first_dim:
            raise RuntimeError(
                f"expecting {tensor_name}.size(0) of {expected_first_dim}, but got "
                f"{static_tensor.size(0)}."
            )
        projection_dim = (
            model.value_projection_dim
            if tensor_name == "static_values"
            else model.query_key_projection_dim
        )
        expected_head_dim = (projection_dim or model.embedding_dim) // model.num_heads
        if static_tensor.size(2) != expected_head_dim:
            raise RuntimeError(
                f"expecting {tensor_name}.size(2) of {expected_head_dim}, but got "
                f"{static_tensor.size(2)}."
            )

    @classmethod
    def validate_mask_shapes(
        cls,
        key_padding_mask: Tensor | None,
        attention_mask: Tensor | None,
        *,
        expected_key_padding_shape: tuple[int, int],
        expected_attention_sequence_shape: tuple[int, int],
        standard_branch_count: int,
    ) -> None:
        cls.validate_key_padding_mask_shape(
            key_padding_mask,
            expected_key_padding_shape,
        )
        if attention_mask is None:
            return
        cls.validate_attention_mask_rank(attention_mask)
        cls.validate_attention_mask_sequence_shape(
            attention_mask,
            expected_attention_sequence_shape,
        )
        cls.validate_attention_mask_leading_dimension(
            attention_mask,
            standard_branch_count,
        )

    @staticmethod
    def validate_key_padding_mask_shape(
        key_padding_mask: Tensor | None,
        expected_shape: tuple[int, int],
    ) -> None:
        if key_padding_mask is None:
            return
        if tuple(key_padding_mask.shape) != expected_shape:
            raise RuntimeError(
                f"key_padding_mask must have shape {expected_shape}, got "
                f"{tuple(key_padding_mask.shape)}."
            )

    @staticmethod
    def validate_attention_mask_rank(
        attention_mask: Tensor,
    ) -> None:
        if attention_mask.dim() in (2, 3):
            return
        raise RuntimeError(
            f"attention_mask must be 2-D or 3-D, got {attention_mask.dim()}-D."
        )

    @staticmethod
    def validate_attention_mask_sequence_shape(
        attention_mask: Tensor,
        expected_shape: tuple[int, int],
    ) -> None:
        if tuple(attention_mask.shape[-2:]) != expected_shape:
            raise RuntimeError(
                "attention_mask must have target/source dimensions "
                f"{expected_shape}, got {tuple(attention_mask.shape[-2:])}."
            )

    @staticmethod
    def validate_attention_mask_leading_dimension(
        attention_mask: Tensor,
        standard_branch_count: int,
    ) -> None:
        if attention_mask.dim() != 3:
            return
        leading_dimension = attention_mask.size(0)
        allowed_dimensions = (1, standard_branch_count)
        if leading_dimension in allowed_dimensions:
            return
        raise RuntimeError(
            "3-D attention_mask leading dimension must be 1 or "
            f"batch_size * num_heads ({standard_branch_count}), got "
            f"{leading_dimension}."
        )

    @staticmethod
    def validate_mask_is_float_or_bool(mask: Tensor, mask_name: str) -> None:
        is_float = torch.is_floating_point(mask)
        is_bool = mask.dtype == torch.bool
        if not is_float and not is_bool:
            raise RuntimeError(
                f"Only bool and floating types of {mask_name} are supported."
            )

    @staticmethod
    def validate_mask_dtype_matches(
        mask: Tensor,
        mask_name: str,
        other_type,
        other_name: str,
        check_other: bool = True,
    ) -> None:
        if check_other and other_type is not None and mask.dtype != other_type:
            raise RuntimeError(
                f"Support for mismatched {mask_name} and {other_name} is deprecated. "
                "Use the same type for both instead."
            )

    @staticmethod
    def validate_attention_mask_for_required_causal_mask(
        attention_mask: Tensor | None,
        causal_attention_mask_flag: bool,
    ) -> None:
        if causal_attention_mask_flag and attention_mask is None:
            raise RuntimeError(
                "Need an attention_mask when the causal_attention_mask_flag is set. "
                "Use the Transformer module method generate_square_subsequent_mask to "
                "create this mask."
            )


class MultiHeadAttentionValidator(AttentionValidatorBase, ValidatorBase):
    OPTIONAL_FIELDS = {"relative_positional_embedding_config", "batch_first_flag"}

    @classmethod
    def validate(cls, model: "MultiHeadAttentionAbstract") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.validate_batch_first_flag(model.batch_first_flag)
        cls.validate_integer_dimension_types(model)
        cls.validate_projection_dimensions(model)
        cls.validate_dropout_probability(model.dropout_probability)
        cls.validate_target_dtype(model.target_dtype)
        cls.validate_nested_configurations(model.cfg)
        cls.validate_relative_configuration(model)
        cls.validate_dimensions(
            batch_size=model.batch_size,
            num_heads=model.num_heads,
            embedding_dim=model.embedding_dim,
            target_sequence_length=model.target_sequence_length,
            source_sequence_length=model.source_sequence_length,
        )
        cls.validate_head_divisibility(model)

    @staticmethod
    def validate_batch_first_flag(batch_first_flag: bool | None) -> None:
        if batch_first_flag is not None and type(batch_first_flag) is not bool:
            raise TypeError(
                "batch_first_flag must be True, False, or None, received "
                f"{batch_first_flag!r}."
            )

    @staticmethod
    def validate_integer_dimension_types(
        model: "MultiHeadAttentionAbstract",
    ) -> None:
        for name in (
            "batch_size",
            "num_heads",
            "embedding_dim",
            "query_key_projection_dim",
            "value_projection_dim",
            "target_sequence_length",
            "source_sequence_length",
        ):
            value = getattr(model, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be int, received {type(value).__name__}.")

    @staticmethod
    def validate_projection_dimensions(
        model: "MultiHeadAttentionAbstract",
    ) -> None:
        for name, value in (
            ("query_key_projection_dim", model.query_key_projection_dim),
            ("value_projection_dim", model.value_projection_dim),
        ):
            if value < 0:
                raise ValueError(f"{name} must be 0 or greater, received {value}.")

    @staticmethod
    def validate_dropout_probability(dropout_probability: float) -> None:
        if not 0.0 <= dropout_probability <= 1.0:
            raise ValueError(
                "dropout probability is invalid: dropout_probability must be between "
                f"0 and 1 inclusive, received {dropout_probability}."
            )

    @staticmethod
    def validate_target_dtype(target_dtype: torch.dtype) -> None:
        if not isinstance(target_dtype, torch.dtype):
            raise TypeError(
                "target_dtype must be a torch dtype, received "
                f"{type(target_dtype).__name__}."
            )
        if not target_dtype.is_floating_point:
            raise ValueError(
                f"target_dtype must be a floating torch dtype, received {target_dtype}."
            )

    @staticmethod
    def validate_nested_configurations(cfg) -> None:
        from emperor.embedding.relative import (
            RelativePositionalEmbeddingConfig,
        )
        from emperor.layers import LayerStackConfig, RecurrentLayerConfig

        if not isinstance(
            cfg.projection_model_config,
            (LayerStackConfig, RecurrentLayerConfig),
        ):
            raise TypeError(
                "projection model configuration must be a LayerStackConfig or "
                "RecurrentLayerConfig, received "
                f"{type(cfg.projection_model_config).__name__}."
            )
        relative_config = cfg.relative_positional_embedding_config
        if relative_config is not None and not isinstance(
            relative_config,
            RelativePositionalEmbeddingConfig,
        ):
            raise TypeError(
                "relative positional embedding configuration must be a "
                "RelativePositionalEmbeddingConfig or None, received "
                f"{type(relative_config).__name__}."
            )

    @staticmethod
    def validate_relative_configuration(
        model: "MultiHeadAttentionAbstract",
    ) -> None:
        relative_config = model.cfg.relative_positional_embedding_config
        if relative_config is None:
            return
        if relative_config.num_heads != model.num_heads:
            raise ValueError(
                "relative positional embedding num_heads must match attention "
                f"num_heads, got {relative_config.num_heads} and {model.num_heads}."
            )
        effective_qk_width = model.query_key_projection_dim or model.embedding_dim
        if relative_config.embedding_dim != effective_qk_width:
            raise ValueError(
                "relative positional embedding embedding_dim must match effective "
                "query/key projection width, got "
                f"{relative_config.embedding_dim} and {effective_qk_width}."
            )

    @classmethod
    def validate_forward_inputs(
        cls,
        model: "MultiHeadAttentionAbstract",
        qkv: "QKV",
        masks: "AttentionMasks",
    ) -> None:
        cls.validate_input_shapes(
            qkv.query,
            qkv.key,
            qkv.value,
            masks.key_padding_mask,
            masks.attention_mask,
        )

    @staticmethod
    def validate_runtime_shape(
        model: "MultiHeadAttentionAbstract",
        runtime_shape: "AttentionRuntimeShape",
    ) -> None:
        configured_and_actual = (
            ("batch_size", model.batch_size, runtime_shape.batch_size),
            (
                "target_sequence_length",
                model.target_sequence_length,
                runtime_shape.target_sequence_length,
            ),
            (
                "source_sequence_length",
                model.source_sequence_length,
                runtime_shape.source_sequence_length,
            ),
        )
        for name, configured_maximum, actual in configured_and_actual:
            if actual > configured_maximum:
                raise ValueError(
                    f"Runtime {name} ({actual}) exceeds configured maximum "
                    f"({configured_maximum})."
                )

    @staticmethod
    def validate_runtime_tensors(
        model: "MultiHeadAttentionAbstract",
        qkv: "QKV",
    ) -> None:
        if qkv.query.size(0) <= 0:
            raise RuntimeError("query sequence length must be greater than 0.")
        if qkv.key.size(0) <= 0 or qkv.value.size(0) <= 0:
            raise RuntimeError("key and value sequence lengths must be greater than 0.")
        batch_sizes = (qkv.query.size(1), qkv.key.size(1), qkv.value.size(1))
        if len(set(batch_sizes)) != 1:
            raise RuntimeError(
                "query, key, and value batch sizes must match, got "
                f"{batch_sizes[0]}, {batch_sizes[1]}, and {batch_sizes[2]}."
            )
        for name, tensor in (
            ("query", qkv.query),
            ("key", qkv.key),
            ("value", qkv.value),
        ):
            if tensor.size(-1) != model.embedding_dim:
                raise RuntimeError(
                    f"{name} embedding width must be {model.embedding_dim}, got "
                    f"{tensor.size(-1)}."
                )
        dtypes = (qkv.query.dtype, qkv.key.dtype, qkv.value.dtype)
        if len(set(dtypes)) != 1:
            raise RuntimeError(
                "query, key, and value dtypes must match, got "
                f"{dtypes[0]}, {dtypes[1]}, and {dtypes[2]}."
            )
        if not all(
            torch.is_floating_point(tensor)
            for tensor in (qkv.query, qkv.key, qkv.value)
        ):
            raise RuntimeError("query, key, and value must be floating point tensors.")
        devices = (qkv.query.device, qkv.key.device, qkv.value.device)
        if len(set(devices)) != 1:
            raise RuntimeError(
                "query, key, and value devices must match, got "
                f"{devices[0]}, {devices[1]}, and {devices[2]}."
            )

    @classmethod
    def validate_static_key_value_inputs(
        cls,
        model: "MultiHeadAttentionAbstract",
        qkv: "QKV",
        static_keys: Tensor | None,
        static_values: Tensor | None,
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> None:
        cls.validate_static_projection_shapes(
            model,
            static_keys,
            static_values,
            runtime_shape,
        )
        for name, tensor in (
            ("static_keys", static_keys),
            ("static_values", static_values),
        ):
            if tensor is None:
                continue
            if tensor.dtype != qkv.query.dtype:
                raise RuntimeError(
                    f"{name} dtype must match query dtype, got "
                    f"{tensor.dtype} and {qkv.query.dtype}."
                )
            if tensor.device != qkv.query.device:
                raise RuntimeError(
                    f"{name} device must match query device, got "
                    f"{tensor.device} and {qkv.query.device}."
                )
        key_source_length = (
            static_keys.size(1) if static_keys is not None else qkv.key.size(0)
        )
        value_source_length = (
            static_values.size(1) if static_values is not None else qkv.value.size(0)
        )
        if key_source_length != value_source_length:
            raise RuntimeError(
                "Selected key and value sources must have equal sequence lengths, "
                f"got {key_source_length} and {value_source_length}."
            )
