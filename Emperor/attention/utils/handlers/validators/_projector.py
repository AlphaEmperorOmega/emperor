import torch

from torch.types import Tensor
from Emperor.base.layer import LayerStackConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.attention.utils.handlers.projector import IndependentProjector
    from Emperor.attention.utils.handlers.projector import SelfAttentionProjector
    from Emperor.attention.utils.handlers.projector import (
        MixtureOfAttentionHeadsProjector,
    )


class SelfAttentionProjectorValidator:
    def __init__(self, model: "SelfAttentionProjector"):
        self.model = model

    def ensure_qkv_are_equal_for_self_attention(
        self, key: Tensor, query: Tensor, value: Tensor
    ):
        are_qkv_same = key is value and query is key
        if not are_qkv_same:
            raise RuntimeError(
                "Self attention can only be computed when `query`, `key`, and `value` are the same tensor."
            )


class IndependentProjectorValidator:
    def __init__(self, model: "IndependentProjector"):
        self.model = model

    def ensure_attention_weights_returned_for_self_attention_only(self):
        if self.model.return_attention_weights_flag:
            raise RuntimeError(
                "`attention_weights` can be returned only when self attention is computed, ensure that `attention_option` is set to `True` and the `query`, `key` and `value` tensors are the same tensor."
            )

    def ensure_propper_kv_shapes_for_independent_projector(
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


class MixtureOfAttentionHeadsProjectorValidator:
    def __init__(self, model: "MixtureOfAttentionHeadsProjector"):
        self.model = model
        self.__ensure_required_config_options_are_not_none()
        self.__ensure_required_config_options_are_correct_types()

    def __ensure_required_config_options_are_not_none(self):
        if self.model.experts_config is None:
            raise ValueError("Configuration Error: 'experts_config' is None")
        if self.model.use_kv_expert_models_flag is None:
            raise ValueError("Configuration Error: 'use_kv_expert_models_flag' is None")

    def __ensure_required_config_options_are_correct_types(self):
        if not isinstance(self.model.experts_config, LayerStackConfig):
            raise TypeError(
                f"Configuration Error: 'experts_config' must be of type LayerStackConfig, received type {type(self.model.experts_config).__name__}"
            )
        if not isinstance(self.model.use_kv_expert_models_flag, bool):
            raise TypeError(
                f"Configuration Error: 'use_kv_expert_models_flag' must be of type bool, received type {type(self.model.use_kv_expert_models_flag).__name__}"
            )

    def ensure_attention_weights_returned_for_self_attention_only(self):
        if self.model.return_attention_weights_flag:
            raise RuntimeError(
                "`attention_weights` can be returned only when self attention is computed, ensure that `attention_option` is set to `True` and the `query`, `key` and `value` tensors are the same tensor."
            )

    def ensure_propper_kv_shapes_for_independent_projector(
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
