"""Private mixer-attention validation implementation."""

from typing import TYPE_CHECKING

import torch

from emperor._validation import ValidatorBase
from emperor.layers import LayerState

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.attention._variants.mixer.layer import MixerAttention


class MixerAttentionValidator(ValidatorBase):
    @classmethod
    def validate(cls, model: "MixerAttention") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.validate_dimensions(
            embedding_dim=model.embedding_dim,
            sequence_length=model.sequence_length,
        )
        cls._validate_non_causal(model.causal_attention_mask_flag)
        cls._validate_mixing_model_config(model.mixing_model_config)

    @staticmethod
    def _validate_non_causal(causal_attention_mask_flag: bool) -> None:
        if causal_attention_mask_flag:
            raise ValueError(
                "causal_attention_mask_flag must be False for MixerAttention."
            )

    @staticmethod
    def _validate_mixing_model_config(mixing_model_config: object) -> None:
        from emperor.experts import MixtureOfExpertsModelConfig
        from emperor.layers import LayerStackConfig, RecurrentLayerConfig

        if not isinstance(
            mixing_model_config,
            (
                LayerStackConfig,
                MixtureOfExpertsModelConfig,
                RecurrentLayerConfig,
            ),
        ):
            raise TypeError(
                "mixing_model_config must be a LayerStackConfig or "
                "RecurrentLayerConfig, or a MixtureOfExpertsModelConfig for "
                "MixerAttention, got "
                f"{type(mixing_model_config).__name__}."
            )

    @classmethod
    def validate_forward_inputs(
        cls,
        model: "MixerAttention",
        q: "Tensor",
        k: "Tensor",
        v: "Tensor",
        key_padding_mask: "Tensor | None",
        attention_mask: "Tensor | None",
        static_k: "Tensor | None",
        static_v: "Tensor | None",
    ) -> None:
        cls._validate_self_processing(q, k, v)
        cls._validate_unsupported_inputs(
            key_padding_mask,
            attention_mask,
            static_k,
            static_v,
        )
        cls._validate_runtime_tensor(model, q)

    @staticmethod
    def _validate_self_processing(q: object, k: object, v: object) -> None:
        if q is not k or q is not v:
            raise RuntimeError(
                "MixerAttention requires q, k, and v to be the same tensor object."
            )

    @staticmethod
    def _validate_unsupported_inputs(
        key_padding_mask: "Tensor | None",
        attention_mask: "Tensor | None",
        static_k: "Tensor | None",
        static_v: "Tensor | None",
    ) -> None:
        if key_padding_mask is not None:
            raise RuntimeError("MixerAttention does not support key padding masks.")
        if attention_mask is not None:
            raise RuntimeError("MixerAttention does not support attention masks.")
        if static_k is not None or static_v is not None:
            raise RuntimeError(
                "MixerAttention does not support static key/value projections."
            )

    @staticmethod
    def _validate_runtime_tensor(
        model: "MixerAttention",
        tensor: "Tensor",
    ) -> None:
        if not torch.is_tensor(tensor):
            raise TypeError(
                f"MixerAttention input must be a Tensor, got {type(tensor).__name__}."
            )
        if tensor.dim() != 3:
            raise RuntimeError(
                f"MixerAttention input must be rank three, got rank {tensor.dim()}."
            )
        if not torch.is_floating_point(tensor):
            raise RuntimeError("MixerAttention input must be floating point.")
        if any(dimension <= 0 for dimension in tensor.shape):
            raise RuntimeError(
                "MixerAttention batch, sequence, and embedding dimensions "
                "must be non-empty."
            )

        sequence_axis = 1 if model.batch_first_flag else 0
        actual_sequence_length = tensor.size(sequence_axis)
        if actual_sequence_length != model.sequence_length:
            raise RuntimeError(
                "MixerAttention sequence length must be exactly "
                f"{model.sequence_length}, got {actual_sequence_length}."
            )
        actual_embedding_dim = tensor.size(-1)
        if actual_embedding_dim != model.embedding_dim:
            raise RuntimeError(
                "MixerAttention embedding width must be exactly "
                f"{model.embedding_dim}, got {actual_embedding_dim}."
            )

    @staticmethod
    def validate_mixing_state(
        state: object,
        flattened_input: "Tensor",
    ) -> LayerState:
        if not isinstance(state, LayerState):
            raise RuntimeError(
                "MixerAttention mixing model must return a LayerState, got "
                f"{type(state).__name__}."
            )
        hidden = state.hidden
        if not torch.is_tensor(hidden):
            raise RuntimeError(
                "MixerAttention mixing model LayerState.hidden must be a Tensor."
            )
        if hidden.shape != flattened_input.shape:
            raise RuntimeError(
                "MixerAttention mixing model must preserve flattened shape "
                f"{tuple(flattened_input.shape)}, got {tuple(hidden.shape)}."
            )
        if hidden.dtype != flattened_input.dtype:
            raise RuntimeError(
                "MixerAttention mixing model must preserve dtype "
                f"{flattened_input.dtype}, got {hidden.dtype}."
            )
        if hidden.device != flattened_input.device:
            raise RuntimeError(
                "MixerAttention mixing model must preserve device "
                f"{flattened_input.device}, got {hidden.device}."
            )
        return state
