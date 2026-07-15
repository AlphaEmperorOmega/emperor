from typing import TYPE_CHECKING

from torch import Tensor

from emperor.attention.core._validator import MultiHeadAttentionValidator

if TYPE_CHECKING:
    from emperor.attention.core.layers import MultiHeadAttentionAbstract
    from emperor.attention.core.runtime import (
        QKV,
        AttentionMasks,
        AttentionRuntimeShape,
    )


class MixtureOfAttentionHeadsValidator(MultiHeadAttentionValidator):
    @staticmethod
    def validate_relative_position_query_shape(
        query: Tensor,
        expected_leading_shape: tuple[int, int, int],
    ) -> None:
        if query.dim() != 5:
            raise RuntimeError(
                "mixture relative-position query must be rank 5 with layout "
                "[batch, selected_expert, head, target, head_width], got "
                f"rank {query.dim()}."
            )
        if tuple(query.shape[:3]) != expected_leading_shape:
            raise RuntimeError(
                "mixture relative-position query must have leading dimensions "
                "(batch_size, top_k, num_heads) "
                f"{expected_leading_shape}, got {tuple(query.shape[:3])}."
            )

    @staticmethod
    def validate_expert_projection_branch_count(
        branch_count: int,
        expected_branch_count: int,
    ) -> None:
        if branch_count != expected_branch_count:
            raise RuntimeError(
                "Mixture attention-ready key/value projections must have a leading "
                "dimension equal to batch_size * top_k * num_heads "
                f"({expected_branch_count}), got {branch_count}."
            )

    @classmethod
    def validate_mixture_mask_shapes(
        cls,
        key_padding_mask: Tensor | None,
        attention_mask: Tensor | None,
        *,
        expected_key_padding_shape: tuple[int, int],
        expected_attention_sequence_shape: tuple[int, int],
        standard_branch_count: int,
        expert_branch_count: int,
    ) -> None:
        cls.validate_key_padding_mask_shape(
            key_padding_mask,
            expected_key_padding_shape,
        )
        if attention_mask is None:
            return
        cls.validate_mixture_attention_mask_rank(attention_mask)
        cls.validate_attention_mask_sequence_shape(
            attention_mask,
            expected_attention_sequence_shape,
        )
        cls.validate_mixture_attention_mask_leading_dimension(
            attention_mask,
            standard_branch_count,
            expert_branch_count,
        )

    @staticmethod
    def validate_mixture_attention_mask_rank(attention_mask: Tensor) -> None:
        if attention_mask.dim() in (2, 3):
            return
        raise RuntimeError(
            "attention_mask must be 2-D or 3-D for mixture of attention heads, "
            f"got {attention_mask.dim()}-D."
        )

    @staticmethod
    def validate_mixture_attention_mask_leading_dimension(
        attention_mask: Tensor,
        standard_branch_count: int,
        expert_branch_count: int,
    ) -> None:
        if attention_mask.dim() != 3:
            return
        leading_dimension = attention_mask.size(0)
        if leading_dimension in (1, standard_branch_count, expert_branch_count):
            return
        raise RuntimeError(
            "3-D attention_mask leading dimension must be 1, "
            "batch_size * num_heads, or batch_size * top_k * num_heads "
            f"(1, {standard_branch_count}, or {expert_branch_count}), got "
            f"{leading_dimension}."
        )

    @classmethod
    def validate(cls, model: "MultiHeadAttentionAbstract") -> None:
        super().validate(model)
        cls.validate_experts_configuration(model)
        cls.validate_expert_key_value_sequence_lengths(model)

    @staticmethod
    def validate_experts_configuration(
        model: "MultiHeadAttentionAbstract",
    ) -> None:
        from emperor.experts.core.config import MixtureOfExpertsConfig
        from emperor.sampler.core.config import RouterConfig, SamplerConfig

        experts_config = model.cfg.experts_config
        use_kv_expert_models_flag = model.cfg.use_kv_expert_models_flag
        if experts_config is None:
            raise ValueError(
                "experts_config is required for mixture of attention heads."
            )
        if not isinstance(experts_config, MixtureOfExpertsConfig):
            raise TypeError(
                "experts_config must be a MixtureOfExpertsConfig, received "
                f"{type(experts_config).__name__}."
            )
        if use_kv_expert_models_flag is None:
            raise ValueError(
                "use_kv_expert_models_flag is required for mixture of attention heads."
            )
        if not isinstance(use_kv_expert_models_flag, bool):
            raise TypeError(
                "use_kv_expert_models_flag must be a bool, received "
                f"{type(use_kv_expert_models_flag).__name__}."
            )

        sampler_config = experts_config.sampler_config
        if not isinstance(sampler_config, SamplerConfig):
            raise TypeError(
                "experts_config.sampler_config must be a SamplerConfig, received "
                f"{type(sampler_config).__name__}."
            )
        if sampler_config.top_k != experts_config.top_k:
            raise ValueError(
                "experts_config.top_k must match "
                "experts_config.sampler_config.top_k, got "
                f"{experts_config.top_k} and {sampler_config.top_k}."
            )
        if sampler_config.num_experts != experts_config.num_experts:
            raise ValueError(
                "experts_config.num_experts must match "
                "experts_config.sampler_config.num_experts, got "
                f"{experts_config.num_experts} and {sampler_config.num_experts}."
            )
        if experts_config.top_k == experts_config.num_experts:
            raise ValueError(
                "MixtureOfAttentionHeads requires sparse indexed routing, so "
                "experts_config.top_k must be less than "
                "experts_config.num_experts; dense routing is not supported."
            )

        router_config = sampler_config.router_config
        if not isinstance(router_config, RouterConfig):
            raise TypeError(
                "experts_config.sampler_config.router_config must be a "
                f"RouterConfig, received {type(router_config).__name__}."
            )
        if router_config.num_experts != experts_config.num_experts:
            raise ValueError(
                "experts_config.num_experts must match experts_config.sampler_config."
                "router_config.num_experts, got "
                f"{experts_config.num_experts} and {router_config.num_experts}."
            )

    @staticmethod
    def validate_expert_key_value_sequence_lengths(
        model: "MultiHeadAttentionAbstract",
    ) -> None:
        if not model.cfg.use_kv_expert_models_flag:
            return
        if model.target_sequence_length != model.source_sequence_length:
            raise ValueError(
                "target_sequence_length and source_sequence_length must be equal "
                "when use_kv_expert_models_flag is True, got "
                f"target_sequence_length={model.target_sequence_length} and "
                f"source_sequence_length={model.source_sequence_length}."
            )

    @classmethod
    def validate_forward_inputs(
        cls,
        model: "MultiHeadAttentionAbstract",
        qkv: "QKV",
        masks: "AttentionMasks",
    ) -> None:
        super().validate_forward_inputs(model, qkv, masks)
        cls.validate_expert_key_value_inputs(
            model, qkv.query, qkv.key, qkv.value
        )
        cls.validate_attention_weights_are_not_requested(model)
        cls.validate_key_value_projection_shapes(qkv.key, qkv.value)

    @staticmethod
    def validate_attention_weights_are_not_requested(
        model: "MultiHeadAttentionAbstract",
    ) -> None:
        if model.return_attention_weights_flag:
            raise RuntimeError(
                "MixtureOfAttentionHeads does not support returning attention_weights; "
                "set return_attention_weights_flag to False."
            )

    @staticmethod
    def validate_expert_key_value_inputs(
        model: "MultiHeadAttentionAbstract",
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> None:
        if not model.cfg.use_kv_expert_models_flag:
            return
        if not (query is key and key is value):
            raise ValueError(
                "query, key, and value must be the same tensor when "
                "use_kv_expert_models_flag is True."
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
        if model.cfg.use_kv_expert_models_flag:
            if static_keys is not None or static_values is not None:
                raise ValueError(
                    "static key/value projections are not supported when "
                    "use_kv_expert_models_flag is True."
                )
            return
        super().validate_static_key_value_inputs(
            model,
            qkv,
            static_keys,
            static_values,
            runtime_shape,
        )
