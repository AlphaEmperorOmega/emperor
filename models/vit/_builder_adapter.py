from __future__ import annotations

from dataclasses import replace
from types import ModuleType
from typing import Any

from models.transformer._builder_options import (
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
    TransformerPositionalEmbeddingOptions,
    VitOutputOptions,
    VitPatchOptions,
)

_TOP_LEVEL_KEYS = ("batch_size", "learning_rate", "input_dim", "output_dim")


def linear_builder_kwargs_from_flat(
    flat_kwargs: dict[str, Any],
    config_module: ModuleType,
) -> dict[str, Any]:
    kwargs = dict(flat_kwargs)
    builder_kwargs = _pop_top_level_kwargs(kwargs)
    builder_kwargs.update(
        {
            "patch_options": _patch_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("patch_options", None),
            ),
            "encoder_options": _encoder_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("encoder_options", None),
            ),
            "positional_embedding_options": _positional_embedding_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("positional_embedding_options", None),
            ),
            "attention_options": _attention_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("attention_options", None),
            ),
            "feed_forward_options": _feed_forward_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("feed_forward_options", None),
            ),
            "output_options": _output_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("output_options", None),
            ),
        }
    )
    builder_kwargs.update(kwargs)
    return builder_kwargs


def _pop_top_level_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: kwargs.pop(key) for key in _TOP_LEVEL_KEYS if key in kwargs}


def _patch_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: VitPatchOptions | None,
) -> VitPatchOptions:
    options = provided or VitPatchOptions(
        patch_size=config_module.IMAGE_PATCH_SIZE,
        input_channels=config_module.INPUT_CHANNELS,
        image_height=config_module.IMAGE_HEIGHT,
        dropout_probability=config_module.PATCH_DROPOUT_PROBABILITY,
        bias_flag=config_module.PATCH_BIAS_FLAG,
    )
    updates = _pop_updates(
        kwargs,
        {
            "image_patch_size": "patch_size",
            "input_channels": "input_channels",
            "image_height": "image_height",
            "patch_dropout_probability": "dropout_probability",
            "patch_bias_flag": "bias_flag",
        },
    )
    return replace(options, **updates) if updates else options


def _encoder_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: TransformerEncoderOptions | None,
) -> TransformerEncoderOptions:
    options = provided or TransformerEncoderOptions(
        hidden_dim=config_module.STACK_HIDDEN_DIM,
        num_layers=config_module.STACK_NUM_LAYERS,
        activation=config_module.STACK_ACTIVATION,
        dropout_probability=config_module.STACK_DROPOUT_PROBABILITY,
        layer_norm_position=config_module.LAYER_NORM_POSITION,
        causal_attention_mask_flag=False,
    )
    updates = _pop_updates(
        kwargs,
        {
            "stack_hidden_dim": "hidden_dim",
            "stack_num_layers": "num_layers",
            "stack_activation": "activation",
            "stack_dropout_probability": "dropout_probability",
            "layer_norm_position": "layer_norm_position",
            "stack_layer_norm_position": "layer_norm_position",
        },
    )
    return replace(options, **updates) if updates else options


def _positional_embedding_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: TransformerPositionalEmbeddingOptions | None,
) -> TransformerPositionalEmbeddingOptions:
    options = provided or TransformerPositionalEmbeddingOptions(
        option=config_module.POSITIONAL_EMBEDDING_OPTION,
        padding_idx=config_module.POSITIONAL_EMBEDDING_PADDING_IDX,
        auto_expand_flag=config_module.POSITIONAL_EMBEDDING_AUTO_EXPAND_FLAG,
    )
    updates = _pop_updates(
        kwargs,
        {
            "positional_embedding_option": "option",
            "positional_embedding_padding_idx": "padding_idx",
            "positional_embedding_auto_expand_flag": "auto_expand_flag",
        },
    )
    return replace(options, **updates) if updates else options


def _attention_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: TransformerAttentionOptions | None,
) -> TransformerAttentionOptions:
    options = provided or TransformerAttentionOptions(
        num_heads=config_module.ATTN_NUM_HEADS,
        num_layers=config_module.ATTN_NUM_LAYERS,
        bias_flag=config_module.ATTN_BIAS_FLAG,
        add_key_value_bias_flag=config_module.ATTN_ADD_KEY_VALUE_BIAS_FLAG,
    )
    updates = _pop_updates(
        kwargs,
        {
            "attn_num_heads": "num_heads",
            "attn_num_layers": "num_layers",
            "attn_bias_flag": "bias_flag",
            "attn_add_key_value_bias_flag": "add_key_value_bias_flag",
        },
    )
    return replace(options, **updates) if updates else options


def _feed_forward_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: TransformerFeedForwardOptions | None,
) -> TransformerFeedForwardOptions:
    options = provided or TransformerFeedForwardOptions(
        num_layers=config_module.FF_NUM_LAYERS,
        bias_flag=config_module.FF_BIAS_FLAG,
    )
    updates = _pop_updates(
        kwargs,
        {
            "ff_num_layers": "num_layers",
            "ff_bias_flag": "bias_flag",
        },
    )
    return replace(options, **updates) if updates else options


def _output_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: VitOutputOptions | None,
) -> VitOutputOptions:
    options = provided or VitOutputOptions(
        num_layers=config_module.OUTPUT_NUM_LAYERS,
        bias_flag=config_module.OUTPUT_BIAS_FLAG,
    )
    updates = _pop_updates(
        kwargs,
        {
            "output_num_layers": "num_layers",
            "output_bias_flag": "bias_flag",
        },
    )
    return replace(options, **updates) if updates else options


def _pop_updates(
    kwargs: dict[str, Any],
    field_map: dict[str, str],
) -> dict[str, Any]:
    return {
        dataclass_field: kwargs.pop(flat_field)
        for flat_field, dataclass_field in field_map.items()
        if flat_field in kwargs
    }
