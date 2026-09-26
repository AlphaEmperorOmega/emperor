from dataclasses import fields

import torch
from torch import Tensor

from emperor.augmentations.adaptive_parameters import WeightDecayScheduleOptions
from emperor.config import ConfigBase
from emperor.embedding.absolute import (
    TextLearnedPositionalEmbeddingConfig,
    TextSinusoidalPositionalEmbeddingConfig,
)
from emperor.experts import MixtureOfExpertsLayerConfig, MixtureOfExpertsModelConfig
from emperor.layers import LayerConfig, LayerStackConfig, RecurrentCompositionConfig

from ._config import HierarchicalByteDecoderConfig
from ._records import OUTPUT_SYMBOLS

_INTEGER_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


def _config_items(config, path):
    if isinstance(config, ConfigBase):
        yield path, config
        for field in fields(config):
            yield from _config_items(
                getattr(config, field.name), f"{path}.{field.name}"
            )


class HierarchicalByteDecoderValidator:
    @staticmethod
    def validate_config_types(cfg, overrides):
        if not isinstance(cfg, HierarchicalByteDecoderConfig):
            raise TypeError("cfg must be HierarchicalByteDecoderConfig")
        if overrides is not None and not isinstance(
            overrides, HierarchicalByteDecoderConfig
        ):
            raise TypeError("overrides must be HierarchicalByteDecoderConfig or None")

    @classmethod
    def resolve_config(cls, cfg):
        for name in ("conditioning_dim", "byte_embedding_dim", "max_token_bytes"):
            cls._positive_integer(getattr(cfg, name), name)
        for name in (
            "byte_position_config",
            "decoder_config",
            "conditioning_projection_config",
            "output_projection_config",
        ):
            if not isinstance(getattr(cfg, name), ConfigBase):
                raise TypeError(f"{name} must be a supplied ConfigBase configuration")
        cls._resolve_positions(cfg)
        cls._resolve_projection(
            cfg.conditioning_projection_config,
            cfg.conditioning_dim,
            cfg.byte_embedding_dim,
            "conditioning_projection_config",
        )
        cls._resolve_projection(
            cfg.output_projection_config,
            cfg.byte_embedding_dim,
            OUTPUT_SYMBOLS,
            "output_projection_config",
        )
        cls._validate_independence(cfg)
        return cls._resolve_decoder(cfg)

    @classmethod
    def _resolve_positions(cls, cfg):
        position = cfg.byte_position_config
        if not isinstance(
            position,
            (
                TextLearnedPositionalEmbeddingConfig,
                TextSinusoidalPositionalEmbeddingConfig,
            ),
        ):
            raise TypeError(
                "byte_position_config must supply learned or sinusoidal text positions"
            )
        cls._dimension(
            position, "embedding_dim", cfg.byte_embedding_dim, "byte_position_config"
        )
        cls._positive_integer(
            position.num_embeddings, "byte_position_config.num_embeddings"
        )
        if position.num_embeddings < cfg.max_token_bytes + 1:
            raise ValueError(
                "byte_position_config.num_embeddings must cover max_token_bytes + 1"
            )
        if position.padding_idx is not None:
            raise ValueError(
                "byte_position_config.padding_idx must be None; all positions are real"
            )

    @classmethod
    def _resolve_projection(cls, projection, input_dim, output_dim, path):
        if not isinstance(
            projection,
            (
                LayerConfig,
                LayerStackConfig,
                RecurrentCompositionConfig,
                MixtureOfExpertsModelConfig,
            ),
        ) or isinstance(projection, MixtureOfExpertsLayerConfig):
            raise TypeError(
                f"{path} must build a LayerState model; wrap tensor layers in LayerConfig"
            )
        cls._dimension(projection, "input_dim", input_dim, path)
        cls._dimension(projection, "output_dim", output_dim, path)
        if (
            isinstance(projection, MixtureOfExpertsModelConfig)
            and projection.stack_config is not None
        ):
            cls._dimension(
                projection.stack_config, "input_dim", input_dim, f"{path}.stack_config"
            )
            cls._dimension(
                projection.stack_config,
                "output_dim",
                output_dim,
                f"{path}.stack_config",
            )

    @classmethod
    def _resolve_decoder(cls, cfg):
        decoder = cfg.decoder_config
        stack = getattr(decoder, "decoder_stack_config", None)
        if not isinstance(stack, ConfigBase) or not hasattr(
            decoder, "encoder_stack_config"
        ):
            raise TypeError(
                "decoder_config must supply a decoder-only Transformer configuration"
            )
        if decoder.encoder_stack_config is not None:
            raise ValueError(
                "decoder_config must be decoder-only (encoder_stack_config=None)"
            )
        for name in ("input_dim", "output_dim"):
            cls._dimension(
                stack,
                name,
                cfg.byte_embedding_dim,
                "decoder_config.decoder_stack_config",
            )
        if isinstance(stack, LayerStackConfig):
            cls._dimension(
                stack,
                "hidden_dim",
                cfg.byte_embedding_dim,
                "decoder_config.decoder_stack_config",
            )
        allowed_attention = set()
        for path, child in _config_items(decoder, "decoder_config"):
            if hasattr(child, "self_attention_config") and hasattr(
                child, "feed_forward_config"
            ):
                cls._dimension(child, "embedding_dim", cfg.byte_embedding_dim, path)
                if child.cross_attention_config is not None:
                    raise ValueError(
                        f"{path}.cross_attention_config must be None; conditioning is a prefix"
                    )
                allowed_attention.add(f"{path}.self_attention_config")
        bounds = []
        for path, child in _config_items(cfg, "hierarchical"):
            if not hasattr(child, "batch_first_flag"):
                continue
            if path.removeprefix("hierarchical.") not in allowed_attention:
                raise ValueError(
                    f"{path}: attention is only supported on the decoder byte sequence axis"
                )
            if child.batch_first_flag is not True:
                raise ValueError(f"{path}.batch_first_flag must explicitly be True")
            if getattr(child, "causal_attention_mask_flag", None) is not True:
                raise ValueError(f"{path}.causal_attention_mask_flag must be True")
            cls._dimension(child, "embedding_dim", cfg.byte_embedding_dim, path)
            cls._positive_integer(child.batch_size, f"{path}.batch_size")
            bounds.append(child.batch_size)
            for name in ("source_sequence_length", "target_sequence_length"):
                limit = getattr(child, name, None)
                cls._positive_integer(limit, f"{path}.{name}")
                if limit < cfg.max_token_bytes + 1:
                    raise ValueError(f"{path}.{name} must cover max_token_bytes + 1")
        if not bounds:
            raise ValueError(
                "decoder_config requires causal byte attention with explicit batch_first_flag"
            )
        return min(bounds)

    @staticmethod
    def _validate_independence(cfg):
        for path, child in _config_items(cfg, "hierarchical"):
            for name in (
                "memory_config",
                "shared_memory_config",
                "context_config",
                "grouping_config",
                "token_sampler_config",
            ):
                if getattr(child, name, None) is not None:
                    raise ValueError(
                        f"{path}.{name} must be None to preserve independent causal tokens"
                    )
            if hasattr(child, "capacity_factor") and child.capacity_factor != 0.0:
                raise ValueError(
                    f"{path}.capacity_factor must be 0.0 to disable batch-dependent dropping"
                )
            schedule = getattr(child, "decay_schedule", None)
            if schedule is not None and schedule != WeightDecayScheduleOptions.DISABLED:
                raise ValueError(
                    f"{path}.decay_schedule must be disabled for grouped decoding"
                )
            if isinstance(child, RecurrentCompositionConfig):
                maximum = next(
                    (
                        getattr(child, name)
                        for name in ("max_steps", "answer_update_count", "high_cycles")
                        if hasattr(child, name)
                    ),
                    None,
                )
                if child.initial_iterations != maximum:
                    raise ValueError(
                        f"{path}.initial_iterations must equal the recurrent maximum"
                    )
                if getattr(child, "sampler_config", None) is not None:
                    raise ValueError(f"{path}.sampler_config must be None")

    @staticmethod
    def _positive_integer(value, path):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{path} must be a supplied positive integer")

    @staticmethod
    def _dimension(config, name, expected, path):
        if not hasattr(config, name):
            raise TypeError(f"{path} must declare {name}")
        value = getattr(config, name)
        if value is None:
            setattr(config, name, expected)
        elif type(value) is not int or value != expected:
            raise ValueError(f"{path}.{name} must equal {expected}, received {value}")

    @staticmethod
    def validate_forward_inputs(cfg, conditioning, prefixes, lengths, mask, device):
        if (
            not isinstance(conditioning, Tensor)
            or conditioning.ndim != 3
            or not conditioning.is_floating_point()
        ):
            raise TypeError(
                "conditioning must be a floating [batch, tokens, conditioning_dim] tensor"
            )
        if (
            min(conditioning.shape[:2]) <= 0
            or conditioning.shape[-1] != cfg.conditioning_dim
        ):
            raise ValueError(
                "conditioning must have nonempty batch/token dimensions and configured conditioning_dim"
            )
        if conditioning.device != device:
            raise ValueError("conditioning must be on the decoder device")
        if (
            not isinstance(prefixes, Tensor)
            or prefixes.ndim != 3
            or prefixes.dtype not in _INTEGER_DTYPES
        ):
            raise TypeError(
                "byte_prefix_ids must be an integer [batch, tokens, bytes] tensor"
            )
        if prefixes.shape[:2] != conditioning.shape[:2]:
            raise ValueError(
                "byte_prefix_ids must share conditioning batch/token dimensions"
            )
        if not isinstance(lengths, Tensor) or lengths.dtype not in _INTEGER_DTYPES:
            raise TypeError("byte_lengths must be an integer tensor")
        if lengths.shape != conditioning.shape[:2]:
            raise ValueError("byte_lengths must have shape [batch, tokens]")
        if mask is None:
            mask = torch.ones(lengths.shape, dtype=torch.bool, device=device)
        elif not isinstance(mask, Tensor) or mask.dtype != torch.bool:
            raise TypeError("attention_mask must be a bool tensor (True means valid)")
        elif mask.shape != lengths.shape:
            raise ValueError("attention_mask must have shape [batch, tokens]")
        lengths = lengths.to(device=device, dtype=torch.long)
        prefixes = prefixes.to(device=device, dtype=torch.long)
        mask = mask.to(device=device)
        valid_lengths = lengths[mask]
        if bool(
            (
                (valid_lengths < 0)
                | (valid_lengths > min(prefixes.shape[-1], cfg.max_token_bytes))
            ).any()
        ):
            raise ValueError(
                "valid byte_lengths must fit byte_prefix_ids and max_token_bytes"
            )
        positions = torch.arange(prefixes.shape[-1], device=device)
        valid_bytes = prefixes[mask.unsqueeze(-1) & (positions < lengths.unsqueeze(-1))]
        if bool(((valid_bytes < 0) | (valid_bytes > 255)).any()):
            raise ValueError(
                "valid byte_prefix_ids must contain only byte values 0 through 255"
            )
        return prefixes, lengths, mask
