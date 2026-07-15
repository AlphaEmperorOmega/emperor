from typing import TYPE_CHECKING

from torch import Tensor

from emperor.base.validator import ValidatorBase
from emperor.parametric.core.mixtures.config import (
    AdaptiveMixtureConfig,
    GeneratorBiasMixtureConfig,
    GeneratorWeightsMixtureConfig,
    VectorWeightsMixtureConfig,
)

if TYPE_CHECKING:
    from emperor.parametric.core.mixtures.base import AdaptiveMixtureBase


class AdaptiveMixtureValidator(ValidatorBase):
    @classmethod
    def validate(cls, model: "AdaptiveMixtureBase") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.validate_dimensions(
            input_dim=model.input_dim,
            output_dim=model.output_dim,
            top_k=model.top_k,
            num_experts=model.num_experts,
        )
        cls._validate_positive_integer("input_dim", model.input_dim)
        cls._validate_positive_integer(
            "output_dim", model.output_dim
        )
        cls._validate_positive_integer("top_k", model.top_k)
        cls._validate_positive_integer(
            "num_experts", model.num_experts
        )
        cls._validate_top_k(model.cfg)
        cls._validate_clip_range(model.cfg)
        cls._validate_vector_dimensions(model.cfg)
        cls._validate_generator_config(model.cfg)

    @staticmethod
    def validate_input_batch_2d(input_batch: Tensor) -> None:
        if not isinstance(input_batch, Tensor):
            raise TypeError(
                "input_batch must be a Tensor, "
                f"received {type(input_batch).__name__}."
            )
        if input_batch.dim() != 2:
            raise ValueError(
                f"Input batch must be a 2D tensor, got {input_batch.ndim}D "
                f"with shape {tuple(input_batch.shape)}."
            )

    @staticmethod
    def validate_weighted_probabilities(
        cfg: AdaptiveMixtureConfig,
        probabilities: Tensor | None,
    ) -> None:
        if cfg.weighted_parameters_flag and probabilities is None:
            raise ValueError(
                "Probabilities must be provided when weighted_parameters_flag is True."
            )

    @staticmethod
    def _validate_positive_integer(name: str, value: int) -> None:
        if isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, received {value!r}.")

    @staticmethod
    def _validate_top_k(cfg: AdaptiveMixtureConfig) -> None:
        if cfg.top_k > cfg.num_experts:
            raise ValueError(
                "top_k cannot exceed num_experts for AdaptiveMixtureConfig, "
                f"received top_k={cfg.top_k}, num_experts={cfg.num_experts}."
            )

    @staticmethod
    def _validate_clip_range(cfg: AdaptiveMixtureConfig) -> None:
        if cfg.clip_range < 0.0:
            raise ValueError(
                f"clip_range must be non-negative, received {cfg.clip_range}."
            )

    @staticmethod
    def _validate_vector_dimensions(cfg: AdaptiveMixtureConfig) -> None:
        if not isinstance(cfg, VectorWeightsMixtureConfig):
            return
        if cfg.input_dim != cfg.output_dim:
            raise ValueError(
                "input_dim and output_dim must match for VectorWeightsMixtureConfig, "
                f"received input_dim={cfg.input_dim}, output_dim={cfg.output_dim}."
            )

    @staticmethod
    def _validate_generator_config(cfg: AdaptiveMixtureConfig) -> None:
        if not isinstance(
            cfg,
            (GeneratorWeightsMixtureConfig, GeneratorBiasMixtureConfig),
        ):
            return

        from emperor.experts.core.config import MixtureOfExpertsConfig

        if not isinstance(cfg.generator_config, MixtureOfExpertsConfig):
            raise TypeError(
                "generator_config must be a MixtureOfExpertsConfig for generator "
                f"mixtures, got {type(cfg.generator_config).__name__}."
            )

        generator_config = cfg.generator_config
        if generator_config.top_k != cfg.top_k:
            raise ValueError(
                "generator_config.top_k must match the mixture top_k, "
                f"received {generator_config.top_k} and {cfg.top_k}."
            )
        if generator_config.num_experts != cfg.num_experts:
            raise ValueError(
                "generator_config.num_experts must match the mixture num_experts, "
                f"received {generator_config.num_experts} and {cfg.num_experts}."
            )

        sampler_config = generator_config.sampler_config
        if sampler_config is None:
            return
        if sampler_config.top_k != cfg.top_k:
            raise ValueError(
                "generator_config.sampler_config.top_k must match the mixture top_k, "
                f"received {sampler_config.top_k} and {cfg.top_k}."
            )
        if sampler_config.num_experts != cfg.num_experts:
            raise ValueError(
                "generator_config.sampler_config.num_experts must match the mixture "
                f"num_experts, received {sampler_config.num_experts} and "
                f"{cfg.num_experts}."
            )
