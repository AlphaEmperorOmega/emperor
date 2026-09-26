from dataclasses import dataclass

import models.gpt.linear_adaptive.config as config
from models.gpt.linear_adaptive import _config_defaults as config_defaults
from models.gpt.linear_adaptive.runtime_options import (
    GptEmbeddingOptions,
    GptLmHeadOptions,
)


@dataclass(frozen=True)
class GptBoundaryConfig:
    embedding_options: GptEmbeddingOptions
    lm_head_options: GptLmHeadOptions


@dataclass(frozen=True)
class BoundaryConfigDependencies:
    input_dim: int
    hidden_dim: int
    output_dim: int
    sequence_length: int
    embedding_options: GptEmbeddingOptions | None
    lm_head_options: GptLmHeadOptions | None


class BoundaryConfigFactory:
    def __init__(self, dependencies: BoundaryConfigDependencies) -> None:
        self.input_dim = dependencies.input_dim
        self.hidden_dim = dependencies.hidden_dim
        self.output_dim = dependencies.output_dim
        self.sequence_length = dependencies.sequence_length
        self.embedding_options = (
            dependencies.embedding_options
            or config_defaults.gpt_embedding_options(config)
        )
        self.lm_head_options = (
            dependencies.lm_head_options or config_defaults.gpt_lm_head_options(config)
        )

    def build_boundary_config(self) -> GptBoundaryConfig:
        self._validate()
        return GptBoundaryConfig(
            embedding_options=self.embedding_options,
            lm_head_options=self.lm_head_options,
        )

    def _validate(self) -> None:
        options = self.embedding_options
        if type(options.hierarchical_language_model_flag) is not bool:
            raise TypeError("hierarchical_language_model_flag must be bool")
        if options.hierarchical_language_model_flag:
            if self.lm_head_options.weight_tying_flag:
                raise ValueError(
                    "Hierarchical language modeling requires lm_head_weight_tying_flag=False"
                )
            for name in (
                "byte_embedding_dim",
                "byte_encoder_num_layers",
                "byte_decoder_num_layers",
                "byte_num_heads",
                "byte_feed_forward_dim",
                "byte_limit",
            ):
                if (
                    type(getattr(options, name)) is not int
                    or getattr(options, name) <= 0
                ):
                    raise ValueError(f"hierarchical {name} must be a positive integer")
            if options.byte_embedding_dim % options.byte_num_heads:
                raise ValueError(
                    "hierarchical byte width must be divisible by byte heads"
                )
        for name, value in {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than 0, received {value}.")
        probability = self.embedding_options.dropout_probability
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "embedding dropout_probability must be in [0.0, 1.0], "
                f"received {probability}."
            )
        if self.lm_head_options.weight_tying_flag and self.input_dim != self.output_dim:
            raise ValueError(
                "GPT LM head weight tying requires input_dim to equal output_dim, "
                f"received input_dim={self.input_dim} and output_dim={self.output_dim}."
            )
