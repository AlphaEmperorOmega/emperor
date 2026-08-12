from dataclasses import dataclass

from models.gpt.expert_linear_adaptive.runtime_options import (
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
    embedding_options: GptEmbeddingOptions
    lm_head_options: GptLmHeadOptions


class BoundaryConfigFactory:
    def __init__(self, dependencies: BoundaryConfigDependencies) -> None:
        self.input_dim = dependencies.input_dim
        self.hidden_dim = dependencies.hidden_dim
        self.output_dim = dependencies.output_dim
        self.sequence_length = dependencies.sequence_length
        self.embedding_options = dependencies.embedding_options
        self.lm_head_options = dependencies.lm_head_options

    def build_boundary_config(self) -> GptBoundaryConfig:
        self._validate()
        return GptBoundaryConfig(
            embedding_options=self.embedding_options,
            lm_head_options=self.lm_head_options,
        )

    def _validate(self) -> None:
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
