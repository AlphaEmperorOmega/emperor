from dataclasses import dataclass
from emperor.base.utils import ConfigBase, optional_field
from emperor.parametric.core.mixtures.options import ClipParameterOptions

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.experts.core.config import MixtureOfExpertsConfig


@dataclass
class AdaptiveMixtureConfig(ConfigBase):
    input_dim: int | None = optional_field("Mixture model input dimension.")
    output_dim: int | None = optional_field("Mixture model output dimension.")
    top_k: int | None = optional_field(
        "Top-k probabilities and indices selected from a distribution."
    )
    num_experts: int | None = optional_field("Number of mixture experts.")
    weighted_parameters_flag: bool | None = optional_field(
        "When True, selected parameters are multiplied by their probabilities."
    )
    clip_parameter_option: ClipParameterOptions | None = optional_field(
        "Clipping strategy for generated parameters."
    )
    clip_range: float | None = optional_field(
        "Symmetric clipping range for generated parameters."
    )


@dataclass
class VectorWeightsMixtureConfig(AdaptiveMixtureConfig):
    def _registry_owner(self) -> type:
        from emperor.parametric.core.mixtures.types.vector import VectorWeightsMixture

        return VectorWeightsMixture


@dataclass
class MatrixWeightsMixtureConfig(AdaptiveMixtureConfig):
    def _registry_owner(self) -> type:
        from emperor.parametric.core.mixtures.types.matrix import MatrixWeightsMixture

        return MatrixWeightsMixture


@dataclass
class MatrixBiasMixtureConfig(AdaptiveMixtureConfig):
    def _registry_owner(self) -> type:
        from emperor.parametric.core.mixtures.types.matrix import MatrixBiasMixture

        return MatrixBiasMixture


@dataclass
class GeneratorWeightsMixtureConfig(AdaptiveMixtureConfig):
    generator_config: "MixtureOfExpertsConfig | None" = optional_field(
        "Mixture-of-experts config used to generate weight factors."
    )

    def _registry_owner(self) -> type:
        from emperor.parametric.core.mixtures.types.generator import (
            GeneratorWeightsMixture,
        )

        return GeneratorWeightsMixture


@dataclass
class GeneratorBiasMixtureConfig(AdaptiveMixtureConfig):
    generator_config: "MixtureOfExpertsConfig | None" = optional_field(
        "Mixture-of-experts config used to generate bias parameters."
    )

    def _registry_owner(self) -> type:
        from emperor.parametric.core.mixtures.types.generator import (
            GeneratorBiasMixture,
        )

        return GeneratorBiasMixture
