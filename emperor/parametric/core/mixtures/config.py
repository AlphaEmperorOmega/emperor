from dataclasses import dataclass, field
from emperor.base.utils import ConfigBase
from emperor.parametric.core.mixtures.types.utils.enums import ClipParameterOptions


@dataclass
class AdaptiveMixtureConfig(ConfigBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Mixture model input dimension"},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Mixture model output dimension"},
    )
    top_k: int | None = field(
        default=None,
        metadata={
            "help": "Inidicates the top-k probs and indices to be selected from a distribution"
        },
    )
    num_experts: int | None = field(
        default=None,
        metadata={"help": "Router output dimension"},
    )
    weighted_parameters_flag: bool | None = field(
        default=None,
        metadata={
            "help": "When `True` the sepected parameters will be multiplied by their probs"
        },
    )
    clip_parameter_option: ClipParameterOptions | None = field(
        default=None,
        metadata={"help": "Specifies the clipping strategy for the mixture parameters"},
    )
    clip_range: float | None = field(
        default=None,
        metadata={
            "help": "Specifies the clipping range for the generated mixture parameters. The range will be between +- `clip_range`"
        },
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
    def _registry_owner(self) -> type:
        from emperor.parametric.core.mixtures.types.generator import (
            GeneratorWeightsMixture,
        )

        return GeneratorWeightsMixture


@dataclass
class GeneratorBiasMixtureConfig(AdaptiveMixtureConfig):
    def _registry_owner(self) -> type:
        from emperor.parametric.core.mixtures.types.generator import (
            GeneratorBiasMixture,
        )

        return GeneratorBiasMixture
