from dataclasses import dataclass, field
from Emperor.base.utils import Module, ConfigBase
from Emperor.adaptive.utils.mixtures.types.utils.enums import ClipParameterOptions
from Emperor.adaptive.utils.mixtures._validator import _AdaptiveMixtureBaseValidator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


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


class AdaptiveMixtureBase(Module):
    def __init__(
        self,
        cfg: "AdaptiveMixtureConfig | ModelConfig",
        overrides: "AdaptiveMixtureConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "mixture_model_config", cfg)
        self.cfg: "AdaptiveMixtureConfig" = self._overwrite_config(config, overrides)
        self.main_cfg = self._resolve_main_config(self.cfg, cfg)

        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.top_k = self.cfg.top_k
        self.num_experts = self.cfg.num_experts
        self.weighted_parameters_flag = self.cfg.weighted_parameters_flag
        self.clip_parameter_option = self.cfg.clip_parameter_option
        self.clip_range = self.cfg.clip_range
        self._validator = _AdaptiveMixtureBaseValidator(self)
