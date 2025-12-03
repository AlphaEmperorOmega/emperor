from Emperor.base.utils import Module, ConfigBase
from dataclasses import dataclass, field

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


@dataclass
class MixtureConfig(ConfigBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Mixture model input dimension"},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Mixture model output dimension"},
    )
    depth_dim: int | None = field(
        default=None,
        metadata={"help": "Mixture model depth dimension"},
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


class MixtureBase(Module):
    def __init__(
        self,
        cfg: "MixtureConfig | ModelConfig",
        overrides: "MixtureConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "mixture_model_config", cfg)
        self.cfg: "MixtureConfig" = self._overwrite_config(config, overrides)

        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.depth_dim = self.cfg.depth_dim
        self.top_k = self.cfg.top_k
        self.num_experts = self.cfg.num_experts
        self.weighted_parameters_flag = self.cfg.weighted_parameters_flag
        self.__validate_inputs()

    def __validate_inputs(self) -> None:
        assert self.depth_dim == self.num_experts, (
            "The `depth_dim` needs to be equal with `num_experts` since this is the dimension the router creates a distribution over."
        )
        if self.depth_dim == self.top_k:
            assert self.weighted_parameters_flag is True, (
                "If `full_mixture` is performed the `weighted_parameters_flag` must be True. Because the `weight_bank` or `bias_bank` needs to be broadcasted across the batch."
            )
