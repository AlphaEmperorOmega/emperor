from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import Tensor

from emperor.config import ConfigBase, optional_field
from emperor.layers import Layer
from emperor.nn import Module
from emperor.transformer._validation import FeedForwardValidator

if TYPE_CHECKING:
    from emperor.experts import MixtureOfExpertsModelConfig
    from emperor.layers import LayerStackConfig, RecurrentLayerConfig


@dataclass
class FeedForwardConfig(ConfigBase):
    input_dim: int | None = optional_field("Feed-forward input feature dimension.")
    output_dim: int | None = optional_field("Feed-forward output feature dimension.")
    stack_config: "LayerStackConfig | MixtureOfExpertsModelConfig | RecurrentLayerConfig | None" = optional_field(  # noqa: E501
        "Either a LayerStackConfig (plain feed-forward stack), a "
        "MixtureOfExpertsModelConfig (mixture-of-experts model that manages its "
        "own routing state), or a RecurrentLayerConfig wrapping either form. "
        "Depth lives on the inner config."
    )

    def _registry_owner(self) -> type:
        return FeedForward


class FeedForward(Module):
    VALIDATOR = FeedForwardValidator

    def __init__(
        self,
        cfg: "FeedForwardConfig",
        overrides: "FeedForwardConfig | None" = None,
    ) -> None:
        super().__init__()
        self.cfg: FeedForwardConfig = self._override_config(cfg, overrides)

        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.stack_config: ConfigBase = self.cfg.stack_config

        self.VALIDATOR.validate(self)

        self.model = self.__build_stack_model()

    def __build_stack_model(self) -> Module:
        overrides = type(self.stack_config)(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return self.stack_config.build(overrides)

    def forward(self, input_batch: Tensor) -> tuple[Tensor, Tensor]:
        original_shape = input_batch.shape
        flattened_input = input_batch.reshape(-1, self.input_dim)
        state = Layer.run_model_returning_state(self.model, flattened_input)
        output = state.hidden.view(*original_shape[:-1], self.output_dim)
        loss = state.loss if state.loss is not None else input_batch.new_zeros(())
        return output, loss
