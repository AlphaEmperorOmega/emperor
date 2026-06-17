from torch import Tensor
from emperor.base.utils import ConfigBase, Module
from emperor.base.layer import Layer
from emperor.transformer.feed_forward.core._validator import FeedForwardValidator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.transformer.feed_forward.core.config import FeedForwardConfig


class FeedForward(Module):
    def __init__(
        self,
        cfg: "FeedForwardConfig",
        overrides: "FeedForwardConfig | None" = None,
    ) -> None:
        super().__init__()
        self.cfg: "FeedForwardConfig" = self._override_config(cfg, overrides)

        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.stack_config: ConfigBase = self.cfg.stack_config

        FeedForwardValidator.validate(self)

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
