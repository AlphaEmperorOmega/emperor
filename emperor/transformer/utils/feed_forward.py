import torch

from torch import Tensor
from torch.nn import Sequential
from dataclasses import dataclass, field
from emperor.base.utils import ConfigBase, Module
from emperor.linears.options import LinearLayerStackOptions
from emperor.adaptive.options import AdaptiveLayerStackOptions
from emperor.experts.options import MixtureOfExpertsStackOptions
from emperor.base.layer import Layer, LayerStack, LayerStackConfig
from emperor.transformer.utils._validator import FeedForwardValidator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


@dataclass
class FeedForwardConfig(ConfigBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    layer_stack_option: "LinearLayerStackOptions | AdaptiveLayerStackOptions | MixtureOfExpertsStackOptions | None" = field(
        default=None,
        metadata={"help": "Number of layers added to the router"},
    )
    num_layers: int | None = field(
        default=None,
        metadata={"help": "Number of layers added to the router"},
    )


class FeedForward(Module):
    def __init__(
        self,
        cfg: "FeedForwardConfig | ModelConfig",
        overrides: "FeedForwardConfig | None" = None,
    ) -> None:
        super().__init__()
        self.cfg: FeedForwardConfig = self._overwrite_config(cfg, overrides)
        self.main_cfg = self._resolve_main_config(self.cfg, cfg)

        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.layer_stack_option = self.cfg.layer_stack_option.value
        self.num_layers = self.cfg.num_layers
        self.validator = FeedForwardValidator(self)
        self.model = self._create_model()
        self._store_shape_attributes()

    def _create_model(self) -> Layer | Sequential:
        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_layers=self.num_layers,
        )
        model = self.layer_stack_option(self.main_cfg, overrides)
        if isinstance(model, LayerStack):
            return model.build_model()
        return model

    def _store_shape_attributes(self):
        self.batch_size = None
        self.sequence_length = None
        self.output_shape = None

    def forward(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        input_batch, skip_mask = self._ensure_correct_shape(input_batch, skip_mask)
        projected_inputs = self.model(input_batch)
        if isinstance(projected_inputs, tuple):
            if len(projected_inputs) == 2:
                output, loss = projected_inputs
            else:
                output, skip_mask, loss = projected_inputs
            output = self._revert_to_original_shape(output)
            return output, loss
        return self._revert_to_original_shape(projected_inputs), torch.tensor(0.0)

    def _ensure_correct_shape(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if input_batch.dim() == 2:
            return input_batch, skip_mask
        self.__resolve_output_shape(input_batch)
        input_batch = input_batch.reshape(self.batch_size * self.sequence_length, -1)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        return input_batch, skip_mask

    def __resolve_output_shape(self, input_batch: Tensor) -> None:
        input_shape = input_batch.shape
        if self.batch_size is not None:
            return
        if input_batch.dim() > 2:
            self.batch_size, self.sequence_length, _ = input_shape
            self.output_shape = [self.batch_size, self.sequence_length, -1]
            return
        self.sequence_length = 1
        self.batch_size, _ = input_shape
        self.output_shape = [self.batch_size, -1]

    def _revert_to_original_shape(self, output_projection: Tensor):
        if self.output_shape is None:
            return output_projection
        return output_projection.view(self.output_shape)
