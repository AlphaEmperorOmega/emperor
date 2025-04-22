import torch
from torch import Tensor

from ..base.utils import Module
from ..base.decorators import timer
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from Emperor.config import Configuration, ParameterGeneratorOptions


class ParameterGeneratorLayer(Module):
    def __init__(
        self,
        cfg: Optional["Configuration"] = None,
        track_time_flag: Optional[bool] = None,
        parameter_geneartor_type: Optional[ParameterGeneratorOptions] = None,
    ):
        super().__init__()
        self.cfg: "ParameterGeneratorConfig" = self._get_config(
            cfg, "parameter_geneartor_config"
        )
        self.track_time_flag = self._resolve(track_time_flag, self.cfg.track_time_flag)
        self.parameter_geneartor_type = self._resolve(
            parameter_geneartor_type, self.cfg.parameter_geneartor_type
        )
        self._init_parameter_generator(cfg)

    def _init_parameter_generator(self, cfg: Optional["Configuration"]) -> None:
        assert self.parameter_geneartor_type is not None, (
            f"Expected `parameter_generator` to be `ParameterGeneratorOptions`, received {type(self.parameter_geneartor_type)}"
        )
        self.parameter_generator = self.parameter_geneartor_type.create(cfg)

    def forward(self, input_batch: Tensor) -> Tensor:
        # TODO: find out the shape of `input_batch` tensor
        if self.track_time_flag:
            return self.track_forward_time(input_batch)
        return self.compute_layer_output(input_batch)

    @timer
    def track_forward_time(self, input_batch: Tensor) -> Tensor:
        return self.compute_layer_output(input_batch)

    def compute_layer_output(self, input_batch: Tensor) -> Tensor:
        generated_weights, generated_biases = self.parameter_generator(input_batch)

        input_batch = input_batch.unsqueeze(1)
        output = torch.matmul(input_batch, generated_weights)
        output = output.squeeze(1)

        if generated_biases is not None:
            output = output + generated_biases

        return output
