from torch import Tensor
from Emperor.base.utils import Module

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class ParameterGenerator(Module):
    def __init__(
        self,
        cfg: "ModelConfig",
        batch_size: Optional[int] = None,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        depth_dim: Optional[int] = None,
        num_router_layers: Optional[int] = None,
        bias_flag: Optional[bool] = None,
        gather_frequency_flag: Optional[bool] = None,
        noisy_topk_flag: Optional[bool] = None,
        top_k: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = self._resolve(batch_size, self.cfg.batch_size)
        self.input_dim = self._resolve(input_dim, self.cfg.input_dim)
        self.output_dim = self._resolve(output_dim, self.cfg.output_dim)
        self.depth_dim = self._resolve(depth_dim, self.cfg.depth_dim)
        self.num_router_layers = self._resolve(
            num_router_layers, self.cfg.num_router_layers
        )
        self.bias_flag = self._resolve(bias_flag, self.cfg.bias_flag)
        self.gather_frequency_flag = self._resolve(
            gather_frequency_flag, self.cfg.gather_frequency_flag
        )
        self.noisy_topk_flag = self._resolve(noisy_topk_flag, self.cfg.noisy_topk_flag)
        self.top_k = self._resolve(top_k, self.cfg.top_k)

        self.num_experts = (
            2 * self.depth_dim if self.noisy_topk_flag else self.depth_dim
        )

    def _init_frequency_model(self):
        # TODO: later implement the frequency gathering methanism
        # frequencyClass = cfg.weightAndBiasGeneratorType.value + "Frequency"
        # self.gatherFrequencyCheck = (
        #     self.gatherFrequencyFlag and frequencyClass in globals()
        # )
        # if self.gatherFrequencyCheck:
        #     self.frequency = globals()[frequencyClass](cfg)
        pass

    def gather_frequency(self, sparse_indexes) -> None:
        if self.gather_frequency_flag:
            self.frequency.update(sparse_indexes)

    def _handle_probabilities_shape_hook(self, probabilities: Tensor) -> Tensor:
        return probabilities

    def _select_parameters(self, weight_indexes, bias_indexes) -> Tuple:
        pass
