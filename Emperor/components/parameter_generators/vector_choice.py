import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter, Linear
from torch.optim.optimizer import Kwargs

from .utils.base import ParameterGenerator
from .utils.routers import VectorChoiceRouterModel
from .utils.mixture import (
    SparseMixtureBehaviour,
    TopkMixtureBehaviour,
    FullMixtureBehaviour,
)
from Emperor.base.utils import randn, arange, reshape, device

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class VectorChoiceBase(ParameterGenerator):
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
    ) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg
        self.weight_bank, self.bias_bank = self._initialize_parameter_banks()
        self.choice_range_weights, self.choice_range_biases = (
            self._init_parameter_choice_ranges()
        )

    def _init_router_model(self) -> VectorChoiceRouterModel:
        return VectorChoiceRouterModel(self.cfg)

    def _init_parameter_banks(self) -> Tuple[Parameter, Optional[Parameter]]:
        weight_bank = Parameter(randn(self.input_dim, self.depth_dim, self.output_dim))
        self._initialize_parameters(weight_bank)

        bias_bank = None
        if self.bias_flag:
            bias_bank = Parameter(randn(self.output_dim, self.depth_dim))
            self._initialize_parameters(bias_bank)

        return weight_bank, bias_bank

    def _init_parameter_choice_ranges(self) -> Tuple[Tensor, Tensor]:
        input_range = arange(self.input_dim)
        output_range = arange(self.output_dim)
        choice_range_weights = reshape(input_range, [1, self.input_dim]).to(device)
        choice_range_biases = reshape(output_range, [1, self.output_dim]).to(device)
        return choice_range_weights, choice_range_biases

    def _select_parameters(self, weight_indexes: Tensor, bias_indexes: Tensor) -> Tuple:
        selected_weights = self._select_parameter_vectors(
            weight_indexes, self.weight_bank, self.choice_range_weights
        )
        selected_biases = None
        if self.biasFlag:
            selected_biases = self._select_parameter_vectors(
                bias_indexes, self.biasBank, self.choice_range_biases
            )

        return selected_weights, selected_biases

    def _select_parameter_vectors(
        self, weight_indexes: Tensor, weight_bank: Parameter, choice_range: Tensor
    ) -> Tensor:
        transposed_indexes = weight_indexes.transpose(1, 0)
        return weight_bank[choice_range, transposed_indexes]

    def calculate_parameter_mixture(
        self,
        selected_weight_parameters: Tensor,
        weight_probs: Tensor,
        selected_bias_parameters: Optional[Tensor],
        bias_probs: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        weight_mixture = self._calculate_mixture(
            selected_weight_parameters, weight_probs, -2, True
        )
        bias_mixture = None
        if self.bias_flag:
            assert selected_bias_parameters is not None and bias_probs is not None
            bias_mixture = self._calculate_mixture(
                selected_bias_parameters, bias_probs, -1
            )
        return weight_mixture, bias_mixture

    def _calculate_mixture(
        self, selected_parameters: Tensor, probs: Tensor, dim: int, is_weight=False
    ) -> Tensor:
        probs = probs.transpose(1, 0)
        if is_weight:
            probs = probs.unsqueeze(-1)
        weighted_weights = selected_parameters * probs
        return weighted_weights.sum(dim=dim)


class VectorChoiceSparse(VectorChoiceBase):
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
    ):
        super().__init__(**kwargs)

        self.mixtureBehaviour = SparseMixtureBehaviour(self.cfg, self)

    def forward(self, inputBatch):
        return self.mixtureBehaviour.calculateMixture(inputBatch)

    def handleProbabilitiesShapeHook(self, probabilities):
        meanProbabilities = L.mean(probabilities)  # [batchSize]
        reshapedMeanProbabilities = L.unsqueeze(
            meanProbabilities, dim=-1
        )  # [batchSize, 1]
        return reshapedMeanProbabilities


class VectorChoiceMixture(VectorChoiceBase):
    def __init__(self, cfg: "ModelConfig"):
        super().__init__(cfg)

        self.topK = min(cfg.topK, self.depthDim)
        assert cfg.topK < self.depthDim, "topK needs to be smaller than the depthDim"

        self.mixtureBehaviour = TopkMixtureBehaviour(self.cfg, self)

    def _initializeParameterChoiceRanges(self):
        inputRange = L.arange(self.inputDim)
        outputRange = L.arange(self.outputDim)
        choiceRangeWeights = L.reshape(inputRange, [1, self.inputDim, 1])
        choiceRangeBiases = L.reshape(outputRange, [1, self.outputDim, 1])
        return choiceRangeWeights, choiceRangeBiases

    def forward(self, inputBatch):
        return self.mixtureBehaviour.calculateMixture(inputBatch)


class VectorChoiceSoftMixture(VectorChoiceBase):
    def __init__(self, cfg: "ModelConfig"):
        super().__init__(cfg)

        self.mixtureBehaviour = FullMixtureBehaviour(self.cfg, self)

    def forward(self, inputBatch):
        return self.mixtureBehaviour.calculateMixture(inputBatch)
