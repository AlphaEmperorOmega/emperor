from .layer import ParameterGeneratorLayer
from Emperor.library.choice import Library as L
from Emperor.base.utils import Module

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class ParallelExperts(Module):
    def __init__(self, cfg: "ModelConfig"):
        super().__init__()

        self.cfg = cfg
        self.numExperts = cfg.numExperts
        self.experts = self._createExperts()

    def _createExperts(self):
        experts = []
        for _ in range(self.numExperts):
            layer = ParameterGeneratorLayer(self.cfg)
            experts.append(layer)

        return L.ModuleList(experts)

    def forward(self, expertOrderedInput, expertFrequency):
        expertInputs = self._splitInputExperts(expertOrderedInput, expertFrequency)
        expertOutputs = []
        for expertIndex in range(self.numExperts):
            input = expertInputs[expertIndex]
            output = self.experts[expertIndex](input)
            expertOutputs.append(output)

        return L.cat(expertOutputs, dim=0)

    def _splitInputExperts(self, expertOrderedInput, expertFrequency):
        expertFrequencyList = expertFrequency.tolist()
        return expertOrderedInput.split(expertFrequencyList)


class ExpertSelector:
    def computeGating(self, topKProbabilities, topKIndexes):
        expert_sroted_probabilities, expert_sorted_indices = (
            self.__sort_probabilitied_and_indices_by_expert_order(
                topKProbabilities, topKIndexes
            )
        )
        scattered_probabilities = self.__scattered_probabiliites(
            topKProbabilities, topKIndexes
        )
        expert_frequency = self.__batch_frequency(topKProbabilities, topKIndexes)
        expert_sorted_nonzero_probability_indices = (
            self.__expert_sorted_nonzero_probability_indexes(
                topKProbabilities, topKIndexes
            )
        )

        return (
            expert_sorted_indices,
            expert_sroted_probabilities,
            scattered_probabilities,
            expert_frequency,
            expert_sorted_nonzero_probability_indices,
        )

    def __sort_probabilitied_and_indices_by_expert_order(
        self, topKProbabilities, topKIndexes
    ):
        (
            _,
            topKProbabilitiesFlattened,
            nonzeroProbabilityIndexes,
            _expertSortedNonzeroTopKIndexes,
        ) = self.__required_inputs(topKProbabilities, topKIndexes)

        expertSortedNonzeroProbabilityIndexes = nonzeroProbabilityIndexes[
            _expertSortedNonzeroTopKIndexes
        ]

        expertSortedTopKBatchIndexes = expertSortedNonzeroProbabilityIndexes.div(
            self.cfg.topK, rounding_mode="trunc"
        )

        expertSortedTopKBatchProbabilities = topKProbabilitiesFlattened[
            expertSortedNonzeroProbabilityIndexes
        ]
        return expertSortedTopKBatchProbabilities, expertSortedTopKBatchIndexes

    def __scattered_probabiliites(self, topKProbabilities, topKIndexes):
        topKProbabilitiesScatteredPlaceholder = torch.zeros(
            [topKProbabilities.shape[0], self.cfg.numExperts], device=L.Device
        )
        topKProbabilitiesScattered = topKProbabilitiesScatteredPlaceholder.scatter(
            1, topKIndexes, topKProbabilities
        )
        return topKProbabilitiesScattered

    def __batch_frequency(self, topKProbabilities, topKIndexes):
        batchExpertFrequency = (
            self.__scattered_probabiliites(topKProbabilities, topKIndexes) > 0
        )
        batchExpertFrequency = batchExpertFrequency.long()
        batchExpertFrequency = batchExpertFrequency.sum(dim=0)

        return batchExpertFrequency

    def __expert_sorted_nonzero_probability_indexes(
        self, topKProbabilities, topKIndexes
    ):
        _, _, nonzeroProbabilityIndexes, _expertSortedNonzeroTopKIndexes = (
            self.__required_inputs(topKProbabilities, topKIndexes)
        )

        expertSortedNonzeroProbabilityIndexes = nonzeroProbabilityIndexes[
            _expertSortedNonzeroTopKIndexes
        ]
        return expertSortedNonzeroProbabilityIndexes

    def __required_inputs(self, topKProbabilities, topKIndexes):
        topKIndexesFlattened = topKIndexes.flatten()
        topKProbabilitiesFlattened = topKProbabilities.flatten()

        nonzeroProbabilityIndexes = topKProbabilitiesFlattened.nonzero()
        nonzeroProbabilityIndexes = nonzeroProbabilityIndexes.squeeze(-1)

        nonzeroTopKIndexes = topKIndexesFlattened[nonzeroProbabilityIndexes]
        _, _expertSortedNonzeroTopKIndexes = nonzeroTopKIndexes.sort(0)

        return (
            topKIndexesFlattened,
            topKProbabilitiesFlattened,
            nonzeroProbabilityIndexes,
            _expertSortedNonzeroTopKIndexes,
        )
