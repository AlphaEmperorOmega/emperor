import torch
import torch.nn as nn
from torch import Tensor, Size
from dataclasses import replace
from Emperor.base.utils import Module
from Emperor.library.choice import Library as L
from .experts import ParallelExperts
from .parameter_generators.utils.probabilitySamplers import ProbabilitySamplerTopk


from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class MixtureOfExperts(Module):
    def __init__(
        self,
        cfg: "ModelConfig",
        inputDim: Optional[int] = None,
        hiddenDim: Optional[int] = None,
        outputDim: Optional[int] = None,
        multiplyByGatesFlag: Optional[bool] = None,
        activationFunction: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.inputDim: int = self._getValue(inputDim, cfg.inputDim)
        self.hiddenDim: int = self._getValue(hiddenDim, cfg.hiddenDim)
        self.outputDim: int = self._getValue(outputDim, cfg.outputDim)
        self.multiplyByGatesFlag: int = self._getValue(
            multiplyByGatesFlag, cfg.multiplyByGatesFlag
        )
        self.attentionOutputDim: int = self._getValue(
            activationFunction, cfg.attentionOutputDim
        )

        self.inputBatchShape: Optional[Size] = None
        # TODO: figure out later if you need to dynamically return indices
        # returnIndices = False

        self._prepareExpertLayers()
        self._prepareExpertSampler()

    def _prepareExpertLayers(self):
        inputExpertsConfig = replace(
            self.cfg, inputDim=self.cfg.inputDim, outputDim=self.cfg.hiddenDim
        )
        self.inputExperts = ParallelExperts(inputExpertsConfig)
        outputExpertsConfig = replace(
            self.cfg, inputDim=self.cfg.hiddenDim, outputDim=self.cfg.outputDim
        )
        self.outputExperts = ParallelExperts(outputExpertsConfig)

    def _prepareExpertSampler(self):
        cfg = replace(
            self.cfg,
            depthDim=self.cfg.numExperts,
            auxiliaryLosses=self.cfg.moeAuxiliaryLosses,
        )
        self.sampler = ProbabilitySamplerTopk(cfg)

    def forward(self, inputBatch: Tensor, skipMask: Optional[Tensor] = None):
        self.inputBatchShape = inputBatch.size()

        (
            expertSortedBatchInputs,
            expertSortedTopKBatchIndexes,
            expertSortedTopKBatchProbabilities,
            batchExpertFrequency,
            expertSortedNonzeroProbabilityIndexes,
        ) = self._prepareInputBatchForExpertProcessing(inputBatch, skipMask)

        feedForwardOutput = self._computeProjections(
            expertSortedBatchInputs, batchExpertFrequency
        )

        expertMixtureOutput = self._computeExpertMixture(
            feedForwardOutput,
            expertSortedTopKBatchIndexes,
            expertSortedTopKBatchProbabilities,
        )

        return expertMixtureOutput

    def _prepareInputBatchForExpertProcessing(self, inputBatch, skipMask=None):
        if skipMask is not None:
            skipMask = skipMask.view(-1, 1)

        inputBatchMatrix = inputBatch.reshape(-1, self.inputDim)

        topKProbabilities, topKIndexes = self.sampler.sampleProbabilitiesAndIndexes(
            inputBatchMatrix, skipMask=skipMask, isTrainingFlag=self.training
        )
        (
            expertSortedTopKBatchIndexes,
            expertSortedTopKBatchProbabilities,
            topKProbabilitiesScattered,
            batchExpertFrequency,
            expertSortedNonzeroProbabilityIndexes,
        ) = self._computeGating(topKProbabilities, topKIndexes)

        expertSortedBatchInputs = inputBatchMatrix[expertSortedTopKBatchIndexes]

        return (
            expertSortedBatchInputs,
            expertSortedTopKBatchIndexes,
            expertSortedTopKBatchProbabilities,
            batchExpertFrequency,
            expertSortedNonzeroProbabilityIndexes,
        )

    def _computeGating(self, topKProbabilities, topKIndexes):
        topKProbabilitiesFlattened = topKProbabilities.flatten()
        topKIndexesFlattened = topKIndexes.flatten()
        nonzeroProbabilityIndexes = topKProbabilitiesFlattened.nonzero()
        nonzeroProbabilityIndexes = nonzeroProbabilityIndexes.squeeze(-1)
        nonzeroTopKIndexes = topKIndexesFlattened[nonzeroProbabilityIndexes]
        _, _expertSortedNonzeroTopKIndexes = nonzeroTopKIndexes.sort(0)

        expertSortedNonzeroProbabilityIndexes = nonzeroProbabilityIndexes[
            _expertSortedNonzeroTopKIndexes
        ]

        expertSortedTopKBatchIndexes = expertSortedNonzeroProbabilityIndexes.div(
            self.cfg.topK, rounding_mode="trunc"
        )
        expertSortedTopKBatchProbabilities = topKProbabilitiesFlattened[
            expertSortedNonzeroProbabilityIndexes
        ]

        topKProbabilitiesScatteredPlaceholder = L.zeros(
            [topKProbabilities.shape[0], self.cfg.numExperts], device=L.Device
        )
        topKProbabilitiesScattered = topKProbabilitiesScatteredPlaceholder.scatter(
            1, topKIndexes, topKProbabilities
        )

        batchExpertFrequency = topKProbabilitiesScattered > 0
        batchExpertFrequency = batchExpertFrequency.long()
        batchExpertFrequency = batchExpertFrequency.sum(dim=0)

        return (
            expertSortedTopKBatchIndexes,
            expertSortedTopKBatchProbabilities,
            topKProbabilitiesScattered,
            batchExpertFrequency,
            expertSortedNonzeroProbabilityIndexes,
        )

        expandedProjection = self.inputExperts(
            expertSortedBatchInputs, batchExpertFrequency
        )
        appliedActivation = self.activation(expandedProjection)
        reductionProjection = self.outputExperts(
            appliedActivation, batchExpertFrequency
        )

        return reductionProjection

    def _computeExpertMixture(
        self,
        feedForwardOutput,
        expertSortedTopKBatchIndexes,
        expertSortedTopKBatchProbabilities,
    ):
        if len(self.inputBatchShape) > 2:
            batchSize, sequenceLength, _ = self.inputBatchShape
            outputShape = [batchSize, sequenceLength, self.outputDim]
        else:
            sequenceLength = 1
            batchSize, _ = self.inputBatchShape
            outputShape = [batchSize, self.outputDim]

        if self.multiplyByGates:
            feedForwardOutput = (
                feedForwardOutput * expertSortedTopKBatchProbabilities.view(-1, 1)
            )

        inputBatchZeros = L.zeros(
            (batchSize * sequenceLength, self.outputDim),
            dtype=feedForwardOutput.dtype,
            device=L.Device,
        )

        expertOutputMixture = inputBatchZeros.index_add(
            0, expertSortedTopKBatchIndexes, feedForwardOutput
        )

        return expertOutputMixture.view(outputShape)


class MixtureOfAttentionHeads(MixtureOfExperts):
    def __init__(self, cfg: "ModelConfig") -> None:
        super().__init__(cfg)

    def computeHiddenProjection(self, inputBatch, skipMask=None):
        self.inputBatchShape = inputBatch.size()
        batchSize, sequenceLength, _ = self.inputBatchShape

        (
            expertSortedBatchInputs,
            expertSortedTopKBatchIndexes,
            expertSortedTopKBatchProbabilities,
            batchExpertFrequency,
            expertSortedNonzeroProbabilityIndexes,
        ) = self._prepareInputBatchForExpertProcessing(inputBatch, skipMask)

        expertsOutput = self.inputExperts(expertSortedBatchInputs, batchExpertFrequency)

        inputBatchZeros = L.zeros(
            (batchSize * sequenceLength * self.cfg.topK, self.hiddenDim),
            dtype=expertsOutput.dtype,
            device=L.Device,
        )
        expertOutputMixture = inputBatchZeros.index_add(
            0, expertSortedNonzeroProbabilityIndexes, expertsOutput
        )

        # topKIndexes = topKIndexes.view(batchSize, sequenceLength, -1)
        expandProjectionOutput = expertOutputMixture.view(
            batchSize, sequenceLength, self.cfg.topK, self.hiddenDim
        )

        self.expertSortedTopKBatchIndexes = expertSortedTopKBatchIndexes
        self.expertSortedTopKBatchProbabilities = expertSortedTopKBatchProbabilities
        self.batchExpertFrequency = batchExpertFrequency
        self.expertSortedNonzeroProbabilityIndexes = (
            expertSortedNonzeroProbabilityIndexes
        )

        return expandProjectionOutput

    def conputeOutputProjection(self, attentionOutput):
        """
        The self variables are computed in MixtureOfExperts.expandProjection
        (or the method above in case you don't see)
        """
        batchSize, sequenceLength, _, _ = attentionOutput.shape

        attentionOutput = attentionOutput.reshape(-1, self.hiddenDim)

        expertInputs = attentionOutput[self.expertSortedNonzeroProbabilityIndexes]

        expertsOutput = self.outputExperts(expertInputs, self.batchExpertFrequency)

        if self.multiplyByGates:
            expertsOutput = (
                expertsOutput * self.expertSortedTopKBatchProbabilities.view(-1, 1)
            )

        inputBatchZeros = L.zeros(
            (batchSize * sequenceLength, self.outputDim),
            dtype=expertsOutput.dtype,
            device=L.Device,
        )
        expertOutputMixture = inputBatchZeros.index_add(
            0, self.expertSortedTopKBatchIndexes, expertsOutput
        )
        return expertOutputMixture.view(batchSize, sequenceLength, self.outputDim)
