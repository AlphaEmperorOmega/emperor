from dataclasses import replace
from Emperor.library.libs.torch import Library as L
from .utils.base import ParameterGenerator
from .utils.routers import VectorChoiceRouterModel
from .utils.mixture import (
    SparseMixtureBehaviour,
    TopkMixtureBehaviour,
    FullMixtureBehaviour,
)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ParameterGeneratorConfig


class VectorChoiceBase(ParameterGenerator):
    def __init__(self, cfg: "ParameterGeneratorConfig"):
        super().__init__(cfg)

        self.cfg = replace(cfg, router=VectorChoiceRouterModel(cfg))
        self.biasFlag = cfg.biasFlag
        self.weightBank, self.biasBank = self._initializeParameterBanks()
        self.choiceRangeWeights, self.choiceRangeBiases = (
            self._initializeParameterChoiceRanges()
        )

    def _initializeParameterBanks(self):
        weightBank = L.parameterWeights(self.inputDim, self.depthDim, self.outputDim)
        L.initializeWeights(weightBank)

        biasBank = None
        if self.biasFlag:
            biasBank = L.parameterWeights(self.outputDim, self.depthDim)
            L.initializeWeights(biasBank)

        return weightBank, biasBank

    def _initializeParameterChoiceRanges(self):
        inputRange = L.arange(self.inputDim)
        outputRange = L.arange(self.outputDim)
        choiceRangeWeights = L.reshape(inputRange, [1, self.inputDim]).to(L.Device)
        choiceRangeBiases = L.reshape(outputRange, [1, self.outputDim]).to(L.Device)
        return choiceRangeWeights, choiceRangeBiases

    def selectParameters(self, weightIndexes, biasIndexes):  # [inputDim, batchSize]
        selectedWeights = self._selectParameterVectors(
            weightIndexes, self.weightBank, self.choiceRangeWeights
        )
        selectedBiases = None
        if self.biasFlag:
            selectedBiases = self._selectParameterVectors(
                biasIndexes, self.biasBank, self.choiceRangeBiases
            )

        return selectedWeights, selectedBiases

    def _selectParameterVectors(self, weightIndexes, weightBank, choiceRange):
        transposedIndexes = L.transpose(weightIndexes, [1, 0])
        return weightBank[choiceRange, transposedIndexes]

    def calculateParameterMixture(
        self,
        selectedWeightParameters,
        weightProbabilities,
        selectedBiasParameters,
        biasProbabilities,
    ):
        weightMixture = self._calculateMixture(
            selectedWeightParameters, weightProbabilities, -2, True
        )
        biasMixture = None
        if self.biasFlag:
            biasMixture = self._calculateMixture(
                selectedBiasParameters, biasProbabilities, -1
            )
        return weightMixture, biasMixture

    def _calculateMixture(
        self, selectedParameters, probabilities, sumDimension: int, isWeight=False
    ):
        probabilities = L.transpose(probabilities, [1, 0])
        if isWeight:
            probabilities = L.unsqueeze(probabilities, dim=-1)
        weightedWeights = selectedParameters * probabilities
        return L.sum(weightedWeights, dim=sumDimension)


class VectorChoiceSparse(VectorChoiceBase):
    def __init__(self, cfg: "ParameterGeneratorConfig"):
        super().__init__(cfg)

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
    def __init__(self, cfg: "ParameterGeneratorConfig"):
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
    def __init__(self, cfg: "ParameterGeneratorConfig"):
        super().__init__(cfg)

        self.mixtureBehaviour = FullMixtureBehaviour(self.cfg, self)

    def forward(self, inputBatch):
        return self.mixtureBehaviour.calculateMixture(inputBatch)
