from typing import TYPE_CHECKING
from Emperor.library.libs.torch import Library as L

from .utils.base import ParameterGenerator
from .utils.mixture import (
    SparseMixtureBehaviour,
    TopkMixtureBehaviour,
    FullMixtureBehaviour,
)

if TYPE_CHECKING:
    from Emperor.config import ParameterGeneratorConfig


class MatrixChoiceBase(ParameterGenerator):
    def __init__(self, cfg: "ParameterGeneratorConfig"):
        super().__init__(cfg)

        self.weightBank, self.biasBank = self._initializeParameterBanks()
        self.weightMixturePorbabilityShape, self.biasMixturePorbabilityShape = (
            self._storeMixtureProbabilityShapes()
        )

    def _initializeParameterBanks(self):
        weightBank = L.parameterWeights(self.depthDim, self.inputDim, self.outputDim)
        L.initializeWeights(weightBank)

        biasBank = None
        if self.biasFlag:
            biasBank = L.parameterWeights(self.depthDim, self.outputDim)
            L.initializeWeights(biasBank)

        return weightBank, biasBank

    def _storeMixtureProbabilityShapes(self):
        weightMixtureShape = [self.batchSize, self.depthDim, 1, 1]
        biasMixtureShape = [self.batchSize, self.depthDim, 1]
        return weightMixtureShape, biasMixtureShape

    def selectParameters(self, weightIndexes, biasIndexes):  # [inputDim, batchSize]
        selectedWeights = self.weightBank[weightIndexes]

        selectedBiases = None
        if self.biasFlag:
            selectedBiases = self.biasBank[biasIndexes]

        return selectedWeights, selectedBiases

    def calculateParameterMixture(
        self,
        selectedWeightParameters,
        weightProbabilities,
        selectedBiasParameters,
        biasProbabilities,
    ):
        weightMixture = self._calculateMixture(
            selectedWeightParameters, weightProbabilities, True
        )
        biasMixture = None
        if self.biasFlag:
            biasMixture = self._calculateMixture(
                selectedBiasParameters, biasProbabilities
            )
        return weightMixture, biasMixture

    def _calculateMixture(self, selectedParameters, probabilities, isWeight=False):
        probabilitiesShape = self.biasMixturePorbabilityShape
        if isWeight:
            probabilitiesShape = self.weightMixturePorbabilityShape

        reshapedProbabilities = L.reshape(probabilities, probabilitiesShape)
        weightedWeights = selectedParameters * reshapedProbabilities
        return L.sum(weightedWeights, dim=1)


class MatrixChoiceSparse(MatrixChoiceBase):
    def __init__(self, cfg: "ParameterGeneratorConfig"):
        super().__init__(cfg)

        self.mixtureBehaviour = SparseMixtureBehaviour(cfg, self)

    def forward(self, inputBatch):
        return self.mixtureBehaviour.calculateMixture(inputBatch)

    def handleProbabilitiesShapeHook(self, probabilities):
        probabilitiesReshaped = L.reshape(probabilities, [-1, 1])
        return probabilitiesReshaped


class MatrixChoiceMixture(MatrixChoiceBase):
    def __init__(self, cfg: "ParameterGeneratorConfig"):
        super().__init__(cfg)

        self.topK = min(cfg.topK, self.depthDim)
        self.routerOutputDim = self.topK

        self.mixtureBehaviour = TopkMixtureBehaviour(cfg, self)

    def _storeMixtureProbabilityShapes(self):
        weightMixtureShape = [self.batchSize, self.topK, 1, 1]
        biasMixtureShape = [self.batchSize, self.topK, 1]
        return weightMixtureShape, biasMixtureShape

    def forward(self, inputBatch):
        return self.mixtureBehaviour.calculateMixture(inputBatch)


class MatrixChoiceSoftMixture(MatrixChoiceBase):
    def __init__(self, cfg: "ParameterGeneratorConfig"):
        super().__init__(cfg)

        self.mixtureBehaviour = FullMixtureBehaviour(cfg, self)

    def forward(self, inputBatch):
        return self.mixtureBehaviour.calculateMixture(inputBatch)
