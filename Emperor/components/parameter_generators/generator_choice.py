from Emperor.library.libs.torch import Library as L

from .utils.base import ParameterGenerator
from .utils.probabilitySamplers import (
    ProbabilitySamplerTopk,
    ProbabilitySamplerFull,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ParameterGeneratorConfig


class GeneratorChoiceBase(ParameterGenerator):
    def __init__(self, cfg: "ParameterGeneratorConfig"):
        super().__init__(cfg)

        self.inputWeightBank, self.outputWeightBank, self.biasBank = (
            self._initializeParameterBanks()
        )

    def _initializeParameterBanks(self):
        inputWeightBank = L.parameterWeights(
            self.depthDim, self.inputDim, self.inputDim
        )
        outputWeightBank = L.parameterWeights(
            self.depthDim, self.inputDim, self.outputDim
        )
        L.initializeWeights(inputWeightBank, outputWeightBank)

        biasBank = None
        if self.biasFlag:
            biasBank = L.parameterWeights(self.depthDim, self.inputDim, self.outputDim)
            L.initializeWeights(biasBank)

        return inputWeightBank, outputWeightBank, biasBank

    def _normalizeVectors(
        self, inputWeightVectors, outputWeightVectors, outputBiasVectors
    ):
        inputWeightVectorsNormalized = L.normalize(inputWeightVectors)
        outputWeightVectorsNormalized = L.normalize(outputWeightVectors)

        biasVectorsNormalized = None
        if self.biasFlag:
            biasVectorsNormalized = L.normalize(outputBiasVectors)

        return (
            inputWeightVectorsNormalized,
            outputWeightVectorsNormalized,
            biasVectorsNormalized,
        )


class GeneratorChoiceSum(GeneratorChoiceBase):
    def __init__(self, cfg: "ParameterGeneratorConfig"):
        super().__init__(cfg)

    def forward(self, inputBatch):
        inputWeightVectors, outputWeightVectors, outputBiasVectors = (
            self._calculateParameterVectors(inputBatch)
        )  # [depthDim, batchSize, inputDim], [depthDim, batchSize, inputDim]

        generatedWeights, generatedBiases = self._generateParameters(
            inputWeightVectors, outputWeightVectors, outputBiasVectors
        )  # [batchSize, inputDim, outputDim]

        # [batchSize, inputDim, outputDim]
        return generatedWeights, generatedBiases, None

    def _calculateParameterVectors(self, inputBatch):  # [batch_size, d_input]
        inputWeightVectors = L.tensorProduct(
            inputBatch, self.inputWeightBank
        )  # [depthDim, batchSize, inputDim]

        outputWeightVectors = L.tensorProduct(
            inputBatch, self.outputWeightBank
        )  # [depthDim, batchSize, outputDim]

        biasVectors = None
        if self.biasFlag:
            biasVectors = L.tensorProduct(
                inputBatch, self.biasBank
            )  # [depthDim, batchSize, outputDim]

        inputWeightVectorsNorm, outputWeightVectorsNorm, outputBiasVectorsNorm = (
            self._normalizeVectors(inputWeightVectors, outputWeightVectors, biasVectors)
        )

        # [depthDim, batchSize, inputDim], [depthDim, batchSize, inputDim]
        return inputWeightVectorsNorm, outputWeightVectorsNorm, outputBiasVectorsNorm

    def _generateParameters(
        self, inputWeightVectors, outputWeightVectors, outputBiasVectors
    ):
        weightVectorsOuterPorduct = L.einsum(
            "bij,bik->bijk", inputWeightVectors, outputWeightVectors
        )  # [depthDim, batchSize, inputDim, outputDim]

        generatedWeights = L.sum(
            weightVectorsOuterPorduct, dim=0
        )  # [batchSize, inputDim, outputDim]
        generatedBiases = None
        if self.biasFlag:
            generatedBiases = L.sum(outputBiasVectors, dim=0)

        return generatedWeights, generatedBiases


class GeneratorChoiceMixture(GeneratorChoiceBase):
    def __init__(self, cfg: "ParameterGeneratorConfig"):
        super().__init__(cfg)
        self.topK = min(cfg.topK, self.depthDim)

        self.probabilitySampler = ProbabilitySamplerTopk(cfg)

    def forward(self, inputBatch):
        (
            weightTopKProbabilities,
            weightTopKIndexes,
            biasTopKProbabilities,
            biasTopKIndexes,
        ) = self._sampleTopKProbabilitiesAndIndexes(inputBatch)

        selectedInputParameters, selectedOutputParameters, selectedBiasesParameters = (
            self._selectGeneratorParameters(weightTopKIndexes, biasTopKIndexes)
        )  # [batchSize, topKExperts, inputDim, inputDim],[batchSize, topKExperts, inputDim, outputDim]

        inputWeightVectors, outputWeightVectors, biasVectorsNorm = (
            self._calculateParameterVectors(
                inputBatch,
                selectedInputParameters,
                selectedOutputParameters,
                selectedBiasesParameters,
            )
        )  # [batchSize, topKExperts, inputDim], [batchSize, topKExperts, outputDim]

        # [batchSize, inputDim, outputDim]
        generatedWeights, generatedBiases = self._generateWeightedParameters(
            weightTopKProbabilities,
            biasTopKProbabilities,
            inputWeightVectors,
            outputWeightVectors,
            biasVectorsNorm,
        )

        # self.gatherFrequency(gatherFrequency)

        return generatedWeights, generatedBiases, None

    def _sampleTopKProbabilitiesAndIndexes(self, inputBatch):
        topKWeightProbabilities, topKWeightIndices = (
            self._sampleProbabilitiesAndIndexes(inputBatch)
        )  # [batchSize, topKExperts], [batchSize, topKExperts]

        topKBiasIndices, topKBiasProbabilities = (None, None)
        if self.biasFlag:
            topKBiasProbabilities, topKBiasIndices = (
                self._sampleProbabilitiesAndIndexes(
                    inputBatch, computeWeightsFlag=False
                )
            )

        return (
            topKWeightProbabilities,
            topKWeightIndices,
            topKBiasProbabilities,
            topKBiasIndices,
        )

    def _sampleProbabilitiesAndIndexes(self, inputBatch, computeWeightsFlag=True):
        return self.probabilitySampler.sampleProbabilitiesAndIndexes(
            inputBatch,
            isTrainingFlag=self.training,
            computeWeightsFlag=computeWeightsFlag,
        )

    def _selectGeneratorParameters(
        self, weightIndexes, biasIndexes
    ):  # [batchSize, topKExperts]
        # [batchSize, topKExperts, inputDim, inputDim]
        selectedInputParameters = self.inputWeightBank[weightIndexes]
        # [batchSize, topKExperts, inputDim, outputDim]
        selectedOutputParameters = self.outputWeightBank[weightIndexes]
        selectedBiasesParameters = None
        if self.biasFlag:
            selectedBiasesParameters = self.biasBank[biasIndexes]
        # [batchSize, topKExperts, inputDim, inputDim],
        # [batchSize, topKExperts, inputDim, outputDim]
        return (
            selectedInputParameters,
            selectedOutputParameters,
            selectedBiasesParameters,
        )

    def _calculateParameterVectors(
        self,
        inputBatch,
        selectedInputWeights,
        selectedOutputWeights,
        selectedOutputBiases,
    ):
        inputWeightVectors = L.einsum(
            "bi,bkij->bkj", inputBatch, selectedInputWeights
        )  # [batchSize, topKExperts, inputDim]

        outputWeightVectors = L.einsum(
            "bi,bkij->bkj", inputBatch, selectedOutputWeights
        )  # [batchSize, topKExperts, outputDim]
        biasVectors = None
        if self.biasFlag:
            biasVectors = L.einsum(
                "bi,bkij->bkj", inputBatch, selectedOutputBiases
            )  # [batchSize, topKExperts, outputDim]

        # [batchSize, topKExperts, inputDim]
        # [batchSize, topKExperts, outputDim]
        inputWeightVectorsNorm, outputWeightVectorsNorm, outputBiasVectorsNorm = (
            self._normalizeVectors(inputWeightVectors, outputWeightVectors, biasVectors)
        )

        # [batchSize, topKExperts, inputDim]
        # [batchSize, topKExperts, outputDim]
        return inputWeightVectorsNorm, outputWeightVectorsNorm, outputBiasVectorsNorm

    def _generateWeightedParameters(
        self,
        topKWeightProbabilities,
        topKBiasProbabilities,
        inputWeightVectors,
        outputWeightVectors,
        biasVectorsNorm,
    ):
        generatedWeights = self._generateWeightParameters(
            topKWeightProbabilities, inputWeightVectors, outputWeightVectors
        )

        generatedBiases = self._generateBiasParameters(
            topKBiasProbabilities, biasVectorsNorm
        )

        return generatedWeights, generatedBiases  # [batchSize, inputDim, outputDim]

    def _generateWeightParameters(
        self, topKWeightProbabilities, inputWeightVectors, outputWeightVectors
    ):
        topKWeightsProbabilitiesReshaped = L.reshape(
            topKWeightProbabilities, [self.batchSize, self.topK, 1, 1]
        )  # [batchSize, topKExperts, 1, 1]

        generatedWeightsRaw = L.einsum(
            "bij,bik->bijk", inputWeightVectors, outputWeightVectors
        )  # [batchSize, topKExperts, inputDim, outputDim]

        weightedGeneratedWeightsRaw = (
            generatedWeightsRaw * topKWeightsProbabilitiesReshaped
        )  # [batchSize, topKExperts, inputDim, outputDim]
        generatedWeights = L.sum(weightedGeneratedWeightsRaw, dim=1)

        return generatedWeights

    def _generateBiasParameters(self, topKBiasProbabilities, biasVectorsNorm):
        if not self.biasFlag:
            return None

        topKBiasProbabilitiesReshaped = L.reshape(
            topKBiasProbabilities, [self.batchSize, self.topK, 1]
        )
        weightedGeneratedBiasesRaw = biasVectorsNorm * topKBiasProbabilitiesReshaped
        generatedBiases = L.sum(weightedGeneratedBiasesRaw, dim=1)
        return generatedBiases


class GeneratorChoiceSoftMixture(GeneratorChoiceBase):
    def __init__(self, cfg: "ParameterGeneratorConfig"):
        super().__init__(cfg)

        self.probabilitySampler = ProbabilitySamplerFull(cfg)

    def forward(self, inputBatch):
        # [batchSize, depthDim]
        weightFullProbabilities, biasFullProbabilities = self._sampleFullProbabilities(
            inputBatch
        )
        # [batchSize, topKExperts, inputDim]
        # [batchSize, topKExperts, outputDim]
        inputWeightVectors, outputWeightVectors, biasVectors = (
            self._calculateParameterVectors(inputBatch)
        )
        # [batchSize, inputDim, outputDim]
        generatedWeights, generatedBiases = self._generateWeightedParameters(
            weightFullProbabilities,
            biasFullProbabilities,
            inputWeightVectors,
            outputWeightVectors,
            biasVectors,
        )

        return generatedWeights, generatedBiases, None

    def _sampleFullProbabilities(self, inputBatch):
        weightFullProbabilities, _ = self._sampleProbabilities(inputBatch)

        biasFullProbabilities = None
        if self.biasFlag:
            biasFullProbabilities, _ = self._sampleProbabilities(
                inputBatch, computeWeightsFlag=False
            )

        return weightFullProbabilities, biasFullProbabilities

    def _sampleProbabilities(self, inputBatch, computeWeightsFlag=True):
        return self.probabilitySampler.sampleProbabilitiesAndIndexes(
            inputBatch,
            isTrainingFlag=self.training,
            computeWeightsFlag=computeWeightsFlag,
        )

    def _calculateParameterVectors(self, inputBatch):
        # [batchSize, depthDim, inputDim]
        inputWeightVectors = L.einsum("bi,kij->bkj", inputBatch, self.inputWeightBank)
        # [batchSize, depthDim, outputDim]
        outputWeightVectors = L.einsum("bi,kij->bkj", inputBatch, self.outputWeightBank)

        biasVectors = None
        if self.biasFlag:
            # [batchSize, depthDim, outputDim]
            biasVectors = L.einsum("bi,kij->bkj", inputBatch, self.biasBank)

        inputWeightVectorsNorm, outputWeightVectorsNorm, biasVectorsNorm = (
            self._normalizeVectors(inputWeightVectors, outputWeightVectors, biasVectors)
        )  # [batchSize, depthDim, inputDim], [batchSize, depthDim, outputDim]

        # [batchSize, depthDim, inputDim]
        # [batchSize, depthDim, outputDim]
        return inputWeightVectorsNorm, outputWeightVectorsNorm, biasVectorsNorm

    def _generateWeightedParameters(
        self,
        maskedWeightProbabilities,
        maskedBiasProbabilities,
        inputWeightVectors,
        outputWeightVectors,
        biasVectors,
    ):  # [batchSize, depthDim, 1, 1]
        generatedWeights = self._generateWeightParameters(
            maskedWeightProbabilities, inputWeightVectors, outputWeightVectors
        )

        generatedBiases = self._generateBiasParameters(
            maskedBiasProbabilities, biasVectors
        )

        return generatedWeights, generatedBiases  # [batchSize, inputDim, outputDim]

    def _generateWeightParameters(
        self, maskedWeightProbabilities, inputWeightVectors, outputWeightVectors
    ):
        weightsProbabilitiesReshaped = L.reshape(
            maskedWeightProbabilities, [self.batchSize, self.depthDim, 1, 1]
        )  # [batchSize, depthDim, 1, 1]

        generatedWeightsRaw = L.einsum(
            "bij,bik->bijk", inputWeightVectors, outputWeightVectors
        )  # [batchSize, depthDim, inputDim, outputDim]

        # [batchSize, depthDim, inputDim, outputDim]
        weightedGeneratedWeightsRaw = generatedWeightsRaw * weightsProbabilitiesReshaped

        # [batchSize, inputDim, outputDim]
        generatedWeights = L.sum(weightedGeneratedWeightsRaw, dim=1)

        return generatedWeights

    def _generateBiasParameters(self, maskedBiasProbabilities, biasVectors):
        if not self.biasFlag:
            return None
        generatedBiases = None
        biasesProbabilitiesReshaped = L.reshape(
            maskedBiasProbabilities, [self.batchSize, self.depthDim, 1]
        )
        weightedGeneratedBiasesRaw = biasVectors * biasesProbabilitiesReshaped
        generatedBiases = L.sum(weightedGeneratedBiasesRaw, dim=1)

        return generatedBiases
