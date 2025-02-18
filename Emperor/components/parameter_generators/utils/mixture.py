from .probabilitySamplers import (
    ProbabilitySamplerSparse,
    ProbabilitySamplerTopk,
    ProbabilitySamplerFull,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import ParameterGenerator
    from Emperor.config import ParameterGeneratorConfig


class SparseMixtureBehaviour:
    def __init__(
        self,
        cfg: "ParameterGeneratorConfig",
        model: "ParameterGenerator",
    ):
        self.model = model
        self.probabilitySampler = ProbabilitySamplerSparse(cfg)

    def calculateMixture(self, inputBatch):  # [batch_size, d_input]
        weightSparseIndexes, biasSparseIndexes, probabilities = (
            self._sampleSparseProbabilitiesAndIndexes(inputBatch)
        )

        selectedWeights, selectedBiases = self.model.selectParameters(
            weightSparseIndexes, biasSparseIndexes
        )  # [batchSize, inputDim, outputDim]

        # self.gatherFrequency(weightIndexes)

        # [batchSize, inputDim, outputDim], [batchSize, 1]
        return selectedWeights, selectedBiases, probabilities

    def _sampleSparseProbabilitiesAndIndexes(self, inputBatch):
        weightSparseProbabilities, weightSparseIndexes = (
            self._sampleProbabilitiesAndIndexes(inputBatch)
        )
        probabilities = weightSparseProbabilities

        biasSparseIndexes = None
        if self.model.biasFlag:
            biasSparseProbabilities, biasSparseIndexes = (
                self._sampleProbabilitiesAndIndexes(
                    inputBatch, computeWeightsFlag=False
                )
            )
            probabilities += biasSparseProbabilities

        return weightSparseIndexes, biasSparseIndexes, probabilities

    def _sampleProbabilitiesAndIndexes(self, inputBatch, computeWeightsFlag=True):
        probabilities, indexes = self.probabilitySampler.sampleProbabilitiesAndIndexes(
            inputBatch,
            isTrainingFlag=self.model.training,
            computeWeightsFlag=computeWeightsFlag,
        )
        probabilitiesReshaped = self.model.handleProbabilitiesShapeHook(probabilities)
        return probabilitiesReshaped, indexes


class TopkMixtureBehaviour:
    def __init__(
        self,
        cfg: "ParameterGeneratorConfig",
        model: "ParameterGenerator",
    ):
        self.model = model
        self.probabilitySampler = ProbabilitySamplerTopk(cfg)

    def calculateMixture(self, inputBatch):
        (
            weightTopKProbabilities,
            weightTopKIndexes,
            biasTopKProbabilities,
            biasTopKIndexes,
        ) = self._sampleTopKProbabilitiesAndIndexes(inputBatch)

        # [batchSize, topK, inputDim, outputDim]
        selectedWeightParameters, selectedBiaseParameters = self.model.selectParameters(
            weightTopKIndexes, biasTopKIndexes
        )

        # [batchSize, inputDim, outputDim]
        weightMixture, biasMixture = self.model.calculateParameterMixture(
            selectedWeightParameters,
            weightTopKProbabilities,
            selectedBiaseParameters,
            biasTopKProbabilities,
        )

        # self.gatherFrequency(topKIndices)

        # [batchSize, inputDim, outputDim]
        return weightMixture, biasMixture, None

    def _sampleTopKProbabilitiesAndIndexes(self, inputBatch):
        # [inputDim, batchSize, topK], [inputDim, batchSize, topK]
        weightTopKProbabilities, weightTopKIndexes = (
            self._sampleProbabilitiesAndIndexes(inputBatch)
        )

        biasTopKProbabilities, biasTopKIndexes = (None, None)
        if self.model.biasFlag:
            biasTopKProbabilities, biasTopKIndexes = (
                self._sampleProbabilitiesAndIndexes(
                    inputBatch, computeWeightsFlag=False
                )
            )

        return (
            weightTopKProbabilities,
            weightTopKIndexes,
            biasTopKProbabilities,
            biasTopKIndexes,
        )

    def _sampleProbabilitiesAndIndexes(self, inputBatch, computeWeightsFlag=True):
        return self.probabilitySampler.sampleProbabilitiesAndIndexes(
            inputBatch,
            isTrainingFlag=self.model.training,
            computeWeightsFlag=computeWeightsFlag,
        )


class FullMixtureBehaviour:
    def __init__(
        self,
        cfg: "ParameterGeneratorConfig",
        model: "ParameterGenerator",
    ):
        self.model = model
        self.probabilitySampler = ProbabilitySamplerFull(cfg)

    def calculateMixture(self, inputBatch):  # [batchSize, inputDim]
        weightFullProbabilities, biasFullProbabilities = self._sampleFullProbabilities(
            inputBatch
        )  # [inputDim, batchSize, depthDim]

        weightMixture, biasMixture = self.model.calculateParameterMixture(
            self.model.weightBank,
            weightFullProbabilities,
            self.model.biasBank,
            biasFullProbabilities,
        )

        # [batchSize, inputDim, outputDim]
        return weightMixture, biasMixture, None

    def _sampleFullProbabilities(self, inputBatch):
        weightFullProbabilities, _ = self._sampleProbabilities(inputBatch)

        biasFullProbabilities = None
        if self.model.biasFlag:
            biasFullProbabilities, _ = self._sampleProbabilities(
                inputBatch, computeWeightsFlag=False
            )

        return weightFullProbabilities, biasFullProbabilities

    def _sampleProbabilities(self, inputBatch, computeWeightsFlag=True):
        return self.probabilitySampler.sampleProbabilitiesAndIndexes(
            inputBatch,
            isTrainingFlag=self.model.training,
            computeWeightsFlag=computeWeightsFlag,
        )
