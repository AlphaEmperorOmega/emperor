import torch
from .routers import RouterModel, VectorChoiceRouterModel
from Emperor.library.choice import Library as L

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ParameterGeneratorConfig


class ProbabilitySamplerBehaviour:
    def __init__(self, cfg: "ParameterGeneratorConfig"):
        self.cfg = cfg
        self.isTrainingFlag = None
        self.noisyTopkFlag = cfg.noisyTopkFlag
        self.numProbabilitiesToRandomlySample = cfg.randomSampleTopK
        self.topK = cfg.topK
        self.topKTresholdFlag = True
        self.noiseEpsilon = 1e-2
        self.topKTreshold = L.toTensor(1 / cfg.depthDim) - 1e-6
        assert self.numProbabilitiesToRandomlySample <= self.topK

        self.routerModel = cfg.router if cfg.router is not None else RouterModel(cfg)

        self.auxiliaryLosses = cfg.auxiliaryLosses
        self.calcSoftmaxCustomFlag = True
        self.computeWeightsFlag = True

    def sampleProbabilitiesAndIndexes(
        self, inputBatch, skipMask=None, isTrainingFlag=False, computeWeightsFlag=True
    ):
        self.isTrainingFlag = isTrainingFlag

        fullProbabilities, logitScores = self._calcProbabilities(
            inputBatch, skipMask, computeWeightsFlag
        )
        sampledProbabilities, indexes = self._sampleProbabilities(fullProbabilities)
        self.computeLossHook(
            logitScores, fullProbabilities, sampledProbabilities, indexes
        )
        if self.calcSoftmaxCustomFlag:
            sampledProbabilities = self.calcSoftmaxCustom(sampledProbabilities)

        return sampledProbabilities, indexes

    def _calcProbabilities(self, inputBatch, skipMask=None, computeWeightsFlag=True):
        logitScores = self.routerModel.calcLogitScores(inputBatch, computeWeightsFlag)
        noisyLogits = self._addNoiseToLogits(logitScores)
        probabilities = L.softmax(noisyLogits)
        return self._maskScores(probabilities, noisyLogits, skipMask)

    def _addNoiseToLogits(self, logitScores):
        if self.noisyTopkFlag:
            # Because the router now generates `self.depthDim * 2` scores
            # one half of those will be used as standard deviation scores
            logitScores, rawNoiseStandardDeviation = L.chunk(logitScores, 2)
            if self.isTrainingFlag:
                noiseStandardDeviation = (
                    L.sigmoid(rawNoiseStandardDeviation) + self.noiseEpsilon
                )

                noise = L.randn_like(logitScores)
                noisyLogitScores = logitScores + noise * noiseStandardDeviation
                logits = noisyLogitScores
            else:
                logits = logitScores
        else:
            logits = logitScores

        return logits

    def _maskScores(self, probabilities, logitsScores, skipMask=None):
        if skipMask is not None:
            probabilities = torch.masked_fill(probabilities, (skipMask == 0), 0)
            logitsScores = torch.masked_fill(logitsScores, (skipMask == 0), 0)
        return probabilities, logitsScores

    def _sampleProbabilities(self, probabilities):
        return self.probabilitySamplerHook(probabilities)

    def computeLossHook(self, logits, fullProbabilities, probabilities, indices):
        return 0.0

    def probabilitySamplerHook(self, probabilities):
        pass

    def calcSoftmaxCustom(self, probabilities):
        return probabilities / (L.sum(probabilities, dim=-1, keepdim=True) + 1e-6)


class ProbabilitySamplerSparse(ProbabilitySamplerBehaviour):
    def __init__(self, cfg: "ParameterGeneratorConfig") -> None:
        super().__init__(cfg)
        self.calcSoftmaxCustomFlag = False

    def probabilitySamplerHook(self, probabilities):
        return L.getTopProbabilityAndIndex(probabilities)

    def computeLossHook(self, logits, fullProbabilities, probabilities, indices):
        dim0 = torch.prod(torch.tensor(probabilities.shape))
        gatesBuffer = torch.zeros(dim0, self.cfg.depthDim).to(L.Device)

        gates = gatesBuffer.scatter(
            1,
            indices.view(-1, 1),
            probabilities.view(-1, 1),
        ).to(L.Device)

        logits = logits.reshape(-1, self.cfg.depthDim)
        fullProbabilities = logits.reshape(-1, self.cfg.depthDim)

        self.auxiliaryLosses.updateAccumulatedStatistics(
            logits, fullProbabilities, gates
        )


class ProbabilitySamplerTopk(ProbabilitySamplerBehaviour):
    def __init__(self, cfg: "ParameterGeneratorConfig") -> None:
        super().__init__(cfg)

    def probabilitySamplerHook(self, probabilities):
        probabilities, indices = self._sampleTopKProbabilities(probabilities)

        return probabilities, indices

    def _sampleTopKProbabilities(self, probabilities):
        if self.isTrainingFlag and (self.numProbabilitiesToRandomlySample > 0):
            topKProbabilities, topKIndices = self._sampleRandomTopKProbabilities(
                probabilities
            )
        else:
            topKProbabilities, topKIndices = L.topk(probabilities, self.topK)

        return topKProbabilities, topKIndices

    def _sampleRandomTopKProbabilities(self, probabilities):
        # Select the `(topk - numProbabilitiesToRandomlySample)` top probabilities
        numTrueProbabilities = self.topK - self.numProbabilitiesToRandomlySample
        _, trueTopKIndices = L.topk(probabilities, numTrueProbabilities)

        # Hide probabilitie that have allready been selected
        maskedProbabilities = probabilities + 1e-6
        rangeIndexes = L.unsqueeze(L.arange(probabilities.size(0)), 1)
        maskedProbabilities[rangeIndexes, trueTopKIndices] = 0

        sampledProbabilityIndices = L.multinomial(
            maskedProbabilities, self.numProbabilitiesToRandomlySample
        )

        topKIndices = L.cat([trueTopKIndices, sampledProbabilityIndices])

        # Retrieve the probabilities of the incides in the above step
        topKProbabilities = L.gather(probabilities, 1, topKIndices)

        return topKProbabilities, topKIndices

    def computeLossHook(self, logits, fullProbabilities, probabilities, indices):
        gatesBuffer = torch.zeros(
            torch.prod(torch.tensor(probabilities.shape)), self.cfg.depthDim
        ).to(L.Device)
        gates = gatesBuffer.scatter(
            1,
            indices.view(-1, self.cfg.topK),
            probabilities.view(-1, self.cfg.topK),
        ).to(L.Device)

        logits = logits.reshape(-1, self.cfg.depthDim)
        fullProbabilities = logits.reshape(-1, self.cfg.depthDim)

        self.auxiliaryLosses.updateAccumulatedStatistics(
            logits, fullProbabilities, gates
        )


class ProbabilitySamplerFull(ProbabilitySamplerBehaviour):
    def __init__(self, cfg: "ParameterGeneratorConfig") -> None:
        super().__init__(cfg)

    def probabilitySamplerHook(self, probabilities):
        probabilities, indices = self._sampleFullProbabilities(probabilities)
        return probabilities, indices

    def _sampleFullProbabilities(self, probabilities):
        if self.topKTresholdFlag:
            return self._maskProbsTopKTreshold(probabilities)
        return probabilities, None

    def _maskProbsTopKTreshold(self, probabilities):
        tresholdTopKMask = probabilities < self.topKTreshold
        maskedProbabilities = L.where(tresholdTopKMask, 0.0, probabilities)
        maskedProbabilities = maskedProbabilities / (
            L.sum(maskedProbabilities, dim=-1, keepdim=True) + 1e-6
        )
        return maskedProbabilities, None
