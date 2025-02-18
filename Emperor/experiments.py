from typing import List
import torch
import torch.nn as nn


from Emperor.components.attention import Attention, PatchEmbedding
from Emperor.components.moe import MixtureOfExperts
from .base.decorators import timer
from dataclasses import replace
from .library.choice import Library as L
from .components.layer import ParameterGeneratorLayer
from .components.parameter_generators.utils.losses import AuxiliaryLosses
from .config import ModelConfig, ParameterGeneratorMultiLayerConfig
from .base.models import Classifier


class ClassifierExperiment(Classifier):
    def __init__(self, learningRate, cfg: "ModelConfig"):
        super().__init__()
        self.lr = learningRate
        self.plotProgress = cfg.plotProgress


class SingleLayerWeightGeneratorModel(ClassifierExperiment):
    def __init__(self, learningRate, cfg: "ModelConfig"):
        super().__init__(learningRate, cfg)
        cfg = replace(
            cfg,
            auxiliaryLosses=AuxiliaryLosses(cfg),
        )

        self.model = L.Sequential(L.Flatten(), ParameterGeneratorLayer(cfg))

    def forward(self, inputBatch):
        self.model[1].loss = 0.0

        output = self.model(inputBatch)
        auxiliaryLosses = self.auxiliaryLosses.getAuxiliaryLossAndClear()

        return (output, auxiliaryLosses)


class MixtureOfExpertsSingleLayerModel(ClassifierExperiment):
    def __init__(self, learningRate, cfg: "ModelConfig"):
        super().__init__(learningRate, cfg)

        self.auxiliaryLosses = AuxiliaryLosses(cfg)
        self.moeAuxiliaryLosses = AuxiliaryLosses(cfg)

        cfg = replace(
            cfg,
            auxiliaryLosses=self.auxiliaryLosses,
            moeAuxiliaryLosses=self.moeAuxiliaryLosses,
        )

        self.model = nn.Sequential(nn.Flatten(), MixtureOfExperts(cfg))

    def forward(self, inputBatch):
        output = self.model(inputBatch)

        expertsAuxiliaryLosses = self.auxiliaryLosses.getAuxiliaryLossAndClear()
        moeAuxiliaryLosses = self.moeAuxiliaryLosses.getAuxiliaryLossAndClear()
        auxiliaryLosses = expertsAuxiliaryLosses + moeAuxiliaryLosses

        return (output, auxiliaryLosses)


class AttentionSingleLayerModel(ClassifierExperiment):
    def __init__(self, learningRate, cfg: "ModelConfig"):
        super().__init__(learningRate, cfg)

        self.auxiliaryLosses = AuxiliaryLosses(cfg)
        self.moeAuxiliaryLosses = AuxiliaryLosses(cfg)
        self.plotProgress = False

        imageSize = 28
        patchSize = 4
        numPatches = (imageSize // patchSize) ** 2
        self.patcherModel = PatchEmbedding(
            inputChannels=1,
            embeddingDim=16,
            patchSize=patchSize,
            numPatches=numPatches,
        )

        cfg = replace(
            cfg,
            auxiliaryLosses=self.auxiliaryLosses,
            moeAuxiliaryLosses=self.moeAuxiliaryLosses,
        )

        self.attention = Attention(
            cfg=cfg,
            embeddingDim=cfg.embeddingDim,
            qkvHiddenDim=cfg.qkvHiddenDim,
            attentionOutputDim=cfg.attentionOutputDim,
            numExperts=cfg.numExperts,
            topK=cfg.topK,
            headDim=cfg.headDim,
        )

        self.model = nn.Sequential(nn.Flatten())

    def forward(self, inputBatch):
        imagePatches = self.patcherModel(inputBatch)
        imagePatches = imagePatches.permute(1, 0, 2)
        attentionOutput, _ = self.attention(
            imagePatches,
            imagePatches,
            imagePatches,
        )
        attentionOutput = attentionOutput.permute(1, 0, 2)

        # output = attentionOutput[:, 0, :]
        output = attentionOutput.mean(dim=1)

        expertsAuxiliaryLosses = self.auxiliaryLosses.getAuxiliaryLossAndClear()
        moeAuxiliaryLosses = self.moeAuxiliaryLosses.getAuxiliaryLossAndClear()
        auxiliaryLosses = expertsAuxiliaryLosses + moeAuxiliaryLosses

        return (output, auxiliaryLosses)


class MultiLayerWeightGeneratorModel(Classifier):
    def __init__(self, learningRate, cfg: ParameterGeneratorMultiLayerConfig):
        super().__init__()
        self.lr = learningRate
        self.auxiliaryLosses = AuxiliaryLosses(cfg.weightGeneratorLayerConfig)
        cfg.weightGeneratorLayerConfig.auxiliaryLosses = self.auxiliaryLosses
        self.plotProgress = cfg.weightGeneratorLayerConfig.plotProgress
        self.inputDim = cfg.weightGeneratorLayerConfig.inputDim
        self.hiddenDim = cfg.hiddenDim
        self.outputDim = cfg.weightGeneratorLayerConfig.outputDim
        self.numberOfLayers = cfg.numberOfLayers
        self.weightAndBiasGeneratorType = (
            cfg.weightGeneratorLayerConfig.weightAndBiasGeneratorType
        )
        self.firstWeightAndBiasGeneratorType = cfg.firstWeightAndBiasGeneratorType
        self.activationFunction = cfg.activatonFunction

        tempInputDim = self.inputDim
        tempHiddenDim = self.hiddenDim
        tempOutputDim = self.outputDim

        layerInputOutputShapes = self.__createLayerInputOutputShapes()
        layers = self.__generateModelLayers(cfg, layerInputOutputShapes)

        self.model = nn.Sequential(*layers)

        cfg.weightGeneratorLayerConfig.inputDim = tempInputDim
        cfg.hiddenDim = tempHiddenDim
        cfg.weightGeneratorLayerConfig.outputDim = tempOutputDim

    def __createLayerInputOutputShapes(self) -> List:
        inputOutput = []
        inputOutput.append([self.inputDim, self.hiddenDim])
        for _ in range(self.numberOfLayers):
            inputOutput.append([self.hiddenDim, self.hiddenDim])
        inputOutput.append([self.hiddenDim, self.outputDim])

        return inputOutput

    def __generateModelLayers(
        self, cfg: ParameterGeneratorMultiLayerConfig, layerInputOutput: List
    ):
        layers = [L.Flatten()]
        first = True
        for inputOutput in layerInputOutput:
            cfg.weightGeneratorLayerConfig.weightAndBiasGeneratorType = (
                self.firstWeightAndBiasGeneratorType
                if first
                else self.weightAndBiasGeneratorType
            )
            (
                cfg.weightGeneratorLayerConfig.inputDim,
                cfg.weightGeneratorLayerConfig.outputDim,
            ) = inputOutput
            layers.append(ParameterGeneratorLayer(cfg.weightGeneratorLayerConfig))
            if inputOutput[1] != self.outputDim:
                layers.append(nn.LayerNorm(cfg.weightGeneratorLayerConfig.outputDim))
                layers.append(self.activationFunction())
            first = False
        return layers

    @timer
    def forwardProcessTime(self, inputBatch):
        return self(inputBatch)

    def forward(self, inputBatch):
        return self.model(inputBatch), None


class SingleLayerSharedWeightGeneratorModel(Classifier):
    def __init__(self, learningRate, cfg: ParameterGeneratorMultiLayerConfig):
        super().__init__()
        self.auxiliaryLosses = AuxiliaryLosses(cfg.weightGeneratorLayerConfig)
        cfg.weightGeneratorLayerConfig.auxiliaryLosses = self.auxiliaryLosses
        self.lr = learningRate
        self.numLayers = cfg.numberOfLayers
        self.layerConfig = cfg.weightGeneratorLayerConfig
        self.plotProgress = self.layerConfig.plotProgress
        self.activationFunction = cfg.activatonFunction()
        self.flatten = L.Flatten()
        firstLayerType = cfg.firstWeightAndBiasGeneratorType
        layerType = cfg.weightGeneratorLayerConfig.weightAndBiasGeneratorType
        inputDim = cfg.weightGeneratorLayerConfig.inputDim
        hiddenDim = cfg.hiddenDim
        outputDim = cfg.weightGeneratorLayerConfig.outputDim

        cfg.weightGeneratorLayerConfig.weightAndBiasGeneratorType = firstLayerType
        cfg.weightGeneratorLayerConfig.inputDim = inputDim
        cfg.weightGeneratorLayerConfig.outputDim = hiddenDim
        self.firstLayer = ParameterGeneratorLayer(cfg.weightGeneratorLayerConfig)
        self.firstlayerNorm = nn.LayerNorm(hiddenDim)

        cfg.weightGeneratorLayerConfig.weightAndBiasGeneratorType = layerType
        cfg.weightGeneratorLayerConfig.inputDim = hiddenDim
        cfg.weightGeneratorLayerConfig.outputDim = hiddenDim
        self.sharedLayer = ParameterGeneratorLayer(cfg.weightGeneratorLayerConfig)
        self.sharedlayerNorm = nn.LayerNorm(hiddenDim)

        cfg.weightGeneratorLayerConfig.inputDim = hiddenDim
        cfg.weightGeneratorLayerConfig.outputDim = outputDim
        cfg.weightGeneratorLayerConfig.biasFlag = False
        self.lastLayer = ParameterGeneratorLayer(cfg.weightGeneratorLayerConfig)

    def forward(self, inputBatch):
        flattendInputBatch = self.flatten(inputBatch)

        firstLayerOutput = self.firstLayer(flattendInputBatch)
        firstLayerNorm = self.sharedlayerNorm(firstLayerOutput)
        firstLayerActivation = self.activationFunction(firstLayerNorm)
        output = firstLayerActivation
        for _ in range(self.numLayers):
            output = self.sharedLayer(output)
            output = self.sharedlayerNorm(output)
            output = self.activationFunction(output)

        return self.lastLayer(output), None
