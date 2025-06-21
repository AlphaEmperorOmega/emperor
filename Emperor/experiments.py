from typing import List
import torch.nn as nn

from Emperor.base.preprocess import PatchEmbeddingConv
from Emperor.components.attention import Attention
from Emperor.components.sut_layer import (
    TransformerEncoderLayerBase,
    TransformerDecorderLayerBase,
)
from Emperor.components.transformer_decoder import TransformerDecoderBase
from Emperor.components.transformer_encoder import TransformerEncoderBase
from Emperor.components.moe import MixtureOfExperts
from .base.decorators import timer
from dataclasses import replace
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

        self.model = nn.Sequential(nn.Flatten(), ParameterGeneratorLayer(cfg))

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
        self.patcherModel = PatchEmbeddingConv(
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

        self.attention = Attention(cfg=cfg)

    def forward(self, inputBatch):
        imagePatches = self.patcherModel(inputBatch)
        imagePatches = imagePatches.permute(1, 0, 2)
        _, rawEncoderOutput = self.attention(
            imagePatches,
            imagePatches,
            imagePatches,
        )
        # rawEncoderOutput = rawEncoderOutput.permute(1, 0, 2)

        output = rawEncoderOutput[:, 0, :]
        # output = attentionOutput.mean(dim=1)

        expertsAuxiliaryLosses = self.auxiliaryLosses.getAuxiliaryLossAndClear()
        moeAuxiliaryLosses = self.moeAuxiliaryLosses.getAuxiliaryLossAndClear()
        auxiliaryLosses = expertsAuxiliaryLosses + moeAuxiliaryLosses

        return (output, auxiliaryLosses)


class TransformerEncoderLayerBaseSingleLayerModel(ClassifierExperiment):
    def __init__(self, learningRate, cfg: "ModelConfig"):
        super().__init__(learningRate, cfg)

        self.auxiliaryLosses = AuxiliaryLosses(cfg)
        self.moeAuxiliaryLosses = AuxiliaryLosses(cfg)
        self.plotProgress = False

        imageSize = 28
        patchSize = 4
        numPatches = (imageSize // patchSize) ** 2
        self.patcherModel = PatchEmbeddingConv(
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

        self.encoderModel = TransformerEncoderLayerBase(
            cfg=cfg,
            returnRawFFNOutputFlag=True,
        )
        self.classificationModel = nn.Linear(cfg.embeddingDim, 10)

        self.useRawOutputFlag = False

    def forward(self, inputBatch):
        imagePatches = self.patcherModel(inputBatch)
        imagePatches = imagePatches.permute(1, 0, 2)
        _, rawEncoderOutput = self.encoderModel(
            imagePatches,
        )

        encoderOutput = rawEncoderOutput.permute(1, 0, 2)
        # output = attentionOutput[:, 0, :]
        output = encoderOutput.sum(dim=1)

        output = self.classificationModel(output)

        expertsAuxiliaryLosses = self.auxiliaryLosses.getAuxiliaryLossAndClear()
        moeAuxiliaryLosses = self.moeAuxiliaryLosses.getAuxiliaryLossAndClear()
        auxiliaryLosses = expertsAuxiliaryLosses + moeAuxiliaryLosses

        return (output, auxiliaryLosses)


class TransformerDecoderLayerBaseSingleLayerModel(ClassifierExperiment):
    def __init__(self, learningRate, cfg: "ModelConfig"):
        super().__init__(learningRate, cfg)

        self.auxiliaryLosses = AuxiliaryLosses(cfg)
        self.moeAuxiliaryLosses = AuxiliaryLosses(cfg)
        self.plotProgress = False

        imageSize = 28
        patchSize = 4
        numPatches = (imageSize // patchSize) ** 2
        self.patcherModel = PatchEmbeddingConv(
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

        self.decoderModel = TransformerDecorderLayerBase(
            cfg=cfg, returnRawFFNOutputFlag=True, crossSelfAttentionFlag=False
        )

        self.classificationModel = nn.Linear(cfg.embeddingDim, 10)

        self.useRawOutputFlag = False

    def forward(self, inputBatch):
        imagePatches = self.patcherModel(inputBatch)
        imagePatches = imagePatches.permute(1, 0, 2)
        decoderOutput, attentionWeights, selfAttentionState = self.decoderModel(
            imagePatches,
        )

        decoderOutput = decoderOutput.permute(1, 0, 2)
        # decoderOutput = decoderOutput[:, 0, :]
        decoderOutput = decoderOutput.sum(dim=1)

        output = self.classificationModel(decoderOutput)
        # output = output.sum(dim=1)

        expertsAuxiliaryLosses = self.auxiliaryLosses.getAuxiliaryLossAndClear()
        moeAuxiliaryLosses = self.moeAuxiliaryLosses.getAuxiliaryLossAndClear()
        auxiliaryLosses = expertsAuxiliaryLosses + moeAuxiliaryLosses

        return (output, auxiliaryLosses)


class TransformerEncoderBaseSingleLayerModel(ClassifierExperiment):
    def __init__(
        self,
        learningRate,
        cfg: "ModelConfig",
        encoderHaltingFlag: bool = False,
    ):
        super().__init__(learningRate, cfg)

        self.auxiliaryLosses = AuxiliaryLosses(cfg)
        self.moeAuxiliaryLosses = AuxiliaryLosses(cfg)
        self.plotProgress = False

        imageSize = 28
        patchSize = 4
        numPatches = (imageSize // patchSize) ** 2
        self.patcherModel = PatchEmbeddingConv(
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

        tokenEmbeddingModule = nn.Embedding(
            num_embeddings=20,
            embedding_dim=cfg.embeddingDim,
            padding_idx=1,
        )

        self.model = TransformerEncoderBase(
            cfg=cfg,
            tokenEmbeddingModule=tokenEmbeddingModule,
            encoderHaltingFlag=encoderHaltingFlag,
        )

        self.classificationModel = nn.Linear(cfg.embeddingDim, 10)
        self.useRawOutputFlag = False

    def forward(self, inputBatch):
        imagePatches = self.patcherModel(inputBatch)
        modelOutput = self.model.forward(
            tokenEmbeddings=imagePatches,
        )

        encoderOutput = modelOutput["encoderOutput"][0]

        encoderOutput = encoderOutput.permute(1, 0, 2)
        # output = encoderOutput[:, 0, :]
        output = encoderOutput.sum(dim=1)

        output = self.classificationModel(output)

        expertsAuxiliaryLosses = self.auxiliaryLosses.getAuxiliaryLossAndClear()
        moeAuxiliaryLosses = self.moeAuxiliaryLosses.getAuxiliaryLossAndClear()
        auxiliaryLosses = expertsAuxiliaryLosses + moeAuxiliaryLosses

        # if auxiliaryLosses > 0:
        #     auxiliaryLosses *= 50
        # else:
        #     auxiliaryLosses *= -50

        return (output, auxiliaryLosses)


class TransformerDecoderBaseSingleLayerModel(ClassifierExperiment):
    def __init__(
        self,
        learningRate,
        cfg: "ModelConfig",
        encoderHaltingFlag: bool = False,
    ):
        super().__init__(learningRate, cfg)

        self.auxiliaryLosses = AuxiliaryLosses(cfg)
        self.moeAuxiliaryLosses = AuxiliaryLosses(cfg)
        self.plotProgress = False

        imageSize = 28
        patchSize = 4
        numPatches = (imageSize // patchSize) ** 2
        self.patcherModel = PatchEmbeddingConv(
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

        tokenEmbeddingModule = nn.Embedding(
            num_embeddings=16,
            embedding_dim=cfg.embeddingDim,
            padding_idx=1,
        )

        self.model = TransformerDecoderBase(
            cfg=cfg,
            dictionary=[i for i in range(16)],
            tokenEmbeddingModule=tokenEmbeddingModule,
            crossSelfAttentionFlag=False,
        )

        self.classificationModel = nn.Linear(cfg.embeddingDim, 10)
        self.useRawOutputFlag = False

    def forward(self, inputBatch):
        imagePatches = self.patcherModel(inputBatch)
        modelOutput, _ = self.model.forward(
            tokenEmbeddings=imagePatches,
        )

        # encoderOutput = modelOutput["encoderOutput"][0]

        # encoderOutput = modelOutput.permute(1, 0, 2)
        # output = encoderOutput[:, 0, :]
        output = modelOutput.sum(dim=1)

        output = self.classificationModel(output)

        expertsAuxiliaryLosses = self.auxiliaryLosses.getAuxiliaryLossAndClear()
        moeAuxiliaryLosses = self.moeAuxiliaryLosses.getAuxiliaryLossAndClear()
        auxiliaryLosses = expertsAuxiliaryLosses + moeAuxiliaryLosses

        # if auxiliaryLosses > 0:
        #     auxiliaryLosses *= 50
        # else:
        #     auxiliaryLosses *= -50

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
        layers = [nn.Flatten()]
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
        self.flatten = nn.Flatten()
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
