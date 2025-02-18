import unittest
from copy import deepcopy
from typing import Dict, Optional, List

from Emperor.base.utils import Trainer
from Emperor.base.datasets import FashionMNIST
from Emperor.library.choice import Library as L
from Emperor.components.parameter_generators.utils.base import ParameterGenerator
from Emperor.components.parameter_generators.utils.losses import AuxiliaryLosses
from Emperor.experiments import (
    SingleLayerWeightGeneratorModel,
    MultiLayerWeightGeneratorModel,
    SingleLayerSharedWeightGeneratorModel,
)
from Emperor.config import (
    ParameterGeneratorConfig,
    ParameterGeneratorMultiLayerConfig,
    ParameterGeneratorOptions,
)


class GeneralTestMethods(unittest.TestCase):
    def checkForwardMethod(
        self, model: ParameterGenerator, areWeightProbabilities=False
    ):
        config = ParameterGeneratorConfig()
        config.auxiliaryLosses = AuxiliaryLosses(config)

        batchSize = config.batchSize
        inputDim = config.inputDim
        outputDim = config.outputDim
        biasFlag = config.biasFlag

        input = L.randn(batchSize, inputDim)
        selectedWeights, selectedBiases, weightProbabilities = model.forward(input)

        expectedSelectedWeightsShape = [batchSize, inputDim, outputDim]
        expectedWeightProbabilityMean = [batchSize, 1]
        expectedSelectedBiasesShape = [batchSize, outputDim]

        self.__checkExpectedShape(selectedWeights, expectedSelectedWeightsShape)
        if biasFlag:
            self.__checkExpectedShape(selectedBiases, expectedSelectedBiasesShape)
        if areWeightProbabilities:
            self.__checkExpectedShape(
                weightProbabilities, expectedWeightProbabilityMean
            )

    def __checkExpectedShape(self, outputTensor, expectedShape: List):
        outputTensorShape = L.shapeList(outputTensor)
        self.assertEqual(outputTensorShape, expectedShape)

    def areWeightsUpdated(
        self, weightAndBiasGeneratorType: ParameterGeneratorOptions
    ) -> None:
        config = ParameterGeneratorConfig()
        config.auxiliaryLosses = AuxiliaryLosses(config)
        config = deepcopy(config)
        config.plotProgress = False
        config.weightAndBiasGeneratorType = weightAndBiasGeneratorType
        batchSize = config.batchSize
        data = FashionMNIST(
            batch_size=batchSize, testDatasetFalg=True, testDatasetNumSamples=128
        )
        model = SingleLayerWeightGeneratorModel(
            learningRate=1,
            cfg=config,
        )

        biasFlag = config.biasFlag
        modelBefore = model.model[1].parameterGenerator
        parametersBeforeTraining = self.getLayerParametersHook(modelBefore, biasFlag)

        trainer = Trainer(max_epochs=1)
        trainer.fit(model, data)

        modelAfter = trainer.model.model[1].parameterGenerator
        parametersAfterTraining = self.getLayerParametersHook(modelAfter, biasFlag)

        self.__checkIfWeightsAreTheSame(
            parametersBeforeTraining, parametersAfterTraining
        )

    def getLayerParametersHook(self, model: ParameterGenerator, biasFlag: bool) -> Dict:
        parameters = {}

        parameters["WEIGHT_BANK"] = model.weightBank.clone().detach()
        if biasFlag:
            parameters["BIAS_BANK"] = model.biasBank.clone().detach()

        return parameters

    def areWeightsUpdatedMultiLayer(
        self, weightAndBiasGeneratorType: ParameterGeneratorOptions
    ):
        config = ParameterGeneratorMultiLayerConfig()
        config.weightGeneratorLayerConfig.auxiliaryLosses = AuxiliaryLosses(
            config.weightGeneratorLayerConfig
        )
        config = deepcopy(config)
        config.weightGeneratorLayerConfig.plotProgress = False
        config.weightGeneratorLayerConfig.weightAndBiasGeneratorType = (
            weightAndBiasGeneratorType
        )
        batchSize = config.weightGeneratorLayerConfig.batchSize
        biasFlag = config.weightGeneratorLayerConfig.biasFlag

        data = FashionMNIST(
            batch_size=batchSize, testDatasetFalg=True, testDatasetNumSamples=128
        )
        model = MultiLayerWeightGeneratorModel(
            learningRate=1,
            cfg=config,
        )
        parametersBeforeTraining = self.__getAllLayerParameters(model, biasFlag)

        trainer = Trainer(max_epochs=3)
        trainer.fit(model, data)

        parametersAfterTraining = self.__getAllLayerParameters(trainer.model, biasFlag)

        self.__checkIfAllLayerWeightsAreTheSame(
            parametersBeforeTraining, parametersAfterTraining
        )

    def __getAllLayerParameters(self, model, biasFlag: bool) -> Dict:
        identifier = "LAYER_"
        layerIndex = 1

        allLayerParameters = {}
        for modelLayer in model.model:
            if isinstance(modelLayer, ParameterGenerator):
                layerIdentifier = identifier + str(layerIndex)
                allLayerParameters[layerIdentifier] = self.getLayerParametersHook(
                    modelLayer, biasFlag
                )
                layerIndex += 1

        return allLayerParameters

    def __checkIfAllLayerWeightsAreTheSame(
        self, paramsBeforeTraining: Dict, paramsAfterTraining: Dict
    ) -> None:
        paramsKeys = paramsBeforeTraining.keys()
        for paramsKey in paramsKeys:
            beforeTraining = paramsBeforeTraining[paramsKey]
            afterTraining = paramsAfterTraining[paramsKey]
            self.__checkIfWeightsAreTheSame(beforeTraining, afterTraining, paramsKey)

    def __checkIfWeightsAreTheSame(
        self,
        paramsBeforeTraining: Dict,
        paramsAfterTraining: Dict,
        layerIdentifier: Optional[str] = None,
    ) -> None:
        paramsKeys = paramsBeforeTraining.keys()
        for paramKey in paramsKeys:
            paramBefore = paramsBeforeTraining[paramKey]
            paramAfter = paramsAfterTraining[paramKey]

            message = f"The the gradient does not update the {paramKey}"
            if isinstance(layerIdentifier, str):
                message += f" for {layerIdentifier}"
            checkParameterDifferences = L.areTensorsIdentical(paramBefore, paramAfter)
            self.assertFalse(checkParameterDifferences, message)

    def areWeightsUpdatedSharedLayer(
        self, weightAndBiasGeneratorType: ParameterGeneratorOptions
    ) -> None:
        config = ParameterGeneratorMultiLayerConfig()
        config = deepcopy(config)
        config.weightGeneratorLayerConfig.plotProgress = False
        config.weightGeneratorLayerConfig.weightAndBiasGeneratorType = (
            weightAndBiasGeneratorType
        )
        batchSize = config.weightGeneratorLayerConfig.batchSize
        data = FashionMNIST(
            batch_size=batchSize, testDatasetFalg=True, testDatasetNumSamples=128
        )
        model = SingleLayerSharedWeightGeneratorModel(
            learningRate=1,
            cfg=config,
        )

        biasFlag = config.weightGeneratorLayerConfig.biasFlag
        parametersBeforeTraining = [
            self.getLayerParametersHook(model.firstLayer.parameterGenerator, biasFlag),
            self.getLayerParametersHook(model.sharedLayer.parameterGenerator, biasFlag),
            self.getLayerParametersHook(model.lastLayer.parameterGenerator, biasFlag),
        ]

        trainer = Trainer(max_epochs=1)
        trainer.fit(model, data)

        parametersAfterTraining = [
            self.getLayerParametersHook(model.firstLayer.parameterGenerator, biasFlag),
            self.getLayerParametersHook(model.sharedLayer.parameterGenerator, biasFlag),
            self.getLayerParametersHook(model.lastLayer.parameterGenerator, biasFlag),
        ]
        for paramsBefore, paramsAfter in zip(
            parametersBeforeTraining, parametersAfterTraining
        ):
            self.__checkIfWeightsAreTheSame(paramsBefore, paramsAfter)
