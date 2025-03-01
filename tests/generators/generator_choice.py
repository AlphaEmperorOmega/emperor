from .general import GeneralTestMethods

from Emperor.library.choice import Library as L
from Emperor.components.parameter_generators.utils.losses import AuxiliaryLosses
from Emperor.config import ParameterGeneratorConfig, ParameterGeneratorOptions
from Emperor.components.parameter_generators.utils.base import ParameterGenerator
from Emperor.components.parameter_generators.generator_choice import (
    GeneratorChoiceBase,
    GeneratorChoiceSum,
    GeneratorChoiceMixture,
    GeneratorChoiceSoftMixture,
)


class GeneralGeneratorMethods(GeneralTestMethods):
    def getLayerParametersHook(self, model: ParameterGenerator, biasFlag: bool) -> None:
        isFirstLayerGeneratorChoice = isinstance(model, GeneratorChoiceBase)
        if isFirstLayerGeneratorChoice:
            parameters = {}

            parameters["INPUT_WEIGHT_BANK"] = model.inputWeightBank.clone().detach()
            parameters["OUTPUT_WEIGHT_BANK"] = model.outputWeightBank.clone().detach()
            if biasFlag:
                parameters["BIAS_BANK"] = model.biasBank.clone().detach()

            return parameters
        else:
            return super().getLayerParametersHook(model, biasFlag)


class TestGeneratorSumClass(GeneralGeneratorMethods):
    def setUp(self) -> None:
        pass

    def testForwardMethod(self):
        config = ParameterGeneratorConfig()
        config.auxiliaryLosses = AuxiliaryLosses(config)
        model = GeneratorChoiceSum(config)
        self.checkForwardMethod(model)

    def testIfWeightsAreUpdated(self):
        self.areWeightsUpdated(ParameterGeneratorOptions.generator_sum)

    def testIfSharedLayerWeightsAreUpdated(self):
        self.areWeightsUpdatedSharedLayer(ParameterGeneratorOptions.generator_sum)

    def testIfWeightsAreUpdatedMultiLayer(self):
        self.areWeightsUpdatedMultiLayer(ParameterGeneratorOptions.generator_sum)

    def testCalcLogitScores(self):
        """
        Input tensor:
            [[0, 1, 2],
             [3, 4, 5]]

        Input Weight Bank:
            [[[ 0,  1,  2],
              [ 3,  4,  5],
              [ 6,  7,  8]],

             [[ 9, 10, 11],
              [12, 13, 14],
              [15, 16, 17]],

             [[18, 19, 20],
              [21, 22, 23],
              [24, 25, 26]],

             [[27, 28, 29],
              [30, 31, 32],
              [33, 34, 35]]]

        Output Weight Bank:
            [[[ 0,  1,  2,  3,  4],
              [ 5,  6,  7,  8,  9],
              [10, 11, 12, 13, 14]],

             [[15, 16, 17, 18, 19],
              [20, 21, 22, 23, 24],
              [25, 26, 27, 28, 29]],

             [[30, 31, 32, 33, 34],
              [35, 36, 37, 38, 39],
              [40, 41, 42, 43, 44]],

             [[45, 46, 47, 48, 49],
              [50, 51, 52, 53, 54],
              [55, 56, 57, 58, 59]]]

        Input Weight Vectors:
            [[[ 15,  18,  21],
              [ 42,  54,  66]],

             [[ 42,  45,  48],
              [150, 162, 174]],

             [[ 69,  72,  75],
              [258, 270, 282]],

             [[ 96,  99, 102],
              [366, 378, 390]]]

        Output Weight Vectors:
            [[[ 25,  28,  31,  34,  37],
              [ 70,  82,  94, 106, 118]],

             [[ 70,  73,  76,  79,  82],
              [250, 262, 274, 286, 298]],

             [[115, 118, 121, 124, 127],
              [430, 442, 454, 466, 478]],

             [[160, 163, 166, 169, 172],
              [610, 622, 634, 646, 658]]]
        """

        inputDim = 3
        depthDim = 4
        outputDim = 5
        batchSize = 2

        # 6 = 2 * 3 = batchSize * inputDim
        inputTensor = L.createTensor(batchSize * inputDim, [batchSize, inputDim])

        # 18 = 2 * 3 * 3 = depthDim * inputDim * inputDim
        inputWeightBankCount = depthDim * inputDim * inputDim
        inputWeightBank = L.createTensor(
            inputWeightBankCount, [depthDim, inputDim, inputDim]
        )
        # 30 = 2 * 3 * 5 = depthDim * inputDim * outputDim
        outputWeightBankCount = depthDim * inputDim * outputDim
        outputWeightBank = L.createTensor(
            outputWeightBankCount, [depthDim, inputDim, outputDim]
        )

        # [depthDim, batchSize, inputDim]
        inputWeightVectors = L.tensorProduct(inputTensor, inputWeightBank)
        # [depthDim, batchSize, outputDim]
        outputWeightVectors = L.tensorProduct(inputTensor, outputWeightBank)

        inputWeightVectorsShape = L.shapeList(inputWeightVectors)
        expectedInputWeightVectorsShape = [depthDim, batchSize, inputDim]
        outputWeightVectorsShape = L.shapeList(outputWeightVectors)
        expectedOutputWeightVectorsShape = [depthDim, batchSize, outputDim]

        checkIfTheMultiplicationIsCorrect = False
        for i in range(depthDim):
            testInputVectors = L.matrixProduct(inputTensor, inputWeightBank[i])
            testOutputVectors = L.matrixProduct(inputTensor, outputWeightBank[i])

            if not L.areTensorsIdentical(testInputVectors, inputWeightVectors[i]):
                checkIfTheMultiplicationIsCorrect = True

            if not L.areTensorsIdentical(testOutputVectors, outputWeightVectors[i]):
                checkIfTheMultiplicationIsCorrect = True

        self.assertFalse(
            checkIfTheMultiplicationIsCorrect,
            "Generated weight vector are not correctly generated.",
        )
        self.assertEqual(inputWeightVectorsShape, expectedInputWeightVectorsShape)
        self.assertEqual(outputWeightVectorsShape, expectedOutputWeightVectorsShape)

    def testGenerateWeightsSteps(self):
        """
        Input tensor:
            [[0, 1, 2],
             [3, 4, 5]]

        Input Weight Bank:
            [[[ 0,  1,  2],
              [ 3,  4,  5],
              [ 6,  7,  8]],

             [[ 9, 10, 11],
              [12, 13, 14],
              [15, 16, 17]],

             [[18, 19, 20],
              [21, 22, 23],
              [24, 25, 26]],

             [[27, 28, 29],
              [30, 31, 32],
              [33, 34, 35]]]

        Input Weight Vectors:
            [[[ 15,  18,  21],
              [ 42,  54,  66]],

             [[ 42,  45,  48],
              [150, 162, 174]],

             [[ 69,  72,  75],
              [258, 270, 282]],

             [[ 96,  99, 102],
              [366, 378, 390]]]

        Output Weight Bank:
            [[[ 0,  1,  2,  3,  4],
              [ 5,  6,  7,  8,  9],
              [10, 11, 12, 13, 14]],

             [[15, 16, 17, 18, 19],
              [20, 21, 22, 23, 24],
              [25, 26, 27, 28, 29]],

             [[30, 31, 32, 33, 34],
              [35, 36, 37, 38, 39],
              [40, 41, 42, 43, 44]],

             [[45, 46, 47, 48, 49],
              [50, 51, 52, 53, 54],
              [55, 56, 57, 58, 59]]]

        Output Weight Vectors:
            [[[ 25,  28,  31,  34,  37],
              [ 70,  82,  94, 106, 118]],

             [[ 70,  73,  76,  79,  82],
              [250, 262, 274, 286, 298]],

             [[115, 118, 121, 124, 127],
              [430, 442, 454, 466, 478]],

             [[160, 163, 166, 169, 172],
              [610, 622, 634, 646, 658]]]

        Weight Vectors Outer Porduct
            [[[[   375,    420,    465,    510,    555],
               [   450,    504,    558,    612,    666],
               [   525,    588,    651,    714,    777]],

              [[  2940,   3444,   3948,   4452,   4956],
               [  3780,   4428,   5076,   5724,   6372],
               [  4620,   5412,   6204,   6996,   7788]]],

             [[[  2940,   3066,   3192,   3318,   3444],
               [  3150,   3285,   3420,   3555,   3690],
               [  3360,   3504,   3648,   3792,   3936]],

              [[ 37500,  39300,  41100,  42900,  44700],
               [ 40500,  42444,  44388,  46332,  48276],
               [ 43500,  45588,  47676,  49764,  51852]]],

             [[[  7935,   8142,   8349,   8556,   8763],
               [  8280,   8496,   8712,   8928,   9144],
               [  8625,   8850,   9075,   9300,   9525]],

              [[110940, 114036, 117132, 120228, 123324],
               [116100, 119340, 122580, 125820, 129060],
               [121260, 124644, 128028, 131412, 134796]]],

             [[[ 15360,  15648,  15936,  16224,  16512],
               [ 15840,  16137,  16434,  16731,  17028],
               [ 16320,  16626,  16932,  17238,  17544]],

              [[223260, 227652, 232044, 236436, 240828],
               [230580, 235116, 239652, 244188, 248724],
               [237900, 242580, 247260, 251940, 256620]]]]

        Weight generator:
            [[[ 26610,  27276,  27942,  28608,  29274],
              [ 27720,  28422,  29124,  29826,  30528],
              [ 28830,  29568,  30306,  31044,  31782]],

             [[374640, 384432, 394224, 404016, 413808],
              [390960, 401328, 411696, 422064, 432432],
              [407280, 418224, 429168, 440112, 451056]]]
        """

        inputDim = 3
        depthDim = 4
        outputDim = 5
        batchSize = 2

        # 6 = 2 * 3 = batchSize * inputDim
        inputTensor = L.createTensor(batchSize * inputDim, [batchSize, inputDim])

        # 18 = 2 * 3 * 3 = depthDim * inputDim * inputDim
        inputWeightBankCount = depthDim * inputDim * inputDim
        inputWeightBank = L.createTensor(
            inputWeightBankCount, [depthDim, inputDim, inputDim]
        )
        # 30 = 2 * 3 * 5 = depthDim * inputDim * outputDim
        outputWeightBankCount = depthDim * inputDim * outputDim
        outputWeightBank = L.createTensor(
            outputWeightBankCount, [depthDim, inputDim, outputDim]
        )

        # [depthDim, batchSize, inputDim]
        inputWeightVectors = L.tensorProduct(inputTensor, inputWeightBank)
        # [depthDim, batchSize, outputDim]
        outputWeightVectors = L.tensorProduct(inputTensor, outputWeightBank)

        # [depthDim, batchSize, inputDim, outputDim]
        weightVectorsOuterPorduct = L.einsum(
            "bij,bik->bijk", inputWeightVectors, outputWeightVectors
        )

        # [batchSize, inputDim, outputDim]
        generatedWeights = L.sum(weightVectorsOuterPorduct, dim=0)

        generatedWeightsShape = L.shapeList(generatedWeights)
        expectedGeneratedWeightsShape = [batchSize, inputDim, outputDim]

        checkIfTheOuterProductIsCorrect = False
        for i in range(depthDim):
            for j in range(batchSize):
                inputOuterProduct = L.vectorOuterProduct(
                    inputWeightVectors[i, j], outputWeightVectors[i, j]
                )

                if not L.areTensorsIdentical(
                    inputOuterProduct, weightVectorsOuterPorduct[i, j]
                ):
                    checkIfTheOuterProductIsCorrect = True

        self.assertEqual(generatedWeightsShape, expectedGeneratedWeightsShape)
        self.assertFalse(
            checkIfTheOuterProductIsCorrect,
            "Generated weights are not correctly generated.",
        )


class TestGeneratorChoiceMixtureClass(GeneralGeneratorMethods):
    def setUp(self) -> None:
        pass

    def testForwardMethod(self):
        config = ParameterGeneratorConfig()
        config.auxiliaryLosses = AuxiliaryLosses(config)
        model = GeneratorChoiceMixture(config)
        self.checkForwardMethod(model)

    def testIfWeightsAreUpdated(self):
        self.areWeightsUpdated(ParameterGeneratorOptions.generator_choice_mixture)

    def testIfSharedLayerWeightsAreUpdated(self):
        self.areWeightsUpdatedSharedLayer(
            ParameterGeneratorOptions.generator_choice_mixture
        )

    def testIfWeightsAreUpdatedMultiLayer(self):
        self.areWeightsUpdatedMultiLayer(
            ParameterGeneratorOptions.generator_choice_mixture
        )

    def testFullForwardProcess(self):
        """
        # Input Weight Bank:
            [[[ 0,  1,  2],
              [ 3,  4,  5],
              [ 6,  7,  8]],

             [[ 9, 10, 11],
              [12, 13, 14],
              [15, 16, 17]],

             [[18, 19, 20],
              [21, 22, 23],
              [24, 25, 26]],

             [[27, 28, 29],
              [30, 31, 32],
              [33, 34, 35]]]

        # Output Weight Bank:
            [[[ 0,  1,  2,  3,  4],
              [ 5,  6,  7,  8,  9],
              [10, 11, 12, 13, 14]],

             [[15, 16, 17, 18, 19],
              [20, 21, 22, 23, 24],
              [25, 26, 27, 28, 29]],

             [[30, 31, 32, 33, 34],
              [35, 36, 37, 38, 39],
              [40, 41, 42, 43, 44]],

             [[45, 46, 47, 48, 49],
              [50, 51, 52, 53, 54],
              [55, 56, 57, 58, 59]]]

        # Top k Probabilities:
            [[0.1000, 0.4000, 0.5000],
             [0.3000, 0.3000, 0.4000]]

        # Top k Indexes:
            [[1, 0, 2],
             [2, 3, 1]]

        # Selected Input Weights:
            [[[[ 9, 10, 11],
               [12, 13, 14],
               [15, 16, 17]],

              [[ 0,  1,  2],
               [ 3,  4,  5],
               [ 6,  7,  8]],

              [[18, 19, 20],
               [21, 22, 23],
               [24, 25, 26]]],


             [[[18, 19, 20],
               [21, 22, 23],
               [24, 25, 26]],

              [[27, 28, 29],
               [30, 31, 32],
               [33, 34, 35]],

              [[ 9, 10, 11],
               [12, 13, 14],
               [15, 16, 17]]]]

        # Selected Output Weights:
            [[[[15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24],
               [25, 26, 27, 28, 29]],

              [[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14]],

              [[30, 31, 32, 33, 34],
               [35, 36, 37, 38, 39],
               [40, 41, 42, 43, 44]]],

             [[[30, 31, 32, 33, 34],
               [35, 36, 37, 38, 39],
               [40, 41, 42, 43, 44]],

              [[45, 46, 47, 48, 49],
               [50, 51, 52, 53, 54],
               [55, 56, 57, 58, 59]],

              [[15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24],
               [25, 26, 27, 28, 29]]]]

        # Input Weight Vectors:
            [[[ 42,  45,  48],
              [ 15,  18,  21],
              [ 69,  72,  75]],

             [[258, 270, 282],
              [366, 378, 390],
              [150, 162, 174]]]

        # Selected Output Weights:
            [[[ 70,  73,  76,  79,  82],
              [ 25,  28,  31,  34,  37],
              [115, 118, 121, 124, 127]],

             [[430, 442, 454, 466, 478],
              [610, 622, 634, 646, 658],
              [250, 262, 274, 286, 298]]]

        # Top k Probabilities Reshaped:
            [[[[0.1000]],
              [[0.4000]],
              [[0.5000]]],

             [[[0.3000]],
              [[0.3000]],
              [[0.4000]]]]

        # Raw Generated Weights:
            [[[[  2940,   3066,   3192,   3318,   3444],
               [  3150,   3285,   3420,   3555,   3690],
               [  3360,   3504,   3648,   3792,   3936]],

              [[   375,    420,    465,    510,    555],
               [   450,    504,    558,    612,    666],
               [   525,    588,    651,    714,    777]],

              [[  7935,   8142,   8349,   8556,   8763],
               [  8280,   8496,   8712,   8928,   9144],
               [  8625,   8850,   9075,   9300,   9525]]],


             [[[110940, 114036, 117132, 120228, 123324],
               [116100, 119340, 122580, 125820, 129060],
               [121260, 124644, 128028, 131412, 134796]],

              [[223260, 227652, 232044, 236436, 240828],
               [230580, 235116, 239652, 244188, 248724],
               [237900, 242580, 247260, 251940, 256620]],

              [[ 37500,  39300,  41100,  42900,  44700],
               [ 40500,  42444,  44388,  46332,  48276],
               [ 43500,  45588,  47676,  49764,  51852]]]]

        # Weighted Raw Generated Weights (after being multiplied by weights):
            [[[[  294,   306,   319,   331,   344],
               [  315,   328,   342,   355,   369],
               [  336,   350,   364,   379,   393]],

              [[  150,   168,   186,   204,   222],
               [  180,   201,   223,   244,   266],
               [  210,   235,   260,   285,   310]],

              [[ 3967,  4071,  4174,  4278,  4381],
               [ 4140,  4248,  4356,  4464,  4572],
               [ 4312,  4425,  4537,  4650,  4762]]],


             [[[33282, 34210, 35139, 36068, 36997],
               [34830, 35802, 36774, 37746, 38718],
               [36378, 37393, 38408, 39423, 40438]],

              [[66978, 68295, 69613, 70930, 72248],
               [69174, 70534, 71895, 73256, 74617],
               [71370, 72774, 74178, 75582, 76986]],

              [[15000, 15720, 16440, 17160, 17880],
               [16200, 16977, 17755, 18532, 19310],
               [17400, 18235, 19070, 19905, 20740]]]]

        # Generated Weights (after topKexperts):
            [[[  4411,   4545,   4679,   4813,   4947],
              [  4635,   4778,   4921,   5064,   5207],
              [  4858,   5010,   5162,   5314,   5466]],

             [[115260, 118226, 121192, 124159, 127125],
              [120204, 123314, 126424, 129535, 132645],
              [125148, 128402, 131656, 134911, 138165]]]
        """
        inputDim = 3
        depthDim = 4
        outputDim = 5
        topKExperts = 3
        batchSize = 2

        input = L.createTensor(batchSize * inputDim, [batchSize, inputDim])

        # 36 = 4 * 3 * 3
        numInputWeights = depthDim * inputDim * inputDim
        # 60 = 4 * 3 * 5
        numOutputWeights = depthDim * inputDim * outputDim

        # [depthDim, inputDim, inputDim]
        inputWeightBank = L.createTensor(
            numInputWeights, [depthDim, inputDim, inputDim]
        )

        # [depthDim, inputDim, outputDim]
        outputWeightBank = L.createTensor(
            numOutputWeights, [depthDim, inputDim, outputDim]
        )

        # [batchSize, topK]
        topKProbabilities = L.toTensor([[0.1, 0.4, 0.5], [0.3, 0.3, 0.4]])
        # [batchSize, topK]
        topkIndexes = L.toTensor([[1, 0, 2], [2, 3, 1]])

        # [batchSize, topKExperts, inputDim, inputDim]
        selectedInputWeights = inputWeightBank[topkIndexes]
        # [batchSize, topKExperts, inputDim, outputDim]
        selectedOutputWeights = outputWeightBank[topkIndexes]

        # [batchSize, topKExperts, inputDim]
        inputWeightVectors = L.einsum("bi,bkij->bkj", input, selectedInputWeights)
        # [batchSize, topKExperts, outputDim]
        outputWeightVectors = L.einsum("bi,bkij->bkj", input, selectedOutputWeights)

        # [depthDim, batchSize, outputDim]
        # inputWeightVectorsNormalized = L.normalize(inputWeightVectors)
        # [depthDim, batchSize, outputDim]
        # outputWeightVectorsNormalized = L.normalize(outputWeightVectors)

        # [batchSize, topKExperts, 1, 1]
        topKProbabilitiesReshaped = L.reshape(
            topKProbabilities, [batchSize, topKExperts, 1, 1]
        )

        # TODO: write a loop to confirm that einsum calculates
        # weights correctly

        # [batchSize, topKExperts, inputDim, outputDim]
        generatedWeightsRaw = L.einsum(
            "bij,bik->bijk", inputWeightVectors, outputWeightVectors
        )

        # [batchSize, topKExperts, inputDim, outputDim]
        weightedGeneratedWeightsRaw = generatedWeightsRaw * topKProbabilitiesReshaped

        # [batchSize, inputDim, outputDim]
        generatedWeights = weightedGeneratedWeightsRaw.sum(dim=1)

        checkWeightVectorsFlag = False
        for batchIndex in range(batchSize):
            for topkIndex in range(topKExperts):
                expectedInputWeightVector = L.tensorProduct(
                    input[batchIndex], selectedInputWeights[batchIndex, topkIndex]
                )
                expectedOutputWeightVector = L.tensorProduct(
                    input[batchIndex], selectedOutputWeights[batchIndex, topkIndex]
                )
                einsumInputWeightVector = inputWeightVectors[batchIndex, topkIndex]
                einsumOutputWeightVector = outputWeightVectors[batchIndex, topkIndex]

                checkInputWeightVectors = L.areTensorsIdentical(
                    expectedInputWeightVector, einsumInputWeightVector
                )
                checkOutputWeightVectors = L.areTensorsIdentical(
                    expectedOutputWeightVector, einsumOutputWeightVector
                )

                checkIfCorrectlyGenerated = (
                    checkInputWeightVectors and checkOutputWeightVectors
                )

                if checkIfCorrectlyGenerated:
                    checkWeightVectorsFlag = True
                    break

            if not checkWeightVectorsFlag:
                break

        checkGeneratedWeightsRawFlag = False
        for batchIndex in range(batchSize):
            for topkIndex in range(topKExperts):
                expectedGeneratedWeightsRaw = L.vectorOuterProduct(
                    inputWeightVectors[batchIndex, topkIndex],
                    outputWeightVectors[batchIndex, topkIndex],
                )
                einsumGeneratedWeightsRaw = generatedWeightsRaw[batchIndex, topkIndex]

                checkInputWeightVectors = L.areTensorsIdentical(
                    einsumGeneratedWeightsRaw, expectedGeneratedWeightsRaw
                )

                if checkInputWeightVectors:
                    checkGeneratedWeightsRawFlag = True
                    break

            if not checkGeneratedWeightsRawFlag:
                break

        self.assertTrue(
            checkWeightVectorsFlag,
            "Weight Vectors of GeneratorMixture are not correclty generated by einsum",
        )

        self.assertTrue(
            checkGeneratedWeightsRawFlag,
            "Weight Vectors of GeneratorMixture are not correclty generated by einsum",
        )

        checkShowProcess = False
        if checkShowProcess:
            print("\n# Input Weight Bank: \n", inputWeightBank)
            print("\n# Output Weight Bank: \n", outputWeightBank)
            print("\n# Top k Probabilities: \n", topKProbabilities)
            print("\n# Top k Indexes: \n", topkIndexes)
            print("\n# Selected Input Weights: \n", selectedInputWeights)
            print("\n# Selected Output Weights: \n", selectedOutputWeights)
            print("\n# Input Weight Vectors: \n", inputWeightVectors)
            print("\n# Selected Output Weights: \n", outputWeightVectors)
            print("\n# Top k Probabilities Reshaped: \n", topKProbabilitiesReshaped)
            print("\n# Raw Generated Weights: \n", generatedWeightsRaw)
            print("\n# Weighted Raw Generated Weights: \n", weightedGeneratedWeightsRaw)
            print("\n# Generated Weights (after topKexperts): \n", generatedWeights)


class TestGeneratorChoiceSoftMixtureClass(GeneralGeneratorMethods):
    def setUp(self) -> None:
        pass

    def testForwardMethod(self):
        config = ParameterGeneratorConfig()
        config.auxiliaryLosses = AuxiliaryLosses(config)
        model = GeneratorChoiceSoftMixture(config)
        self.checkForwardMethod(model)

    def testIfWeightsAreUpdated(self):
        self.areWeightsUpdated(ParameterGeneratorOptions.generator_choice_soft_mixture)

    def testIfSharedLayerWeightsAreUpdated(self):
        self.areWeightsUpdatedSharedLayer(
            ParameterGeneratorOptions.generator_choice_soft_mixture
        )

    def testIfWeightsAreUpdatedMultiLayer(self):
        self.areWeightsUpdatedMultiLayer(
            ParameterGeneratorOptions.generator_choice_soft_mixture
        )

    def testFullForwardProcess(self):
        """
        # Input Weight Bank:
            tensor([[[ 0,  1,  2],
                     [ 3,  4,  5],
                     [ 6,  7,  8]],

                    [[ 9, 10, 11],
                     [12, 13, 14],
                     [15, 16, 17]],

                    [[18, 19, 20],
                     [21, 22, 23],
                     [24, 25, 26]],

                    [[27, 28, 29],
                     [30, 31, 32],
                     [33, 34, 35]]])

        # Output Weight Bank:
            tensor([[[ 0,  1,  2,  3,  4],
                     [ 5,  6,  7,  8,  9],
                     [10, 11, 12, 13, 14]],

                    [[15, 16, 17, 18, 19],
                     [20, 21, 22, 23, 24],
                     [25, 26, 27, 28, 29]],

                    [[30, 31, 32, 33, 34],
                     [35, 36, 37, 38, 39],
                     [40, 41, 42, 43, 44]],

                    [[45, 46, 47, 48, 49],
                     [50, 51, 52, 53, 54],
                     [55, 56, 57, 58, 59]]])

        # Probabilities:
            tensor([[0.1000, 0.4000, 0.3000, 0.2000],
                    [0.3000, 0.3000, 0.3000, 0.1000]])

        # Input Weight Vectors:
            tensor([[[ 15,  18,  21],
                     [ 42,  45,  48],
                     [ 69,  72,  75],
                     [ 96,  99, 102]],

                    [[ 42,  54,  66],
                     [150, 162, 174],
                     [258, 270, 282],
                     [366, 378, 390]]])

        # Output Weight Vectors:
            tensor([[[ 25,  28,  31,  34,  37],
                     [ 70,  73,  76,  79,  82],
                     [115, 118, 121, 124, 127],
                     [160, 163, 166, 169, 172]],

                    [[ 70,  82,  94, 106, 118],
                     [250, 262, 274, 286, 298],
                     [430, 442, 454, 466, 478],
                     [610, 622, 634, 646, 658]]])

        # Probabilities Reshaped:
            tensor([[[[0.1]],
                     [[0.4]],
                     [[0.3]],
                     [[0.2]]],

                    [[[0.3000]],
                     [[0.3000]],
                     [[0.3000]],
                     [[0.1000]]]])

        # Raw Generated Weights:
            tensor([[[[   375,    420,    465,    510,    555],
                      [   450,    504,    558,    612,    666],
                      [   525,    588,    651,    714,    777]],

                     [[  2940,   3066,   3192,   3318,   3444],
                      [  3150,   3285,   3420,   3555,   3690],
                      [  3360,   3504,   3648,   3792,   3936]],

                     [[  7935,   8142,   8349,   8556,   8763],
                      [  8280,   8496,   8712,   8928,   9144],
                      [  8625,   8850,   9075,   9300,   9525]],

                     [[ 15360,  15648,  15936,  16224,  16512],
                      [ 15840,  16137,  16434,  16731,  17028],
                      [ 16320,  16626,  16932,  17238,  17544]]],


                    [[[  2940,   3444,   3948,   4452,   4956],
                      [  3780,   4428,   5076,   5724,   6372],
                      [  4620,   5412,   6204,   6996,   7788]],

                     [[ 37500,  39300,  41100,  42900,  44700],
                      [ 40500,  42444,  44388,  46332,  48276],
                      [ 43500,  45588,  47676,  49764,  51852]],

                     [[110940, 114036, 117132, 120228, 123324],
                      [116100, 119340, 122580, 125820, 129060],
                      [121260, 124644, 128028, 131412, 134796]],

                     [[223260, 227652, 232044, 236436, 240828],
                      [230580, 235116, 239652, 244188, 248724],
                      [237900, 242580, 247260, 251940, 256620]]]])

        # Weighted Raw Generated Weights:
            tensor([[[[3.7500e+01, 4.2000e+01, 4.6500e+01, 5.1000e+01, 5.5500e+01],
                      [4.5000e+01, 5.0400e+01, 5.5800e+01, 6.1200e+01, 6.6600e+01],
                      [5.2500e+01, 5.8800e+01, 6.5100e+01, 7.1400e+01, 7.7700e+01]],

                     [[1.1760e+03, 1.2264e+03, 1.2768e+03, 1.3272e+03, 1.3776e+03],
                      [1.2600e+03, 1.3140e+03, 1.3680e+03, 1.4220e+03, 1.4760e+03],
                      [1.3440e+03, 1.4016e+03, 1.4592e+03, 1.5168e+03, 1.5744e+03]],

                     [[2.3805e+03, 2.4426e+03, 2.5047e+03, 2.5668e+03, 2.6289e+03],
                      [2.4840e+03, 2.5488e+03, 2.6136e+03, 2.6784e+03, 2.7432e+03],
                      [2.5875e+03, 2.6550e+03, 2.7225e+03, 2.7900e+03, 2.8575e+03]],

                     [[3.0720e+03, 3.1296e+03, 3.1872e+03, 3.2448e+03, 3.3024e+03],
                      [3.1680e+03, 3.2274e+03, 3.2868e+03, 3.3462e+03, 3.4056e+03],
                      [3.2640e+03, 3.3252e+03, 3.3864e+03, 3.4476e+03, 3.5088e+03]]],


                    [[[8.8200e+02, 1.0332e+03, 1.1844e+03, 1.3356e+03, 1.4868e+03],
                      [1.1340e+03, 1.3284e+03, 1.5228e+03, 1.7172e+03, 1.9116e+03],
                      [1.3860e+03, 1.6236e+03, 1.8612e+03, 2.0988e+03, 2.3364e+03]],

                     [[1.1250e+04, 1.1790e+04, 1.2330e+04, 1.2870e+04, 1.3410e+04],
                      [1.2150e+04, 1.2733e+04, 1.3316e+04, 1.3900e+04, 1.4483e+04],
                      [1.3050e+04, 1.3676e+04, 1.4303e+04, 1.4929e+04, 1.5556e+04]],

                     [[3.3282e+04, 3.4211e+04, 3.5140e+04, 3.6068e+04, 3.6997e+04],
                      [3.4830e+04, 3.5802e+04, 3.6774e+04, 3.7746e+04, 3.8718e+04],
                      [3.6378e+04, 3.7393e+04, 3.8408e+04, 3.9424e+04, 4.0439e+04]],

                     [[2.2326e+04, 2.2765e+04, 2.3204e+04, 2.3644e+04, 2.4083e+04],
                      [2.3058e+04, 2.3512e+04, 2.3965e+04, 2.4419e+04, 2.4872e+04],
                      [2.3790e+04, 2.4258e+04, 2.4726e+04, 2.5194e+04, 2.5662e+04]]]])

        # Generated Weights (after topKexperts):
            tensor([[[ 6666,  6840,  7015,  7189,  7364],
                     [ 6957,  7140,  7324,  7507,  7691],
                     [ 7248,  7440,  7633,  7825,  8018]],

                    [[67740, 69799, 71858, 73917, 75976],
                     [71172, 73375, 75578, 77781, 79984],
                     [74604, 76951, 79298, 81645, 83992]]])
        """
        inputDim = 3
        depthDim = 4
        outputDim = 5
        topKExperts = 3
        batchSize = 2

        input = L.createTensor(batchSize * inputDim, [batchSize, inputDim])

        # 36 = 4 * 3 * 3
        numInputWeights = depthDim * inputDim * inputDim
        # 60 = 4 * 3 * 5
        numOutputWeights = depthDim * inputDim * outputDim

        # [depthDim, inputDim, inputDim]
        inputWeightBank = L.createTensor(
            numInputWeights, [depthDim, inputDim, inputDim]
        )

        # [depthDim, inputDim, outputDim]
        outputWeightBank = L.createTensor(
            numOutputWeights, [depthDim, inputDim, outputDim]
        )

        # [batchSize, topK]
        probabilities = L.toTensor([[0.1, 0.4, 0.3, 0.2], [0.3, 0.3, 0.3, 0.1]])

        # [batchSize, depthDim, inputDim]
        inputWeightVectors = L.einsum("bi,kij->bkj", input, inputWeightBank)
        # [batchSize, depthDim, outputDim]
        outputWeightVectors = L.einsum("bi,kij->bkj", input, outputWeightBank)

        # [depthDim, batchSize, outputDim]
        # inputWeightVectorsNormalized = L.normalize(inputWeightVectors)
        # [depthDim, batchSize, outputDim]
        # outputWeightVectorsNormalized = L.normalize(outputWeightVectors)

        # [batchSize, depthDim, 1, 1]
        probabilitiesReshaped = L.reshape(probabilities, [batchSize, depthDim, 1, 1])

        # [batchSize, depthDim, inputDim, outputDim]
        generatedWeightsRaw = L.einsum(
            "bij,bik->bijk", inputWeightVectors, outputWeightVectors
        )

        # [batchSize, depthDim, inputDim, outputDim]
        weightedGeneratedWeightsRaw = generatedWeightsRaw * probabilitiesReshaped

        # [batchSize, inputDim, outputDim]
        generatedWeights = L.sum(weightedGeneratedWeightsRaw, dim=1)

        checkWeightVectorsFlag = False
        for batchIndex in range(batchSize):
            for depthIndex in range(topKExperts):
                expectedInputWeightVector = L.tensorProduct(
                    input[batchIndex], inputWeightBank[depthIndex]
                )
                expectedOutputWeightVector = L.tensorProduct(
                    input[batchIndex], outputWeightBank[depthIndex]
                )
                einsumInputWeightVector = inputWeightVectors[batchIndex, depthIndex]
                einsumOutputWeightVector = outputWeightVectors[batchIndex, depthIndex]

                checkInputWeightVectors = L.areTensorsIdentical(
                    expectedInputWeightVector, einsumInputWeightVector
                )
                checkOutputWeightVectors = L.areTensorsIdentical(
                    expectedOutputWeightVector, einsumOutputWeightVector
                )

                checkIfCorrectlyGenerated = (
                    checkInputWeightVectors and checkOutputWeightVectors
                )

                if checkIfCorrectlyGenerated:
                    checkWeightVectorsFlag = True
                    break

            if not checkWeightVectorsFlag:
                break

        checkGeneratedWeightsRawFlag = False
        for batchIndex in range(batchSize):
            for depthIndex in range(topKExperts):
                expectedGeneratedWeightsRaw = L.vectorOuterProduct(
                    inputWeightVectors[batchIndex, depthIndex],
                    outputWeightVectors[batchIndex, depthIndex],
                )
                einsumGeneratedWeightsRaw = generatedWeightsRaw[batchIndex, depthIndex]

                checkInputWeightVectors = L.areTensorsIdentical(
                    einsumGeneratedWeightsRaw, expectedGeneratedWeightsRaw
                )

                if checkInputWeightVectors:
                    checkGeneratedWeightsRawFlag = True
                    break

            if not checkGeneratedWeightsRawFlag:
                break

        self.assertTrue(
            checkWeightVectorsFlag,
            "Weight Vectors of GeneratorMixture are not correclty generated by einsum",
        )

        self.assertTrue(
            checkGeneratedWeightsRawFlag,
            "Weight Vectors of GeneratorMixture are not correclty generated by einsum",
        )

        checkShowProcess = False
        if checkShowProcess:
            print("\n# Input Weight Bank: \n", inputWeightBank)
            print("\n# Output Weight Bank: \n", outputWeightBank)
            print("\n# Probabilities: \n", probabilities)
            print("\n# Input Weight Vectors: \n", inputWeightVectors)
            print("\n# Output Weight Vectors: \n", outputWeightVectors)
            print("\n# Probabilities Reshaped: \n", probabilitiesReshaped)
            print("\n# Raw Generated Weights: \n", generatedWeightsRaw)
            print("\n# Weighted Raw Generated Weights: \n", weightedGeneratedWeightsRaw)
            print("\n# Generated Weights (after topKexperts): \n", generatedWeights)
