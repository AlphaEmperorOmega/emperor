from .general import GeneralTestMethods

from Emperor.library.choice import Library as L
from Emperor.components.parameter_generators.utils.losses import AuxiliaryLosses
from Emperor.components.parameter_generators.vector_choice import (
    VectorChoiceMixture,
    VectorChoiceSparse,
    VectorChoiceSoftMixture,
)
from Emperor.config import (
    ParameterGeneratorConfig,
    ParameterGeneratorOptions,
)


class TestVectorChoiceSparseClass(GeneralTestMethods):
    def setUp(self) -> None:
        pass

    def testForwardMethod(self):
        config = ParameterGeneratorConfig()
        config.auxiliaryLosses = AuxiliaryLosses(config)
        model = VectorChoiceSparse(config)
        self.checkForwardMethod(model, True)

    def testIfWeightsAreUpdated(self):
        self.areWeightsUpdated(ParameterGeneratorOptions.vector_choice_sparse)

    def testIfSharedLayerWeightsAreUpdated(self):
        self.areWeightsUpdatedSharedLayer(
            ParameterGeneratorOptions.vector_choice_sparse
        )

    def testIfWeightsAreUpdatedMultiLayer(self):
        self.areWeightsUpdatedMultiLayer(ParameterGeneratorOptions.vector_choice_sparse)

    def testSelectWeightsMethodSteps(self):
        """
        Weight bank:
            [[[ 0,  1,  2,  3,  4],
              [ 5,  6,  7,  8,  9],
              [10, 11, 12, 13, 14],
              [15, 16, 17, 18, 19]],

             [[20, 21, 22, 23, 24],
              [25, 26, 27, 28, 29],
              [30, 31, 32, 33, 34],
              [35, 36, 37, 38, 39]],

             [[40, 41, 42, 43, 44],
              [45, 46, 47, 48, 49],
              [50, 51, 52, 53, 54],
              [55, 56, 57, 58, 59]]]

        Input indexes:
            [[1, 2],
             [2, 0],
             [1, 3]]

        Choice range:
            [[0, 1, 2]]

        STEPS:
        Transposed indexes:
            [[1, 2, 1],
             [2, 0, 3]]

        Selected weights:
            [[[ 5,  6,  7,  8,  9],
              [30, 31, 32, 33, 34],
              [45, 46, 47, 48, 49]],

             [[10, 11, 12, 13, 14],
              [20, 21, 22, 23, 24],
              [55, 56, 57, 58, 59]]]
        """
        inputDim = 3
        depthDim = 4
        outputDim = 5
        batchSize = 2

        # [inputDim, batchSize]
        inputIndexes = L.toTensor([[1, 2], [2, 0], [1, 3]])
        # [inputDim, depthDim, outputDim]
        weightBank = L.createTensor(60, [inputDim, depthDim, outputDim])
        # [batchSize, inputDim]
        choiceRange = L.reshape(L.arange(inputDim), [1, inputDim])

        # [batchSize, inputDim]
        transposedIndexes = L.transpose(inputIndexes, [1, 0])
        # [inputDim, batchSize, outputDim]
        selectedWeights = weightBank[choiceRange, transposedIndexes]

        # CHECKS
        selectedWeightsShape = L.shapeList(selectedWeights)
        expectedSelectedWeightsShape = [batchSize, inputDim, outputDim]

        self.assertEqual(selectedWeightsShape, expectedSelectedWeightsShape)

    def testSelectBiasesMethodSteps(self):
        """
        Weight bank:
            [[ 0,  1,  2,  3],
             [ 4,  5,  6,  7],
             [ 8,  9, 10, 11],
             [12, 13, 14, 15],
             [16, 17, 18, 19]]

        Input indexes:
            [[1, 2],
             [2, 0],
             [1, 3],
             [3, 0],
             [2, 3]]

        Choice range:
            [[0, 1, 2, 3, 4]]

        STEPS:
        Transposed indexes:
            [[1, 2, 1, 3, 2],
             [2, 0, 3, 0, 3]]

        Selected weights:
            [[ 1,  6,  9, 15, 18],
             [ 2,  4, 11, 12, 19]]
        """
        depthDim = 4
        outputDim = 5
        batchSize = 2

        # [inputDim, batchSize]
        inputIndexes = L.toTensor([[1, 2], [2, 0], [1, 3], [3, 0], [2, 3]])
        # [inputDim, depthDim, outputDim]
        biasBank = L.createTensor(20, [outputDim, depthDim])
        # [batchSize, inputDim]
        choiceRange = L.reshape(L.arange(outputDim), [1, outputDim])
        # [batchSize, inputDim]
        transposedIndexes = L.transpose(inputIndexes, [1, 0])
        # [inputDim, batchSize, outputDim]
        selectedBias = biasBank[choiceRange, transposedIndexes]

        # CHECKS
        selectedWeightsShape = L.shapeList(selectedBias)
        expectedSelectedWeightsShape = [batchSize, outputDim]

        self.assertEqual(selectedWeightsShape, expectedSelectedWeightsShape)


class TestVectorChoiceMixtureClass(GeneralTestMethods):
    def setUp(self) -> None:
        pass

    def testForwardMethod(self):
        config = ParameterGeneratorConfig()
        config.auxiliaryLosses = AuxiliaryLosses(config)
        model = VectorChoiceMixture(config)
        self.checkForwardMethod(model)

    def testIfWeightsAreUpdated(self):
        self.areWeightsUpdated(ParameterGeneratorOptions.vector_choice_mixture)

    def testIfSharedLayerWeightsAreUpdated(self):
        self.areWeightsUpdatedSharedLayer(
            ParameterGeneratorOptions.vector_choice_mixture
        )

    def testIfWeightsAreUpdatedMultiLayer(self):
        self.areWeightsUpdatedMultiLayer(
            ParameterGeneratorOptions.vector_choice_mixture
        )

    def testSelectWeightsMethodSteps(self):
        """
        Weight bank:
            [[[ 0,  1,  2,  3,  4],
              [ 5,  6,  7,  8,  9],
              [10, 11, 12, 13, 14],
              [15, 16, 17, 18, 19]],

             [[20, 21, 22, 23, 24],
              [25, 26, 27, 28, 29],
              [30, 31, 32, 33, 34],
              [35, 36, 37, 38, 39]],

             [[40, 41, 42, 43, 44],
              [45, 46, 47, 48, 49],
              [50, 51, 52, 53, 54],
              [55, 56, 57, 58, 59]]]

        Input indexes:
            [[[1, 0, 2],
              [2, 3, 1]],

             [[2, 1, 3],
              [0, 1, 3]],

             [[1, 3, 0],
              [3, 2, 0]]]

        Probabilities:
            [[[0.1, 0.4, 0.5],
              [0.3, 0.3, 0.4]],

             [[0.6, 0.2, 0.2],
              [0.3, 0.4, 0.3]],

             [[0.1, 0.7, 0.2],
              [0.3, 0.6, 0.1]]]

        Choice range:
            [[[0],
              [1],
              [2]]]

        Selected Weights muliplied by their probabilities:
            [[[[ 5,  6,  7,  8,  9],        [[[[0.1],
               [ 0,  1,  2,  3,  4],           [0.4],
               [10, 11, 12, 13, 14]],          [0.5]],

              [[30, 31, 32, 33, 34],          [[0.6],
               [25, 26, 27, 28, 29],           [0.2],
               [35, 36, 37, 38, 39]],          [0.2]],

              [[45, 46, 47, 48, 49],          [[0.1],
               [55, 56, 57, 58, 59],           [0.7],
               [40, 41, 42, 43, 44]]],         [0.2]]],
                                         *
             [[[10, 11, 12, 13, 14],         [[[0.3],
               [15, 16, 17, 18, 19],           [0.3],
               [ 5,  6,  7,  8,  9]],          [0.4]],

              [[20, 21, 22, 23, 24],          [[0.3],
               [25, 26, 27, 28, 29],           [0.4],
               [35, 36, 37, 38, 39]],          [0.3]],

              [[55, 56, 57, 58, 59],          [[0.3],
               [50, 51, 52, 53, 54],           [0.6],
               [40, 41, 42, 43, 44]]]]         [0.1]]]]

        Weighted selected weights
            [[[[ 0.5,  0.6,  0.7,  0.8,  0.9],
               [ 0.0,  0.4,  0.8,  1.2,  1.6],
               [ 5.0,  5.5,  6.0,  6.5,  7.0]],

              [[18.0, 18.6, 19.2, 19.8, 20.4],
               [ 5.0,  5.2,  5.4,  5.6,  5.8],
               [ 7.0,  7.2,  7.4,  7.6,  7.8]],

              [[ 4.5,  4.6,  4.7,  4.8,  4.9],
               [38.5, 39.2, 39.9, 40.6, 41.3],
               [ 8.0,  8.2,  8.4,  8.6,  8.8]]],

             [[[ 3.0,  3.3,  3.6,  3.9,  4.2],
               [ 4.5,  4.8,  5.1,  5.4,  5.7],
               [ 2.0,  2.4,  2.8,  3.2,  3.6]],

              [[ 6.0,  6.3,  6.6,  6.9,  7.2],
               [10.0, 10.4, 10.8, 11.2, 11.6],
               [10.5, 10.8, 11.1, 11.4, 11.7]],

              [[16.5, 16.8, 17.1, 17.4, 17.7],
               [30.0, 30.6, 31.2, 31.8, 32.4],
               [ 4.0,  4.1,  4.2,  4.3,  4.4]]]]

        Mixed weights
            [[[ 5.5,  6.5,  7.5,  8.5,  9.5],
              [30.0, 31.0, 32.0, 33.0, 34.0],
              [51.0, 52.0, 53.0, 54.0, 55.0]],

             [[ 9.5, 10.5, 11.5, 12.5, 13.5],
              [26.5, 27.5, 28.5, 29.5, 30.5],
              [50.5, 51.5, 52.5, 53.5, 54.5]]]
        """
        inputDim = 3
        depthDim = 4
        outputDim = 5
        topKExperts = 3
        batchSize = 2

        weightBank = L.createTensor(
            60, [inputDim, depthDim, outputDim]
        )  # [inputDim, depthDim, outputDim]

        choiceRange = L.reshape(
            L.arange(inputDim), [1, inputDim, 1]
        )  # [batchSize, inputDim]

        topkIndexes = L.toTensor(
            [[[1, 0, 2], [2, 3, 1]], [[2, 1, 3], [0, 1, 3]], [[1, 3, 0], [3, 2, 0]]]
        )  # [inputDim, batchSize, topKExperts]
        topKProbabilities = L.toTensor(
            [
                [[0.1, 0.4, 0.5], [0.3, 0.3, 0.4]],
                [[0.6, 0.2, 0.2], [0.3, 0.4, 0.3]],
                [[0.1, 0.7, 0.2], [0.3, 0.6, 0.1]],
            ]
        )  # [inputDim, batchSize, topKExperts]

        # Operation performed method: selectWeights
        transposedTopkIndexes = L.transpose(topkIndexes, [1, 0])
        # [batchSize, inputDim, topKExperts, outputDim]
        selectedWeights = weightBank[choiceRange, transposedTopkIndexes]

        # Operation performed method: sumSelectedWeightMixture
        transposedProbabilities = L.transpose(
            topKProbabilities, [1, 0]
        )  # [batchSize, inputDim, topKExperts]
        reshapedProbabilities = L.unsqueeze(
            transposedProbabilities, dim=-1
        )  # [batchSize, inputDim, topKExperts, 1]

        selectedWeightsWeighted = (
            selectedWeights * reshapedProbabilities
        )  # [batchSize, inputDim, topKExperts, outputDim]
        weightMixture = L.sum(
            selectedWeightsWeighted, dim=-2
        )  # [batchSize, inputDim, outputDim]

        # CHECKS
        selectedWeightsShape = L.shapeList(selectedWeights)
        expectedSelectedWeightsShape = [batchSize, inputDim, topKExperts, outputDim]
        weightMixtureShape = L.shapeList(weightMixture)
        expectedWeightsMixtureShape = [batchSize, inputDim, outputDim]

        self.assertEqual(weightMixtureShape, expectedWeightsMixtureShape)
        self.assertEqual(selectedWeightsShape, expectedSelectedWeightsShape)


class TestVectorChoiceSoftMixtureClass(GeneralTestMethods):
    def setUp(self) -> None:
        pass

    def testForwardMethod(self):
        config = ParameterGeneratorConfig()
        config.auxiliaryLosses = AuxiliaryLosses(config)
        model = VectorChoiceSoftMixture(config)
        self.checkForwardMethod(model)

    def testIfWeightsAreUpdated(self):
        self.areWeightsUpdated(ParameterGeneratorOptions.vector_choice_soft_mixture)

    def testIfSharedLayerWeightsAreUpdated(self):
        self.areWeightsUpdatedSharedLayer(
            ParameterGeneratorOptions.vector_choice_soft_mixture
        )

    def testIfWeightsAreUpdatedMultiLayer(self):
        self.areWeightsUpdatedMultiLayer(
            ParameterGeneratorOptions.vector_choice_soft_mixture
        )

    def testMaskProbsBasedOnTopKTresholdSteps(self):
        """
        Probabilities:
            [[[0.0, 0.4, 0.5, 0.0],
              [1.0, 0.0, 0.0, 0.0]],
             [[0.0, 0.5, 0.0, 0.5],
              [0.2, 0.4, 0.1, 0.3]],
             [[0.1, 0.7, 0.1, 0.1],
              [0.3, 0.1, 0.3, 0.3]]]

        Probabilities < 0.4 = True
            [[[ True, False, False,  True],
              [False,  True,  True,  True]],

             [[ True, False,  True, False],
              [ True, False,  True,  True]],

             [[ True, False,  True,  True],
              [ True,  True,  True,  True]]]

        1e-6 is replaced probabilties where the mask is True
            [[[1.e-6, 0.400, 0.500, 1.e-6],
              [1.000, 1.e-6, 0.000, 1.e-6]],

             [[1.e-6, 0.500, 1.e-6, 0.500],
              [1.e-6, 0.400, 1.e-6, 1.e-6]],

             [[1.e-6, 0.700, 1.e-6, 1.e-6],
              [1.e-6, 1.e-6, 1.e-6, 1.e-6]]]

        New probabilities on the maksed probabilities
            [[[0.0, 0.4, 0.5, 0.0],
              [1.0, 0.0, 0.0, 0.0]],

             [[0.0, 0.5, 0.0, 0.5],
              [0.0, 1.0, 0.0, 0.0]],

             [[0.0, 1.0, 0.0, 0.0],
              [0.2, 0.2, 0.2, 0.2]]]

        WARNING: of `softmax` is used this is what the probabilities will be
        this is because of operatios done for numerical stability
            [[[0.1945, 0.2902, 0.3207, 0.1945],
              [0.4754, 0.1749, 0.1749, 0.1749]],

             [[0.1888, 0.3112, 0.1888, 0.3112],
              [0.2226, 0.3321, 0.2226, 0.2226]],

             [[0.1995, 0.4016, 0.1995, 0.1995],
              [0.2500, 0.2500, 0.2500, 0.2500]]]
        """

        inputDim = 3
        depthDim = 4
        batchSize = 2

        topKTreshold = 0.4
        # [batchSize, inputDim, outputDim]
        probabilities = L.toTensor(
            [
                [[0.0, 0.4, 0.5, 0.0], [1.0, 0.0, 0.0, 0.0]],
                [[0.0, 0.5, 0.0, 0.5], [0.2, 0.4, 0.1, 0.3]],
                [[0.1, 0.7, 0.1, 0.1], [0.3, 0.1, 0.3, 0.3]],
            ]
        )

        # [inputDim, batchSize, depthDim]
        tresholdTopKMask = probabilities < topKTreshold
        # [inputDim, batchSize, depthDim]
        maskedProbabilities = L.where(tresholdTopKMask, 1e-6, probabilities)
        # [inputDim, batchSize, depthDim]
        customSoftmax = maskedProbabilities / (
            L.sum(maskedProbabilities, -1, keepdim=True) + 1e-6
        )

        customSoftmaxShape = L.shapeList(customSoftmax)
        expectedcustomSoftmaxShape = [inputDim, batchSize, depthDim]

        self.assertEqual(customSoftmaxShape, expectedcustomSoftmaxShape)

    def testSumWeightBankSoftMixtureSteps(self):
        """
        Weight bank:
            [[[ 0,  1,  2,  3,  4],
              [ 5,  6,  7,  8,  9],
              [10, 11, 12, 13, 14],
              [15, 16, 17, 18, 19]],

             [[20, 21, 22, 23, 24],
              [25, 26, 27, 28, 29],
              [30, 31, 32, 33, 34],
              [35, 36, 37, 38, 39]],

             [[40, 41, 42, 43, 44],
              [45, 46, 47, 48, 49],
              [50, 51, 52, 53, 54],
              [55, 56, 57, 58, 59]]]

        Probabilities:
            [[[0.0, 0.4, 0.5, 0.0],
              [1.0, 0.0, 0.0, 0.0]],
             [[0.0, 0.5, 0.0, 0.5],
              [0.2, 0.4, 0.1, 0.3]],
             [[0.1, 0.7, 0.1, 0.1],
              [0.3, 0.1, 0.3, 0.3]]]

        Probabilities transposed:
            [[[0.0, 0.4, 0.5, 0.0],
              [0.0, 0.5, 0.0, 0.5],
              [0.1, 0.7, 0.1, 0.1]],

             [[1.0, 0.0, 0.0, 0.0],
              [0.2, 0.4, 0.1, 0.3],
              [0.3, 0.1, 0.3, 0.3]]]

        Weight bank multiplied by reshaped probabilities:
             [[[ 0,  1,  2,  3,  4]    [[[[0.0],
               [ 5,  6,  7,  8,  9],      [0.4],
               [10, 11, 12, 13, 14],      [0.5],
               [15, 16, 17, 18, 19]],     [0.0]],

              [[20, 21, 22, 23, 24],     [[0.0],
               [25, 26, 27, 28, 29],      [0.5],
               [30, 31, 32, 33, 34],      [0.0],
               [35, 36, 37, 38, 39]],     [0.5]],

             [[40, 41, 42, 43, 44],      [[0.1],
               [45, 46, 47, 48, 49],      [0.7],
               [50, 51, 52, 53, 54],      [0.1],
               [55, 56, 57, 58, 59]]]     [0.1]]],
                                      *
             [[[ 0,  1,  2,  3,  4],    [[[1.0],
               [ 5,  6,  7,  8,  9],      [0.0],
               [10, 11, 12, 13, 14],      [0.0],
               [15, 16, 17, 18, 19]],     [0.0]],

              [[20, 21, 22, 23, 24],     [[0.2],
               [25, 26, 27, 28, 29],      [0.4],
               [30, 31, 32, 33, 34],      [0.1],
               [35, 36, 37, 38, 39]],     [0.3]],

              [[40, 41, 42, 43, 44],     [[0.3],
               [45, 46, 47, 48, 49],      [0.1],
               [50, 51, 52, 53, 54],      [0.3],
               [55, 56, 57, 58, 59]]]     [0.3]]]]

        Weighted weight bank:
            [[[[ 0.0,  0.0,  0.0,  0.0,  0.0],
               [ 2.0,  2.4,  2.8,  3.2,  3.6],
               [ 5.0,  5.5,  6.0,  6.5,  7.0],
               [ 0.0,  0.0,  0.0,  0.0,  0.0]],

              [[ 0.0,  0.0,  0.0,  0.0,  0.0],
               [12.5, 13.0, 13.5, 14.0, 14.5],
               [ 0.0,  0.0,  0.0,  0.0,  0.0],
               [17.5, 18.0, 18.5, 19.0, 19.5]],

              [[ 4.0,  4.1,  4.2,  4.3,  4.4],
               [31.5, 32.2, 32.9, 33.6, 34.3],
               [ 5.0,  5.1,  5.2,  5.3,  5.4],
               [ 5.5,  5.6,  5.7,  5.8,  5.9]]],

             [[[ 0.0,  1.0,  2.0,  3.0,  4.0],
               [ 0.0,  0.0,  0.0,  0.0,  0.0],
               [ 0.0,  0.0,  0.0,  0.0,  0.0],
               [ 0.0,  0.0,  0.0,  0.0,  0.0]],

              [[ 4.0,  4.2,  4.4,  4.6,  4.8],
               [10.0, 10.4, 10.8, 11.2, 11.6],
               [ 3.0,  3.1,  3.2,  3.3,  3.4],
               [10.5, 10.8, 11.1, 11.4, 11.7]],

              [[12.0, 12.3, 12.6, 12.9, 13.2],
               [ 4.5,  4.6,  4.7,  4.8,  4.9],
               [15.0, 15.3, 15.6, 15.9, 16.2],
               [16.5, 16.8, 17.1, 17.4, 17.7]]]]

        Weight mixture
            [[[ 7.0,  7.9,  8.8,  9.7, 10.6],
              [30.0, 31.0, 32.0, 33.0, 34.0],
              [46.0, 47.0, 48.0, 49.0, 50.0]],

             [[ 0.0,  1.0,  2.0,  3.0,  4.0],
              [27.5, 28.5, 29.5, 30.5, 31.5],
              [48.0, 49.0, 50.0, 51.0, 52.0]]]
        """

        inputDim = 3
        depthDim = 4
        outputDim = 5
        batchSize = 2

        maskedProbabilities = L.toTensor(
            [
                [[0.0, 0.4, 0.5, 0.0], [1.0, 0.0, 0.0, 0.0]],
                [[0.0, 0.5, 0.0, 0.5], [0.2, 0.4, 0.1, 0.3]],
                [[0.1, 0.7, 0.1, 0.1], [0.3, 0.1, 0.3, 0.3]],
            ]
        )  # [inputDim, batchSize,  depthDim]

        weightBank = L.createTensor(
            60, [inputDim, depthDim, outputDim]
        )  # [inputDim, depthDim, outputDim]

        # [batchSize, inputDim, depthDim]
        maskedProbabilitiesTransposed = L.transpose(maskedProbabilities, [1, 0])
        # [batchSize, inputDim, depthDim, 1]
        maskedProbabilitiesUnsqueezed = L.unsqueeze(
            maskedProbabilitiesTransposed, dim=-1
        )

        # [batchSize, inputDim, depthDim, outputDim]
        weightedWeights = L.unsqueeze(weightBank) * maskedProbabilitiesUnsqueezed
        # [inputDim, batchSize, outputDim]
        weightMixture = L.sum(weightedWeights, dim=-2)

        customSoftmaxShape = L.shapeList(weightMixture)
        expectedcustomSoftmaxShape = [batchSize, inputDim, outputDim]

        self.assertEqual(customSoftmaxShape, expectedcustomSoftmaxShape)
