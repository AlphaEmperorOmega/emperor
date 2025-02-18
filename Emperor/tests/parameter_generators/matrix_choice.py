from .general import GeneralTestMethods

from Emperor.library.choice import Library as L
from Emperor.components.parameter_generators.utils.losses import AuxiliaryLosses
from Emperor.components.parameter_generators.matrix_choice import (
    MatrixChoiceSparse,
    MatrixChoiceMixture,
    MatrixChoiceSoftMixture,
)
from Emperor.config import (
    ParameterGeneratorOptions,
    ParameterGeneratorConfig,
)


class TestMatrixChoiceSparseClass(GeneralTestMethods):
    def setUp(self) -> None:
        pass

    def testForwardMethod(self):
        config = ParameterGeneratorConfig()
        config.auxiliaryLosses = AuxiliaryLosses(config)
        model = MatrixChoiceSparse(config)
        self.checkForwardMethod(model, areWeightProbabilities=True)

    def testIfWeightsAreUpdated(self):
        self.areWeightsUpdated(ParameterGeneratorOptions.matrix_choice_sparse)

    def testIfSharedLayerWeightsAreUpdated(self):
        self.areWeightsUpdatedSharedLayer(
            ParameterGeneratorOptions.matrix_choice_sparse
        )

    def testIfWeightsAreUpdatedMultiLayer(self):
        self.areWeightsUpdatedMultiLayer(ParameterGeneratorOptions.matrix_choice_sparse)

    def testSelectWeightsMethodSteps(self):
        """
        Weight bank:
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
              [55, 56, 57, 58, 59]],

             [[60, 61, 62, 63, 64],
              [65, 66, 67, 68, 69],
              [70, 71, 72, 73, 74]],

             [[75, 76, 77, 78, 79],
              [80, 81, 82, 83, 84],
              [85, 86, 87, 88, 89]]]

        Input indexes:
            [1, 2, 2, 0, 1, 3]

        Selected weights:
            [[[15, 16, 17, 18, 19],
              [20, 21, 22, 23, 24],
              [25, 26, 27, 28, 29]],

             [[30, 31, 32, 33, 34],
              [35, 36, 37, 38, 39],
              [40, 41, 42, 43, 44]],

             [[30, 31, 32, 33, 34],
              [35, 36, 37, 38, 39],
              [40, 41, 42, 43, 44]],

             [[ 0,  1,  2,  3,  4],
              [ 5,  6,  7,  8,  9],
              [10, 11, 12, 13, 14]],

             [[15, 16, 17, 18, 19],
              [20, 21, 22, 23, 24],
              [25, 26, 27, 28, 29]],

             [[45, 46, 47, 48, 49],
              [50, 51, 52, 53, 54],
              [55, 56, 57, 58, 59]]]
        """
        inputDim = 3
        depthDim = 6
        outputDim = 5
        batchSize = 6
        # [inputDim, batchSize]
        inputIndexes = L.toTensor([1, 2, 2, 0, 1, 3])
        # [inputDim, depthDim, outputDim]
        weightBank = L.createTensor(90, [depthDim, inputDim, outputDim])

        # [inputDim, batchSize, outputDim]
        selectedWeights = weightBank[inputIndexes]

        # CHECKS
        selectedWeightsShape = L.shapeList(selectedWeights)
        expectedSelectedWeightsShape = [batchSize, inputDim, outputDim]

        self.assertEqual(selectedWeightsShape, expectedSelectedWeightsShape)


class TestMatrixChoiceMixtureClass(GeneralTestMethods):
    def setUp(self) -> None:
        pass

    def testForwardMethod(self):
        config = ParameterGeneratorConfig()
        config.auxiliaryLosses = AuxiliaryLosses(config)
        model = MatrixChoiceMixture(config)
        self.checkForwardMethod(model)

    def testIfWeightsAreUpdated(self):
        self.areWeightsUpdated(ParameterGeneratorOptions.matrix_choice_mixture)

    def testIfSharedLayerWeightsAreUpdated(self):
        self.areWeightsUpdatedSharedLayer(
            ParameterGeneratorOptions.matrix_choice_mixture
        )

    def testIfWeightsAreUpdatedMultiLayer(self):
        self.areWeightsUpdatedMultiLayer(
            ParameterGeneratorOptions.matrix_choice_mixture
        )

    def testSelectWeightsMethodSteps(self):
        """
        Weight Bank
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

        Topk indices of matrices to select:
            [[1, 0, 2],
             [2, 3, 1]]

        Selected weight matrices for each sample in the batch:
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
        """
        inputDim = 3
        depthDim = 4
        outputDim = 5
        topKExperts = 3
        batchSize = 2

        # [inputDim, depthDim, outputDim]
        weightBank = L.createTensor(60, [depthDim, inputDim, outputDim])
        # [batchSize, topK]
        topkIndexes = L.toTensor([[1, 0, 2], [2, 3, 1]])

        # [batchSize, topKExperts, inputDim, outputDim]
        selectedWeights = weightBank[topkIndexes]

        selectedWeightsShape = L.shapeList(selectedWeights)
        expectedSelectedWeightsShape = [batchSize, topKExperts, inputDim, outputDim]

        self.assertEqual(selectedWeightsShape, expectedSelectedWeightsShape)

    def testCalcWeightMixtureMethodSteps(self):
        """
        Weight bank:
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

        Given a `batchSize` of 2 and `topK` of 3 here
        are probabilities and their indices:
            - Probabilities:
            [[0.1, 0.4, 0.5],
             [0.3, 0.3, 0.4]]
            - Indices:
            [[1, 0, 2],
             [2, 3, 1]]

        Selected weights multiplied by reshaped probabilities:
            [[[[15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24],        [[[[0.1000]]
               [25, 26, 27, 28, 29]],

              [[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],          [[0.4000]],
               [10, 11, 12, 13, 14]],

              [[30, 31, 32, 33, 34],
               [35, 36, 37, 38, 39],          [[0.5000]]],
               [40, 41, 42, 43, 44]]],
                                        *
             [[[30, 31, 32, 33, 34],
               [35, 36, 37, 38, 39],         [[[0.3000]],
               [40, 41, 42, 43, 44]],

              [[45, 46, 47, 48, 49],
               [50, 51, 52, 53, 54],          [[0.3000]],
               [55, 56, 57, 58, 59]],

              [[15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24],          [[0.4000]]]]
               [25, 26, 27, 28, 29]]]]

        Weighted weights:
            [[[[ 1.5,  1.6,  1.7,  1.8,  1.9],
               [ 2.0,  2.1,  2.2,  2.3,  2.4],
               [ 2.5,  2.6,  2.7,  2.8,  2.9]],

              [[ 0.0,  0.4,  0.8,  1.2,  1.6],
               [ 2.0,  2.4,  2.8,  3.2,  3.6],
               [ 4.0,  4.4,  4.8,  5.2,  5.6]],

              [[15.0, 15.5, 16.0, 16.5, 17.0],
               [17.5, 18.0, 18.5, 19.0, 19.5],
               [20.0, 20.5, 21.0, 21.5, 22.0]]],

             [[[ 9.0,  9.3,  9.6,  9.9, 10.2],
               [10.5, 10.8, 11.1, 11.4, 11.7],
               [12.0, 12.3, 12.6, 12.9, 13.2]],

              [[13.5, 13.8, 14.1, 14.4, 14.7],
               [15.0, 15.3, 15.6, 15.9, 16.2],
               [16.5, 16.8, 17.1, 17.4, 17.7]],

              [[ 6.0,  6.4,  6.8,  7.2,  7.6],
               [ 8.0,  8.4,  8.8,  9.2,  9.6],
               [10.0, 10.4, 10.8, 11.2, 11.6]]]]

        Mixed weights:
            [[[16.5, 17.5, 18.5, 19.5, 20.5],
              [21.5, 22.5, 23.5, 24.5, 25.5],
              [26.5, 27.5, 28.5, 29.5, 30.5]],

             [[28.5, 29.5, 30.5, 31.5, 32.5],
              [33.5, 34.5, 35.5, 36.5, 37.5],
              [38.5, 39.5, 40.5, 41.5, 42.5]]]

        """
        inputDim = 3
        depthDim = 4
        outputDim = 5
        topK = 3
        batchSize = 2

        weightBank = L.createTensor(
            60, [depthDim, inputDim, outputDim]
        )  # [inputDim, depthDim, outputDim]

        topkIndexes = L.toTensor([[1, 0, 2], [2, 3, 1]])  # [batchSize, topK]
        topKProbabilities = L.toTensor(
            [[0.1, 0.4, 0.5], [0.3, 0.3, 0.4]]
        )  # [batchSize, topK]

        # Operation performed method: selectWeights
        # [batchSize, topKExperts, inputDim, outputDim]
        selectedWeights = weightBank[topkIndexes]

        reshapedProbabilities = L.reshape(topKProbabilities, [batchSize, topK, 1, 1])

        # Operation performed method: sumSelectedWeightMixture
        # [batch_size, topk, inputDim, outputDim]
        weightedWeights = selectedWeights * reshapedProbabilities

        # [batchSize, inputDim, outputDim]
        weightMixture = L.sum(weightedWeights, dim=1)

        # CHECKS
        selectedWeightsShape = L.shapeList(selectedWeights)
        expectedSelectedWeightsShape = [batchSize, topK, inputDim, outputDim]

        weightMixtureShape = L.shapeList(weightMixture)
        expectedWeightsMixtureShape = [batchSize, inputDim, outputDim]

        self.assertEqual(weightMixtureShape, expectedWeightsMixtureShape)
        self.assertEqual(selectedWeightsShape, expectedSelectedWeightsShape)


class TestMatrixChoiceSoftMixtureClass(GeneralTestMethods):
    def setUp(self) -> None:
        pass

    def testForwardMethod(self):
        config = ParameterGeneratorConfig()
        config.auxiliaryLosses = AuxiliaryLosses(config)
        model = MatrixChoiceSoftMixture(config)
        self.checkForwardMethod(model)

    def testIfWeightsAreUpdated(self):
        self.areWeightsUpdated(ParameterGeneratorOptions.matrix_choice_soft_mixture)

    def testIfSharedLayerWeightsAreUpdated(self):
        self.areWeightsUpdatedSharedLayer(
            ParameterGeneratorOptions.matrix_choice_soft_mixture
        )

    def testIfWeightsAreUpdatedMultiLayer(self):
        self.areWeightsUpdatedMultiLayer(
            ParameterGeneratorOptions.matrix_choice_soft_mixture
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

    def testCalcWeightMixtureMethodSteps(self):
        """
        Weight bank:
            [[[ 0,  1,  2,  3,  4],
              [ 5,  6,  7,  8,  9],
              [10, 11, 12, 13, 14]],

             [[15, 16, 17, 18, 19],
              [20, 21, 22, 23, 24],
              [25, 26, 27, 28, 29]],

             [[30, 31, 32, 33, 34],
              [35, 36, 37, 38, 39],
              [40, 41, 42, 43, 44]]]

        Given a `batchSize` of 2 and `depthDim` of 3 here
        are probabilities and their indices:
            [[0.1, 0.4, 0.5],
             [0.3, 0.3, 0.4]]

        Selected weights multiplied by reshaped probabilities:
            [[[ 0,  1,  2,  3,  4],
              [ 5,  6,  7,  8,  9],        [[[[0.1]]
              [10, 11, 12, 13, 14]],

             [[15, 16, 17, 18, 19],
              [20, 21, 22, 23, 24],          [[0.4]],
              [25, 26, 27, 28, 29]],

             [[30, 31, 32, 33, 34],
              [35, 36, 37, 38, 39],          [[0.5]]],
              [40, 41, 42, 43, 44]]]
                                        *
            [[[ 0,  1,  2,  3,  4],
              [ 5,  6,  7,  8,  9],         [[[0.3]],
              [10, 11, 12, 13, 14]],

             [[15, 16, 17, 18, 19],
              [20, 21, 22, 23, 24],          [[0.3]],
              [25, 26, 27, 28, 29]],

             [[30, 31, 32, 33, 34],
              [35, 36, 37, 38, 39],          [[0.4]]]]
              [40, 41, 42, 43, 44]]]


        Weighted weights:
            [[[[ 0.0,  0.1,  0.2,  0.3,  0.4],
               [ 0.5,  0.6,  0.7,  0.8,  0.9],
               [ 1.0,  1.1,  1.2,  1.3,  1.4]],

              [[ 6.0,  6.4,  6.8,  7.2,  7.6],
               [ 8.0,  8.4,  8.8,  9.2,  9.6],
               [10.0, 10.4, 10.8, 11.2, 11.6]],

              [[15.0, 15.5, 16.0, 16.5, 17.0],
               [17.5, 18.0, 18.5, 19.0, 19.5],
               [20.0, 20.5, 21.0, 21.5, 22.0]]],

             [[[ 0.0,  0.3,  0.6,  0.9,  1.2],
               [ 1.5,  1.8,  2.1,  2.4,  2.7],
               [ 3.0,  3.3,  3.6,  3.9,  4.2]],

              [[ 4.5,  4.8,  5.1,  5.4,  5.7],
               [ 6.0,  6.3,  6.6,  6.9,  7.2],
               [ 7.5,  7.8,  8.1,  8.4,  8.7]],

              [[12.0, 12.4, 12.8, 13.2, 13.6],
               [14.0, 14.4, 14.8, 15.2, 15.6],
               [16.0, 16.4, 16.8, 17.2, 17.6]]]]

        Mixed weights:
            [[[21.0, 22.0, 23.0, 24.0, 25.0],
              [26.0, 27.0, 28.0, 29.0, 30.0],
              [31.0, 32.0, 33.0, 34.0, 35.0]],

             [[16.5, 17.5, 18.5, 19.5, 20.5],
              [21.5, 22.5, 23.5, 24.5, 25.5],
              [26.5, 27.5, 28.5, 29.5, 30.5]]]

        """
        inputDim = 3
        depthDim = 3
        outputDim = 5
        batchSize = 2

        weightBank = L.createTensor(
            45, [depthDim, inputDim, outputDim]
        )  # [inputDim, depthDim, outputDim]

        topKProbabilities = L.toTensor(
            [[0.1, 0.4, 0.5], [0.3, 0.3, 0.4]]
        )  # [batchSize, depthDim]

        reshapedProbabilities = L.reshape(
            topKProbabilities, [batchSize, depthDim, 1, 1]
        )

        # Operation performed method: sumSelectedWeightMixture
        # [batch_size, depthDim, inputDim, outputDim]
        weightedWeights = weightBank * reshapedProbabilities

        # [batchSize, inputDim, outputDim]
        weightMixture = L.sum(weightedWeights, dim=1)

        # CHECKS
        weightMixtureShape = L.shapeList(weightMixture)
        expectedWeightsMixtureShape = [batchSize, inputDim, outputDim]

        self.assertEqual(weightMixtureShape, expectedWeightsMixtureShape)
