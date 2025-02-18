from Emperor.base.utils import Module
from Emperor.base.decorators import timer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ParameterGeneratorConfig


class ParameterGenerator(Module):
    def __init__(self, cfg: "ParameterGeneratorConfig"):
        super().__init__()
        self.cfg = cfg
        self.batchSize = cfg.batchSize
        self.inputDim = cfg.inputDim
        self.outputDim = cfg.outputDim
        self.depthDim = cfg.depthDim
        self.mlpRouterFlag = cfg.mlpRouterFlag
        self.biasFlag = cfg.biasFlag
        self.gatherFrequencyFlag = cfg.gatherFrequencyFlag
        self.noisyTopkFlag = cfg.noisyTopkFlag
        self.topK = cfg.topK

        # TODO: later implement the frequency gathering methanism
        #
        # frequencyClass = cfg.weightAndBiasGeneratorType.value + "Frequency"
        # self.gatherFrequencyCheck = (
        #     self.gatherFrequencyFlag and frequencyClass in globals()
        # )
        # if self.gatherFrequencyCheck:
        #     self.frequency = globals()[frequencyClass](cfg)

        self.routerOutputDim = (
            2 * self.depthDim if self.noisyTopkFlag else self.depthDim
        )

        self.calcWeightsFlag = False

    @timer
    def forwardProcessTime(self, inputBatch):
        return self(inputBatch)

    def gatherFrequency(self, sparseIndexes):
        if self.gatherFrequencyCheck:
            self.frequency.update(sparseIndexes)

    def handleProbabilitiesShapeHook(self, probabilities):
        return probabilities

    def selectParameters(self, weightIndexes, biasIndexes):
        pass
