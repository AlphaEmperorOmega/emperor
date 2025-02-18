from ..library.choice import Library as L
from ..base.utils import Module
from ..base.decorators import timer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ParameterGeneratorConfig


class ParameterGeneratorLayer(Module):
    def __init__(self, cfg: "ParameterGeneratorConfig"):
        super().__init__()

        self.layerTimeFlag = False
        self.parameterGenerator = cfg.parameterGeneartorType.create(cfg)
        self.gatherFrequency = cfg.gatherFrequencyFlag

    def forward(self, inputBatch):
        if self.layerTimeFlag:
            return self.forwardWithTimer(inputBatch)
        return self.forwardDefault(inputBatch)

    @timer
    def forwardWithTimer(self, inputBatch):
        return self.forwardDefault(inputBatch)

    def forwardDefault(self, inputBatch):
        generatedWeights, generatedBiases, probabilities = self.parameterGenerator(
            inputBatch
        )

        output = L.tensorProduct(inputBatch.unsqueeze(1), generatedWeights).squeeze(1)
        output = (output + generatedBiases) if generatedBiases is not None else output

        # Scale each output vector by the mean of probabilities of it's selected weights
        # if probabilities is not None:
        #     output = output * probabilities

        return output
