from .layer import ParameterGeneratorLayer
from Emperor.library.choice import Library as L
from Emperor.base.utils import Module

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class ParallelExperts(Module):
    def __init__(self, cfg: "ModelConfig"):
        super().__init__()

        self.cfg = cfg
        self.numExperts = cfg.numExperts
        self.experts = self._createExperts()

    def _createExperts(self):
        experts = []
        for _ in range(self.numExperts):
            layer = ParameterGeneratorLayer(self.cfg)
            experts.append(layer)

        return L.ModuleList(experts)

    def forward(self, expertOrderedInput, expertFrequency):
        expertInputs = self._splitInputExperts(expertOrderedInput, expertFrequency)
        expertOutputs = []
        for expertIndex in range(self.numExperts):
            input = expertInputs[expertIndex]
            output = self.experts[expertIndex](input)
            expertOutputs.append(output)

        return L.cat(expertOutputs, dim=0)

    def _splitInputExperts(self, expertOrderedInput, expertFrequency):
        expertFrequencyList = expertFrequency.tolist()
        return expertOrderedInput.split(expertFrequencyList)
