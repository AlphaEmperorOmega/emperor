from Emperor.base.utils import Module

from Emperor.library.choice import Library as L

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ParameterGeneratorConfig


class RouterModel(Module):
    def __init__(self, cfg: "ParameterGeneratorConfig") -> None:
        super().__init__()
        self.cfg = cfg
        self.inputDim = cfg.inputDim
        self.outputDim = cfg.outputDim
        self.mlpRouterFlag = cfg.mlpRouterFlag
        self.biasFlag = cfg.biasFlag

        self.routerOutputDim = 2 * cfg.depthDim if cfg.noisyTopkFlag else cfg.depthDim
        self._storeRoutersHook()

    def _storeRoutersHook(self) -> None:
        self.weightRouterModel = self._addMlpRouter()
        L.initializeWeights(self.weightRouterModel)
        if self.biasFlag:
            self.biasRouterModel = self._addMlpRouter()
            L.initializeWeights(self.biasRouterModel)

    def _addMlpRouter(self):
        if self.mlpRouterFlag:
            routerModel = L.Sequential(
                L.Linear(self.inputDim, self.inputDim),
                L.Tanh(),
                L.Linear(self.inputDim, self.routerOutputDim, bias=False),
            )
        else:
            routerModel = L.Linear(self.inputDim, self.routerOutputDim, bias=False)
        L.initializeWeights(routerModel)
        return routerModel

    def calcLogitScores(self, inputBatch, computeWeightsFlag=True):
        if computeWeightsFlag:
            return self.weightRouterModel(inputBatch)
        return self.biasRouterModel(inputBatch)


class VectorChoiceRouterModel(RouterModel):
    def __init__(self, cfg: "ParameterGeneratorConfig") -> None:
        super().__init__(cfg)

    def _storeRoutersHook(self) -> None:
        self.routerWeights = L.parameterWeights(
            self.inputDim, self.inputDim, self.routerOutputDim
        )
        L.initializeWeights(self.routerWeights)
        self.routerBiases = None
        if self.biasFlag:
            self.routerBiases = L.parameterWeights(
                self.outputDim, self.inputDim, self.routerOutputDim
            )
            L.initializeWeights(self.routerBiases)

    def calcLogitScores(self, inputBatch, computeWeightsFlag=True):
        if computeWeightsFlag:
            return L.tensorProduct(inputBatch, self.routerWeights)
        return L.tensorProduct(inputBatch, self.routerBiases)
