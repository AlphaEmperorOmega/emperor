from Emperor.config import ModelConfig
from .probabilitySamplers import SamplerModel
from Emperor.base.utils import Module

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import ParameterGenerator
    from Emperor.config import ParameterGeneratorConfig


class MixtureModel(Module):
    def __init__(
        self,
        model: "ParameterGenerator",
        cfg: "MixtureConfig | ModelConfig | None" = None,
        bias_flag: bool | None = None,
    ):
        super().__init__()
        self.model = model

        self.cfg_main = cfg
        self.cfg: "MixtureModel | None" = self._resolve_config(
            cfg, "sampler_model_config"
        )
        self.bias_flag = self._resolve(bias_flag, "bias_flag", cfg)
        self.probability_sampler_model = SamplerModel(cfg)

    def _sample_weight_bias_probabilities_and_indexes(self, inputBatch, skip_mask):
        weight_probabilities, weight_indexes = (
            self._sample_probabilities_and_indexes(inputBatch, skip_mask)
        )

        bias_indexes = bias_probabilities = None
        if self.bias_flag:
            self.set_router_weight_flag(False)
            bias_probabilities, bias_indexes = (
                self.__sample_probabilities_and_indexes(
                    inputBatch, skip_mask
                )
            )
            self.set_router_weight_flag(False)

        return weight_indexes, bias_indexes, weight_probabilities, bias_probabilities

    def set_router_weight_flag(self, compute_weight_logit_scores_flag: bool=True):
        self.probability_sampler_model.sampler_model.set_compute_weight_flag(
            compute_weight_logit_scores_flag
        )

    def _sampleProbabilities(self, inputBatch, skip_mask):
        return self.probability_sampler_model(
            input_matrix=inputBatch,
            skip_mask=skip_mask,
            is_training_flag=self.model.training,
            custom_softmax_flag=False,
        )


class SparseMixtureBehaviour(MixtureModel):
    def __init__(
        self,
        cfg: "MixtureModel | ModelConfig | None" = None,
        model: "ParameterGenerator",
    ):
        super().__init__(cfg, model)

    def compute_mixture(self, inputBatch):
        weight_indexes, bias_indexes, weight_probabilities, bias_probabilities = (
            self._sample_probabilities_and_indexes(inputBatch)
        )

        selected_weights, selected_biases = self.model.select_parameters(
            weight_indexes, bias_indexes
        )

        return selected_weights, selected_biases

    # def _sample_sparse_probabilities_and_indexes(self, inputBatch):
    #     weight_probabilities, weight_indexes = (
    #         self._sample_probabilities_and_indexes(inputBatch)
    #     )
    #
    #     bias_indexes = bias_probabilities = None
    #     if self.bias_flag:
    #         bias_probabilities, bias_indexes = (
    #             self.__sample_probabilities_and_indexes(
    #                 inputBatch 
    #             )
    #         )
    #
    #     return weight_indexes, bias_indexes, weight_probabilities, bias_probabilities 

    def _sample_probabilities_and_indexe(self, inputBatch):
        probabilities, indexes = self.probabilitySampler.sampleProbabilitiesAndIndexes(
            inputBatch,
            isTrainingFlag=self.model.training,
        )
        probabilitiesReshaped = self.model.handleProbabilitiesShapeHook(probabilities)
        return probabilitiesReshaped, indexes


class TopkMixtureBehaviour:
    def __init__(
        self,
        cfg: "ParameterGeneratorConfig",
        model: "ParameterGenerator",
    ):
        super().__init__(cfg, model)

    def calculateMixture(self, inputBatch):
        (
            weightTopKProbabilities,
            weightTopKIndexes,
            biasTopKProbabilities,
            biasTopKIndexes,
        ) = self._sampleTopKProbabilitiesAndIndexes(inputBatch)

        # [batchSize, topK, inputDim, outputDim]
        selectedWeightParameters, selectedBiaseParameters = self.model.selectParameters(
            weightTopKIndexes, biasTopKIndexes
        )

        # [batchSize, inputDim, outputDim]
        weightMixture, biasMixture = self.model.calculateParameterMixture(
            selectedWeightParameters,
            weightTopKProbabilities,
            selectedBiaseParameters,
            biasTopKProbabilities,
        )

        # [batchSize, inputDim, outputDim]
        return weightMixture, biasMixture, None

    def _sampleTopKProbabilitiesAndIndexes(self, inputBatch):
        # [inputDim, batchSize, topK], [inputDim, batchSize, topK]
        weightTopKProbabilities, weightTopKIndexes = (
            self._sampleProbabilitiesAndIndexes(inputBatch)
        )

        biasTopKProbabilities, biasTopKIndexes = (None, None)
        if self.biasFlag:
            biasTopKProbabilities, biasTopKIndexes = (
                self._sampleProbabilitiesAndIndexes(
                    inputBatch
                )
            )

        return (
            weightTopKProbabilities,
            weightTopKIndexes,
            biasTopKProbabilities,
            biasTopKIndexes,
        )

    def _sampleProbabilitiesAndIndexes(self, inputBatch):
        return self.probabilitySampler.sampleProbabilitiesAndIndexes(
            inputBatch,
            isTrainingFlag=self.model.training,
        )


class FullMixtureBehaviour:
    def __init__(
        self,
        cfg: "ParameterGeneratorConfig",
        model: "ParameterGenerator",
    ):
        super().__init__(cfg, model)

    def calculateMixture(self, inputBatch):  # [batchSize, inputDim]
        weightFullProbabilities, biasFullProbabilities = self._sampleFullProbabilities(
            inputBatch
        )

        weightMixture, biasMixture = self.model.calculateParameterMixture(
            self.model.weightBank,
            weightFullProbabilities,
            self.model.biasBank,
            biasFullProbabilities,
        )

        # [batchSize, inputDim, outputDim]
        return weightMixture, biasMixture, None

    def _sampleFullProbabilities(self, inputBatch):
        weightFullProbabilities, _ = self._sampleProbabilities(inputBatch)

        biasFullProbabilities = None
        if self.model.biasFlag:
            biasFullProbabilities, _ = self._sampleProbabilities(
                inputBatch, 
            )

        return weightFullProbabilities, biasFullProbabilities

    def _sampleProbabilities(self, inputBatch):
        return self.probabilitySampler.sampleProbabilitiesAndIndexes(
            inputBatch,
            isTrainingFlag=self.model.training,
        )
