import torch
import torch.nn as nn
from dataclasses import dataclass, field
from torch import Tensor
from dataclasses import replace
from dataclasses import dataclass, field
from Emperor.base.utils import ConfigBase, Module
from Emperor.library.choice import Library as L


from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


@dataclass
class MixtureOfExpertsConfig(ConfigBase):
    input_dim: Optional[int] = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    hidden_dim: Optional[int] = field(
        default=None,
        metadata={"help": "Expert hidden dimension"},
    )
    output_dim: Optional[int] = field(
        default=None,
        metadata={"help": "Expert output dimension"},
    )
    multiply_by_gates_flag: Optional[bool] = field(
        default=None,
        metadata={"help": "If true, multiply expert output by gate values."},
    )
    activation_function: Optional[nn.Module] = field(
        default=None,
        metadata={"help": "Activation function for expert layers."},
    )
    hidden_dropout_probability: Optional[float] = field(
        default=None,
        metadata={"help": "Dropout probability for expert hidden layers."},
    )
    hidden_layer_norm_flag: Optional[bool] = field(
        default=None,
        metadata={
            "help": "If true, apply layer normalization to expert hidden layers."
        },
    )


class MixtureOfExperts(Module):
    def __init__(
        self,
        cfg: "MixtureOfExpertsConfig | ModelConfig",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_dim = self.cfg.input_dim
        self.hidden_dim = self.cfg.hidden_dim
        self.output_dim = self.cfg.output_dim
        self.multiply_by_gates_flag = self.cfg.multiply_by_gates_flag
        self.activation_function = self.cfg.activation_function
        self.hidden_dropout_probability = self.cfg.hidden_dropout_probability
        self.hidden_layer_norm_flag = self.cfg.hidden_layer_norm_flag
        self.input_batch_shape = None

        self._init_expert_layers()
        self._init_expert_sampler()
        self._init_hidden_modules()

    def _init_expert_layers(self):
        inputExpertsConfig = replace(
            self.cfg, inputDim=self.inputDim, outputDim=self.hiddenDim
        )
        self.inputExperts = ParallelExperts(inputExpertsConfig)
        outputExpertsConfig = replace(
            self.cfg, inputDim=self.hiddenDim, outputDim=self.outputDim
        )
        self.outputExperts = ParallelExperts(outputExpertsConfig)

    def _init_expert_sampler(self):
        cfg = replace(
            self.cfg,
            depthDim=self.cfg.numExperts,
            auxiliaryLosses=self.cfg.moeAuxiliaryLosses,
        )
        self.sampler = ProbabilitySamplerTopk(cfg)

    def _init_hidden_modules(self):
        self.dropoutModule = self.hiddenLayerNormModule = None
        if (
            self.hiddenDropoutProbability is not None
            and self.hiddenDropoutProbability > 0
        ):
            self.dropoutModule = nn.Dropout(self.hiddenDropoutProbability)
        if self.hiddenLayerNormFlag:
            self.hiddenLayerNormModule = nn.LayerNorm(self.hiddenDim)

    def forward(self, inputBatch: Tensor, skipMask: Optional[Tensor] = None):
        self.inputBatchShape = inputBatch.size()

        (
            expertSortedBatchInputs,
            expertSortedTopKBatchIndexes,
            expertSortedTopKBatchProbabilities,
            batchExpertFrequency,
            expertSortedNonzeroProbabilityIndexes,
        ) = self._prepareInputBatchForExpertProcessing(inputBatch, skipMask)

        feedForwardOutput = self._computeProjections(
            expertSortedBatchInputs, batchExpertFrequency
        )

        expertMixtureOutput = self._computeExpertMixture(
            feedForwardOutput,
            expertSortedTopKBatchIndexes,
            expertSortedTopKBatchProbabilities,
        )

        return expertMixtureOutput

    def _prepareInputBatchForExpertProcessing(self, inputBatch, skipMask=None):
        if skipMask is not None:
            skipMask = skipMask.view(-1, 1)

        inputBatchMatrix = inputBatch.reshape(-1, self.inputDim)

        topKProbabilities, topKIndexes = self.sampler.sampleProbabilitiesAndIndexes(
            inputBatchMatrix, skipMask=skipMask, isTrainingFlag=self.training
        )
        (
            expertSortedTopKBatchIndexes,
            expertSortedTopKBatchProbabilities,
            topKProbabilitiesScattered,
            batchExpertFrequency,
            expertSortedNonzeroProbabilityIndexes,
        ) = self._computeGating(topKProbabilities, topKIndexes)

        expertSortedBatchInputs = inputBatchMatrix[expertSortedTopKBatchIndexes]

        return (
            expertSortedBatchInputs,
            expertSortedTopKBatchIndexes,
            expertSortedTopKBatchProbabilities,
            batchExpertFrequency,
            expertSortedNonzeroProbabilityIndexes,
        )

    def _computeGating(self, topKProbabilities, topKIndexes):
        topKProbabilitiesFlattened = topKProbabilities.flatten()
        topKIndexesFlattened = topKIndexes.flatten()
        nonzeroProbabilityIndexes = topKProbabilitiesFlattened.nonzero()
        nonzeroProbabilityIndexes = nonzeroProbabilityIndexes.squeeze(-1)
        nonzeroTopKIndexes = topKIndexesFlattened[nonzeroProbabilityIndexes]
        _, _expertSortedNonzeroTopKIndexes = nonzeroTopKIndexes.sort(0)

        expertSortedNonzeroProbabilityIndexes = nonzeroProbabilityIndexes[
            _expertSortedNonzeroTopKIndexes
        ]

        expertSortedTopKBatchIndexes = expertSortedNonzeroProbabilityIndexes.div(
            self.cfg.topK, rounding_mode="trunc"
        )
        expertSortedTopKBatchProbabilities = topKProbabilitiesFlattened[
            expertSortedNonzeroProbabilityIndexes
        ]

        topKProbabilitiesScatteredPlaceholder = L.zeros(
            [topKProbabilities.shape[0], self.cfg.numExperts], device=L.Device
        )
        topKProbabilitiesScattered = topKProbabilitiesScatteredPlaceholder.scatter(
            1, topKIndexes, topKProbabilities
        )

        batchExpertFrequency = topKProbabilitiesScattered > 0
        batchExpertFrequency = batchExpertFrequency.long()
        batchExpertFrequency = batchExpertFrequency.sum(dim=0)

        return (
            expertSortedTopKBatchIndexes,
            expertSortedTopKBatchProbabilities,
            topKProbabilitiesScattered,
            batchExpertFrequency,
            expertSortedNonzeroProbabilityIndexes,
        )

    def _computeProjections(
        self,
        expertSortedBatchInputs: Tensor,
        batchExpertFrequency: Tensor,
    ):
        expandedProjection = self.inputExperts(
            expertSortedBatchInputs, batchExpertFrequency
        )
        hiddenRepresentation = self.activationFunction(expandedProjection)
        if self.dropoutModule:
            hiddenRepresentation = self.dropoutModule(hiddenRepresentation)
        if self.hiddenLayerNormModule:
            hiddenRepresentation = self.hiddenLayerNormModule(hiddenRepresentation)
        reductionProjection = self.outputExperts(
            hiddenRepresentation, batchExpertFrequency
        )

        return reductionProjection

    def _computeExpertMixture(
        self,
        feedForwardOutput,
        expertSortedTopKBatchIndexes,
        expertSortedTopKBatchProbabilities,
    ):
        if len(self.inputBatchShape) > 2:
            batchSize, sequenceLength, _ = self.inputBatchShape
            outputShape = [batchSize, sequenceLength, self.outputDim]
        else:
            sequenceLength = 1
            batchSize, _ = self.inputBatchShape
            outputShape = [batchSize, self.outputDim]

        if self.multiplyByGatesFlag:
            feedForwardOutput = (
                feedForwardOutput * expertSortedTopKBatchProbabilities.view(-1, 1)
            )

        inputBatchZeros = L.zeros(
            (batchSize * sequenceLength, self.outputDim),
            dtype=feedForwardOutput.dtype,
            device=L.Device,
        )

        expertOutputMixture = inputBatchZeros.index_add(
            0, expertSortedTopKBatchIndexes, feedForwardOutput
        )

        return expertOutputMixture.view(outputShape)


class ParallelExperts(Module):
    def __init__(self, cfg: "ModelConfig"):
        super().__init__()

        self.cfg = cfg
        self.num_experts = cfg.num_experts
        self.experts = self.__create_experts()

    def __create_experts(self):
        experts = []
        for _ in range(self.num_experts):
            layer = ParameterLayerBase(self.cfg)
            experts.append(layer)

        return nn.ModuleList(experts)

    def forward(self, expert_ordered_input, expert_frequency):
        expert_inputs = self._split_input_experts(
            expert_ordered_input, expert_frequency
        )
        expert_outputs = []
        for expert_index in range(self.num_experts):
            input_tensor = expert_inputs[expert_index]
            output = self.experts[expert_index](input_tensor)
            expert_outputs.append(output)

        return torch.cat(expert_outputs, dim=0)

    def _split_input_experts(self, expert_ordered_input, expert_frequency):
        expert_frequency_list = expert_frequency.tolist()
        return expert_ordered_input.split(expert_frequency_list)


class MixtureOfAttentionHeads(MixtureOfExperts):
    def __init__(self, cfg: "ModelConfig") -> None:
        super().__init__(cfg)

    def computeHiddenProjection(self, inputBatch, skipMask=None):
        self.inputBatchShape = inputBatch.size()
        batchSize, sequenceLength, _ = self.inputBatchShape

        (
            expertSortedBatchInputs,
            expertSortedTopKBatchIndexes,
            expertSortedTopKBatchProbabilities,
            batchExpertFrequency,
            expertSortedNonzeroProbabilityIndexes,
        ) = self._prepareInputBatchForExpertProcessing(inputBatch, skipMask)

        expertsOutput = self.inputExperts(expertSortedBatchInputs, batchExpertFrequency)

        inputBatchZeros = L.zeros(
            (batchSize * sequenceLength * self.cfg.topK, self.hiddenDim),
            dtype=expertsOutput.dtype,
            device=L.Device,
        )
        expertOutputMixture = inputBatchZeros.index_add(
            0, expertSortedNonzeroProbabilityIndexes, expertsOutput
        )

        # topKIndexes = topKIndexes.view(batchSize, sequenceLength, -1)
        expandProjectionOutput = expertOutputMixture.view(
            batchSize, sequenceLength, self.cfg.topK, self.hiddenDim
        )

        self.expertSortedTopKBatchIndexes = expertSortedTopKBatchIndexes
        self.expertSortedTopKBatchProbabilities = expertSortedTopKBatchProbabilities
        self.batchExpertFrequency = batchExpertFrequency
        self.expertSortedNonzeroProbabilityIndexes = (
            expertSortedNonzeroProbabilityIndexes
        )

        return expandProjectionOutput

    def conputeOutputProjection(self, attentionOutput):
        """
        The self variables are computed in MixtureOfExperts.expandProjection
        (or the method above in case you don't see)
        """
        batchSize, sequenceLength, _, _ = attentionOutput.shape

        attentionOutput = attentionOutput.reshape(-1, self.hiddenDim)

        expertInputs = attentionOutput[self.expertSortedNonzeroProbabilityIndexes]

        expertsOutput = self.outputExperts(expertInputs, self.batchExpertFrequency)

        if self.multiplyByGatesFlag:
            expertsOutput = (
                expertsOutput * self.expertSortedTopKBatchProbabilities.view(-1, 1)
            )

        inputBatchZeros = L.zeros(
            (batchSize * sequenceLength, self.outputDim),
            dtype=expertsOutput.dtype,
            device=L.Device,
        )
        expertOutputMixture = inputBatchZeros.index_add(
            0, self.expertSortedTopKBatchIndexes, expertsOutput
        )
        return expertOutputMixture.view(batchSize, sequenceLength, self.outputDim)
