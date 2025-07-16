import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass, field

from Emperor.base.utils import DataClassBase, Module, device
from Emperor.components.parameter_generators.utils.base import LayerBlock
from Emperor.components.parameter_generators.utils.enums import (
    ActivationFunctionOptions,
    LayerTypes,
)
from Emperor.components.parameter_generators.utils.routers import RouterModel
from Emperor.components.parameter_generators.utils.samplers import SamplerModel

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


@dataclass
class MixtureOfExpertsConfig(DataClassBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    hidden_dim: int | None = field(
        default=None,
        metadata={"help": "Expert hidden dimension"},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Expert output dimension"},
    )
    multiply_by_gates_flag: bool | None = field(
        default=None,
        metadata={"help": "If true, multiply expert output by gate values."},
    )
    activation_function: nn.Module | None = field(
        default=None,
        metadata={"help": "Activation function for expert layers."},
    )
    dropout_probability: float | None = field(
        default=None,
        metadata={"help": "Dropout probability for expert hidden layers."},
    )
    hidden_layer_norm_flag: bool | None = field(
        default=None,
        metadata={
            "help": "If true, apply layer normalization to expert hidden layers."
        },
    )
    input_expert_config: "ExpertsConfig | None" = field(
        default=None,
        metadata={"help": "Configuration for the input experts"},
    )
    output_expert_config: "ExpertsConfig | None" = field(
        default=None,
        metadata={"help": "Configuration for the output experts"},
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
        self.dropout_probability = self.cfg.dropout_probability
        self.layer_norm_flag = self.cfg.layer_norm_flag
        self.input_experts_config = self.cfg.input_expert_config
        self.output_experts_config = self.cfg.output_expert_config

        self.utils = ExpertSelectorUtils()
        self.router = RouterModel(cfg)
        self.sampler = SamplerModel(cfg)
        self.input_experts = ParallelExperts(self.input_experts_config)
        # Output experts should be raw
        self.output_experts = ParallelExperts(self.output_experts_config)

    def forward(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ):
        self._input_batch_shape = input_batch.size()

        input_batch_matrix, skip_mask = self.__reshape_inputs(input_batch, skip_mask)

        logits = self.router.compute_logit_scores(input_batch)
        probabilities, indices, skip_mask, loss = (
            self.sampler.sample_probabilities_and_indices(logits, skip_mask)
        )

        (
            expert_sorted_batch_inputs,
            expert_sorted_top_k_batch_indexes,
            expert_sorted_top_k_batch_probabilities,
            batch_expert_frequency,
            expert_sorted_nonzero_probability_indexes,
        ) = self._prepare_inputs(input_batch_matrix, probabilities, indices)

        feed_forward_output = self._compute_projections(
            expert_sorted_batch_inputs, batch_expert_frequency
        )

        expert_mixture_output = self._compute_mixture(
            feed_forward_output,
            expert_sorted_top_k_batch_indexes,
            expert_sorted_top_k_batch_probabilities,
        )

        return expert_mixture_output

    def _prepare_inputs(
        self,
        input_batch: Tensor,
        probabilities: Tensor,
        indices: Tensor,
    ):
        (
            expert_sorted_indices,
            expert_sroted_probabilities,
            scattered_probabilities,
            expert_frequency,
            expert_sorted_nonzero_probability_indices,
        ) = self.utils.compute_gating(probabilities, indices)

        expert_sorted_input_batch = input_batch[expert_sorted_indices]

        return (
            expert_sorted_input_batch,
            expert_sroted_probabilities,
            scattered_probabilities,
            expert_frequency,
            expert_sorted_nonzero_probability_indices,
        )

    def __reshape_inputs(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        input_batch = input_batch.reshape(-1, self.inputDim)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        return input_batch, skip_mask

    def _compute_projections(
        self,
        expert_sorted_inputs: Tensor,
        frequency: Tensor,
    ):
        expanded_projection = self.input_experts(expert_sorted_inputs, frequency)
        # hidden_representation = self.activationFunction(expanded_projection)
        # if self.dropoutModule:
        #     hidden_representation = self.dropoutModule(hidden_representation)
        # if self.hiddenLayerNormModule:
        #     hidden_representation = self.hiddenLayerNormModule(hidden_representation)
        reduction_projection = self.output_experts(expanded_projection, frequency)

        return reduction_projection

    def _compute_expert_mixture(
        self,
        experts_output,
        expert_sorted_indices,
        expert_sorted_probabilities,
    ):
        if len(self._input_batch_shape) > 2:
            batch_size, sequence_length, _ = self._input_batch_shape
            output_shape = [batch_size, sequence_length, self.output_dim]
        else:
            sequence_length = 1
            batch_size, _ = self._input_batch_shape
            output_shape = [batch_size, self.output_dim]

        if self.multiply_by_gates_flag:
            flattened_probabilities = expert_sorted_probabilities.view(-1, 1)
            experts_output = experts_output * flattened_probabilities

        experts_shaped_zeros = torch.zeros(
            (batch_size * sequence_length, self.output_dim),
            dtype=experts_output.dtype,
            device=L.Device,
        )

        output = experts_shaped_zeros.index_add(
            0, expert_sorted_indices, experts_output
        )

        return output.view(output_shape)


class ExpertSelectorUtils:
    def compute_gating(self, probabilities, indexes):
        expert_sroted_probabilities, expert_sorted_indices = (
            self.__sort_probabilities_and_indices_by_expert_order(
                probabilities, indexes
            )
        )
        scattered_probabilities = self.__scatter_probabilities(probabilities, indexes)
        expert_frequency = self.__compute_expert_frequency(probabilities, indexes)
        expert_sorted_nonzero_probability_indices = (
            self.__sort_nonzero_probability_indices_by_expert_order(
                probabilities, indexes
            )
        )

        return (
            expert_sorted_indices,
            expert_sroted_probabilities,
            scattered_probabilities,
            expert_frequency,
            expert_sorted_nonzero_probability_indices,
        )

    def __sort_probabilities_and_indices_by_expert_order(
        self,
        probabilities: Tensor,
        indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        (
            _,
            flattened_probabilities,
            nonzero_probability_indices,
            _expert_sorted_nonzero_indices,
        ) = self.__required_inputs(probabilities, indices)
        expert_sorted_nonzero_probability_indices = nonzero_probability_indices[
            _expert_sorted_nonzero_indices
        ]
        expert_sorted_nonzero_indices = expert_sorted_nonzero_probability_indices.div(
            self.top_k, rounding_mode="trunc"
        )
        expert_sorted_nonzero_probabilities = flattened_probabilities[
            expert_sorted_nonzero_probability_indices
        ]
        return expert_sorted_nonzero_probabilities, expert_sorted_nonzero_indices

    def __scatter_probabilities(
        self,
        probabilities: Tensor,
        indices: Tensor,
    ) -> Tensor:
        batch_size = probabilities.shape[0]
        placeholder = torch.zeros([batch_size, self.num_experts], device=device)
        return placeholder.scatter(1, indices, probabilities)

    def __compute_expert_frequency(self, probabilities, indices):
        scattered_probabilities = self.__scatter_probabilities(probabilities, indices)
        expert_frequency = scattered_probabilities > 0
        expert_frequency = expert_frequency.long()
        return expert_frequency.sum(dim=0)

    def __sort_nonzero_probability_indices_by_expert_order(
        self,
        probabilities: Tensor,
        indices: Tensor,
    ) -> Tensor:
        _, _, nonzero_probability_indices, _expert_sorted_nonzero_indices = (
            self.__required_inputs(probabilities, indices)
        )

        return nonzero_probability_indices[_expert_sorted_nonzero_indices]

    def __required_inputs(
        self,
        probabilities: Tensor,
        indices: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        flattened_indices = indices.flatten()
        flattened_probabilities = probabilities.flatten()

        nonzero_probability_indices = flattened_probabilities.nonzero()
        nonzero_probability_indices = nonzero_probability_indices.squeeze(-1)

        nonzero_indices = flattened_indices[nonzero_probability_indices]
        _, _expert_sorted_nonzero_indices = nonzero_indices.sort(0)

        return (
            flattened_indices,
            flattened_probabilities,
            nonzero_probability_indices,
            _expert_sorted_nonzero_indices,
        )


@dataclass
class ExpertsConfig(DataClassBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Expert output dimension"},
    )
    activation: ActivationFunctionOptions | None = field(
        default=None,
        metadata={"help": "Activation function for the experts."},
    )
    model_type: LayerTypes | None = field(
        default=None,
        metadata={"help": "Type of layer used for the experts."},
    )
    num_experts: int | None = field(
        default=None,
        metadata={"help": "Number of experts in the model"},
    )


class ParallelExperts(Module):
    def __init__(
        self,
        cfg: "ExpertsConfig | ModelConfig",
        overrides: "ExpertsConfig | None" = None,
        is_output_expert: bool = False,
    ):
        super().__init__()
        config = getattr(cfg, "expert_model_config", cfg)
        self.cfg: "ExpertsConfig" = self._overwrite_config(config, overrides)

        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.num_experts = self.cfg.num_experts
        self.activation: ActivationFunctionOptions = self.cfg.activation.value
        self.model_type: LayerTypes = self.cfg.model_type.value
        self._valudate_fields(self.cfg, ExpertsConfig)
        self.experts_module = self.__create_experts()

    def __create_experts(self) -> nn.ModuleList:
        expert_list = []
        for _ in range(self.num_experts):
            model = self.model_type(self.cfg)
            layer_block = LayerBlock(
                model=model,
                activation_function=self.activation,
                dropout_probability=0.0,
                layer_norm_flag=True,
            )
            expert_list.append(layer_block)
        return nn.ModuleList(expert_list)

    def forward(self, expert_ordered_input, expert_frequency):
        expert_inputs = self.__split_input_experts(
            expert_ordered_input, expert_frequency
        )
        expert_outputs = []
        for expert_index in range(self.num_experts):
            input_tensor = expert_inputs[expert_index]
            output = self.experts[expert_index](input_tensor)
            expert_outputs.append(output)
        return torch.cat(expert_outputs, dim=0)

    def __split_input_experts(self, expert_ordered_input, expert_frequency):
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
