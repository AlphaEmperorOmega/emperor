import torch
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


class ExpertSelector:
    def computeGating(self, topKProbabilities, topKIndexes):
        expert_sroted_probabilities, expert_sorted_indices = (
            self.__sort_probabilitied_and_indices_by_expert_order(
                topKProbabilities, topKIndexes
            )
        )
        scattered_probabilities = self.__scattered_probabiliites(
            topKProbabilities, topKIndexes
        )
        expert_frequency = self.__batch_frequency(topKProbabilities, topKIndexes)
        expert_sorted_nonzero_probability_indices = (
            self.__expert_sorted_nonzero_probability_indexes(
                topKProbabilities, topKIndexes
            )
        )

        return (
            expert_sorted_indices,
            expert_sroted_probabilities,
            scattered_probabilities,
            expert_frequency,
            expert_sorted_nonzero_probability_indices,
        )

    def __sort_probabilitied_and_indices_by_expert_order(
        self, topKProbabilities, topKIndexes
    ):
        (
            _,
            topKProbabilitiesFlattened,
            nonzeroProbabilityIndexes,
            _expertSortedNonzeroTopKIndexes,
        ) = self.__required_inputs(topKProbabilities, topKIndexes)

        expertSortedNonzeroProbabilityIndexes = nonzeroProbabilityIndexes[
            _expertSortedNonzeroTopKIndexes
        ]

        expertSortedTopKBatchIndexes = expertSortedNonzeroProbabilityIndexes.div(
            self.cfg.topK, rounding_mode="trunc"
        )

        expertSortedTopKBatchProbabilities = topKProbabilitiesFlattened[
            expertSortedNonzeroProbabilityIndexes
        ]
        return expertSortedTopKBatchProbabilities, expertSortedTopKBatchIndexes

    def __scattered_probabiliites(self, topKProbabilities, topKIndexes):
        topKProbabilitiesScatteredPlaceholder = torch.zeros(
            [topKProbabilities.shape[0], self.cfg.numExperts], device=L.Device
        )
        topKProbabilitiesScattered = topKProbabilitiesScatteredPlaceholder.scatter(
            1, topKIndexes, topKProbabilities
        )
        return topKProbabilitiesScattered

    def __batch_frequency(self, probabilities, indices):
        expert_frequency = self.__scattered_probabiliites(probabilities, indices)
        expert_frequency = expert_frequency > 0
        expert_frequency = expert_frequency.long()
        expert_frequency = expert_frequency.sum(dim=0)

        return expert_frequency

    def __expert_sorted_nonzero_probability_indexes(
        self, topKProbabilities, topKIndexes
    ):
        _, _, nonzeroProbabilityIndexes, _expertSortedNonzeroTopKIndexes = (
            self.__required_inputs(topKProbabilities, topKIndexes)
        )

        expertSortedNonzeroProbabilityIndexes = nonzeroProbabilityIndexes[
            _expertSortedNonzeroTopKIndexes
        ]
        return expertSortedNonzeroProbabilityIndexes

    def __required_inputs(self, topKProbabilities, topKIndexes):
        topKIndexesFlattened = topKIndexes.flatten()
        topKProbabilitiesFlattened = topKProbabilities.flatten()

        nonzeroProbabilityIndexes = topKProbabilitiesFlattened.nonzero()
        nonzeroProbabilityIndexes = nonzeroProbabilityIndexes.squeeze(-1)

        nonzeroTopKIndexes = topKIndexesFlattened[nonzeroProbabilityIndexes]
        _, _expertSortedNonzeroTopKIndexes = nonzeroTopKIndexes.sort(0)

        return (
            topKIndexesFlattened,
            topKProbabilitiesFlattened,
            nonzeroProbabilityIndexes,
            _expertSortedNonzeroTopKIndexes,
        )

    def __computed_gating(
        self,
        probabilities: Tensor,
        indices: Tensor,
    ):
        flattened_probabilities = probabilities.flatten()
        nonzero_probability_indices = flattened_probabilities.nonzero()
        nonzero_probability_indices = nonzero_probability_indices.squeeze(-1)
        flattened_indices = indices.flatten()
        filtered_nonzero_indexes = flattened_indices[nonzero_probability_indices]
        _, _expert_sorted_filtered_nonzero_indexes = filtered_nonzero_indexes.sort(0)

        expert_sorted_filtered_nonzero_indexes = nonzero_probability_indices[
            _expert_sorted_filtered_nonzero_indexes
        ]

        expert_sorted_batch_indices = expert_sorted_filtered_nonzero_indexes.div(
            self.top_k, rounding_mode="trunc"
        )
        expert_sperted_probabilities = flattened_probabilities[
            expert_sorted_filtered_nonzero_indexes
        ]
        batch_experts_frequency, scattered_probabilities = self.__compute_frequency(
            probabilities,
            indices,
        )

        return (
            expert_sorted_batch_indices,
            expert_sperted_probabilities,
            scattered_probabilities,
            batch_experts_frequency,
            expert_sorted_filtered_nonzero_indexes,
        )

    def __compute_frequency(
        self,
        probabilities: Tensor,
        indices: Tensor,
    ):
        batch_size = probabilities.shape[0]
        probabilities_scattered_placeholder = torch.zeros(
            [batch_size, self.num_experts], device=device
        )
        scattered_probabilities = probabilities_scattered_placeholder.scatter(
            1, indices, probabilities
        )

        batch_experts_frequency = scattered_probabilities > 0
        batch_experts_frequency = batch_experts_frequency.long()
        batch_experts_frequency = batch_experts_frequency.sum(dim=0)
        return batch_experts_frequency, scattered_probabilities


class MixtureOfExperts(Module):
    def __init__(
        self,
        cfg: "MixtureOfExpertsConfig | ModelConfig",
    ) -> None:
        super().__init__()
        self.cfg = cfg
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
        self.input_batch_shape = None

    def forward(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        self.input_batch_shape = input_batch.size()

        input_batch_matrix, skip_mask = self.__reshape_inputs(input_batch, skip_mask)

        logits = self.router.compute_logit_scores(input_batch_matrix)
        probabilities, indices, skip_mask, loss = (
            self.sampler.sample_probabilities_and_indices(logits, skip_mask)
        )

        (
            expert_sorted_indices,
            expert_sorted_probabilities,
            scattered_probabilities,
            expert_frequency,
            expert_sorted_nonzero_probability_indices,
        ) = self.utils.compute_gating(probabilities, indices)

        expert_sorted_inputs = input_batch[expert_sorted_indices]

        feed_forward_output = self._compute_projections(
            expert_sorted_inputs, expert_frequency
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


class ExpertsLayer(Module):
    def __init__(
        self,
        cfg: "ExpertsConfig | ModelConfig",
        overrides: "ExpertsConfig | None" = None,
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
