import copy
import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass, field

from Emperor.base.utils import DataClassBase, Module, device
from Emperor.layers.utils.base import LayerBlock
from Emperor.layers.utils.enums import (
    ActivationFunctionOptions,
    LayerTypes,
)
from Emperor.layers.utils.linears import LinearLayer
from Emperor.layers.utils.routers import RouterModel
from Emperor.layers.utils.samplers import SamplerModel

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


@dataclass
class MixtureOfExpertsConfig(DataClassBase):
    weighted_parameters_flag: bool | None = field(
        default=None,
        metadata={
            "help": "When `True` the sepected parameters will be multiplied by their probs."
        },
    )


class MixtureOfExperts(Module):
    def __init__(
        self,
        cfg: "MixtureOfExpertsConfig | ModelConfig",
    ) -> None:
        super().__init__()
        self.cfg = cfg

        self.batch_size = None
        self.sequence_length = None
        self.output_shape = None
        self.router = RouterModel(cfg)
        self.sampler = SamplerModel(cfg)
        self.input_experts = ExpertsLayer(cfg)
        self.output_experts = ExpertsLayer(cfg)

    def forward(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        input_batch_matrix, skip_mask = self.__prepare_inputs(input_batch, skip_mask)
        logits = self.router.compute_logit_scores(input_batch_matrix)
        probabilities, indices, skip_mask, loss = (
            self.sampler.sample_probabilities_and_indices(logits, skip_mask)
        )
        input_experts_projection = self.input_experts(input_batch, indices)
        output_experts_projection = self.output_experts(
            input_experts_projection, indices
        )
        expert_mixture_output = self.__compute_expert_mixture(
            output_experts_projection,
            indices,
            probabilities,
        )

        return expert_mixture_output, skip_mask

    def __prepare_inputs(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        self.__resolve_output_shape(input_batch)
        input_batch = input_batch.reshape(-1, self.inputDim)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        return input_batch, skip_mask

    def __resolve_output_shape(self, input_batch: Tensor) -> None:
        input_shape = input_batch.shape
        if len(self._input_batch_shape) > 2:
            self.batch_size, self.sequence_length, _ = input_shape
            self.output_shape = [self.batch_size, self.sequence_length, -1]
            return
        self.sequence_length = 1
        self.batch_size, _ = input_shape
        self.output_shape = [self.batch_size, -1]

    def __compute_expert_mixture(
        self,
        experts_output: Tensor,
        expert_sorted_indices: Tensor,
        expert_sorted_probabilities: Tensor,
    ):
        if self.weighted_parameters_flag:
            flattened_probabilities = expert_sorted_probabilities.view(-1, 1)
            experts_output = experts_output * flattened_probabilities

        _, output_dim = experts_output.shape
        experts_shaped_zeros = torch.zeros(
            (self.batch_size * self.sequence_length, output_dim),
            dtype=experts_output.dtype,
            device=device,
        )

        output = experts_shaped_zeros.index_add(
            0, expert_sorted_indices, experts_output
        )

        return output.view(self.output_shape)


@dataclass
class ExpertsModuleConfig(DataClassBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Expert output dimension"},
    )
    dropout_probability: float | None = field(
        default=None,
        metadata={
            "help": "Float between (0.0, 1.0), that indicates the percentage of features being dropped out."
        },
    )
    layer_norm_flag: bool | None = field(
        default=None,
        metadata={"help": "Type of layer used for the experts."},
    )
    model_type: LayerTypes | None = field(
        default=None,
        metadata={"help": "Type of layer used for the experts."},
    )
    activation: ActivationFunctionOptions | None = field(
        default=None,
        metadata={"help": "Activation function for the experts."},
    )
    num_experts: int | None = field(
        default=None,
        metadata={"help": "Number of experts in the model"},
    )


class ExpertsModule(Module):
    def __init__(
        self,
        cfg: "ExpertsModuleConfig | ModelConfig",
        overrides: "ExpertsModuleConfig | None" = None,
        is_output_layer_flag: bool = False,
    ):
        super().__init__()
        self.is_output_layer_flag = is_output_layer_flag
        config = getattr(cfg, self.__resolve_config_type(), cfg)
        self.cfg: "ExpertsModuleConfig" = self._overwrite_config(config, overrides)

        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.num_experts = self.cfg.num_experts
        self.layer_norm_flag = self.cfg.layer_norm_flag
        self.dropout_probability = self.cfg.dropout_probability
        self.activation = self.cfg.activation
        self.model_type = self.cfg.model_type
        self._valudate_fields(self.cfg, ExpertsModuleConfig)

        self.expert_modules = self.__create_experts(cfg)

    def __resolve_config_type(self) -> str:
        if self.is_output_layer_flag:
            return "output_moe_layer_config"
        return "input_moe_layer_config"

    def __create_experts(self, cfg: "ModelConfig") -> nn.ModuleList:
        expert_list = []
        for _ in range(self.num_experts):
            model = self.model_type.value(self.__resolve_model_type_overrides(cfg))
            layer_norm_output_dim = self.output_dim if self.layer_norm_flag else None
            layer_block = LayerBlock(
                model=model,
                activation_function=self.activation.value,
                layer_norm_output_dim=layer_norm_output_dim,
                dropout_probability=self.dropout_probability,
            )
            expert_list.append(layer_block)
        return nn.ModuleList(expert_list)

    def __resolve_model_type_overrides(self, cfg: "ModelConfig"):
        c = copy.deepcopy(cfg)
        if issubclass(self.model_type.value, LinearLayer):
            c.linear_layer_model_config.input_dim = self.input_dim
            c.linear_layer_model_config.output_dim = self.output_dim
            return c
        c.mixture_model_config.input_dim = self.input_dim
        c.mixture_model_config.output_dim = self.output_dim
        return c

    def compute_expert_outputs(
        self,
        input_batch: Tensor,
        indices: Tensor,
    ) -> Tensor | tuple[Tensor, Tensor]:
        expert_outputs = []
        layer_loss = torch.tensor(0.0)
        is_tuple = False
        for expert_index, expert_model in enumerate(self.expert_modules):
            expert_sample_indices = self.__get_expert_indices(indices, expert_index)
            if expert_sample_indices.numel() == 0:
                continue
            expert_assigned_samples = input_batch[expert_sample_indices]
            output = expert_model(expert_assigned_samples)
            # TODO: Refactor in the future when objects are to store data
            # instead of tuples
            is_tuple = isinstance(output, tuple)
            if is_tuple:
                output, loss = output
                expert_outputs.append(output)
                layer_loss += loss
                continue
            expert_outputs.append(output)
        if is_tuple:
            experts_output = torch.cat(expert_outputs, dim=0)
            return experts_output, layer_loss
        return torch.cat(expert_outputs, dim=0)

    def __get_expert_indices(
        self,
        indices: Tensor,
        expert_index: int,
    ) -> Tensor:
        boolean_tensor = indices == expert_index
        flattened_tensor = boolean_tensor.sum(dim=-1)
        indices_for_expert = flattened_tensor.nonzero()
        return indices_for_expert.squeeze(dim=-1)


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
