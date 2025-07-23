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
class MixtureOfExpertsFeedForwardConfig(DataClassBase):
    weighted_parameters_flag: bool | None = field(
        default=None,
        metadata={
            "help": "When `True` the sepected parameters will be multiplied by their probs."
        },
    )


class MixtureOfExpertsFeedForward(Module):
    def __init__(
        self,
        cfg: "MixtureOfExpertsFeedForwardConfig | ModelConfig",
        overrides: "MixtureOfExpertsFeedForwardConfig | None" = None,
    ) -> None:
        super().__init__()
        config = getattr(cfg, "mixture_of_experts_config", cfg)
        self.cfg: MixtureOfExpertsFeedForwardConfig = self._overwrite_config(
            config, overrides
        )
        self.weighted_parameters_flag = self.cfg.weighted_parameters_flag

        self.router = RouterModel(cfg)
        self.sampler = SamplerModel(cfg)
        self.input_module = MixtureOfExperts(cfg)
        self.output_module = MixtureOfExperts(cfg, is_output_layer_flag=True)

        self.batch_size = None
        self.sequence_length = None
        self.output_shape = None

    def forward(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        input_batch_matrix, skip_mask = self._prepare_inputs(input_batch, skip_mask)
        logits = self.router.compute_logit_scores(input_batch_matrix)
        probabilities, indices, skip_mask, sampler_loss = (
            self.sampler.sample_probabilities_and_indices(logits, skip_mask)
        )
        input_projection, input_loss = self.input_module.compute_expert_outputs(
            input_batch_matrix, indices
        )
        output_projection, output_loss = self.output_module.compute_expert_outputs(
            input_projection, indices, probabilities
        )
        expert_mixture_output = output_projection.view(self.output_shape)
        loss = sampler_loss + input_loss + output_loss
        return expert_mixture_output, skip_mask, loss

    def _prepare_inputs(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        self.__resolve_output_shape(input_batch)
        input_batch = input_batch.reshape(self.batch_size * self.sequence_length, -1)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        return input_batch, skip_mask

    def __resolve_output_shape(self, input_batch: Tensor) -> None:
        input_shape = input_batch.shape
        if self.batch_size is not None:
            return
        if len(input_shape) > 2:
            self.batch_size, self.sequence_length, _ = input_shape
            self.output_shape = [self.batch_size, self.sequence_length, -1]
            return
        self.sequence_length = 1
        self.batch_size, _ = input_shape
        self.output_shape = [self.batch_size, -1]


@dataclass
class MixtureOfExpertsConfig(DataClassBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Expert output dimension"},
    )
    top_k: int | None = field(
        default=None,
        metadata={
            "help": "Top-k probabilities and indices to be selected from a distribution"
        },
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
    compute_expert_mixture_flag: bool | None = field(
        default=None,
        metadata={"help": "When true computes the expert mixture for this layer."},
    )
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
        overrides: "MixtureOfExpertsConfig | None" = None,
        is_output_layer_flag: bool = False,
    ):
        super().__init__()
        self.is_output_layer_flag = is_output_layer_flag
        config = getattr(cfg, self.__resolve_config_type(), cfg)
        self.cfg: "MixtureOfExpertsConfig" = self._overwrite_config(config, overrides)

        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.top_k = self.cfg.top_k
        self.num_experts = self.cfg.num_experts
        self.layer_norm_flag = self.cfg.layer_norm_flag
        self.dropout_probability = self.cfg.dropout_probability
        self.activation = self.cfg.activation
        self.model_type = self.cfg.model_type
        self.compute_expert_mixture_flag = self.cfg.compute_expert_mixture_flag
        self.weighted_parameters_flag = self.cfg.weighted_parameters_flag
        self._valudate_fields(self.cfg, MixtureOfExpertsConfig)

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
        probabilities: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        expert_outputs = []
        experts_indices_list = []
        loss = torch.tensor(0.0)
        for expert_index, expert_model in enumerate(self.expert_modules):
            expert_sample_indices = self.__get_expert_indices(indices, expert_index)
            if expert_sample_indices.numel() == 0:
                continue
            experts_indices_list.append(expert_sample_indices)
            expert_assigned_samples = input_batch[expert_sample_indices]
            output = expert_model(expert_assigned_samples)
            if isinstance(output, tuple):
                output, expert_loss = output
                expert_outputs.append(output)
                loss += expert_loss
                continue
            expert_outputs.append(output)

        experts_indices = torch.cat(experts_indices_list)
        output = torch.cat(expert_outputs, dim=0)
        output = self.__compute_expert_mixture(output, experts_indices, probabilities)

        return output, loss

    def __get_expert_indices(
        self,
        indices: Tensor,
        expert_index: int,
    ) -> Tensor:
        boolean_tensor = indices == expert_index
        flattened_tensor = boolean_tensor.sum(dim=-1)
        indices_for_expert = flattened_tensor.nonzero()
        return indices_for_expert.squeeze(dim=-1)

    def __compute_expert_mixture(
        self,
        experts_output: Tensor,
        indices: Tensor,
        probabilities: Tensor | None = None,
    ) -> Tensor:
        if not self.compute_expert_mixture_flag:
            return experts_output

        if self.weighted_parameters_flag:
            probabilities = probabilities.view(-1, 1)
            experts_output = experts_output * probabilities

        input_dim, output_dim = experts_output.shape
        output_shape = (input_dim // self.top_k, output_dim)
        output = torch.zeros(output_shape, dtype=experts_output.dtype, device=device)
        output.index_add_(0, indices, experts_output)
        return output


class MixtureOfAttentionHeads(MixtureOfExpertsFeedForward):
    def __init__(self, cfg: "ModelConfig") -> None:
        super().__init__(cfg)
        self.probabilities = None
        self.indices = None
        self.skip_mask = None
        self.total_loss

    def compute_expert_input_layer(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> Tensor:
        input_batch_matrix, skip_mask = self._prepare_inputs(input_batch, skip_mask)
        logits = self.router.compute_logit_scores(input_batch_matrix)
        self.probabilities, self.indices, self.skip_mask, sampler_loss = (
            self.sampler.sample_probabilities_and_indices(logits, skip_mask)
        )
        experts_projection, input_loss = self.input_experts.compute_expert_outputs(
            input_batch_matrix, self.indices
        )

        # inputBatchZeros = L.zeros(
        #     (batchSize * sequenceLength * self.cfg.topK, self.hiddenDim),
        #     dtype=expertsOutput.dtype,
        #     device=L.Device,
        # )
        # expertOutputMixture = inputBatchZeros.index_add(
        #     0, expertSortedNonzeroProbabilityIndexes, expertsOutput
        # )
        # topKIndexes = topKIndexes.view(batchSize, sequenceLength, -1)
        # self.expertSortedTopKBatchIndexes = expertSortedTopKBatchIndexes
        # self.expertSortedTopKBatchProbabilities = expertSortedTopKBatchProbabilities
        # self.batchExpertFrequency = batchExpertFrequency
        # self.expertSortedNonzeroProbabilityIndexes = (
        #     expertSortedNonzeroProbabilityIndexes
        # )
        # reshaped_projection_outputs = experts_projection.view(
        #     batch_size, sequence_length, self.cfg.top_k, self.hidden_dim
        # )

        return experts_projection.view(self.output_shape)

    def conputeOutputProjection(self, attentionOutput):
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
