import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass, field

from Emperor.base.layer import LayerStackConfig
from Emperor.sampler.model import SamplerModel
from Emperor.sampler.utils.samplers import SamplerConfig
from Emperor.sampler.utils.routers import RouterConfig, RouterModel
from Emperor.base.utils import ConfigBase, Module
from Emperor.linears.options import LinearLayerStackOptions
from Emperor.experts.utils._validator import _Validator
from Emperor.experts.utils.enums import (
    ExpertWeightingPositionOptions,
    LayerRoleOptions,
)

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


@dataclass
class MixtureOfExpertsConfig(ConfigBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    layer_stack_option: LinearLayerStackOptions | None = field(
        default=None,
        metadata={"help": "Number of layers added to the router"},
    )
    top_k: int | None = field(
        default=None,
        metadata={
            "help": "Top-k probabilities and indices to be selected from a distribution"
        },
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
    weighting_position_option: ExpertWeightingPositionOptions | None = field(
        default=None,
        metadata={
            "help": "Dictates if the weights are applided before or after the experts."
        },
    )
    init_sampler_model_flag: bool | None = field(
        default=None,
        metadata={
            "help": "When `True` the `RouterModel `and `SamplerModel` will be added to the current layer."
        },
    )
    layer_role_option: LayerRoleOptions | None = field(
        default=None,
        metadata={
            "help": "Dictates if the layer is used as an normal, input or output MoE layer."
        },
    )
    router_model_config: "RouterConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )
    sampler_model_config: "SamplerConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )


class MixtureOfExperts(Module):
    def __init__(
        self,
        cfg: "MixtureOfExpertsConfig | ModelConfig",
        overrides: "MixtureOfExpertsConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "mixture_of_experts_config", cfg)
        self.cfg: "MixtureOfExpertsConfig" = self._overwrite_config(config, overrides)
        self.main_cfg = self._resolve_main_config(self.cfg, cfg)

        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.layer_stack_model = self.cfg.layer_stack_option
        self.top_k = self.cfg.top_k
        self.num_experts = self.cfg.num_experts
        self.compute_expert_mixture_flag = self.cfg.compute_expert_mixture_flag
        self.weighted_parameters_flag = self.cfg.weighted_parameters_flag
        self.init_sampler_model_flag = self.cfg.init_sampler_model_flag
        self.weighting_position_option = self.cfg.weighting_position_option
        self.layer_role_option = self.cfg.layer_role_option
        self.router_model_config = self.cfg.router_model_config
        self.sampler_model_config = self.cfg.sampler_model_config

        self.validator = _Validator(self)
        self.router, self.sampler = self.__maybe_create_router_and_sampler()
        self.expert_modules = self.__create_experts()

    def __maybe_create_router_and_sampler(
        self,
    ) -> tuple[RouterModel | None, SamplerModel | None]:
        if not self.cfg.init_sampler_model_flag:
            return None, None
        self.validator.ensure_router_config_exists()
        self.validator.ensure_sampler_config_exists()
        router_overrides = RouterConfig(input_dim=self.input_dim)
        router = RouterModel(self.router_model_config, router_overrides)
        sampler = SamplerModel(self.sampler_model_config)
        return router, sampler

    def __create_experts(self) -> nn.ModuleList:
        expert_list = []
        for _ in range(self.num_experts):
            overrides = LayerStackConfig(
                input_dim=self.input_dim, output_dim=self.output_dim
            )
            model_stack = self.layer_stack_model.value(
                self.main_cfg, overrides
            ).build_model()

            expert_list.append(model_stack)
        return nn.ModuleList(expert_list)

    def forward(
        self,
        input_batch: Tensor,
        probabilities: Tensor | None = None,
        indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor]:
        probabilities, indices, sampler_loss = self.__maybe_compute_expert_indices(
            input_batch, probabilities, indices
        )
        output, expert_loss = self.__compute_experts(
            input_batch, probabilities, indices
        )

        indices = torch.randperm(9)[:3]

        total_loss = sampler_loss + expert_loss
        return output, total_loss

    def __compute_experts(
        self, input_batch: Tensor, probabilities: Tensor, indices: Tensor
    ) -> tuple[Tensor, Tensor]:
        expert_outputs_list = []
        top_k_order_indices_list = []
        total_loss = torch.tensor(0.0)
        for expert_index, expert_model in enumerate(self.expert_modules):
            expert_sample_indices = self.__get_expert_indices(indices, expert_index)
            top_k_order_indices = self.__get_original_order_indices(
                indices, expert_index
            )
            expert_samples_probabilities = self.__get_expert_probabilities(
                top_k_order_indices, probabilities, expert_index
            )
            if expert_sample_indices is not None and expert_sample_indices.numel() == 0:
                continue

            expert_output, loss = self.__compute_expert_output(
                expert_model,
                input_batch,
                expert_sample_indices,
                expert_samples_probabilities,
            )
            total_loss = total_loss + loss
            expert_outputs_list.append(expert_output)
            if self.top_k != self.num_experts:
                top_k_order_indices_list.append(top_k_order_indices)

        expert_outputs = torch.cat(expert_outputs_list, dim=0)
        original_order_indices = None
        if self.top_k != self.num_experts:
            original_order_indices = torch.cat(top_k_order_indices_list)

        output = self.__compute_expert_mixture(
            expert_outputs, original_order_indices, probabilities
        )
        return output, total_loss

    def __maybe_compute_expert_indices(
        self,
        inputs: Tensor,
        probabilities: Tensor | None = None,
        indices: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor]:
        if indices is not None or probabilities is not None:
            self.validator.ensure_no_sampler_with_indices()
            return probabilities, indices, torch.tensor(0.0)
        self.validator.ensure_sampler_is_initialized()
        self.validator.ensure_external_probabilities_are_not_given(
            probabilities, indices
        )
        logits = self.router.compute_logit_scores(inputs)
        skip_mask = None
        probabilities, indices, skip_mask, sampler_loss = (
            self.sampler.sample_probabilities_and_indices(logits, skip_mask)
        )
        return probabilities, indices, sampler_loss

    def __get_expert_indices(
        self,
        indices: Tensor,
        expert_index: int,
    ) -> Tensor | None:
        if self.top_k == self.num_experts:
            return None

        samples_for_current_expert = indices == expert_index
        if self.top_k > 1:
            samples_for_current_expert = samples_for_current_expert.sum(dim=-1)
        sample_indices_for_expert = samples_for_current_expert.nonzero()
        return sample_indices_for_expert.flatten()

    def __get_original_order_indices(
        self,
        indices: Tensor,
        expert_index: int,
    ) -> Tensor | None:
        if self.top_k == self.num_experts:
            return None
        boolean_tensor = indices == expert_index
        original_order_indices = boolean_tensor.flatten()
        original_order_indices = original_order_indices.nonzero()
        return original_order_indices.squeeze(dim=-1)

    def __get_expert_probabilities(
        self,
        indices: Tensor,
        probabilities: Tensor,
        expert_index: int,
    ) -> Tensor:
        if self.__is_before():
            if self.top_k == self.num_experts:
                return probabilities[:, expert_index]
            probabilities = probabilities.flatten()
            probabilities = probabilities[indices]

        return probabilities

    def __compute_expert_output(
        self,
        expert_model: Callable,
        input_batch: Tensor,
        indices: Tensor,
        probabilities: Tensor | None,
    ) -> tuple[list[Tensor], Tensor]:
        expert_samples = input_batch
        if indices is not None:
            expert_samples = input_batch[indices]
        if self.__is_before():
            expert_samples = self.__maybe_apply_probabilities(
                expert_samples, probabilities
            )

        output = expert_model(expert_samples)
        if isinstance(output, tuple):
            return output
        return output, torch.tensor(0.0)

    def __maybe_apply_probabilities(
        self,
        logits: Tensor,
        probabilities: Tensor | None = None,
    ) -> Tensor:
        if not self.weighted_parameters_flag:
            return logits

        self.validator.ensure_probabilities_exist(probabilities)
        return logits * probabilities.reshape(-1, 1)

    def __is_before(self) -> bool:
        position_option = ExpertWeightingPositionOptions.BEFORE_EXPERTS
        return self.weighting_position_option == position_option

    def __compute_expert_mixture(
        self,
        experts_output: Tensor,
        indices: Tensor,
        probabilities: Tensor | None = None,
    ) -> Tensor:
        _, output_dim = experts_output.shape
        if self.top_k != self.num_experts:
            _, _index_sorted_indices = indices.sort(dim=0)
            experts_output = experts_output[_index_sorted_indices]
        if self.__is_after():
            experts_output = self.__maybe_apply_probabilities(
                experts_output, probabilities
            )

        experts_output = experts_output.view(-1, self.top_k, output_dim)
        if not self.compute_expert_mixture_flag or self.top_k == 1:
            return experts_output
        return experts_output.sum(dim=1)

    def __is_after(self) -> bool:
        position_option = ExpertWeightingPositionOptions.AFTER_EXPERTS
        return self.weighting_position_option == position_option
