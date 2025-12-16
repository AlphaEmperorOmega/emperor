import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass, field

from Emperor.base.layer import LayerStackConfig
from Emperor.sampler.model import SamplerModel
from Emperor.sampler.utils.samplers import SamplerConfig
from Emperor.sampler.utils.routers import RouterConfig, RouterModel
from Emperor.base.utils import ConfigBase, Module, device
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

        total_loss = sampler_loss + expert_loss
        return output, total_loss

    def __compute_experts(
        self, input_batch: Tensor, probabilities: Tensor, indices: Tensor
    ) -> tuple[Tensor, Tensor]:
        expert_outputs = []
        experts_indices_list = []
        total_loss = torch.tensor(0.0)
        expert_index = 0
        for expert_model in self.expert_modules:
            expert_sample_indices = self.__get_expert_indices(
                indices, probabilities, expert_index
            )
            expert_sample_probabilities = self.__get_expert_probabilities(
                indices, probabilities, expert_index
            )

            expert_index += 1
            if expert_sample_indices.numel() == 0:
                continue

            expert_output, loss = self.__compute_expert_output(
                expert_model,
                input_batch,
                expert_sample_indices,
                expert_sample_probabilities,
            )
            experts_indices_list.append(expert_sample_indices)
            expert_outputs.append(expert_output)
            total_loss = total_loss + loss

        experts_indices = torch.cat(experts_indices_list)
        output = torch.cat(expert_outputs, dim=0)
        output = self.__compute_expert_mixture(output, experts_indices, probabilities)

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
        probabilities: Tensor,
        expert_index: int,
    ) -> Tensor:
        if self.top_k == self.num_experts:
            num_samples = probabilities.shape[0]
            indices = torch.arange(num_samples)
            return indices

        boolean_tensor = indices == expert_index
        samples_for_current_expert = boolean_tensor
        if self.top_k > 1:
            samples_for_current_expert = samples_for_current_expert.sum(dim=-1)
        sample_indices_for_expert = samples_for_current_expert.nonzero()
        return sample_indices_for_expert.squeeze(dim=-1)

    def __get_expert_probabilities(
        self,
        indices: Tensor,
        probabilities: Tensor,
        expert_index: int,
    ) -> Tensor:
        if self.top_k == self.num_experts:
            return probabilities[:, expert_index]

        if self.__is_before():
            boolean_tensor = indices == expert_index
            probabilities = probabilities * boolean_tensor.float()
            probabilities = probabilities.sum(dim=-1)
            if self.top_k > 1:
                probabilities = probabilities[probabilities.nonzero()].squeeze(dim=-1)

        return probabilities

    def __compute_expert_output(
        self,
        expert_model: Callable,
        input_batch: Tensor,
        indices: Tensor,
        probabilities: Tensor | None,
    ) -> tuple[list[Tensor], Tensor]:
        expert_samples = input_batch[indices]
        if self.__is_before():
            expert_samples = self.__maybe_apply_probabilities(
                expert_samples, probabilities
            )

        output = expert_model(expert_samples)
        if isinstance(output, tuple):
            output, expert_loss = output
            return output, expert_loss
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
        input_dim, output_dim = experts_output.shape
        if not self.compute_expert_mixture_flag:
            output_shape = (input_dim, output_dim)
            output = torch.zeros(
                output_shape, dtype=experts_output.dtype, device=device
            )
            output.index_add_(0, indices, experts_output)

            return output

        if self.__is_after():
            experts_output = self.__maybe_apply_probabilities(
                experts_output, probabilities
            )

        output_shape = (input_dim // self.top_k, output_dim)
        output = torch.zeros(output_shape, dtype=experts_output.dtype, device=device)
        output.index_add_(0, indices, experts_output)
        return output

    def __is_after(self) -> bool:
        position_option = ExpertWeightingPositionOptions.AFTER_EXPERTS
        return self.weighting_position_option == position_option
