import torch
import torch.nn as nn

from torch import Tensor
from dataclasses import dataclass, field
from Emperor.sampler.model import SamplerModel
from Emperor.base.layer import LayerStackConfig
from Emperor.base.utils import ConfigBase, Module
from Emperor.experts.utils._validator import _Validator
from Emperor.experts.utils.enums import (
    ExpertWeightingPositionOptions,
    InitSamplerOptions,
)

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from Emperor.config import ModelConfig
    from Emperor.sampler.utils.samplers import SamplerConfig
    from Emperor.linears.options import LinearLayerStackOptions
    from Emperor.sampler.utils.routers import RouterConfig, RouterModel


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
    layer_stack_option: "LinearLayerStackOptions | None" = field(
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
    init_sampler_option: InitSamplerOptions | None = field(
        default=None,
        metadata={
            "help": "Use `SHARED` for a single router and sampler across all layers, or `LAYER` for one per layer."
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
        self.main_cfg: "LayerStackConfig" = self._resolve_main_config(self.cfg, cfg)

        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.layer_stack_model: "LinearLayerStackOptions" = self.cfg.layer_stack_option
        self.top_k: int = self.cfg.top_k
        self.num_experts: int = self.cfg.num_experts
        self.compute_expert_mixture_flag: bool = self.cfg.compute_expert_mixture_flag
        self.weighted_parameters_flag: bool = self.cfg.weighted_parameters_flag
        self.init_sampler_option: "InitSamplerOptions" = self.cfg.init_sampler_option
        self.weighting_position_option: "ExpertWeightingPositionOptions" = (
            self.cfg.weighting_position_option
        )
        self.router_model_config: "RouterConfig" = self.cfg.router_model_config
        self.sampler_model_config: "SamplerConfig" = self.cfg.sampler_model_config

        self.validator = _Validator(self)
        self.router, self.sampler = self.__maybe_create_router_and_sampler()
        self.expert_modules = self.__create_experts()

    def get_top_k(self) -> int:
        return self.top_k

    def __maybe_create_router_and_sampler(
        self,
    ) -> tuple["RouterModel | None", "SamplerModel | None"]:
        from Emperor.sampler.utils.routers import RouterConfig, RouterModel
        if self.init_sampler_option != InitSamplerOptions.LAYER:
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
    ) -> tuple[Tensor, Tensor]:
        probabilities, indices, sampler_loss = self._maybe_compute_expert_indices(
            input_batch, probabilities, indices
        )

        expert_outputs, sample_indices_for_all_experts, probabilities, expert_loss = (
            self._compute_experts(input_batch, probabilities, indices)
        )

        output = self.__compute_expert_mixture(
            expert_outputs, sample_indices_for_all_experts, probabilities
        )

        total_loss = sampler_loss + expert_loss
        return output, total_loss

    def _maybe_compute_expert_indices(
        self,
        inputs: Tensor,
        probabilities: Tensor | None = None,
        indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.init_sampler_option != InitSamplerOptions.LAYER or (
            indices is not None or probabilities is not None
        ):
            return probabilities, indices, torch.tensor(0.0)
        self.validator.ensure_sampler_is_initialized()
        self.validator.ensure_external_probabilities_are_not_given(
            probabilities, indices
        )
        logits = self.router.compute_logit_scores(inputs)
        # TODO: In the future see if `skip_mask` needs to be implemented
        skip_mask = None
        probabilities, indices, skip_mask, sampler_loss = (
            self.sampler.sample_probabilities_and_indices(logits, skip_mask)
        )
        return probabilities, indices, sampler_loss

    def _compute_experts(
        self, input_batch: Tensor, probabilities: Tensor, indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        expert_outputs_list = []
        sample_indices_for_expert_list = []
        total_loss = torch.tensor(0.0)
        for expert_index, expert_model in enumerate(self.expert_modules):
            expert_sample_indices = self.__get_expert_indices(indices, expert_index)
            sample_indices_for_expert = self._get_sample_indices_for_expert(
                indices, expert_index
            )
            expert_samples_probabilities = self.__get_expert_probabilities(
                sample_indices_for_expert, probabilities, expert_index
            )
            if expert_sample_indices is not None and expert_sample_indices.numel() == 0:
                empty_tensor = torch.tensor([], dtype=torch.int16)
                sample_indices_for_expert_list.append(empty_tensor)
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
                sample_indices_for_expert_list.append(sample_indices_for_expert)

        expert_outputs = torch.cat(expert_outputs_list, dim=0)
        sample_indices_for_all_experts = None
        if self.top_k != self.num_experts:
            sample_indices_for_all_experts = torch.cat(sample_indices_for_expert_list)

        return expert_outputs, sample_indices_for_all_experts, probabilities, total_loss

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

    def _get_sample_indices_for_expert(
        self,
        indices: Tensor,
        expert_index: int,
    ) -> Tensor | None:
        if self.top_k == self.num_experts:
            return None
        boolean_tensor = indices == expert_index
        expert_sample_indices = boolean_tensor.flatten()
        expert_sample_indices = expert_sample_indices.nonzero()
        return expert_sample_indices.squeeze(dim=-1)

    def __get_expert_probabilities(
        self,
        indices: Tensor,
        probabilities: Tensor,
        expert_index: int,
    ) -> Tensor:
        if self._should_apply_probabilities_before():
            if self.top_k == self.num_experts:
                return probabilities[:, expert_index]
            probabilities = probabilities.flatten()
            probabilities = probabilities[indices]

        return probabilities

    def _should_apply_probabilities_before(self) -> bool:
        position_option = ExpertWeightingPositionOptions.BEFORE_EXPERTS
        return self.weighting_position_option == position_option


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

        expert_samples = self._maybe_apply_probabilities_before(
            expert_samples, probabilities
        )

        output = expert_model(expert_samples)
        if isinstance(output, tuple):
            return output
        return output, torch.tensor(0.0)

    def _maybe_apply_probabilities_before(
        self,
        experts_output: Tensor,
        probabilities: Tensor | None = None,
    ) -> Tensor:
        position_option = ExpertWeightingPositionOptions.BEFORE_EXPERTS
        if self.weighting_position_option == position_option:
            experts_output = self.__maybe_apply_probabilities(
                experts_output, probabilities
            )
        return experts_output

    def __maybe_apply_probabilities(
        self,
        logits: Tensor,
        probabilities: Tensor | None = None,
    ) -> Tensor:
        if not self.weighted_parameters_flag:
            return logits

        self.validator.ensure_probabilities_exist(probabilities)
        return logits * probabilities.reshape(-1, 1)


    def __compute_expert_mixture(
        self,
        experts_output: Tensor,
        indices: Tensor,
        probabilities: Tensor | None = None,
    ) -> Tensor:
        output_dim = experts_output.size(-1)
        if self.top_k != self.num_experts:
            _, _index_sorted_indices = indices.sort(dim=0)
            experts_output = experts_output[_index_sorted_indices]

        experts_output = self._maybe_apply_probabilities_after(
            experts_output, probabilities
        )

        if not self.compute_expert_mixture_flag or self.top_k == 1:
            return experts_output

        if self.top_k > 1:
            experts_output = experts_output.view(-1, self.top_k, output_dim)
        return experts_output.sum(dim=1)

    def _maybe_apply_probabilities_after(
        self,
        experts_output: Tensor,
        probabilities: Tensor | None = None,
    ) -> Tensor:
        position_option = ExpertWeightingPositionOptions.AFTER_EXPERTS
        if self.weighting_position_option == position_option:
            experts_output = self.__maybe_apply_probabilities(
                experts_output, probabilities
            )
        return experts_output




class MixtureOfExpertsMap(MixtureOfExperts):
    def __init__(
        self,
        cfg: "MixtureOfExpertsConfig | ModelConfig",
        overrides: "MixtureOfExpertsConfig | None" = None,
    ):
        overrides = self.__update_overrides(overrides)
        super().__init__(cfg, overrides)
        self.sample_indices_for_all_experts = []

        self.sample_indices_for_all_experts = None
        self.sample_probabilities = None

    def __update_overrides(self, overrides: "MixtureOfExpertsConfig | None" = None) -> MixtureOfExpertsConfig:
        if overrides is None:
            return MixtureOfExpertsConfig(
                weighted_parameters_flag=False,
                compute_expert_mixture_flag=False,
                init_sampler_option=InitSamplerOptions.DISABLED
            )
        overrides.weighted_parameters_flag = False
        overrides.compute_expert_mixture_flag = False
        overrides.init_sampler_option = InitSamplerOptions.DISABLED

        return overrides



class MixtureOfExpertsReduce(MixtureOfExperts):
    def __init__(
        self,
        cfg: "MixtureOfExpertsConfig | ModelConfig",
        overrides: "MixtureOfExpertsConfig | None" = None,
    ):
        overrides = self.__update_overrides(overrides)
        super().__init__(cfg, overrides)

    def __update_overrides(self, overrides: "MixtureOfExpertsConfig | None" = None) -> MixtureOfExpertsConfig:
        if overrides is None:
            return MixtureOfExpertsConfig(
                weighted_parameters_flag=True,
                compute_expert_mixture_flag=True,
                weighting_position_option=ExpertWeightingPositionOptions.AFTER_EXPERTS,
                init_sampler_option=InitSamplerOptions.DISABLED,
            )
        overrides.weighted_parameters_flag = True
        overrides.compute_expert_mixture_flag = True
        overrides.weighting_position_option = ExpertWeightingPositionOptions.AFTER_EXPERTS
        overrides.init_sampler_option = InitSamplerOptions.DISABLED

        return overrides

    def forward(
        self,
        input_batch: Tensor,
        probabilities: Tensor | None = None,
        indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:

        expert_outputs, sample_indices_for_all_experts, probabilities, expert_loss = self._compute_experts(
            input_batch, probabilities, indices
        )
        output = self.__compute_expert_mixture(
            expert_outputs, sample_indices_for_all_experts, probabilities
        )

        total_loss = expert_loss
        return output, total_loss

    def _compute_experts(
        self,
        input_batch: Tensor,
        probabilities: Tensor,
        indices: Tensor,
    ) -> tuple[Tensor, Tensor]:

        expert_outputs_list = []
        sample_indices_for_expert_list = []
        total_loss = torch.tensor(0.0)
        for expert_index, expert_model in enumerate(self.expert_modules):
            expert_indices = self._get_sample_indices_for_expert(
                indices, expert_index
            )

            if expert_indices is not None and expert_indices.numel() == 0:
                continue

            expert_output, loss = self.__compute_expert_output(
                expert_model,
                input_batch,
                expert_indices,
            )
            total_loss = total_loss + loss
            expert_outputs_list.append(expert_output)
            sample_indices_for_expert_list.append(expert_indices)

        expert_outputs = torch.cat(expert_outputs_list, dim=0)
        sample_indices_for_expert_tensor = torch.cat(sample_indices_for_expert_list)

        return (
            expert_outputs,
            sample_indices_for_expert_tensor,
            probabilities,
            total_loss,
        )

    def __compute_expert_output(
        self,
        expert_model: Callable,
        input_batch: Tensor,
        indices: Tensor,
    ) -> tuple[list[Tensor], Tensor]:
        expert_samples = input_batch
        if indices is not None:
            expert_samples = input_batch[indices]

        output = expert_model(expert_samples)
        if isinstance(output, tuple):
            return output
        return output, torch.tensor(0.0)

    def __compute_expert_mixture(
        self,
        experts_output: Tensor,
        indices: Tensor,
        probabilities: Tensor | None = None,
    ) -> Tensor:
        output_dim = experts_output.size(-1)
        if self.top_k != self.num_experts:
            _, _index_sorted_indices = indices.sort(dim=0)
            experts_output = experts_output[_index_sorted_indices]

        experts_output = self._maybe_apply_probabilities_after(experts_output, probabilities)

        if not self.compute_expert_mixture_flag or self.top_k == 1:
            return experts_output

        if self.top_k > 1:
            experts_output = experts_output.view(-1, self.top_k, output_dim)
        return experts_output.sum(dim=1)
