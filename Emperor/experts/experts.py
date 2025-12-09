import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass, field

from Emperor.sampler.model import SamplerModel
from Emperor.sampler.utils.samplers import SamplerConfig
from Emperor.sampler.utils.routers import RouterConfig, RouterModel
from Emperor.base.utils import ConfigBase, Module, device
from Emperor.linears.options import LinearLayerStackOptions
from Emperor.experts.utils.enums import (
    ExpertWeightingPositionOptions,
    LayerRoleOptions,
)


from typing import TYPE_CHECKING, Callable


if TYPE_CHECKING:
    from Emperor.config import ModelConfig


__all__ = ["MixtureOfExpertsConfig", "MixtureOfExperts"]


class _Validator:
    @staticmethod
    def ensure_sampler_is_initialized(
        init_sampler_model_flag: bool,
    ) -> None:
        if not init_sampler_model_flag:
            raise ValueError(
                "The `init_sampler_model_flag` must be set to `True` to initialize the `RouterModel` and `SamplerModel` when `indices` are not provided."
            )

    @staticmethod
    def ensure_external_probabilities_are_not_given(
        probabilities: Tensor | None,
        indices: Tensor | None,
    ) -> None:
        if indices is not None or probabilities is not None:
            raise ValueError(
                "Indices must be None. Providing indices where they are not expected is not allowed."
            )

    @staticmethod
    def ensure_no_sampler_with_indices(
        init_sampler_model_flag: bool,
    ) -> None:
        if init_sampler_model_flag:
            raise ValueError(
                "Invalid configuration: `init_sampler_model_flag` must be set to `False` when `indices` are provided. This prevents creating duplicate `RouterModel` and `SamplerModel` instances in the current layer."
            )

    @staticmethod
    def ensure_probabilities_exist(probabilities: Tensor | None) -> None:
        if probabilities is None:
            raise ValueError(
                "Missing input: `probabilities` must be supplied when `indices` are used to ensure accurate weighting and processing of inputs."
            )

    @staticmethod
    def ensure_router_config_exists(router_model_config: "RouterConfig | None") -> None:
        if router_model_config is None:
            raise ValueError(
                "Configuration Error: `router_model_config` must be defined to properly initialize and utilize the router model in the mixture of experts layer."
            )

    @staticmethod
    def ensure_sampler_config_exists(
        sampler_model_config: "SamplerConfig | None",
    ) -> None:
        if sampler_model_config is None:
            raise ValueError(
                "Configuration Error: `sampler_model_config` must be defined to properly initialize and utilize the sampler model in the mixture of experts layer."
            )


@dataclass
class MixtureOfExpertsConfig(ConfigBase):
    layer_stack_option: LinearLayerStackOptions = field(
        default=LinearLayerStackOptions.BASE,
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

        self.layer_stack_model = self.cfg.layer_stack_option.value
        self.top_k = self.cfg.top_k
        self.num_experts = self.cfg.num_experts
        self.compute_expert_mixture_flag = self.cfg.compute_expert_mixture_flag
        self.weighted_parameters_flag = self.cfg.weighted_parameters_flag
        self.init_sampler_model_flag = self.cfg.init_sampler_model_flag
        self.weighting_position_option = self.cfg.weighting_position_option
        self.layer_role_option = self.cfg.layer_role_option
        self.router_model_config = self.cfg.router_model_config
        self.sampler_model_config = self.cfg.sampler_model_config

        self.expert_modules = self.__create_experts()
        self.router, self.sampler = self.__maybe_create_router_and_sampler()

    def __maybe_create_router_and_sampler(
        self,
    ) -> tuple[RouterModel | None, SamplerModel | None]:
        if not self.cfg.init_sampler_model_flag:
            return None, None
        _Validator.ensure_router_config_exists(self.router_model_config)
        _Validator.ensure_sampler_config_exists(self.sampler_model_config)
        router = RouterModel(self.router_model_config)
        sampler = SamplerModel(self.sampler_model_config)
        return router, sampler

    def __create_experts(self) -> nn.ModuleList:
        expert_list = []
        for _ in range(self.num_experts):
            model_stack = self.layer_stack_model(self.main_cfg).build_model()

            expert_list.append(model_stack)
        return nn.ModuleList(expert_list)

    def forward(
        self,
        input_batch: Tensor,
        probabilities: Tensor | None = None,
        indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        probabilities, indices, sampler_loss = self.__maybe_compute_expert_indices(
            input_batch, probabilities, indices
        )

        expert_outputs = []
        experts_indices_list = []
        total_loss = torch.tensor(0.0) + sampler_loss
        for expert_index, expert_model in enumerate(self.expert_modules):
            expert_sample_indices = self.__get_expert_indices(indices, expert_index)
            if expert_sample_indices.numel() == 0:
                continue
            expert_output, loss = self.__compute_experts_output(
                expert_model, input_batch, expert_sample_indices, probabilities
            )
            experts_indices_list.append(expert_sample_indices)
            expert_outputs.append(expert_output)
            total_loss += loss

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
            _Validator.ensure_no_sampler_with_indices(self.init_sampler_model_flag)
            return probabilities, indices, torch.tensor(0.0)
        _Validator.ensure_sampler_is_initialized(self.init_sampler_model_flag)
        _Validator.ensure_external_probabilities_are_not_given(probabilities, indices)
        logits = self.router.compute_logit_scores(inputs)
        skip_mask = None
        probabilities, indices, skip_mask, sampler_loss = (
            self.sampler.sample_probabilities_and_indices(logits, skip_mask)
        )
        return probabilities, indices, sampler_loss

    def __compute_experts_output(
        self,
        expert_model: Callable,
        input_batch: Tensor,
        indices: Tensor,
        probabilities: Tensor | None,
    ) -> tuple[list[Tensor], Tensor]:
        expert_samples = input_batch[indices]
        expert_samples = self.__maybe_apply_probabilities(
            expert_samples, probabilities, indices
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
        indices: Tensor | None = None,
    ) -> Tensor:
        if self.__should_apply_weights():
            return logits

        _Validator.ensure_probabilities_exist(probabilities)
        if indices is not None:
            probabilities = probabilities[indices]
        probabilities = probabilities.view(-1, 1)
        return logits * probabilities

    def __should_apply_weights(self, before_flag=False) -> bool:
        is_weighted = self.weighted_parameters_flag
        position_option = ExpertWeightingPositionOptions.BEFORE_EXPERTS
        if before_flag:
            position_option = ExpertWeightingPositionOptions.AFTER_EXPERTS
        is_before = self.weighting_position_option == position_option
        return is_weighted and is_before

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

        experts_output = self.__maybe_apply_probabilities(experts_output, probabilities)

        input_dim, output_dim = experts_output.shape
        output_shape = (input_dim // self.top_k, output_dim)
        output = torch.zeros(output_shape, dtype=experts_output.dtype, device=device)
        output.index_add_(0, indices, experts_output)
        return output
