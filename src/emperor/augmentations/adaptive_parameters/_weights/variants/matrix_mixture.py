from copy import deepcopy
from dataclasses import replace

from torch import Tensor

from emperor.augmentations.adaptive_parameters._decay import DecayPolicy
from emperor.augmentations.adaptive_parameters._weights.config import (
    MatrixWeightsMixtureConfig,
)
from emperor.augmentations.adaptive_parameters._weights.validation import (
    MatrixWeightsMixtureValidator,
)
from emperor.nn import Module
from emperor.sampler import RouterConfig


class MatrixWeightsMixture(Module):
    VALIDATOR = MatrixWeightsMixtureValidator

    def __init__(
        self,
        cfg: MatrixWeightsMixtureConfig,
        overrides: MatrixWeightsMixtureConfig | None = None,
    ):
        super().__init__()
        self.cfg = self.__resolve_config(self._override_config(cfg, overrides))
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.num_experts = self.cfg.num_experts
        self.top_k = self.cfg.top_k
        self.parameter_bank_shape = (self.num_experts, self.input_dim, self.output_dim)
        self.parameter_bank = self._init_parameter_bank(self.parameter_bank_shape)
        self.sampler = self.__init_sampler()
        self._decay_policy = DecayPolicy(self.cfg)

    @classmethod
    def __resolve_config(cls, cfg, model_config=None):
        cls.VALIDATOR.validate_initialization_config(cfg)
        resolved_config = deepcopy(cfg)
        resolved_config.sampler_config = cls.__resolve_sampler_config(
            resolved_config, model_config
        )
        cls.VALIDATOR.validate_resolved_config(resolved_config)
        return resolved_config

    @classmethod
    def __resolve_sampler_config(cls, cfg, model_config):
        sampler_config = cfg.sampler_config
        cls.VALIDATOR.validate_sampler_fields(sampler_config, cfg)
        sampler_config.num_experts = cfg.num_experts
        sampler_config.top_k = cfg.top_k
        sampler_defaults = dict(
            threshold=0.0,
            filter_above_threshold=False,
            num_topk_samples=0,
            noisy_topk_flag=False,
            coefficient_of_variation_loss_weight=0.0,
            switch_loss_weight=0.0,
            zero_centred_loss_weight=0.0,
            mutual_information_loss_weight=0.0,
        )
        for name, default in sampler_defaults.items():
            cls.VALIDATOR.validate_sampler_option(
                name, getattr(sampler_config, name), default
            )
            setattr(sampler_config, name, default)
        if sampler_config.normalize_probabilities_flag is None:
            sampler_config.normalize_probabilities_flag = cfg.top_k > 1
        sampler_config.router_config = cls.__resolve_router_config(cfg, model_config)
        return sampler_config

    @classmethod
    def __resolve_router_config(cls, cfg, model_config):
        router_config = cfg.sampler_config.router_config
        cls.VALIDATOR.validate_router_config(router_config)
        if router_config is None:
            router_config = RouterConfig()
        cls.VALIDATOR.validate_router_fields(router_config, cfg)
        router_config.input_dim = cfg.input_dim
        router_config.num_experts = cfg.num_experts
        router_config.noisy_topk_flag = False
        if router_config.model_config is None:
            if cfg.model_config is not None:
                model_config = cfg.model_config
            router_config.model_config = deepcopy(model_config)
        cls.VALIDATOR.validate_generator_stack(router_config.model_config)
        return router_config

    def __init_sampler(self):
        self.VALIDATOR.validate_sampler_config(self.cfg.sampler_config)
        return self.cfg.sampler_config.build()

    @classmethod
    def validate_owner_config(cls, cfg, *, input_dim, output_dim, model_config) -> None:
        """Preflight the effective configuration without constructing parameters."""
        for name, value in (("input_dim", input_dim), ("output_dim", output_dim)):
            cls.VALIDATOR.validate_matching_value(name, getattr(cfg, name), value)
        owner_config = replace(cfg, input_dim=input_dim, output_dim=output_dim)
        cls.__resolve_config(owner_config, model_config)

    def forward(
        self,
        weight_params: Tensor,
        context: Tensor,
    ) -> Tensor:
        self.VALIDATOR.validate_forward_inputs(self, weight_params, context)
        self.VALIDATOR.validate_sampler_available(self.sampler)
        probabilities, indices, skip_mask, loss = (
            self.sampler.sample_probabilities_and_indices(context)
        )
        self.VALIDATOR.validate_sampler_result(skip_mask, loss)
        self.VALIDATOR.validate_route(self, probabilities, indices)
        generated_parameters = self.__reduce_mixture(probabilities, indices)
        decayed_parameters = self._decay_policy(weight_params)
        return decayed_parameters + generated_parameters

    def __reduce_mixture(self, probabilities: Tensor, indices: Tensor | None) -> Tensor:
        selected_parameters = self.__select_parameters(indices)
        weighted_parameters = self.__apply_probabilities(
            selected_parameters, probabilities
        )
        return weighted_parameters.sum(dim=1)

    def __select_parameters(self, indices: Tensor | None) -> Tensor:
        if indices is None:
            return self.parameter_bank
        expert_indices_per_context = indices.reshape(-1, self.top_k)
        return self.parameter_bank[expert_indices_per_context]

    def __apply_probabilities(
        self, selected_parameters: Tensor, probabilities: Tensor
    ) -> Tensor:
        probability_shape = (-1, self.top_k) + (1,) * (self.parameter_bank.ndim - 1)
        return selected_parameters * probabilities.reshape(probability_shape)
