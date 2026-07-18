from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.nn import Module
from emperor.sampler._config import SamplerConfig
from emperor.sampler._selection.losses import SamplerAuxiliaryLosses
from emperor.sampler._selection.validation import SamplerBaseValidator

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class SamplerBase(Module):
    VALIDATOR = SamplerBaseValidator

    def __init__(
        self,
        cfg: "SamplerConfig | ModelConfig",
        overrides: "SamplerConfig | None" = None,
    ) -> None:
        super().__init__()

        config = getattr(cfg, "sampler_model_config", cfg)
        self.cfg: SamplerConfig = self._override_config(config, overrides)

        self.top_k = self.cfg.top_k
        self.threshold = self.cfg.threshold
        self.noisy_topk_flag = self.cfg.noisy_topk_flag
        self.filter_above_threshold = self.cfg.filter_above_threshold
        self.num_topk_samples = self.cfg.num_topk_samples
        self.normalize_probabilities_flag = self.cfg.normalize_probabilities_flag
        self.num_experts = self.cfg.num_experts
        self.coefficient_of_variation_loss_weight = (
            self.cfg.coefficient_of_variation_loss_weight
        )
        self.switch_loss_weight = self.cfg.switch_loss_weight
        self.zero_centred_loss_weight = self.cfg.zero_centred_loss_weight
        self.mutual_information_loss_weight = self.cfg.mutual_information_loss_weight
        self._prepare_for_validation()
        self.VALIDATOR.validate(self)

        self.noise_epsilon = 1e-2
        self.auxiliary_loss_model = SamplerAuxiliaryLosses(self.cfg)
        self.register_buffer("default_loss", torch.zeros(()))
        self.updated_skip_mask = None
        self.auxiliary_loss = self.default_loss

    def get_probabilities_and_indices(
        self,
        router_logit_scores: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor]:
        self.VALIDATOR.validate_forward_inputs(self, router_logit_scores, skip_mask)
        masked_full_probabilities, masked_router_logits = (
            self.__compute_masked_probabilities(router_logit_scores, skip_mask)
        )
        selected_probabilities, selected_expert_indices = (
            self._sample_probabilities_and_indices(masked_full_probabilities)
        )
        auxiliary_loss = self._compute_loss(
            masked_router_logits,
            masked_full_probabilities,
            selected_probabilities,
            selected_expert_indices,
            skip_mask,
        )
        normalized_selected_probabilities = self._normalize_probabilities(
            selected_probabilities
        )
        updated_skip_mask = self.__update_mask_given_threshold(
            masked_full_probabilities, skip_mask
        )

        self.set_auxiliary_loss(auxiliary_loss)
        self.set_updated_skip_mask(updated_skip_mask)

        return (
            normalized_selected_probabilities,
            selected_expert_indices,
            updated_skip_mask,
            auxiliary_loss,
        )

    def _prepare_for_validation(self) -> None:
        pass

    def set_updated_skip_mask(self, updated_skip_mask: Tensor | None) -> None:
        self.updated_skip_mask = updated_skip_mask

    def set_auxiliary_loss(self, loss: Tensor) -> None:
        self.auxiliary_loss = loss

    def __compute_masked_probabilities(
        self,
        router_logit_scores: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        noise_adjusted_logits = self.__add_noise_to_logits(router_logit_scores)
        expert_dimension = 1
        full_probabilities = torch.softmax(noise_adjusted_logits, dim=expert_dimension)
        masked_full_probabilities, masked_router_logits = self.__apply_skip_mask(
            full_probabilities, noise_adjusted_logits, skip_mask
        )
        return masked_full_probabilities, masked_router_logits

    def __add_noise_to_logits(self, logit_scores_matrix: Tensor) -> Tensor:
        if not self.noisy_topk_flag:
            return logit_scores_matrix

        num_noisy_topk_components = 2
        expert_dimension = 1
        expert_logit_scores, raw_noise_standard_deviation = logit_scores_matrix.chunk(
            num_noisy_topk_components, dim=expert_dimension
        )
        noise_adjusted_logits = expert_logit_scores

        if self.training:
            noise_standard_deviation = (
                torch.sigmoid(raw_noise_standard_deviation) + self.noise_epsilon
            )
            standard_normal_noise = torch.randn_like(expert_logit_scores)
            # TODO: In the future maybe scale the noise by
            # `raw_noise_standard_deviation` or scale `standard_normal_noise`
            # by some `decaying_noise_amplifier` scalar
            scaled_noise = standard_normal_noise * noise_standard_deviation
            noise_adjusted_logits = expert_logit_scores + scaled_noise

        return noise_adjusted_logits

    def __apply_skip_mask(
        self,
        probabilities: Tensor,
        router_logit_scores: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.threshold == 0.0 or skip_mask is None:
            return probabilities, router_logit_scores
        skipped_input_mask = skip_mask == 0
        masked_probabilities = torch.masked_fill(probabilities, skipped_input_mask, 0)
        masked_router_logits = torch.masked_fill(
            router_logit_scores, skipped_input_mask, 0
        )
        return masked_probabilities, masked_router_logits

    def __update_mask_given_threshold(
        self,
        probabilities: Tensor,
        skip_mask: Tensor | None = None,
    ) -> Tensor | None:
        if skip_mask is None or self.threshold == 0.0:
            return skip_mask
        below_threshold_mask = probabilities < self.threshold
        expert_dimension = 1
        if self.filter_above_threshold:
            has_probability_at_or_above_threshold = (
                ~below_threshold_mask.all(dim=expert_dimension)
            ).unsqueeze(expert_dimension)
            updated_skip_mask = torch.masked_fill(
                skip_mask, has_probability_at_or_above_threshold, 0
            )
            return updated_skip_mask
        all_probabilities_below_threshold = below_threshold_mask.all(
            dim=expert_dimension
        ).unsqueeze(expert_dimension)
        updated_skip_mask = torch.masked_fill(
            skip_mask, all_probabilities_below_threshold, 0
        )
        return updated_skip_mask

    def _compute_loss(
        self,
        logits: Tensor,
        full_probabilities: Tensor,
        sampled_probabilities: Tensor,
        indices: Tensor,
        skip_mask: Tensor | None = None,
    ) -> Tensor:
        return self.default_loss

    def _sample_probabilities_and_indices(
        self, *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        raise NotImplementedError(
            "`_sample_probabilities_and_indices` has to be implemented in "
            "classes that inherit `SamplerBase`."
        )

    def _normalize_probabilities(self, probabilities: Tensor) -> Tensor:
        if self.normalize_probabilities_flag:
            expert_dimension = 1
            normalization_epsilon = 1e-6
            normalization_denominator = (
                torch.sum(probabilities, dim=expert_dimension, keepdim=True)
                + normalization_epsilon
            )
            detached_normalization_denominator = normalization_denominator.detach()
            normalized_probabilities = (
                probabilities / detached_normalization_denominator
            )
            return normalized_probabilities
        return probabilities
