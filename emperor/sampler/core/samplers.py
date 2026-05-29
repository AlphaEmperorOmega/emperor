import torch
from torch import Tensor
from emperor.base.utils import expand_dims
from emperor.base.utils import Module
from emperor.sampler.core.config import SamplerConfig
from emperor.sampler.core.losses import SamplerAuxiliaryLosses
from emperor.sampler.core._validator import (
    SamplerBaseValidator,
    SamplerSparseValidator,
    SamplerTopkValidator,
    SamplerFullValidator,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class SamplerBase(Module):
    def __init__(
        self,
        cfg: "SamplerConfig | ModelConfig",
        overrides: "SamplerConfig | None" = None,
    ) -> None:
        super().__init__()

        config = getattr(cfg, "sampler_model_config", cfg)
        self.cfg: "SamplerConfig" = self._override_config(config, overrides)

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
        SamplerBaseValidator.validate(self)

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
        SamplerBaseValidator.validate_router_logit_scores(self, router_logit_scores)
        SamplerBaseValidator.validate_skip_mask(router_logit_scores, skip_mask)
        full_probabilities, logits = self.__compute_masked_probabilities(
            router_logit_scores, skip_mask
        )
        selected_probs, selected_indices = self._sample_probabilities_and_indices(
            full_probabilities
        )
        loss = self._compute_loss(
            logits, full_probabilities, selected_probs, selected_indices, skip_mask
        )
        probabilities = self._normalize_probabilities(selected_probs)
        skip_mask = self.__update_mask_given_threshold(full_probabilities, skip_mask)

        self.set_auxiliary_loss(loss)
        self.set_updated_skip_mask(skip_mask)

        return probabilities, selected_indices, skip_mask, loss

    def set_updated_skip_mask(self, updated_skip_mask: Tensor | None) -> None:
        self.updated_skip_mask = updated_skip_mask

    def set_auxiliary_loss(self, loss: Tensor) -> None:
        self.auxiliary_loss = loss

    def __compute_masked_probabilities(
        self,
        router_logit_scores: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        logits = self.__add_noise_to_logits(router_logit_scores)
        probabilities = torch.softmax(logits, dim=-1)
        masked_probabilities, masked_logit_scores = self.__apply_skip_mask(
            probabilities, logits, skip_mask
        )
        return masked_probabilities, masked_logit_scores

    def __add_noise_to_logits(self, logit_scores_matrix: Tensor) -> Tensor:
        if not self.noisy_topk_flag:
            return logit_scores_matrix

        num_chunks = 2
        logit_scores_matrix_chunk, raw_noise_std_matrix = logit_scores_matrix.chunk(
            num_chunks, dim=-1
        )
        logit_scores_matrix = logit_scores_matrix_chunk

        if self.training:
            noise_std = torch.sigmoid(raw_noise_std_matrix) + self.noise_epsilon
            noise = torch.randn_like(logit_scores_matrix)
            # TODO: In the future maybe scale the noise by `raw_noise_std`
            # or scale `noise` by some `decaying_noise_amplifier` scalar
            noisy_logit_scores = logit_scores_matrix + noise * noise_std
            logit_scores_matrix = noisy_logit_scores

        return logit_scores_matrix

    def __apply_skip_mask(
        self,
        probabilities: Tensor,
        router_logit_scores: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.threshold == 0.0 or skip_mask is None:
            return probabilities, router_logit_scores
        mask = skip_mask == 0
        probabilities = torch.masked_fill(probabilities, mask, 0)
        router_logit_scores = torch.masked_fill(router_logit_scores, mask, 0)
        return probabilities, router_logit_scores

    def __update_mask_given_threshold(
        self,
        probabilities: Tensor,
        skip_mask: Tensor | None = None,
    ) -> Tensor | None:
        if skip_mask is None or self.threshold == 0.0:
            return skip_mask
        threshold_mask = probabilities < self.threshold
        if self.filter_above_threshold:
            mask_update = (threshold_mask.all(dim=-1) == False).unsqueeze(-1)
            return torch.masked_fill(skip_mask, mask_update, 0)
        mask_update = threshold_mask.all(dim=-1).unsqueeze(-1)
        return torch.masked_fill(skip_mask, mask_update, 0)

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
            "`_sample_probabilities_and_indices` has to be implemented in classes that inherit `SamplerBase`."
        )

    def _normalize_probabilities(self, probabilities: Tensor) -> Tensor:
        if self.normalize_probabilities_flag:
            denominator = torch.sum(probabilities, dim=-1, keepdim=True) + 1e-6
            return probabilities / denominator.detach()
        return probabilities


class SamplerSparse(SamplerBase):
    def __init__(
        self,
        cfg: "SamplerConfig | ModelConfig",
        overrides: "SamplerConfig | None" = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.top_k = 1
        SamplerSparseValidator.validate(self)

    def _sample_probabilities_and_indices(
        self, probabilities: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        probability, indices = torch.max(probabilities, dim=-1)
        return probability, indices

    def _compute_loss(
        self,
        logits: Tensor,
        full_probabilities: Tensor,
        sampled_probabilities: Tensor,
        indices: Tensor,
        skip_mask: Tensor | None = None,
    ) -> Tensor:
        gates = self.__prepare_loss_gates(sampled_probabilities, indices)
        logits = logits.reshape(-1, self.num_experts)
        full_probabilities = full_probabilities.reshape(-1, self.num_experts)
        skip_mask = self.__prepare_loss_skip_mask(skip_mask)

        self.auxiliary_loss_model.update_accumulated_statistics(
            logits, full_probabilities, gates, skip_mask
        )
        return self.auxiliary_loss_model.get_auxiliary_loss_and_clear()

    def __prepare_loss_gates(
        self, sampled_probabilities: Tensor, indices: Tensor
    ) -> Tensor:
        input_dim = sampled_probabilities.numel()
        gates_buffer = torch.zeros(input_dim, self.num_experts).to(self.device)
        gates = gates_buffer.scatter(
            1,
            indices.view(-1, self.top_k),
            sampled_probabilities.view(-1, self.top_k),
        ).to(self.device)

        return gates

    def __prepare_loss_skip_mask(
        self, skip_mask: Tensor | None = None
    ) -> Tensor | None:
        if skip_mask is not None:
            return skip_mask.reshape(-1, 1)
        return None


class SamplerTopk(SamplerBase):
    def __init__(
        self,
        cfg: "SamplerConfig | ModelConfig",
        overrides: "SamplerConfig | None" = None,
    ) -> None:
        super().__init__(cfg, overrides)
        SamplerTopkValidator.validate(self)

    def _sample_probabilities_and_indices(
        self, probabilities: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        if self.training and (self.num_topk_samples > 0):
            probabilities, indices = self.__sample_deterministic_and_random_topk(
                probabilities
            )
            return probabilities, indices

        probabilities, indices = torch.topk(probabilities, self.top_k)
        return probabilities, indices

    def __sample_deterministic_and_random_topk(
        self, probabilities: Tensor
    ) -> tuple[Tensor, Tensor]:
        num_deterministic = self.top_k - self.num_topk_samples
        _, topk_deterministic_indices = probabilities.topk(num_deterministic, dim=-1)

        masked_probs = probabilities + 1e-6
        batch_indices = expand_dims(
            torch.arange(probabilities.size(0), device=probabilities.device), dim=1
        )
        masked_probs[batch_indices, topk_deterministic_indices] = 0
        topk_random_indices = torch.multinomial(masked_probs, self.num_topk_samples)

        final_topk_indices = torch.concat(
            [topk_deterministic_indices, topk_random_indices], dim=-1
        )

        final_topk_probs = torch.gather(probabilities, 1, final_topk_indices)
        return final_topk_probs, final_topk_indices

    def _compute_loss(
        self,
        logits: Tensor,
        full_probabilities: Tensor,
        sampled_probabilities: Tensor,
        indices: Tensor,
        skip_mask: Tensor | None = None,
    ) -> Tensor:
        logits = logits.reshape(-1, self.num_experts)
        gates = self.__prepare_loss_gates(sampled_probabilities, indices)
        full_probabilities = full_probabilities.reshape(-1, self.num_experts)
        skip_mask = self.__prepare_loss_skip_mask(skip_mask)

        self.auxiliary_loss_model.update_accumulated_statistics(
            logits, full_probabilities, gates, skip_mask
        )
        return self.auxiliary_loss_model.get_auxiliary_loss_and_clear()

    def __prepare_loss_gates(
        self, sampled_probabilities: Tensor, indices: Tensor
    ) -> Tensor:
        input_dim = sampled_probabilities.numel()
        gates_buffer = torch.zeros(input_dim, self.num_experts).to(self.device)
        gates = gates_buffer.scatter(
            1,
            indices.view(-1, self.top_k),
            sampled_probabilities.view(-1, self.top_k),
        ).to(self.device)

        return gates

    def __prepare_loss_skip_mask(
        self, skip_mask: Tensor | None = None
    ) -> Tensor | None:
        if skip_mask is not None:
            return skip_mask.reshape(-1, 1)
        return None


class SamplerFull(SamplerBase):
    def __init__(
        self,
        cfg: "SamplerConfig | ModelConfig",
        overrides: "SamplerConfig | None" = None,
    ) -> None:
        super().__init__(cfg, overrides)
        SamplerFullValidator.validate(self)

    def _sample_probabilities_and_indices(
        self, probability_matrix: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        probability_matrix = self.__apply_dynamic_topk_threshold_mask(
            probability_matrix
        )
        return probability_matrix, None

    def __apply_dynamic_topk_threshold_mask(self, probabilities: Tensor) -> Tensor:
        if self.threshold == 0.0:
            return probabilities

        dynamic_topk_mask = probabilities < self.threshold
        masked_probabilities = torch.where(dynamic_topk_mask, 0.0, probabilities)
        normalized_probabilities = self._normalize_probabilities(masked_probabilities)
        return normalized_probabilities
