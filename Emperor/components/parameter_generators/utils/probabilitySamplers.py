import torch
from torch import Tensor

from Emperor.components.parameter_generators.utils.losses import AuxiliaryLosses
from .routers import RouterModel
from Emperor.base.utils import (
    sigmoid,
    randn_like,
    masked_fill,
    tensor,
    zeros,
    arange,
    expand_dims,
    concat,
)
from Emperor.library.choice import Library as L
from Emperor.base.utils import Module

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig, SamplerConfig


class SamplerModel(Module):
    def __init__(
        self,
        cfg: "SamplerConfig | ModelConfig | None" = None,
        top_k: int | None = None,
        top_k_threshold: float | None = None,
        num_topk_samples: int | None = None,
        noisy_topk_flag: bool | None = None,
    ) -> None:
        super().__init__()
        self.cfg_main = cfg
        self.cfg: "SamplerConfig | None" = self._resolve_config(
            cfg, "sampler_model_config"
        )
        self.top_k = self._resolve(top_k, "top_k")
        self.top_k_threshold = self._resolve(top_k_threshold, "top_k_threshold")
        self.noisy_topk_flag = self._resolve(noisy_topk_flag, "noisy_topk_flag", cfg)
        self.num_topk_samples = self._resolve(num_topk_samples, "num_topk_samples")

        assert self.num_topk_samples <= self.top_k
        self.router_model: "RouterModel" = self._router_model_hook()

        self.noise_epsilon = 1e-2
        self.is_training_flag = False
        self.auxiliaryLosses = AuxiliaryLosses(self.cfg_main)

    def _router_model_hook(self) -> RouterModel:
        return RouterModel(self.cfg_main)

    def forward(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
        is_training_flag: bool = False,
        custom_softmax_flag: bool = False,
        compute_weight_flag: bool = True,
    ) -> tuple[Tensor, Tensor | None]:
        self.is_training_flag = is_training_flag
        all_probabilities, logits = self.__calc_probabilities(
            input_batch, skip_mask, compute_weight_flag
        )
        selected_probabilities, selected_indexes = self.__sample_probabilities(
            all_probabilities
        )
        self._compute_loss_hook(
            logits, all_probabilities, selected_probabilities, selected_indexes
        )
        if custom_softmax_flag:
            selected_probabilities = self._calc_softmax_custom(selected_probabilities)

        return selected_probabilities, selected_indexes

    def __calc_probabilities(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
        compute_weight_flag: bool = True,
    ) -> tuple[Tensor, Tensor]:
        logits = self.router_model.compute_logit_scores(
            input_batch, compute_weight_flag
        )
        logits = self.__add_noise(logits)
        probabilities = torch.softmax(logits, dim=-1)
        return self.__mask_scores(probabilities, logits, skip_mask)

    def __add_noise(self, logit_scores: Tensor) -> Tensor:
        if self.noisy_topk_flag:
            # Because the router now generates `self.depthDim * 2` scores
            # one half of those will be used as standard deviation scores
            logit_scores_chunk, raw_noise_std = logit_scores.chunk(2, dim=-1)
            logit_scores = logit_scores_chunk

            if self.is_training_flag:
                noise_std = sigmoid(raw_noise_std) + self.noise_epsilon
                noise = randn_like(logit_scores)
                # TODO: In the future maybe scale the noise by `raw_noise_std`
                # or scale `noise` by some `decaying_noise_amplifier` scalar
                noisy_logit_scores = logit_scores + noise * noise_std
                logit_scores = noisy_logit_scores

        return logit_scores

    def __mask_scores(
        self,
        probabilities: Tensor,
        logits_scores: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if skip_mask is not None:
            mask = skip_mask == 0
            probabilities = masked_fill(probabilities, mask, 0)
            logits_scores = masked_fill(logits_scores, mask, 0)
        return probabilities, logits_scores

    def __sample_probabilities(
        self, probabilities: Tensor
    ) -> Tensor | tuple[Tensor, Tensor]:
        selected_probs, selected_indices = self._probability_sampler_hook(probabilities)
        return selected_probs, selected_indices

    def _compute_loss_hook(
        self,
        logits: Tensor,
        full_probabilities: Tensor,
        probabilities: Tensor,
        indexes: Tensor,
    ) -> float:
        return 0.0

    def _probability_sampler_hook(
        self, probabilities: Tensor
    ) -> Tensor | tuple[Tensor, Tensor]:
        raise NotImplementedError(
            "Derived classes must implement _probability_sampler_hook"
        )

    def _calc_softmax_custom(self, probabilities: Tensor) -> Tensor:
        return probabilities / (torch.sum(probabilities, dim=-1, keepdim=True) + 1e-6)


class ProbabilitySamplerSparse(SamplerModel):
    def __init__(
        self,
        cfg: "SamplerConfig | ModelConfig | None" = None,
        top_k: int | None = None,
        top_k_threshold: float | None = None,
        num_topk_samples: int | None = None,
        noisy_topk_flag: bool | None = None,
    ) -> None:
        super().__init__(
            cfg,
            top_k,
            top_k_threshold,
            num_topk_samples,
            noisy_topk_flag,
        )
        self.calc_softmax_custom_flag = False

    def _probability_sampler_hook(
        self, probabilities: Tensor
    ) -> Tensor | tuple[Tensor, Tensor]:
        return torch.max(probabilities, dim=-1)

    def _compute_loss_hook(
        self,
        logits: Tensor,
        full_probabilities: Tensor,
        probabilities: Tensor,
        indexes: Tensor,
    ):
        first_dim = torch.prod(tensor(probabilities.shape))
        gatesBuffer = zeros(first_dim, self.cfg.depthDim).to(L.Device)

        gates = gatesBuffer.scatter(
            1,
            indexes.view(-1, 1),
            probabilities.view(-1, 1),
        ).to(L.Device)

        logits = logits.reshape(-1, self.cfg.depthDim)
        fullProbabilities = logits.reshape(-1, self.cfg.depthDim)

        self.auxiliaryLosses.updateAccumulatedStatistics(
            logits, fullProbabilities, gates
        )


class ProbabilitySamplerTopk(SamplerModel):
    def __init__(
        self,
        cfg: "SamplerConfig | ModelConfig | None" = None,
        top_k: int | None = None,
        top_k_threshold: float | None = None,
        num_topk_samples: int | None = None,
        noisy_topk_flag: bool | None = None,
    ) -> None:
        super().__init__(
            cfg,
            top_k,
            top_k_threshold,
            num_topk_samples,
            noisy_topk_flag,
        )

    def _probability_sampler_hook(
        self, probabilities: Tensor
    ) -> Tensor | tuple[Tensor, Tensor]:
        probabilities, indexes = self.__sample_topk_probabilities(probabilities)

        return probabilities, indexes

    def __sample_topk_probabilities(
        self, probabilities: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self.is_training_flag and (self.num_topk_samples > 0):
            top_k_Probabilities, top_k_indexes = self._sample_num_topk_samples(
                probabilities
            )
        else:
            top_k_Probabilities, top_k_indexes = L.topk(probabilities, self.top_k)

        return top_k_Probabilities, top_k_indexes

    def _sample_num_topk_samples(self, probabilities: Tensor) -> tuple[Tensor, Tensor]:
        # Determine how many top-k items to keep deterministically
        num_deterministic = self.top_k - self.num_topk_samples
        _, topk_deterministic_indexes = probabilities.topk(num_deterministic, dim=-1)

        # Mask out already selected indexes
        masked_probs = probabilities + 1e-6
        batch_indexes = expand_dims(arange(probabilities.size(0)), 1)
        masked_probs[batch_indexes, topk_deterministic_indexes] = 0

        # Sample the remaining top-k entries randomly
        topk_random_indexes = torch.multinomial(masked_probs, self.num_topk_samples)

        # Combine deterministic and random top-k indexes
        final_topk_indexes = concat(
            [topk_deterministic_indexes, topk_random_indexes], dim=-1
        )

        # Gather the corresponding probabilities
        final_topk_probs = L.gather(probabilities, 1, final_topk_indexes)

        return final_topk_probs, final_topk_indexes

    def _computeLossHook(
        self,
        logits,
        fullProbabilities,
        probabilities,
        indexes,
    ):
        gatesBuffer = torch.zeros(
            torch.prod(torch.tensor(probabilities.shape)), self.cfg.depthDim
        ).to(L.Device)
        gates = gatesBuffer.scatter(
            1,
            indexes.view(-1, self.cfg.topK),
            probabilities.view(-1, self.cfg.topK),
        ).to(L.Device)

        logits = logits.reshape(-1, self.cfg.depthDim)
        fullProbabilities = logits.reshape(-1, self.cfg.depthDim)

        self.auxiliaryLosses.updateAccumulatedStatistics(
            logits, fullProbabilities, gates
        )


class ProbabilitySamplerFull(SamplerModel):
    def __init__(
        self,
        cfg: "SamplerConfig | ModelConfig | None" = None,
        top_k: int | None = None,
        top_k_threshold: float | None = None,
        num_topk_samples: int | None = None,
        noisy_topk_flag: bool | None = None,
    ) -> None:
        super().__init__(
            cfg,
            top_k,
            top_k_threshold,
            num_topk_samples,
            noisy_topk_flag,
        )

    def _probability_sampler_hook(
        self, probabilities: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        probabilities, _ = self.__sample_full_probabilities(probabilities)
        return probabilities, _

    def __sample_full_probabilities(self, probabilities: Tensor) -> tuple[Tensor, None]:
        if self.top_k_threshold > 0.0:
            return self.__mask_probs_by_threshold(probabilities)
        return probabilities, None

    def __mask_probs_by_threshold(self, probabilities: Tensor) -> tuple[Tensor, None]:
        threshold_mask = probabilities < self.top_k_threshold
        masked_probabilities = torch.where(threshold_mask, 0.0, probabilities)
        return masked_probabilities, None
