import torch
from torch import Tensor
from .routers import RouterModel
from Emperor.base.utils import sigmoid, randn_like, masked_fill
from Emperor.library.choice import Library as L
from Emperor.base.utils import Module

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from Emperor.config import ModelConfig, SamplerConfig


class SamplerModel(Module):
    def __init__(
        self,
        cfg: "SamplerConfig | ModelConfig | None" = None,
        top_k: int | None = None,
        top_k_treshold: float | None = None,
        num_top_k_samples: int | None = None,
        noisy_topk_flag: bool | None = None,
    ) -> None:
        self.cfg: "SamplerConfig | None" = self._resolve_config(
            cfg, "sampler_model_config"
        )
        self.top_k = self._resolve(top_k, "top_k")
        self.top_k_treshold = self._resolve(top_k_treshold, "top_k_treshold")
        self.noisy_topk_flag = self._resolve(noisy_topk_flag, "noisy_topk_flag", cfg)
        self.num_top_k_samples = self._resolve(num_top_k_samples, "num_top_k_samples")

        assert self.num_top_k_random_samples <= self.top_k
        self.router_model: "RouterModel" = self._init_router_model()

        self.noise_epsilon = 1e-2
        self.is_training_flag = False

    def _router_model_hook(self) -> RouterModel:
        return RouterModel(self.cfg)

    def sample_probs_and_indices(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
        is_training_flag: bool = False,
        custom_softmax_flag: bool = False,
        compute_weight_flag: bool = True,
    ) -> tuple[Tensor, Tensor]:
        self.is_training_flag = is_training_flag
        all_probabilities, logits = self.__calc_probabilities(
            input_batch, skip_mask, compute_weight_flag
        )
        selected_probabilities, selected_indexes = self.__sample_probabilities(
            all_probabilities
        )
        self.compute_loss_hook(
            logits, all_probabilities, selected_probabilities, selected_indexes
        )
        if custom_softmax_flag:
            selected_probabilities = self.calc_softmax_custom(selected_probabilities)

        return selected_probabilities, selected_indexes

    def __calc_probabilities(
        self,
        input_batch: Tensor,
        skip_mask: Optional[Tensor] = None,
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
        skip_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        if skip_mask is not None:
            mask = skip_mask == 0
            probabilities = masked_fill(probabilities, mask, 0)
            logits_scores = masked_fill(logits_scores, mask, 0)
        return probabilities, logits_scores

    def __sample_probabilities(self, probabilities: Tensor) -> Tensor:
        return self.probability_sampler_hook(probabilities)

    def compute_loss_hook(
        self,
        logits: Tensor,
        full_probabilities: Tensor,
        probabilities: Tensor,
        indices: Tensor,
    ) -> float:
        return 0.0

    def probability_sampler_hook(self, probabilities: Tensor) -> Tensor:
        pass

    def calc_softmax_Custom(self, probabilities: Tensor) -> Tensor:
        return probabilities / (torch.sum(probabilities, dim=-1, keepdim=True) + 1e-6)


class ProbabilitySamplerSparse(SamplerModel):
    def __init__(self, cfg: "ParameterGeneratorConfig") -> None:
        super().__init__(cfg)
        self.calcSoftmaxCustomFlag = False

    def probabilitySamplerHook(self, probabilities):
        return L.getTopProbabilityAndIndex(probabilities)

    def computeLossHook(self, logits, fullProbabilities, probabilities, indices):
        dim0 = torch.prod(torch.tensor(probabilities.shape))
        gatesBuffer = torch.zeros(dim0, self.cfg.depthDim).to(L.Device)

        gates = gatesBuffer.scatter(
            1,
            indices.view(-1, 1),
            probabilities.view(-1, 1),
        ).to(L.Device)

        logits = logits.reshape(-1, self.cfg.depthDim)
        fullProbabilities = logits.reshape(-1, self.cfg.depthDim)

        self.auxiliaryLosses.updateAccumulatedStatistics(
            logits, fullProbabilities, gates
        )


class ProbabilitySamplerTopk(SamplerModel):
    def __init__(self, cfg: "ParameterGeneratorConfig") -> None:
        super().__init__(cfg)

    def probabilitySamplerHook(self, probabilities):
        probabilities, indices = self._sampleTopKProbabilities(probabilities)

        return probabilities, indices

    def _sampleTopKProbabilities(self, probabilities):
        if self.isTrainingFlag and (self.numProbabilitiesToRandomlySample > 0):
            topKProbabilities, topKIndices = self._sampleRandomTopKProbabilities(
                probabilities
            )
        else:
            topKProbabilities, topKIndices = L.topk(probabilities, self.topK)

        return topKProbabilities, topKIndices

    def _sampleRandomTopKProbabilities(self, probabilities):
        # Select the `(topk - numProbabilitiesToRandomlySample)` top probabilities
        numTrueProbabilities = self.topK - self.numProbabilitiesToRandomlySample
        _, trueTopKIndices = L.topk(probabilities, numTrueProbabilities)

        # Hide probabilitie that have allready been selected
        maskedProbabilities = probabilities + 1e-6
        rangeIndexes = L.unsqueeze(L.arange(probabilities.size(0)), 1)
        maskedProbabilities[rangeIndexes, trueTopKIndices] = 0

        sampledProbabilityIndices = L.multinomial(
            maskedProbabilities, self.numProbabilitiesToRandomlySample
        )

        topKIndices = L.cat([trueTopKIndices, sampledProbabilityIndices])

        # Retrieve the probabilities of the incides in the above step
        topKProbabilities = L.gather(probabilities, 1, topKIndices)

        return topKProbabilities, topKIndices

    def computeLossHook(self, logits, fullProbabilities, probabilities, indices):
        gatesBuffer = torch.zeros(
            torch.prod(torch.tensor(probabilities.shape)), self.cfg.depthDim
        ).to(L.Device)
        gates = gatesBuffer.scatter(
            1,
            indices.view(-1, self.cfg.topK),
            probabilities.view(-1, self.cfg.topK),
        ).to(L.Device)

        logits = logits.reshape(-1, self.cfg.depthDim)
        fullProbabilities = logits.reshape(-1, self.cfg.depthDim)

        self.auxiliaryLosses.updateAccumulatedStatistics(
            logits, fullProbabilities, gates
        )


class ProbabilitySamplerFull(SamplerModel):
    def __init__(self, cfg: "ParameterGeneratorConfig") -> None:
        super().__init__(cfg)

    def probabilitySamplerHook(self, probabilities):
        probabilities, indices = self._sampleFullProbabilities(probabilities)
        return probabilities, indices

    def _sampleFullProbabilities(self, probabilities):
        if self.top_k_treshold > 0.0:
            return self._maskProbsTopKTreshold(probabilities)
        return probabilities, None

    def _maskProbsTopKTreshold(self, probabilities):
        tresholdTopKMask = probabilities < self.topKTreshold
        maskedProbabilities = L.where(tresholdTopKMask, 0.0, probabilities)
        maskedProbabilities = maskedProbabilities / (
            L.sum(maskedProbabilities, dim=-1, keepdim=True) + 1e-6
        )
        return maskedProbabilities, None
