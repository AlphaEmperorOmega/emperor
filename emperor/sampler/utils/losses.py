import torch

from torch import Tensor
from torch.nn import functional as F
from emperor.base.utils import Module

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.sampler.utils.samplers import SamplerConfig


class AuxiliaryLossBase:
    def __init__(self, loss_weight: float = 0.0):
        self.loss_weight = loss_weight
        self.default_error = torch.tensor(0.0)

    def get_weighted_loss(self) -> Tensor:
        if self.is_loss_weight_zero():
            return self.default_error
        return self.loss_weight * self._compute_loss()

    def is_loss_weight_zero(self) -> bool:
        return self.loss_weight == 0.0

    def _accumulate(self, attr, value):
        if getattr(self, attr) is None:
            setattr(self, attr, value)
        else:
            getattr(self, attr).add_(value)

    def _is_accumulation_none(self, accumulation: Tensor | None) -> None:
        if accumulation is None:
            raise ValueError(
                "`self.accumulation` is `None`. Please call `update_accumulation` before validating accumulation."
            )

    def _is_valid_input(self, input: Tensor | None, obj: object) -> None:
        if input is None:
            raise ValueError(
                f"A valid input tensor is required when `loss_weight` > 0, for {obj.__class__.__name__} instance."
            )

    def reset_loss(self) -> None:
        raise NotImplementedError(
            "`reset_loss` method must be implemented by subclasses of AuxiliaryLossBase"
        )

    def update_loss(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "`update_loss` method must be implemented by subclasses of AuxiliaryLossBase"
        )

    def _compute_loss(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError(
            "`_cumpute_loss` method must be implemented by subclasses of AuxiliaryLossBase"
        )


class CoefficientOfVariationLoss(AuxiliaryLossBase):
    def __init__(self, loss_weight: float = 0.0):
        super().__init__(loss_weight)
        self.eps = 1e-10
        self.gates_accumulation = None

    def reset_loss(self) -> None:
        self.gates_accumulation = None

    def update_accumulation(self, gates: Tensor | None = None) -> None:
        if self.is_loss_weight_zero():
            return
        self._is_valid_input(gates, self)
        probability = torch.sum(gates, dim=0)
        self._accumulate("gates_accumulation", probability)

    def _compute_loss(self) -> Tensor:
        if self.__is_accumulation_shape_valid():
            return self.default_error
        return self.__compute_coefficient_of_variation()

    def __is_accumulation_shape_valid(self) -> bool:
        self._is_accumulation_none(self.gates_accumulation)
        return self.gates_accumulation.shape[0] == 1

    def __compute_coefficient_of_variation(self) -> Tensor:
        probabilities = F.normalize(self.gates_accumulation, p=1, dim=0)
        variation = probabilities.float().var()
        mean = probabilities.float().mean() ** 2

        return variation / (mean + self.eps)


class SwitchLoss(AuxiliaryLossBase):
    def __init__(self, num_experts: int, loss_weight: float = 0.0):
        super().__init__(loss_weight)
        self.num_experts = num_experts
        self.probability_accumulation = None
        self.frequency_accumulation = None

    def reset_loss(self) -> None:
        self.probability_accumulation = None
        self.frequency_accumulation = None

    def update_accumulation(
        self,
        probabilities: Tensor | None = None,
        gates: Tensor | None = None,
    ) -> None:
        if self.is_loss_weight_zero():
            return
        self._is_valid_input(probabilities, self)
        self._is_valid_input(gates, self)
        probability = torch.sum(probabilities, dim=0)
        self._accumulate("probability_accumulation", probability)
        frequency = torch.sum((gates > 0).float(), dim=0)
        self._accumulate("frequency_accumulation", frequency)

    def _compute_loss(self) -> Tensor:
        self._is_accumulation_none(self.probability_accumulation)
        self._is_accumulation_none(self.frequency_accumulation)
        return self.__compute_switch_loss()

    def __compute_switch_loss(self) -> Tensor:
        p = self.probability_accumulation
        f = self.frequency_accumulation
        normalized_probabilities = F.normalize(p, p=1, dim=0)
        normalized_frequency = F.normalize(f, p=1, dim=0)
        loss = normalized_probabilities * normalized_frequency
        return self.num_experts * loss.sum()


class ZeroCentredLoss(AuxiliaryLossBase):
    def __init__(self, loss_weight: float = 0.0):
        super().__init__(loss_weight)
        self.squared_log_sum_exp_accumulation = None
        self.count_accumulation = None

    def reset_loss(self) -> None:
        self.squared_log_sum_exp_accumulation = None
        self.count_accumulation = None

    def update_accumulation(self, logits: Tensor | None = None) -> None:
        if self.is_loss_weight_zero():
            return
        self._is_valid_input(logits, self)
        squared_log_sum_exp = self.__compute_squared_log_sum_exp(logits)
        self._accumulate("squared_log_sum_exp_accumulation", squared_log_sum_exp)
        count = torch.tensor(logits.size(0))
        self._accumulate("count_accumulation", count)

    def __compute_squared_log_sum_exp(self, logits: Tensor) -> Tensor:
        squared_log_sum_exp = torch.exp(logits).sum(dim=-1)
        squared_log_sum_exp = torch.log(squared_log_sum_exp) ** 2
        squared_log_sum_exp = torch.sum(squared_log_sum_exp)
        return squared_log_sum_exp

    def _compute_loss(self) -> Tensor:
        self._is_accumulation_none(self.squared_log_sum_exp_accumulation)
        self._is_accumulation_none(self.count_accumulation)
        return self.__compute_zero_centred_loss()

    def __compute_zero_centred_loss(self):
        return self.squared_log_sum_exp_accumulation / self.count_accumulation


class MutualInformationLoss(AuxiliaryLossBase):
    def __init__(self, loss_weight: float = 0.0):
        super().__init__(loss_weight)
        self.log_probabilities = []
        self.probabilities = []
        self.skip_masks = []

    def reset_loss(self) -> None:
        self.log_probabilities = []
        self.probabilities = []
        self.skip_masks = []

    def update_accumulation(
        self,
        logits: Tensor | None = None,
        probabilities: Tensor | None = None,
        skip_masks: Tensor | None = None,
    ) -> None:
        if self.is_loss_weight_zero():
            return

        self._is_valid_input(logits, self)
        self._is_valid_input(probabilities, self)
        self._is_valid_input(skip_masks, self)

        log_probabilities = torch.log_softmax(logits, dim=-1)
        self.log_probabilities.append(log_probabilities)
        self.probabilities.append(probabilities)
        self.skip_masks.append(skip_masks)

    def _compute_loss(self) -> Tensor:
        self._is_accumulation_list_empty(self.log_probabilities)
        self._is_accumulation_list_empty(self.probabilities)
        self._is_accumulation_list_empty(self.skip_masks)
        return self.__compute_mutual_information_loss()

    def _is_accumulation_list_empty(self, accumulation: list) -> None:
        if len(accumulation) == 0:
            raise ValueError(
                "`self.accumulation_list` is `empty`. Please call `update_accumulation` before validating accumulation."
            )

    def __compute_mutual_information_loss(self):
        probabilities = torch.cat(self.probabilities, dim=0)
        log_probabilities = torch.cat(self.log_probabilities, dim=0)
        masks = torch.cat(self.skip_masks, dim=0)

        p_x = masks / (masks.sum() + 1e-12)
        p_e = (p_x * probabilities).sum(dim=0)
        # WARNING: Ensure that `skip_mask` does not contain
        # any zeros exist in `p_e.log()` will produce `-inf`
        # `H_e` will store `nan` in it's result
        H_e = (p_e * p_e.log()).sum()

        neg_H_e_given_x = (p_x * probabilities * log_probabilities).sum()
        mi_loss = -(neg_H_e_given_x + H_e)
        return mi_loss


class SamplerAuxiliaryLosses(Module):
    def __init__(
        self,
        cfg: "SamplerConfig",
        overrides: "SamplerConfig | None" = None,
    ) -> None:
        super().__init__()

        config = getattr(cfg, "sampler_model_config", cfg)
        self.cfg: "SamplerConfig" = self._overwrite_config(config, overrides)

        self.num_experts = self.cfg.num_experts
        self.coefficient_of_variation_loss_weight = (
            self.cfg.coefficient_of_variation_loss_weight
        )
        self.switch_loss_weight = self.cfg.switch_loss_weight
        self.zero_centred_loss_weight = self.cfg.zero_centred_loss_weight
        self.mutual_information_loss_weight = self.cfg.mutual_information_loss_weight

        self.coefficient_of_variation_loss = CoefficientOfVariationLoss(
            self.coefficient_of_variation_loss_weight
        )
        self.switch_loss = SwitchLoss(self.num_experts, self.switch_loss_weight)
        self.zero_centred_loss = ZeroCentredLoss(self.zero_centred_loss_weight)
        self.mutual_information_loss = MutualInformationLoss(
            self.mutual_information_loss_weight
        )

    def update_accumulated_statistics(
        self,
        logits: Tensor | None = None,
        probabilities: Tensor | None = None,
        gates: Tensor | None = None,
        skip_mask: Tensor | None = None,
    ) -> None:
        self.coefficient_of_variation_loss.update_accumulation(gates)
        self.switch_loss.update_accumulation(probabilities, gates)
        self.zero_centred_loss.update_accumulation(logits)
        self.mutual_information_loss.update_accumulation(
            logits, probabilities, skip_mask
        )

    def get_auxiliary_loss_and_clear(self) -> Tensor:
        total_loss = self.__compute_total_loss()
        self.__reset_all_accumulations()
        return total_loss

    def __compute_total_loss(self) -> Tensor:
        return (
            self.coefficient_of_variation_loss.get_weighted_loss()
            + self.switch_loss.get_weighted_loss()
            + self.zero_centred_loss.get_weighted_loss()
            + self.mutual_information_loss.get_weighted_loss()
        )

    def __reset_all_accumulations(self) -> None:
        self.coefficient_of_variation_loss.reset_loss()
        self.switch_loss.reset_loss()
        self.zero_centred_loss.reset_loss()
        self.mutual_information_loss.reset_loss()
