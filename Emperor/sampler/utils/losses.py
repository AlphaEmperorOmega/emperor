from torch import Tensor
from Emperor.base.utils import Module
from Emperor.sampler.utils.losses.auxiliary_losses import (
    CoefficientOfVariationLoss,
    MutualInformationLoss,
    SwitchLoss,
    ZeroCentredLoss,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.sampler.utils.samplers import SamplerConfig


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
