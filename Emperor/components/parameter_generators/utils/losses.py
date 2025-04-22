import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxiliaryLosses:
    def __init__(self, cfg):
        super().__init__()
        # Temporarly remove the config requirements until refactored
        self.coefficientOfVariationLoss = 0.0  # cfg.coefficientOfVariationLossWeight
        self.switchLoss = 0.0  # cfg.switchLossWeight
        self.zeroCentredLoss = 0.0  # cfg.zeroCentredLossWeight
        self.mutualInformationLoss = 0.0  # cfg.mutualInformationLossWeight
        # TEMPORARY
        self.topK = 0.0  # cfg.topK

        self.initializeAccumulatedStatistics()

    def initializeAccumulatedStatistics(self):
        self.probability_accumulation = 0.0
        self.gate_accumulation = 0.0
        self.frequency_accumulation = 0.0
        self.squared_log_sum_exp_accumulation = 0.0
        self.count_accumulation = 0

        self.probabilities = []
        self.log_probabilities = []
        self.skip_masks = []

    def update_accumulated_statistics(
        self, logits, probabilities, gates, skipMasks=None
    ):
        squaredLogSumExp = torch.exp(logits)
        squaredLogSumExp = squaredLogSumExp.sum(dim=-1)
        squaredLogSumExp = torch.log(squaredLogSumExp)
        squaredLogSumExp = squaredLogSumExp**2
        logProbabilities = torch.log_softmax(logits, dim=-1)

        self.probability_accumulation += torch.sum(probabilities, dim=0)
        self.gate_accumulation += torch.sum(gates, dim=0)
        self.frequency_accumulation += torch.sum((gates > 0).float(), dim=0)
        self.squared_log_sum_exp_accumulation += torch.sum(squaredLogSumExp)
        self.count_accumulation += logits.size(0)

        self.probabilities.append(probabilities)
        self.log_probabilities.append(logProbabilities)
        if skipMasks is not None:
            self.skip_masks.append(skipMasks)

    def getAuxiliaryLossAndClear(self):
        if (
            isinstance(self.probabilityAccumulation, float)
            and isinstance(self.gateAccumulation, float)
            and isinstance(self.frequencyAccumulation, float)
            and isinstance(self.squaredLogSumExpAccumulation, float)
        ):
            return 0.0

        coefficientOfVariationSquaredLoss = self.computeCoefficientOfVariation(
            self.gateAccumulation
        )
        switchLoss = self.computeSwitchLoss(
            self.probabilityAccumulation, self.frequencyAccumulation
        )
        zLoss = self.computeZeroCentredLoss(
            self.squaredLogSumExpAccumulation, self.countAccumulation
        )
        if len(self.skipMasks) != 0:
            miLoss = self.computeMutualInformationLoss(
                self.probabilities, self.logProbabilities, self.skipMasks
            )
        else:
            miLoss = 0.0

        loss = (
            self.coefficientOfVariationLoss * coefficientOfVariationSquaredLoss
            + self.switchLoss * switchLoss
            + self.zeroCentredLoss * zLoss
            + self.mutualInformationLoss * miLoss
        )

        self.initializeAccumulatedStatistics()
        return loss

    def computeCoefficientOfVariation(self, probabilities):
        normalizedProbabilities = F.normalize(probabilities, p=1, dim=0)
        return self.computeCoefficientOfVariationSquared(normalizedProbabilities)

    def computeCoefficientOfVariationSquared(self, probabilities):
        eps = 1e-10

        if probabilities.shape[0] == 1:
            return 0

        variation = probabilities.float().var()
        mean = probabilities.float().mean() ** 2

        return variation / (mean + eps)

    def computeSwitchLoss(self, probabilities, frequency):
        normalizedProbabilities = F.normalize(probabilities, p=1, dim=0)
        normalizedFrequency = F.normalize(frequency, p=1, dim=0)
        loss = normalizedProbabilities * normalizedFrequency

        return self.topK * loss.sum()

    def computeZeroCentredLoss(self, squaredLogSumExp, count):
        return squaredLogSumExp / count

    def computeMutualInformationLoss(self, probabilities, logProbabilities, masks):
        probabilities = torch.cat(probabilities, dim=0)
        logProbabilities = torch.cat(logProbabilities, dim=0)

        masks = torch.cat(masks, dim=0)

        p_x = masks / (masks.sum() + 1e-12)
        p_e = (p_x * probabilities).sum(0)
        H_e = (p_e * p_e.log()).sum()

        meg_H_e_given_x = (p_x * probabilities * logProbabilities).sum()
        return -(meg_H_e_given_x + H_e)
