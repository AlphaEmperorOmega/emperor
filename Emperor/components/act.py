import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules import padding


from Emperor.base.utils import Module
from Emperor.components.sut_layer import TransformerEncoderLayerBase

from typing import TYPE_CHECKING, Dict, Optional, List, Tuple, Any

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class AdaptiveComputationTimeWrapper(Module):
    """
    Just in case you don't get the memo in the future
    `act` = `Adaptive Computation Time`
    """

    def __init__(
        self,
        cfg: "ModelConfig",
        model: TransformerEncoderLayerBase,
        threshold=ACT_THRESHOLD,
        haltingDropout=0.0,
    ):
        super(AdaptiveComputationTimeWrapper, self).__init__()
        self.cfg = cfg
        self.model = model
        self.embeddingDim = cfg.embeddingDim
        self.haltingDropout = haltingDropout
        self.threshold = threshold

        self.gatingModel = self._createGatingModel()

    def _createGatingModel(self):
        gatingModel = nn.Sequential(
            nn.Linear(self.embeddingDim, self.embeddingDim),
            nn.GELU(),
            nn.Dropout(self.haltingDropout),
            nn.Linear(self.embeddingDim, 2, bias=False),
        )

        nn.init.zeros_(gatingModel[-1].weight)
        return gatingModel

    def forward(
        self,
        previousAdaptiveComputationState: Optional[Tuple],
        previousHiddenState: Tensor,
        selfAttentionInput: Optional[Tensor],
        paddingMask: Tensor,
        layerIdx: Optional[int] = None,
        *args,
        **kwargs,
    ):
        currentAdaptiveComputationState, haltingMask = self._prepareStateAndHaltingMask(
            previousAdaptiveComputationState, previousHiddenState, paddingMask
        )

        modelOutput = self.model.forward(
            inputBatch=previousHiddenState,
            selfAttentionInput=selfAttentionInput,
            haltMask=haltingMask,
            layerIdx=layerIdx,
            *args,
            **kwargs,
        )

        selfAttentionInput, actLoss = self._computeSelfAttentionAndACTLoss(
            modelOutput,
            selfAttentionInput,
            previousAdaptiveComputationState,
            currentAdaptiveComputationState,
            haltingMask,
            paddingMask,
        )

        return (
            currentAdaptiveComputationState,
            modelOutput,
            selfAttentionInput,
            actLoss,
        )

    def _prepareStateAndHaltingMask(
        self,
        previousAdaptiveComputationState: Optional[Tuple],
        previousHiddenState: Tensor,
        paddingMask: Tensor,
    ):
        if previousAdaptiveComputationState is None:
            sequenceLength, batchSize, _ = previousHiddenState.size()
            haltingMaskLogits = torch.zeros(sequenceLength, batchSize)
            accumulatedExpectedDepth = torch.zeros(sequenceLength, batchSize)
            accumulatedHiddenState = torch.zeros_like(previousHiddenState)
            stateIndex = 0
            haltingMask = paddingMask
        else:
            (
                stateIndex,
                haltingMaskLogits,
                accumulatedHiddenState,
                accumulatedExpectedDepth,
            ) = previousAdaptiveComputationState

            gatingLogits = self._computeGatingLogits(previousHiddenState)
            pribabilityLogits, haltingMaskLogits = self._splitGatingLogits(
                gatingLogits, haltingMaskLogits
            )
            accumulatedHiddenState = self._updateAccumulatedHiddenState(
                previousHiddenState,
                accumulatedHiddenState,
                pribabilityLogits,
            )
            accumulatedExpectedDepth = self._updateAccumulatedExpectedDepth(
                accumulatedExpectedDepth,
                stateIndex,
                pribabilityLogits,
            )

            haltingMask = self._updateHaltingMask(
                haltingMaskLogits,
                paddingMask,
            )

            stateIndex = stateIndex + 1

        updatedAdaptiveComputationState = (
            stateIndex,
            haltingMaskLogits,
            accumulatedHiddenState,
            accumulatedExpectedDepth,
        )
        return updatedAdaptiveComputationState, haltingMask

    def _computeGatingLogits(self, h):
        logits = self.gatingModel(h)
        return F.log_softmax(logits, dim=-1)

    def _splitGatingLogits(self, gatingLogits, haltingMaskLogits):
        logHaltMask = haltingMaskLogits.unsqueeze(-1) + gatingLogits
        # the logHaltMask[..., 0] retrieves the first column
        # of every matrix in a 3d tensor, for example given a
        # tensor of shape (4, 3, 2) will be reshaped to (4, 3)
        # meaning the rows of the result consits of the first
        # column of logHaltMask
        haltingMaskLogits = logHaltMask[..., 0]
        pribabilityLogits = torch.exp(logHaltMask[..., 1])
        return pribabilityLogits, haltingMaskLogits

    def _updateAccumulatedHiddenState(
        self,
        previousHiddenState: Tensor,
        accumulatedHiddenState: Tensor,
        pribabilityLogits: Tensor,
    ):
        return (
            accumulatedHiddenState
            + pribabilityLogits.unsqueeze(-1) * previousHiddenState
        )

    def _updateAccumulatedExpectedDepth(
        self,
        accumulatedExpectedDepth: Tensor,
        stateIndex: int,
        pribabilityLogits: Tensor,
    ):
        return accumulatedExpectedDepth + stateIndex * pribabilityLogits

    def _updateHaltingMask(
        self,
        haltingMaskLogits: Tensor,
        paddingMask: Tensor,
    ):
        haltingMask = haltingMaskLogits.exp()
        conditionToHaltTokens = haltingMask < (1 - self.threshold)
        updatedHatlingMask = haltingMask.masked_fill(conditionToHaltTokens, 0)
        appliedPaddingMask = updatedHatlingMask * paddingMask
        return appliedPaddingMask.contiguous()

    def _computeSelfAttentionAndACTLoss(
        self,
        layerOutputTuple: Tuple,
        selfAttentionInput: Tensor,
        previousAdaptiveComputationState: Optional[Tuple],
        currentAdaptiveComputationState: Tuple,
        haltingMask: Tensor,
        paddingMask: Tensor,
    ):
        layerOutput: Tensor = layerOutputTuple[0]
        if previousAdaptiveComputationState is not None:
            (
                stateIndex,
                haltingMaskLogits,
                accumulatedHiddenState,
                accumulatedExpectedDepth,
            ) = currentAdaptiveComputationState

            selfAttentionInput = self._updateSelfAttention(
                haltingMask,
                layerOutput,
                accumulatedHiddenState,
                selfAttentionInput,
            )

            actLoss = self._computeACTLoss(
                accumulatedExpectedDepth,
                haltingMask,
                stateIndex,
                paddingMask,
            )

        else:
            selfAttentionInput = layerOutput
            actLoss = 0

        return selfAttentionInput, actLoss

    def _updateSelfAttention(
        self,
        haltingMask: Tensor,
        layerOutput: Tensor,
        accumulatedHiddenState: Tensor,
        selfAttentionInput: Tensor,
    ):
        replacementCondition = haltingMask.unsqueeze(-1) < (1 - self.threshold)
        layerOutputScaledAndMasked = haltingMask.unsqueeze(-1) * layerOutput
        replacementTensor = accumulatedHiddenState + layerOutputScaledAndMasked
        replacementTensor = replacementTensor.type_as(selfAttentionInput)

        selfAttentionInput = torch.where(
            replacementCondition, selfAttentionInput, replacementTensor
        )
        return selfAttentionInput

    def _computeACTLoss(
        self,
        accumulatedExpectedDepth: Tensor,
        haltingMask: Tensor,
        stateIndex: int,
        paddingMask: Tensor,
    ):
        adaptiveComputationTimeLoss = (
            accumulatedExpectedDepth + haltingMask * stateIndex
        ) * paddingMask
        return adaptiveComputationTimeLoss.sum() / paddingMask.sum()
