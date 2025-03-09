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
        self.gatingModel = self._createGatingModel()
        self.threshold = threshold

    def _createGatingModel(self):
        gatingModel = nn.Sequential(
            nn.Linear(self.embeddingDim, self.embeddingDim),
            nn.GELU(),
            nn.Dropout(self.haltingDropout),
            nn.Linear(self.embeddingDim, 2, bias=False),
        )

        nn.init.zeros_(gatingModel[-1].weight)
        return gatingModel

    def _computeGatingLogits(self, h):
        logits = self.gatingModel(h)
        return F.log_softmax(logits, dim=-1)

    def _updateHalting(self, gatingLogits, log_never_halt):
        log_halt = log_never_halt[..., None] + gatingLogits
        log_never_halt = log_halt[..., 0]
        p = torch.exp(log_halt[..., 1])
        return p, log_never_halt

    def forward(
        self,
        previousAdaptiveComputationState: Optional[Tuple],
        layerInput: Tensor,
        selfAttentionInput: Optional[Tensor] = None,
        paddingMask: Optional[Tthresholdensor] = None,
        layerIdx: Optional[int] = None,
        *args,
        **kwargs,
    ):
        if previousAdaptiveComputationState is None:
            log_never_halt = acc_expect_depth = torch.zeros_like(layerInput[..., 0])
            acc_h = torch.zeros_like(layerInput)
            index = 0
            hatlingMask = paddingMask
        else:
            (index, log_never_halt, acc_h, acc_expect_depth) = (
                previousAdaptiveComputationState
            )
            gatingLogits = self._computeGatingLogits(layerInput)
            p, log_never_halt = self._updateHalting(gatingLogits, log_never_halt)
            acc_h = acc_h + p[..., None] * layerInput
            acc_expect_depth = acc_expect_depth + index * p
            p_never_halt = log_never_halt.exp()
            p_never_halt = (
                p_never_halt.masked_fill((p_never_halt < (1 - self.threshold)), 0)
                * paddingMask
            )
            p_never_halt = p_never_halt.contiguous()
            index = index + 1

        currentAdaptiveComputationState = (
            index,
            log_never_halt,
            acc_h,
            acc_expect_depth,
        )

        layerOutputs = self.model.forward(
            inputBatch=layerInput,
            selfAttentionInput=selfAttentionInput,
            haltMask=hatlingMask,
            layerIdx=layerIdx,
            *args,
            **kwargs,
        )

        layerOutput = layerOutputs[0]
        if previousAdaptiveComputationState is not None:
            selfAttentionInput = torch.where(
                hatlingMask[..., None] < (1 - self.threshold),
                selfAttentionInput,
                (acc_h + p_never_halt[..., None] * curr_h).type_as(selfAttentionInput),
            )
            act_loss = (acc_expect_depth + p_never_halt * i) * pad_mask
            act_loss = act_loss.sum() / pad_mask.sum()
        else:
            selfAttentionInput = layerOutput
            act_loss = 0

        return (
            currentAdaptiveComputationState,
            layerOutputs,
            selfAttentionInput,
            act_loss,
        )
