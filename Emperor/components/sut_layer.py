import torch
import torch.nn as nn
from torch.types import Tensor
from Emperor.base.utils import Module
from Emperor.components.attention import Attention
from Emperor.components.moe import MixtureOfExperts

from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Union, List

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class TransformerEncoderLayerBase(Module):
    def __init__(
        self,
        cfg: "ModelConfig",
        embeddingDim: Optional[int] = None,
        returnRawFFNOutputFlag: Optional[bool] = None,
        normalizeBeforeFlag: Optional[bool] = None,
        activationFunction: Optional[nn.Module] = None,
        attnDropoutProbability: Optional[float] = None,
        ffnDropoutProbability: Optional[float] = None,
    ):
        super().__init__()

        self.cfg = cfg
        self.embeddingDim = self._getValue(embeddingDim, cfg.embeddingDim)
        self.returnRawFFNOutputFlag = self._getValue(
            returnRawFFNOutputFlag, cfg.returnRawFFNOutputFlag
        )
        self.normalizeBeforeFlag = self._getValue(
            normalizeBeforeFlag, cfg.normalizeBeforeFlag
        )
        self.normalizeBeforeFlag = self._getValue(
            activationFunction, cfg.activationFunction
        )
        self.attnDropoutProbability = self._getValue(
            attnDropoutProbability, cfg.attnDropoutProbability
        )
        self.ffnDropoutProbability = self._getValue(
            ffnDropoutProbability, cfg.ffnDropoutProbability
        )

        # self.quant_noise = cfg.quant_noise.pq
        # self.quant_noise_block_size = cfg.quant_noise.pq_block_size

        self._initializeSelfAttentionModules()
        self._initializeFeedForwardModules()

    def _initializeSelfAttentionModules(self):
        self.attnLayerNormModule = nn.LayerNorm(self.embeddingDim)
        self.attnDropoutModule = nn.Dropout(self.attnDropoutProbability)
        self.attentionModel = self._createAttentionModel()

    def _initializeFeedForwardModules(self):
        self.ffnLayerNormModule = nn.LayerNorm(self.embeddingDim)
        self.ffnDroputModule = nn.Dropout(self.ffnDropoutProbability)
        self.ffnModel = self._createFeedForwadModelModel()

    def _createAttentionModel(self):
        return Attention(self.cfg)

    def _createFeedForwadModelModel(self):
        return MixtureOfExperts(self.cfg)

    def forward(
        self,
        inputBatch,
        selfAttentionInput,
        haltMask,
        layerIdx,
        encoderPaddingMask,
        attentionMask,
    ):
        attentionOutput = self._computeAttentionRepresentation(
            inputBatch,
            attentionMask,
            selfAttentionInput,
            layerIdx,
            encoderPaddingMask,
            haltMask,
        )

        ffnOutput, ffnRawOutput = self._computeFeedForwardRepresentation(
            attentionOutput, haltMask
        )

        return ffnOutput, ffnRawOutput

    def _computeAttentionRepresentation(
        self,
        inputBatch,
        attentionMask,
        selfAttentionInput,
        layerIdx,
        encoderPaddingMask,
        haltMask,
    ):
        if selfAttentionInput is None:
            selfAttentionInput = inputBatch

        if attentionMask is not None:
            fillElement = -1e8 if inputBatch.dtype == torch.float32 else -1e4
            attentionMask = attentionMask.masked_fill(
                attentionMask.to(torch.bool), fillElement
            )

        residual = inputBatch
        if self.normalizeBeforeFlag:
            inputBatchTemp = self.attentionLayerNormModule(inputBatch)
            if inputBatch is selfAttentionInput:
                selfAttentionInput = inputBatchTemp
            else:
                selfAttentionInput = self.attentionLayerNormModule(selfAttentionInput)
            inputBatch = inputBatchTemp

        inputBatch, _, selfAuxiliaryLoss = self.selfAttentionModel(
            query=inputBatch,
            key=selfAttentionInput,
            value=selfAttentionInput,
            layerIdx=layerIdx,
            queryPaddingMask=encoderPaddingMask,
            keyPaddingMask=encoderPaddingMask,
            attentionMask=attentionMask,
            skipMask=haltMask,
        )

        inputBatch = self.attentionDropoutModule(inputBatch)
        inputBatch = self._computeResidualConnection(inputBatch, residual)

        if not self.normalizeBeforeFlag:
            inputBatch = self.attentionLayerNormModule(inputBatch)

        return inputBatch

    def _computeFeedForwardRepresentation(self, attentionOutput, haltMask):
        residual = attentionOutput
        normalizedFFNInput = attentionOutput
        if self.normalizeBeforeFlag:
            normalizedFFNInput = self.feedForwardLayerNormModule(attentionOutput)

        ffnRawOutput = self.mixtureOfExpertsModel(normalizedFFNInput, haltMask)
        sparseFFNdOutput = self.feedForwardDroputModule(ffnRawOutput)
        residualConnection = self._computeResidualConnection(sparseFFNdOutput, residual)

        normalizedOutput = residualConnection
        if not self.normalizeBeforeFlag:
            normalizedOutput = self.feedForwardLayerNormModule(residualConnection)

        ffnOutput = normalizedOutput
        if self.returnRawFFNOutputFlag:
            return ffnOutput, ffnRawOutput

        return ffnOutput, None

    def _computeResidualConnection(self, updatedRepresentation, residual):
        return updatedRepresentation + residual


class ModuleWrapper(Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, inputBatch):
        return super().forward(inputBatch)


class TransformerDecorderLayerBase(Module):
    def __init__(self, cfg: "ModelConfig"):
        super().__init__()
        self.embedDim = cfg.embeddingDim

        self.dropoutModule = nn.Dropout(cfg.dropout)

        # self.quantNoise = cfg.qantNoise
        # self.quantNosieBlockNoise = cfg.quantNoise.pqBlockSize

        self.crossSelfAttentionFlag = cfg.crossSelfAttentionFlag
        self.activationFunction = cfg.activationFunction

        self.activationFunctionDropoutModule = nn.Dropout(cfg.dropout)

        act_list = [
            ModuleWrapper(self.activationFunction),
            self.activationFunctionDropoutModule,
        ]

        self.selfAttentionModel = Attention(cfg)
        self.selfAttentionLayerNorm = nn.LayerNorm(self.embedDim)

        self.numHeads = self.selfAttentionModel.numHeads
        self.headDim = self.selfAttentionModel.headDim
        self.scaleHeads = cfg.scaleHeads

        cAttnDefaultParameters = torch.ones((self.numHeads))
        self.c_attn = nn.Parameter(cAttnDefaultParameters, requires_grad=True)

        self.normalizeBeforeFlag = cfg.normalizeBeforeFlag

        if noEncoderAttention:
            self.encoderAttention = None
            self.encoderAttentionLayerNorm = None
        else:
            self.encoderAttention = Attention(cfg)
            self.staticKeyValueFlag = True
            self.encoderAttentionLayerNorm = nn.LayerNorm(self.embeddingDim)

        self.feedForwardLayerNormModule = nn.LayerNorm(cfg.decoder.ffn_embed_dim)

        wResidDefaultParameters = torch.ones(self.embedDim)
        self.wResid = nn.Parameter(wResidDefaultParameters, requires_grad=True)

        if self.feedForwardLayerNormModule is not None:
            act_list = act_list + [self.feedForwardLayerNormModule]

        self.mixtureOfExpertsModel = MixtureOfExperts(cfg)

        self.feedForwardDroputModule = nn.LayerNorm(self.embedDim)

        self.needAttention = True
        self.onnxTrace = False

    def residualConnection(self, inputBatch, residualConnection):
        return residualConnection + inputBatch

    def forward(
        self,
        inputBatch,
        haltMask,
        layerIdx,
        selfAttentionInput: Optional[Tensor],
        encoderOutput: Optional[Tensor] = None,
        encoderPaddingMask: Optional[Tensor] = None,
        previousSelfAttentionState: Optional[List[Tensor]] = None,
        previousEncoderAttentionState: Optional[List[Tensor]] = None,
        incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        selfAttentionMask: Optional[Tensor] = None,
        selfAttentionPaddingMask: Optional[Tensor] = None,
        needHeadWeights: bool = False,
    ):
        if selfAttentionInput is None:
            selfAttentionInput = inputBatch

        if needHeadWeights:
            needAttentionWeights = True

        if previousSelfAttentionState is not None:
            previousKeyProjection, previousValueProjection = previousSelfAttentionState[
                :2
            ]
            savedState: Dict[str, Optional[Tensor]] = {
                "previousKeyProjection": previousKeyProjection,
                "previousValueProjection": previousValueProjection,
            }
            if len(previousSelfAttentionState) >= 3:
                savedState["previousKeyPaddingMask"] = previousSelfAttentionState[2]
            assert incrementalState is not None
            self.selfAttentionModel._updateIncrementalState(
                incrementalState, savedState, layerIdx
            )
        _currentSelfAttentionInputBuffer = self.selfAttentionModel._getInputBuffer(
            incrementalState, layerIdx
        )

        # ------------------------------

        residual = inputBatch
        if self.normalizeBeforeFlag:
            inputBatchTemp = self.selfAttentionLayerNorm(inputBatch)
            if selfAttentionInput is inputBatch:
                selfAttentionInput = inputBatchTemp
            else:
                selfAttentionInput = self.selfAttentionLayerNorm(selfAttentionInput)
            inputBatch = inputBatchTemp

        checkIfIncrementalStateDoesNotExist = not (
            incrementalState is not None
            and _currentSelfAttentionInputBuffer is not None
            and "previousKeyProjection" in _currentSelfAttentionInputBuffer
        )

        if self.crossSelfAttentionFlag and checkIfIncrementalStateDoesNotExist:
            if selfAttentionMask is not None:
                assert encoderOutput is not None
                decoderSequenceLength = inputBatch.size(0)
                encoderSequenceLength = encoderOutput.size(0)
                selfAttentionPadding = inputBatch.new_zeros(
                    decoderSequenceLength, encoderSequenceLength
                )
                selfAttentionMask = torch.cat(
                    (selfAttentionPadding, selfAttentionMask), dim=1
                )
            if selfAttentionPaddingMask is not None:
                if encoderPaddingMask is not None:
                    assert encoderOutput is not None
                    enconderSequenceLength = encoderOutput.size(0)
                    encoderBatchSize = encoderOutput.size(1)
                    encoderPaddingMask = selfAttentionPaddingMask.new_zeros(
                        encoderBatchSize, enconderSequenceLength
                    )
                selfAttentionPaddingMask = torch.cat(
                    (encoderPaddingMask, selfAttentionPaddingMask), dim=1
                )
            assert encoderOutput is not None
            y = torch.cat((encoderOutput, selfAttentionInput), dim=0)
        else:
            y = selfAttentionInput

        inputBatch, attentionWeights, auxiliaryLoss = self.selfAttentionModel(
            query=inputBatch,
            key=y,
            value=y,
            layerIdx=layerIdx,
            queryPaddingMask=queryPaddingMask,
            keyPaddingMask=keyPaddingMask,
            incrementalState=incrementalState,
            needHeadWeights=False,
            selfAttentionMask=selfAttentionMask,
            skipMask=haltMask,
        )

        # TODO: find out what is the point of this
        if self.c_attn is not None:
            targetLength, batchSize = inputBatch.size(0), inputBatch.size(1)
            inputBatch = inputBatch.view(
                targetLength, batchSize, self.numHeads, self.headDim
            )
            inputBatch = torch.einsum("tbhd,h->tbhd", inputBatch, self.c_attn)
            inputBatch = inputBatch.reshape(targetLength, batchSize, self.embedDim)

        # TODO: find out why this layer not is even here ? becaise 6 lines
        # below another layernorm is perfomred
        # if self.attentionLayerNorm is not None:
        #     inputBatch = self.selfAttentionLayerNorm(inputBatch)

        inputBatch = self.dropoutModule(inputBatch)
        inputBatch = self.residualConnection(inputBatch, residual)
        if not self.normalizeBeforeFlag:
            inputBatch = self.selfAttentionLayerNorm(inputBatch)

        inputBatch = self._computeCrossAttention(
            inputBatch,
            encoderOutput,
            previousEncoderAttentionState,
            selfAttentionMask,
            encoderPaddingMask,
            layerIdx,
            incrementalState,
            haltMask,
        )

        inputBatch = self._computeFeedForward(inputBatch, haltMask)

        if self.onnxTrace and incrementalState is not None:
            savedState = self.selfAttentionModel._getInputBuffer(
                incrementalState, layerIdx
            )
            assert savedState is not None
            if selfAttentionPaddingMask is not None:
                selfAttentionState = [
                    savedState["previousKeyProjection"],
                    savedState["previousValueProjection"],
                    savedState["previousKeyPaddingMask"],
                ]
            else:
                selfAttentionState = [
                    savedState["previousKeyProjection"],
                    savedState["previousValueProjection"],
                ]
            return inputBatch, attention, attentionWeights, None
        return inputBatch, attention, None, None

    def _computeCrossAttention(
        self,
        inputBatch,
        encoderOutput,
        previousEncoderAttentionState,
        selfAttentionMask,
        encoderPaddingMask,
        layerIdx,
        incrementalState,
        haltMask,
    ):
        if self.encoderAttention is not None and encoderOutput is not None:
            residual = inputBatch
            if self.normalizeBeforeFlag:
                inputBatch = self.encoderAttentionLayerNorm(inputBatch)

            if previousEncoderAttentionState is not None:
                previousKeyProjection, previousValueProjection = (
                    previousEncoderAttentionState[:2]
                )

                savedState: Dict[str, Optional[Tensor]] = {
                    "previousKeyProjection": previousKeyProjection,
                    "previousValueProjection": previousValueProjection,
                }

                if len(previousEncoderAttentionState) >= 3:
                    savedState["previousKeyPaddingMask"] = previousKeyProjection[2]
                assert incrementalState is not None
                self.encoderAttention._updateIncrementalState(
                    incrementalState, savedState, layerIdx
                )

            inputBatch, attention, crossAuxiliaryAttention = self.encoderAttention(
                query=inputBatch,
                key=encoderOutput,
                value=encoderOutput,
                layerIdx=layerIdx,
                queryPaddingMask=selfAttentionMask,
                keyPaddingMask=encoderPaddingMask,
                incrementalState=incrementalState,
                # needWeights=need_attn or (not self.training and self.need_attn),
                # needHeadWeights = needHeadWeights,
                skipMask=haltMask,
            )
            inputBatch = self.dropoutModule(inputBatch)
            inputBatch = self.residualConnection(inputBatch, residual)
            if not self.normalizeBeforeFlag:
                inputBatch = self.encoderAttentionLayerNorm(inputBatch)

        return inputBatch

    def _computeFeedForward(self, inputBatch, haltMask):
        residual = inputBatch
        if self.wResid:
            residual = torch.mul(self.wResid, residual)

        if self.normalizeBeforeFlag:
            inputBatch = self.feedForwardLayerNormModule(inputBatch)

        inputBatch = self.mixtureOfExpertsModel(inputBatch, haltMask)
        inputBatch = self.feedForwardDroputModule(inputBatch)
        inputBatch = self.residualConnection(inputBatch, residual)

        if not self.normalizeBeforeFlag:
            inputBatch = self.feedForwardLayerNormModule(inputBatch)

        return inputBatch
