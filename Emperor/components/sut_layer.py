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
        qkvHiddenDim: Optional[int] = None,
        ffnHiddenDim: Optional[int] = None,
        activationFunction: Optional[nn.Module] = None,
        attnDropoutProbability: Optional[float] = None,
        ffnDropoutProbability: Optional[float] = None,
        returnRawFFNOutputFlag: Optional[bool] = None,
        normalizeBeforeFlag: Optional[bool] = None,
    ):
        super().__init__()

        self.cfg = cfg
        self.embeddingDim = self._getValue(embeddingDim, cfg.embeddingDim)
        self.qkvHiddenDim = self._getValue(qkvHiddenDim, cfg.qkvHiddenDim)
        self.ffnHiddenDim = self._getValue(ffnHiddenDim, cfg.ffnHiddenDim)
        self.activationFunction = self._getValue(
            activationFunction, cfg.activationFunction
        )
        self.attnDropoutProbability = self._getValue(
            attnDropoutProbability, cfg.attnDropoutProbability
        )
        self.ffnDropoutProbability = self._getValue(
            ffnDropoutProbability, cfg.ffnDropoutProbability
        )
        self.returnRawFFNOutputFlag = self._getValue(
            returnRawFFNOutputFlag, cfg.returnRawFFNOutputFlag
        )
        self.normalizeBeforeFlag = self._getValue(
            normalizeBeforeFlag, cfg.normalizeBeforeFlag
        )

        # self.quant_noise = cfg.quant_noise.pq
        # self.quant_noise_block_size = cfg.quant_noise.pq_block_size

        self._initializeSelfAttentionModules()
        self._initializeFeedForwardModules()

    def _initializeSelfAttentionModules(self):
        self.attnLayerNormModule = nn.LayerNorm(self.embeddingDim)
        self.attnDropoutModule = nn.Dropout(self.attnDropoutProbability)
        self.attentionModel = self._createAttentionModelHook()

    def _createAttentionModelHook(self):
        return Attention(
            self.cfg,
            embeddingDim=self.embeddingDim,
            qkvHiddenDim=self.qkvHiddenDim,
        )

    def _initializeFeedForwardModules(self):
        self.ffnLayerNormModule = nn.LayerNorm(self.embeddingDim)
        self.ffnDroputModule = nn.Dropout(self.ffnDropoutProbability)
        self.ffnModel = self._createFeedForwadModelModelHook()

    def _createFeedForwadModelModelHook(self):
        return MixtureOfExperts(
            self.cfg,
            inputDim=self.embeddingDim,
            hiddenDim=self.ffnHiddenDim,
            outputDim=self.embeddingDim,
        )

    def forward(
        self,
        inputBatch: Tensor,
        selfAttentionInput: Optional[Tensor] = None,
        haltMask: Optional[Tensor] = None,
        encoderPaddingMask: Optional[Tensor] = None,
        attentionMask: Optional[Tensor] = None,
    ):
        if selfAttentionInput is None:
            selfAttentionInput = inputBatch

        attentionOutput = self._computeAttention(
            inputBatch,
            selfAttentionInput,
            haltMask,
            encoderPaddingMask,
            attentionMask,
        )

        ffnOutput, ffnRawOutput = self._computeFeedForward(attentionOutput, haltMask)

        return ffnOutput, ffnRawOutput

    def _computeAttention(
        self,
        inputBatch: Tensor,
        selfAttentionInput: Optional[Tensor],
        haltMask: Optional[Tensor],
        encoderPaddingMask: Optional[Tensor],
        attentionMask: Optional[Tensor],
    ):
        assert inputBatch is not None, (
            f"Ensure the `inputBatch` is a `Tensor`, received {type(inputBatch)}"
        )

        if attentionMask is not None:
            fillElement = -1e8 if inputBatch.dtype == torch.float32 else -1e4
            attentionMask = attentionMask.masked_fill(
                attentionMask.to(torch.bool), fillElement
            )

        residual = inputBatch
        if self.normalizeBeforeFlag:
            inputBatchTemp = self.attnLayerNormModule(inputBatch)
            if inputBatch is selfAttentionInput:
                selfAttentionInput = inputBatchTemp
            else:
                selfAttentionInput = self.attnLayerNormModule(selfAttentionInput)
            inputBatch = inputBatchTemp

        attnOutput, _ = self.attentionModel(
            query=inputBatch,
            key=selfAttentionInput,
            value=selfAttentionInput,
            queryPaddingMask=encoderPaddingMask,
            keyPaddingMask=encoderPaddingMask,
            attentionMask=attentionMask,
            skipMask=haltMask,
        )

        attnOutput = self.attnDropoutModule(attnOutput)
        attnOutput = self._computeResidualConnection(attnOutput, residual)

        if not self.normalizeBeforeFlag:
            attnOutput = self.attnLayerNormModule(inputBatch)

        return attnOutput

    def _computeFeedForward(
        self,
        attentionOutput: Tensor,
        haltMask: Optional[Tensor],
    ):
        residual = attentionOutput
        normalizedFFNInput = attentionOutput
        if self.normalizeBeforeFlag:
            normalizedFFNInput = self.ffnLayerNormModule(attentionOutput)

        ffnRawOutput = self.ffnModel(normalizedFFNInput, haltMask)
        sparseFFNdOutput = self.ffnDroputModule(ffnRawOutput)
        residualConnection = self._computeResidualConnection(sparseFFNdOutput, residual)

        normalizedOutput = residualConnection
        if not self.normalizeBeforeFlag:
            normalizedOutput = self.ffnLayerNormModule(residualConnection)

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
    def __init__(
        self,
        cfg: "ModelConfig",
        embeddingDim: Optional[int] = None,
        qkvHiddenDim: Optional[int] = None,
        ffnHiddenDim: Optional[int] = None,
        activationFunction: Optional[nn.Module] = None,
        attnDropoutProbability: Optional[float] = None,
        ffnDropoutProbability: Optional[float] = None,
        returnRawFFNOutputFlag: Optional[bool] = None,
        normalizeBeforeFlag: Optional[bool] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.embeddingDim = self._getValue(embeddingDim, cfg.embeddingDim)
        self.qkvHiddenDim = self._getValue(qkvHiddenDim, cfg.qkvHiddenDim)
        self.ffnHiddenDim = self._getValue(ffnHiddenDim, cfg.ffnHiddenDim)
        self.activationFunction = self._getValue(
            activationFunction, cfg.activationFunction
        )
        self.attnDropoutProbability = self._getValue(
            attnDropoutProbability, cfg.attnDropoutProbability
        )
        self.ffnDropoutProbability = self._getValue(
            ffnDropoutProbability, cfg.ffnDropoutProbability
        )
        self.returnRawFFNOutputFlag = self._getValue(
            returnRawFFNOutputFlag, cfg.returnRawFFNOutputFlag
        )
        self.normalizeBeforeFlag = self._getValue(
            normalizeBeforeFlag, cfg.normalizeBeforeFlag
        )

        # -----------------------------------------

        self.crossSelfAttentionFlag = cfg.crossSelfAttentionFlag
        self.activationFunction = cfg.activationFunction
        self.numHeads = self.selfAttentionModel.numHeads
        self.headDim = self.selfAttentionModel.headDim
        self.scaleHeads = cfg.scaleHeads
        self.encoderAttentionFlag = cfg.encoderAttentionFlag

        # self.quantNoise = cfg.qantNoise
        # self.quantNosieBlockNoise = cfg.quantNoise.pqBlockSize

        # -----------------------------------------

        # self.dropoutModule = nn.Dropout(cfg.dropout)
        # self.activationFunctionDropoutModule = nn.Dropout(cfg.dropout)

        self.scaleSelfAttentionHeadsFlag = cfg.scaleSelfAttentionHeadsFlag

        self.selfAttnHeadScalers = None
        if self.scaleSelfAttentionHeadsFlag:
            self.selfAttnHeadScalers = nn.Parameter(
                torch.ones((self.numHeads)), requires_grad=True
            )

        self.normalizeBeforeFlag = cfg.normalizeBeforeFlag

        wResidDefaultParameters = torch.ones(self.embedDim)
        self.wResid = nn.Parameter(wResidDefaultParameters, requires_grad=True)

        # act_list = [
        #     ModuleWrapper(self.activationFunction),
        #     self.activationFunctionDropoutModule,
        # ]
        #
        # if self.feedForwardLayerNormModule is not None:
        #     act_list = act_list + [self.feedForwardLayerNormModule]

        # self.mixtureOfExpertsModel = MixtureOfExperts(cfg)
        # self.feedForwardLayerNormModule = nn.LayerNorm(cfg.decoder.ffn_embed_dim)
        #
        # self.feedForwardDroputModule = nn.LayerNorm(self.embedDim)

        self.needAttention = True
        self.onnxTrace = False

        self._initSelfAttentionModules()
        self._initEncoderDecoderAttentionModules()
        self._initFeedForwardModules()

    def _initSelfAttentionModules(self):
        self.selfAttnLayerNorm = nn.LayerNorm(self.embeddingDim)
        self.selfAttnModule = nn.Dropout(self.selfAttnDropoutProbability)
        self.selfAttnModel = self._createSelfAttentionModelHook()

    def _createSelfAttentionModelHook(self):
        return Attention(
            self.cfg,
            embeddingDim=self.embeddingDim,
            qkvHiddenDim=self.qkvHiddenDim,
        )

    def _initEncoderDecoderAttentionModules(self):
        self.encoderDecoderAttn = None
        self.encoderAttnLayerNorm = None
        self.encoderAttnDropout = None
        if self.encoderAttentionFlag:
            self.encoderAttnLayerNorm = nn.LayerNorm(self.embeddingDim)
            self.encoderAttnDropout = nn.Dropout(self.encoderDecoderAttnProbability)
            self.encoderDecoderAttn = self._createEncoderDecoderAttnModelHook()

    def _createEncoderDecoderAttnModelHook(self):
        return Attention(
            self.cfg,
            embeddingDim=self.embeddingDim,
            qkvHiddenDim=self.qkvHiddenDim,
            staticKeyValueFlag=True,
        )

    def _initFeedForwardModules(self):
        self.ffnLayerNormModule = nn.LayerNorm(self.embeddingDim)
        self.ffnDroputModule = nn.Dropout(self.ffnDropoutProbability)
        self.ffnModel = self._createFeedForwadModelModelHook()

    def _createFeedForwadModelModelHook(self):
        return MixtureOfExperts(
            self.cfg,
            inputDim=self.embeddingDim,
            hiddenDim=self.ffnHiddenDim,
            outputDim=self.embeddingDim,
        )

    def residualConnection(self, inputBatch, residualConnection):
        return residualConnection + inputBatch

    def forward(
        self,
        inputBatch,
        haltMask: Tensor,
        layerIdx: int,
        selfAttentionInput: Optional[Tensor],
        encoderOutput: Optional[Tensor] = None,
        encoderPaddingMask: Optional[Tensor] = None,
        previousSelfAttentionState: Optional[List[Tensor]] = None,
        previousEncoderDecoderAttentionState: Optional[List[Tensor]] = None,
        incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        selfAttentionMask: Optional[Tensor] = None,
        selfAttentionPaddingMask: Optional[Tensor] = None,
        needHeadWeights: bool = False,
    ):
        if selfAttentionInput is None:
            selfAttentionInput = inputBatch

        if needHeadWeights:
            needAttentionWeights = True

        _currentSelfAttentionInputBuffer = self._getSelfAttentionSavedState(
            layerIdx, previousSelfAttentionState, incrementalState
        )

        checkIfIncrementalStateDoesExistFlag = (
            incrementalState is not None
            and _currentSelfAttentionInputBuffer is not None
            and "previousKeyProjection" in _currentSelfAttentionInputBuffer
        )

        selfAttentionMask, selfAttentionPaddingMask = (
            self._updateSelfAttentionPaddingMasks(
                inputBatch=inputBatch,
                encoderOutput=encoderOutput,
                encoderPaddingMask=encoderPaddingMask,
                selfAttentionMask=selfAttentionMask,
                selfAttentionPaddingMask=selfAttentionPaddingMask,
                checkIfIncrementalStateDoesExistFlag=checkIfIncrementalStateDoesExistFlag,
            )
        )

        inputBatch = self._computeSelfAttention(
            inputBatch,
            encoderOutput,
            previousEncoderAttentionState,
            selfAttentionMask,
            encoderPaddingMask,
            layerIdx,
            incrementalState,
            haltMask,
        )

        inputBatch = self._computeEncoderDecoderAttention(
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
                    savedState["previousKeyMultiHeadProjection"],
                    savedState["previousValueMultiHeadProjection"],
                    savedState["previousKeyPaddingMask"],
                ]
            else:
                selfAttentionState = [
                    savedState["previousKeyMultiHeadProjection"],
                    savedState["previousValueMultiHeadProjection"],
                ]
            return inputBatch, attention, attentionWeights, None
        return inputBatch, attention, None, None

    def _getSelfAttentionSavedState(
        self,
        layerIdx: int,
        previousSelfAttentionState: Optional[List[Tensor]] = None,
        incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        if previousSelfAttentionState is not None:
            previousKeyProjection, previousValueProjection = previousSelfAttentionState[
                :2
            ]
            savedState: Dict[str, Optional[Tensor]] = {
                "previousKeyMultiHeadProjection": previousKeyProjection,
                "previousValueMultiHeadProjection": previousValueProjection,
            }
            if len(previousSelfAttentionState) >= 3:
                savedState["previousKeyPaddingMask"] = previousSelfAttentionState[2]
            assert incrementalState is not None
            self.selfAttentionModel._updateIncrementalState(
                incrementalState, savedState, layerIdx
            )
        return self.selfAttentionModel._getInputBuffer(incrementalState, layerIdx)

    def _computeSelfAttention(
        self,
        inputBatch: Tensor,
        haltMask: Tensor,
        layerIdx: int,
        selfAttentionInput: Optional[Tensor],
        encoderOutput: Optional[Tensor],
        incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        selfAttentionMask: Optional[Tensor],
        selfAttentionPaddingMask: Optional[Tensor],
        checkIfIncrementalStateDoesExistFlag: bool = False,
    ):
        residual = inputBatch
        if self.normalizeBeforeFlag:
            inputBatchTemp = self.selfAttnLayerNorm(inputBatch)
            if selfAttentionInput is inputBatch:
                selfAttentionInput = inputBatchTemp
            else:
                selfAttentionInput = self.selfAttnLayerNorm(selfAttentionInput)
            inputBatch = inputBatchTemp

        if self.crossSelfAttentionFlag and not checkIfIncrementalStateDoesExistFlag:
            assert encoderOutput is not None
            assert selfAttentionInput is not None
            selfAttention = torch.cat((encoderOutput, selfAttentionInput), dim=0)
        else:
            selfAttention = selfAttentionInput

        attnOutput, _ = self.selfAttnModel(
            query=inputBatch,
            key=selfAttention,
            value=selfAttention,
            layerIdx=layerIdx,
            queryPaddingMask=selfAttentionPaddingMask,
            keyPaddingMask=selfAttentionPaddingMask,
            incrementalState=incrementalState,
            needHeadWeights=False,
            selfAttentionMask=selfAttentionMask,
            skipMask=haltMask,
        )

        attnOutput = self._scaleSelfAttentionHeads(attnOutput)
        attnOutput = self.dropoutModule(attnOutput)
        attnOutput = self.residualConnection(attnOutput, residual)

        if not self.normalizeBeforeFlag:
            attnOutput = self.selfAttentionLayerNorm(inputBatch)

        return attnOutput

    def _scaleSelfAttentionHeads(
        self,
        attnOutput: Tensor,
    ):
        if self.selfAttnHeadScalers is not None:
            """
                TO BE INVESTIGATED LATER:
                I don't get why you scale the atteniton heads after the output projection,
                because those have allready been projected by the output layer of the
                attention mechanism

                One wierd thing here is that this implies that `embeddingDim` == `qkvHiddenDim`
                this is the only way this works.
                Einstein summation notation (einsum) is used here:
                    - "tbhd,h->tbhd" means:
                        - Multiply each attention head (h) with a learnable weight vector self.self.selfAttnHeadScalers.
                        - Keep the original shape (T, B, H, D) after the operation.
                - This acts like a per-head scaling factor applied across all batches and tokens.
            """
            assert self.embeddingDim == self.qkvHiddenDim, (
                f"In order to scale the `heads` of the attention mechanism the `embeddingDim` {self.embeddingDim} must be equal to `qkvHiddenDim` {self.qkvHiddenDim}"
            )
            targetLength, batchSize, _ = attnOutput.size()
            attnOutputHeads = attnOutput.view(
                targetLength, batchSize, self.numHeads, self.headDim
            )
            attnOutputScaledHeads = torch.einsum(
                "tbhd,h->tbhd", attnOutputHeads, self.selfAttnHeadScalers
            )
            return attnOutputScaledHeads.reshape(targetLength, batchSize, self.embedDim)

    def _updateSelfAttentionPaddingMasks(
        self,
        inputBatch: Tensor,
        encoderOutput: Optional[Tensor],
        encoderPaddingMask: Optional[Tensor] = None,
        selfAttentionMask: Optional[Tensor] = None,
        selfAttentionPaddingMask: Optional[Tensor] = None,
        checkIfIncrementalStateDoesExistFlag: bool = False,
    ):
        if self.crossSelfAttentionFlag and not checkIfIncrementalStateDoesExistFlag:
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
            if selfAttentionPaddingMask is not None and encoderPaddingMask is not None:
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
        return selfAttentionMask, selfAttentionPaddingMask

    def _computeEncoderDecoderAttention(
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
