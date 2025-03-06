import torch
import torch.nn as nn
from torch.types import Tensor
from Emperor.base.utils import Module
from Emperor.components.attention import Attention
from Emperor.components.moe import MixtureOfExperts

from typing import TYPE_CHECKING, Dict, Optional, List

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

    def _initializeSelfAttentionModules(self) -> None:
        self.attnLayerNormModule = nn.LayerNorm(self.embeddingDim)
        self.attnDropoutModule = nn.Dropout(self.attnDropoutProbability)
        self.attentionModel = self._createAttentionModelHook()

    def _createAttentionModelHook(self) -> Attention:
        return Attention(
            self.cfg,
            embeddingDim=self.embeddingDim,
            qkvHiddenDim=self.qkvHiddenDim,
        )

    def _initializeFeedForwardModules(self) -> None:
        self.ffnLayerNormModule = nn.LayerNorm(self.embeddingDim)
        self.ffnDroputModule = nn.Dropout(self.ffnDropoutProbability)
        self.ffnModel = self._createFeedForwadModelModelHook()

    def _createFeedForwadModelModelHook(self) -> MixtureOfExperts:
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
        layerIdx: Optional[int] = None,
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
        attnOutput: Tensor,
        haltMask: Optional[Tensor],
    ):
        residual = attnOutput
        if self.normalizeBeforeFlag:
            attnOutput = self.ffnLayerNormModule(attnOutput)

        ffnRawOutput = self.ffnModel(attnOutput, haltMask)
        attnOutput = self.ffnDroputModule(ffnRawOutput)
        attnOutput = self._computeResidualConnection(attnOutput, residual)

        if not self.normalizeBeforeFlag:
            attnOutput = self.ffnLayerNormModule(attnOutput)

        if self.returnRawFFNOutputFlag:
            return attnOutput, ffnRawOutput

        return attnOutput, None

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
        selfAttnDropoutProbability: Optional[float] = None,
        crossAttnProbability: Optional[float] = None,
        ffnDropoutProbability: Optional[float] = None,
        returnRawFFNOutputFlag: Optional[bool] = None,
        normalizeBeforeFlag: Optional[bool] = None,
        crossSelfAttentionFlag: Optional[bool] = None,
        scaleSelfAttentionHeadsFlag: Optional[bool] = None,
        scaleResidualsConnectionFlag: Optional[bool] = None,
        crossAttentionFlag: Optional[bool] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.embeddingDim = self._getValue(embeddingDim, cfg.embeddingDim)
        self.qkvHiddenDim = self._getValue(qkvHiddenDim, cfg.qkvHiddenDim)
        self.ffnHiddenDim = self._getValue(ffnHiddenDim, cfg.ffnHiddenDim)
        self.activationFunction = self._getValue(
            activationFunction, cfg.activationFunction
        )
        self.selfAttnDropoutProbability = self._getValue(
            selfAttnDropoutProbability, cfg.selfAttnDropoutProbability
        )
        self.crossAttnProbability = self._getValue(
            crossAttnProbability, cfg.crossAttnProbability
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
        self.crossSelfAttentionFlag = self._getValue(
            crossSelfAttentionFlag, cfg.crossSelfAttentionFlag
        )
        self.crossAttentionFlag = self._getValue(
            crossAttentionFlag, cfg.crossAttentionFlag
        )
        self.scaleSelfAttentionHeadsFlag = self._getValue(
            scaleSelfAttentionHeadsFlag, cfg.scaleSelfAttentionHeadsFlag
        )
        self.scaleResidualsConnectionFlag = self._getValue(
            scaleResidualsConnectionFlag, cfg.scaleResidualsConnectionFlag
        )

        # -----------------------------------------

        # self.crossSelfAttentionFlag = cfg.crossSelfAttentionFlag
        # self.activationFunction = cfg.activationFunction
        # self.scaleHeads = cfg.scaleHeads
        # self.encoderAttentionFlag = cfg.encoderAttentionFlag

        # self.quantNoise = cfg.qantNoise
        # self.quantNosieBlockNoise = cfg.quantNoise.pqBlockSize

        # -----------------------------------------

        # self.dropoutModule = nn.Dropout(cfg.dropout)
        # self.activationFunctionDropoutModule = nn.Dropout(cfg.dropout)

        self.needAttention = True
        self.onnxTrace = False

        self.__initSelfAttentionModules()
        self.__initCrossAttentionModules()
        self.__initFeedForwardModules()
        self.__initLearnableScalers()

    def __initSelfAttentionModules(self):
        self.selfAttnLayerNorm = nn.LayerNorm(self.embeddingDim)
        self.selfAttnDropout = nn.Dropout(self.selfAttnDropoutProbability)
        self.selfAttnModel = self._createSelfAttentionModelHook()

        self.numHeads = self.selfAttnModel.numHeads
        self.headDim = self.selfAttnModel.headDim

    def _createSelfAttentionModelHook(self):
        return Attention(
            self.cfg,
            embeddingDim=self.embeddingDim,
            qkvHiddenDim=self.qkvHiddenDim,
        )

    def __initCrossAttentionModules(self):
        self.crossAttnLayerNorm = self.crossAttnDropout = self.crossAttnModel = None
        if self.crossSelfAttentionFlag:
            self.crossAttnLayerNorm = nn.LayerNorm(self.embeddingDim)
            self.crossAttnDropout = nn.Dropout(self.crossAttnProbability)
            self.crossAttnModel = self._createCrossAttnModelHook()

    def _createCrossAttnModelHook(self):
        return Attention(
            self.cfg,
            embeddingDim=self.embeddingDim,
            qkvHiddenDim=self.qkvHiddenDim,
            encoderDecoderAttentionFlag=True,
            staticKeyValueFlag=True,
        )

    def __initFeedForwardModules(self):
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

    def __initLearnableScalers(self):
        self.selfAttnHeadScalers = None
        if self.scaleSelfAttentionHeadsFlag:
            self.selfAttnHeadScalers = nn.Parameter(
                torch.ones((self.numHeads)), requires_grad=True
            )

        self.residualConnectionScalers = None
        if self.scaleResidualsConnectionFlag:
            self.residualConnectionScalers = nn.Parameter(
                torch.ones(self.embeddingDim), requires_grad=True
            )

    def forward(
        self,
        inputBatch: Tensor,
        haltMask: Optional[Tensor] = None,
        layerIdx: Optional[int] = None,
        selfAttentionInput: Optional[Tensor] = None,
        encoderOutput: Optional[Tensor] = None,
        encoderPaddingMask: Optional[Tensor] = None,
        previousSelfAttentionState: Optional[List[Tensor]] = None,
        previousCrossAttnState: Optional[List[Tensor]] = None,
        incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        selfAttentionMask: Optional[Tensor] = None,
        selfAttentionPaddingMask: Optional[Tensor] = None,
        needHeadWeights: bool = False,
    ):
        if selfAttentionInput is None:
            selfAttentionInput = inputBatch

        if needHeadWeights:
            needAttentionWeights = True

        self._updateCurrentLayerIncrementalState(
            layerIdx, previousSelfAttentionState, incrementalState
        )

        _currentSelfAttentionInputBuffer = self.selfAttnModel._getSavedState(
            incrementalState, layerIdx
        )

        checkIfIncrementalStateDoesExistFlag = (
            incrementalState is not None
            and _currentSelfAttentionInputBuffer is not None
            and "previousKeyMultiHeadProjection" in _currentSelfAttentionInputBuffer
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

        selfAttentionOutput, attentionWeights = self._computeSelfAttention(
            inputBatch=inputBatch,
            selfAttentionInput=selfAttentionInput,
            haltMask=haltMask,
            layerIdx=layerIdx,
            incrementalState=incrementalState,
            selfAttentionMask=selfAttentionMask,
            selfAttentionPaddingMask=selfAttentionPaddingMask,
            encoderOutput=encoderOutput,
            checkIfIncrementalStateDoesExistFlag=checkIfIncrementalStateDoesExistFlag,
        )

        coressAttentionOutput, attentionWeights = self._computeCrossAttention(
            selfAttnOutput=selfAttentionOutput,
            encoderOutput=encoderOutput,
            encoderPaddingMask=encoderPaddingMask,
            haltMask=haltMask,
            layerIdx=layerIdx,
            incrementalState=incrementalState,
            previousCrossAttnState=previousCrossAttnState,
            selfAttentionPaddingMask=selfAttentionPaddingMask,
        )

        decoderOutput, ffnRawOutput = self._computeFeedForward(
            coressAttentionOutput, haltMask
        )

        # TODO: for later understand what this `self.onnxTrace` flag is
        # and  what it represents
        if self.onnxTrace and incrementalState is not None:
            savedState = self.selfAttnModel._getSavedState(incrementalState, layerIdx)
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
            return decoderOutput, attentionWeights, selfAttentionState, ffnRawOutput
        return decoderOutput, attentionWeights, None, ffnRawOutput

    def _updateCurrentLayerIncrementalState(
        self,
        layerIdx: int,
        previousSavedState: Optional[List[Tensor]] = None,
        incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        if previousSavedState is not None and incrementalState is not None:
            previousKeyProjection, previousValueProjection = previousSavedState[:2]
            savedState: Dict[str, Optional[Tensor]] = {
                "previousKeyMultiHeadProjection": previousKeyProjection,
                "previousValueMultiHeadProjection": previousValueProjection,
            }
            if len(previousSavedState) >= 3:
                savedState["previousKeyPaddingMask"] = previousSavedState[2]
            assert incrementalState is not None

            layerId = "attn_state_%d" % layerIdx
            incrementalState = (
                self.selfAttnModel.incrementalStateModule.setIncrementalState(
                    incrementalState, layerId, savedState
                )
            )

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

    def _computeSelfAttention(
        self,
        inputBatch: Tensor,
        selfAttentionInput: Optional[Tensor],
        haltMask: Tensor,
        layerIdx: int,
        incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        selfAttentionMask: Optional[Tensor],
        selfAttentionPaddingMask: Optional[Tensor],
        encoderOutput: Optional[Tensor],
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

        attnOutput, attnWeights = self.selfAttnModel(
            query=inputBatch,
            key=selfAttention,
            value=selfAttention,
            layerIdx=layerIdx,
            queryPaddingMask=selfAttentionPaddingMask,
            keyPaddingMask=selfAttentionPaddingMask,
            incrementalState=incrementalState,
            # needHeadWeights=False,
            attentionMask=selfAttentionMask,
            skipMask=haltMask,
        )

        attnOutput = self._scaleSelfAttentionHeads(attnOutput)
        attnOutput = self.selfAttnDropout(attnOutput)
        attnOutput = self._computeResidualConnection(attnOutput, residual)

        if not self.normalizeBeforeFlag:
            attnOutput = self.selfAttnLayerNorm(inputBatch)

        return attnOutput, attnWeights

    def _scaleSelfAttentionHeads(
        self,
        attnOutput: Tensor,
    ):
        if self.scaleSelfAttentionHeadsFlag:
            """
                TO BE INVESTIGATED LATER:
                I don't get why you scale the atteniton heads after the output projection,
                because those have allready been projected by the output layer of the
                attention mechanism before the output projection

                One wierd thing here is that this implies that `embeddingDim` == `qkvHiddenDim` is required
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

        return attnOutput

    def _computeCrossAttention(
        self,
        selfAttnOutput: Tensor,
        encoderOutput: Optional[Tensor],
        encoderPaddingMask: Optional[Tensor],
        haltMask: Tensor,
        layerIdx: int,
        incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        previousCrossAttnState: Optional[List[Tensor]],
        selfAttentionPaddingMask: Optional[Tensor],
    ):
        attentionWeights = None
        if self.crossSelfAttentionFlag and encoderOutput is not None:
            residual = selfAttnOutput
            if self.normalizeBeforeFlag:
                selfAttnOutput = self.crossAttnLayerNorm(selfAttnOutput)

            self._updateCurrentLayerIncrementalState(
                layerIdx=layerIdx,
                previousSavedState=previousCrossAttnState,
                incrementalState=incrementalState,
            )

            attnOutput, attentionWeights = self.crossAttnModel(
                query=selfAttnOutput,
                key=encoderOutput,
                value=encoderOutput,
                layerIdx=layerIdx,
                queryPaddingMask=selfAttentionPaddingMask,
                keyPaddingMask=encoderPaddingMask,
                incrementalState=incrementalState,
                # needWeights=need_attn or (not self.training and self.need_attn),
                # needHeadWeights = needHeadWeights,
                skipMask=haltMask,
            )

            attnOutput = self.crossAttnDropout(attnOutput)
            attnOutput = self._computeResidualConnection(attnOutput, residual)

            if not self.normalizeBeforeFlag:
                attnOutput = self.crossAttnLayerNorm(attnOutput)
            selfAttnOutput = attnOutput

        return selfAttnOutput, attentionWeights

    def _computeResidualConnection(self, inputBatch, residualConnection):
        return residualConnection + inputBatch

    def _computeFeedForward(self, attnOutput: Tensor, haltMask: Optional[Tensor]):
        x = attnOutput
        residual = x
        residual = self._scalreResidualConnectionScalers(residual)

        if self.normalizeBeforeFlag:
            x = self.ffnLayerNormModule(x)

        ffnRawOutput = self.ffnModel(x, haltMask)
        x = self.ffnDroputModule(ffnRawOutput)
        x = self._computeResidualConnection(x, residual)

        if not self.normalizeBeforeFlag:
            x = self.ffnLayerNormModule(x)

        if self.returnRawFFNOutputFlag:
            return x, ffnRawOutput

        return x, None

    def _scalreResidualConnectionScalers(self, residualConnection):
        # TODO: when going in more detail into this make sure you understand
        # what the point of scaling the residual connection is
        if self.residualConnectionScalers is not None:
            residualConnection = torch.mul(
                self.residualConnectionScalers, residualConnection
            )
        return residualConnection
