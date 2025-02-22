from dataclasses import replace
from torch import Tensor, save
import math
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
from Emperor.base.utils import Module
from Emperor.library.choice import Library as L
from .moe import MixtureOfAttentionHeads

from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class Attention(Module):
    def __init__(
        self,
        cfg: Optional["ModelConfig"] = None,
        embeddingDim: Optional[int] = None,
        queryInputDim: Optional[int] = None,
        keyInputDim: Optional[int] = None,
        valueInputDim: Optional[int] = None,
        qkvHiddenDim: Optional[int] = None,
        attentionOutputDim: Optional[int] = None,
        dropoutProbability: Optional[float] = None,
        biasFlag: Optional[bool] = None,
        addZeroAttentionFlag: Optional[bool] = None,
        selfAttentionFlag: Optional[bool] = None,
        encoderDecoderAttentionFlag: Optional[bool] = None,
        quantNoise: Optional[float] = None,
        quantBlockSize: Optional[int] = None,
        numExperts: Optional[int] = None,
        topK: Optional[int] = None,
        headDim: Optional[int] = None,
        coefficientOfVariationLossWeight: Optional[float] = None,
        switchLossWeight: Optional[float] = None,
        zeroCenteredLossWeight: Optional[float] = None,
        mutualInformationLossWeight: Optional[float] = None,
        randomSampleTopK: Optional[int] = None,
        gatingDropout: Optional[int] = None,
        addMemoryBiasKeyValuesFlag: Optional[bool] = None,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.embeddingDim: int = self._getValue(embeddingDim, cfg.embeddingDim)
        queryInputDim = self._getValue(queryInputDim, cfg.queryInputDim)
        self.queryInputDim: int = self._getDim(queryInputDim)
        keyInputDim = self._getValue(keyInputDim, cfg.keyInputDim)
        self.keyInputDim: int = self._getDim(keyInputDim)
        valueInputDim = self._getValue(valueInputDim, cfg.valueInputDim)
        self.valueInputDim: int = self._getDim(valueInputDim)
        qkvHiddenDim = self._getValue(qkvHiddenDim, cfg.qkvHiddenDim)
        self.qkvHiddenDim: int = self._getDim(qkvHiddenDim)
        attentionOutputDim = self._getValue(attentionOutputDim, cfg.attentionOutputDim)
        self.attentionOutputDim: int = self._getDim(attentionOutputDim)
        self.dropoutProbability: float = self._getValue(
            dropoutProbability, cfg.dropoutProbability
        )
        self.biasFlag: bool = self._getValue(biasFlag, cfg.biasFlag)
        self.addZeroAttentionFlag: bool = self._getValue(
            addZeroAttentionFlag, cfg.addZeroAttentionFlag
        )
        self.selfAttentionFlag: bool = self._getValue(
            selfAttentionFlag, cfg.selfAttentionFlag
        )
        self.encoderDecorderAttentionFlag: bool = self._getValue(
            encoderDecoderAttentionFlag, cfg.encoderDecoderAttentionFlag
        )

        self.quantNoise: float = self._getValue(quantNoise, cfg.quantNoise)
        self.quantBlockSize: int = self._getValue(quantBlockSize, cfg.quantBlockSize)

        self.numExperts: int = self._getValue(numExperts, cfg.numExperts)
        self.topK: int = self._getValue(topK, cfg.topK)
        self.headDim: int = self._getValue(headDim, cfg.headDim)
        self.coefficientOfVariationLossWeight: float = self._getValue(
            coefficientOfVariationLossWeight, cfg.coefficientOfVariationLossWeight
        )
        self.switchLossWeight: float = self._getValue(
            switchLossWeight, cfg.switchLossWeight
        )
        self.zeroCentredLossWeight: float = self._getValue(
            zeroCenteredLossWeight, cfg.zeroCenteredLossWeight
        )
        self.mutualInformationLossWeight: float = self._getValue(
            mutualInformationLossWeight, cfg.mutualInformationLossWeight
        )
        self.randomSampleTopK: int = self._getValue(
            randomSampleTopK, cfg.randomSampleTopK
        )
        self.gatingDropout: int = self._getValue(gatingDropout, cfg.gatingDropout)
        self.addMemoryBiasKeyValuesFlag = self._getValue(
            addMemoryBiasKeyValuesFlag, cfg.addMemoryBiasKeyValuesFlag
        )
        assert self.qkvHiddenDim % self.headDim == 0, (
            f"Ensure that `qkvHiddenDim` is perfeclty divisible by `headDim` perfect number of heads, `qkvHiddenDim`:{qkvHiddenDim} and `headDim`: {headDim}"
        )

        self.numHeads = self.qkvHiddenDim // self.headDim
        self.scaling = self.headDim**-0.5
        self.maxPositions = 64

        self.inputShape: Optional[Tuple] = None

        # TODO: make sure these flags are necessary
        self.attentionProjectionBiasFlag = False
        self.attentionWeightsBeforeSoftmaxFlag = False
        self.staticKeyValueFlag = False
        self.createEmptyFlag = False

        self._prepareQKVModels()
        self._prepareModules()
        self._prepareMemoryBiases()
        self._prepareRelativePosintionalParameters()
        self._resetParameters()

    def _prepareQKVModels(self):
        self.keyProjectionModel = nn.Linear(
            self.keyInputDim,
            self.qkvHiddenDim,
            bias=self.biasFlag,
        )
        self.valueProjectionModel = nn.Linear(
            self.valueInputDim,
            self.qkvHiddenDim,
            bias=self.biasFlag,
        )
        cfg = replace(
            self.cfg,
            inputDim=self.queryInputDim,
            hiddenDim=self.qkvHiddenDim,
            outputDim=self.attentionOutputDim,
            biasFlag=self.biasFlag,
        )
        self.queryProjectionModel = MixtureOfAttentionHeads(cfg)

    def _prepareModules(self):
        self.dropoutModule = L.Dropout(self.dropoutProbability)
        self.incrementalStateModule = IncrementalState()

    def _prepareMemoryBiases(self):
        self.keyMemoryBiases = self.valueMemoryBiases = None
        if self.addMemoryBiasKeyValuesFlag:
            self.keyMemoryBiases = L.Parameter(L.Tensor(1, 1, self.qkvHiddenDim))
            self.valueMemoryBiases = L.Parameter(L.Tensor(1, 1, self.qkvHiddenDim))

    def _prepareRelativePosintionalParameters(self):
        self.relativePositionEmbedding = None
        if self.selfAttentionFlag:
            parameterDefault = torch.Tensor(
                self.numHeads, self.headDim, self.maxPositions * 2 + 1
            )
            self.relativePositionEmbedding = L.Parameter(parameterDefault)

    def _resetParameters(self):
        # Commented code because of `MixtureOfExperts` model
        if self._areQKVDimEqual():
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(
                self.keyProjectionModel.weight, gain=1 / math.sqrt(2)
            )
            nn.init.xavier_uniform_(
                self.valueProjectionModel.weight, gain=1 / math.sqrt(2)
            )
            # nn.init.xavier_uniform_(self.queryProjectionModel.weight, gain=1/math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.keyProjectionModel.weight)
            nn.init.xavier_uniform_(self.valueProjectionModel.weight)
            # nn.init.xavier_uniform_(self.queryProjectionModel.weight)

        # nn.init.xavier_uniform_(self.attentionOutputModel.weight)
        # if self.attentionOutputModel.bias is not None:
        #     nn.init.constant_(self.attentionOutputModel.bias, 0.0)

        if self.keyMemoryBiases is not None:
            nn.init.xavier_uniform_(self.keyMemoryBiases)

        if self.valueMemoryBiases is not None:
            nn.init.xavier_uniform_(self.valueMemoryBiases)

        if self.selfAttentionFlag and self.relativePositionEmbedding is not None:
            nn.init.zeros_(self.relativePositionEmbedding)

    def _getDim(self, dimension: Optional[int] = None, default: Optional[int] = None):
        default = default if default is not None else self.embeddingDim
        return dimension if dimension is not None else default

    def _areQKVDimEqual(self):
        areQueryKeyDimSame = self.keyInputDim == self.queryInputDim
        areQueryValueDimSame = self.valueInputDim == self.queryInputDim
        return areQueryKeyDimSame and areQueryValueDimSame

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor],
        value: Optional[Tensor],
        layerIdx: Optional[int] = None,
        queryPaddingMask: Optional[Tensor] = None,
        keyPaddingMask: Optional[Tensor] = None,
        incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        skipMask: Optional[Tensor] = None,
        attentionMask: Optional[Tensor] = None,
    ):
        self.inputShape = query.size()
        targetLength, batchSize, embeddingDim = query.size()
        sourceLength = targetLength

        key, value, savedState = self._getLayerSavedStateFromIncrementalState(
            key, value, layerIdx, incrementalState
        )

        queryProjection, keyProjection, valueProjection = (
            self._computeQueryKeyValueProjections(query, key, value, skipMask)
        )

        queryProjection = queryProjection * self.scaling

        (keyProjection, valueProjection, attentionMask, keyPaddingMask) = (
            self._attachMemoryBiasesToKeyValueProjections(
                keyProjection, valueProjection, attentionMask, keyPaddingMask
            )
        )

        sourceLength = keyProjection.size(0)

        (queryProjection, keyProjection, valueProjection) = (
            self._splitQueryKeyValueProjectionsIntoMultipleHeads(
                queryProjection,
                keyProjection,
                valueProjection,
            )
        )

        (keyProjection, valueProjection, keyPaddingMask, incrementalState) = (
            self._updateKeyValueProjectionsUsingLayerSavedState(
                keyProjection,
                valueProjection,
                keyPaddingMask,
                layerIdx,
                incrementalState,
                savedState,
            )
        )

        sourceLength = keyProjection.size(2)

        assert keyProjection is not None, (
            f"`keyProjection` should be a tensor but its {keyProjection}"
        )
        assert keyProjection.size(2) == sourceLength, (
            f"For `keyProjection` the sequence length should be {sourceLength} but it's {keyProjection.size(2)}"
        )

        if keyPaddingMask is not None and keyPaddingMask.dim() == 0:
            keyPaddingMask = None

        if keyPaddingMask is not None:
            assert keyPaddingMask.size(0) == batchSize, (
                f"`keyPaddingMask` batch size should be {batchSize} but it's {keyPaddingMask.size(0)} "
            )
            assert keyPaddingMask.size(1) == sourceLength, (
                f"`keyPaddingMask` source length should be {sourceLength} but it's {keyPaddingMask.size(1)} "
            )

        (keyProjection, valueProjection, attentionMask, keyPaddingMask) = (
            self._addZeroAttention(
                keyProjection,
                valueProjection,
                attentionMask,
                keyPaddingMask,
            )
        )

        sourceLength = keyProjection.size(2)
        totalBatchSize = batchSize * self.topK * self.numHeads

        attentionWeights, attentionMask = self._computeAttentionWeights(
            queryProjection,
            keyProjection,
            attentionMask,
            incrementalState,
            totalBatchSize,
            targetLength,
            sourceLength,
        )

        assert list(attentionWeights.size()) == [
            totalBatchSize,
            targetLength,
            sourceLength,
        ]

        attentionWeights = self._maskAttentionWeightsUsingKeyPaddingMask(
            attentionWeights,
            keyPaddingMask,
            totalBatchSize,
            targetLength,
            sourceLength,
        )

        if self.attentionWeightsBeforeSoftmaxFlag:
            return attentionWeights, valueProjection

        attentionProbabilities = self._computeSoftmaxAttentionWeights(attentionWeights)
        weightedValueProjections, targetLength = self._computeWeightedValueProjections(
            attentionProbabilities, valueProjection
        )

        attentionProjection = self._computeAttentionOutputProjection(
            weightedValueProjections, targetLength
        )
        attentionWeights = None

        return attentionProjection, attentionWeights

    def _getLayerSavedStateFromIncrementalState(
        self,
        key: Optional[Tensor],
        value: Optional[Tensor],
        layerIdx: Optional[int] = None,
        incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        savedState = None
        if incrementalState is not None:
            savedState = self._getSavedState(incrementalState, layerIdx)
            if self.staticKeyValueFlag:
                key = value = None

        return key, value, savedState

    def _getSavedState(
        self,
        incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        layerIdx: Optional[int],
    ) -> Optional[Dict[str, Optional[Tensor]]]:
        layerIdPrefix = "attn_state_%d" % layerIdx if layerIdx else ""
        result = self.incrementalStateModule.getIncrementalState(
            incrementalState, layerIdPrefix
        )

        if result is not None:
            return result
        else:
            if self.createEmptyFlag:
                emptyResult: Dict[str, Optional[Tensor]] = {}
                return emptyResult
            else:
                return None

    def _computeQueryKeyValueProjections(
        self,
        query: Tensor,
        key: Optional[Tensor],
        value: Optional[Tensor],
        skipMask: Optional[Tensor] = None,
    ):
        assert query is not None, (
            f"To compute `queryProjection` the `query` input must be a `Tensor`, received {type(query)}"
        )
        if self.encoderDecorderAttentionFlag:
            queryProjection = self.queryProjectionModel.computeHiddenProjection(
                query, skipMask=skipMask
            )
            if key is not None:
                assert value is not None, (
                    f"When computing `encoderDecoderAttention` the `key` and `value` inputs must be a `Tensor`, received `key` {type(key)} and `value` {type(value)}"
                )
                keyProjection = self.keyProjectionModel(key)
                valueProjection = self.valueProjectionModel(value)
            else:
                keyProjection = valueProjection = None
        else:
            assert key is not None, (
                f"To compute `keyProjection` the `key` input must be a `Tensor`, received {type(key)}"
            )
            assert value is not None, (
                f"To compute `valueProjection` the `value` input must be a `Tensor`, received {type(value)}"
            )

            queryProjection = self.queryProjectionModel.computeHiddenProjection(
                query, skipMask=skipMask
            )
            keyProjection = self.keyProjectionModel(key)
            valueProjection = self.valueProjectionModel(value)

        return queryProjection, keyProjection, valueProjection

    def _attachMemoryBiasesToKeyValueProjections(
        self,
        keyProjection: Optional[Tensor] = None,
        valueProjection: Optional[Tensor] = None,
        attentionMask: Optional[Tensor] = None,
        keyPaddingMask: Optional[Tensor] = None,
    ):
        if self.addMemoryBiasKeyValuesFlag:
            _, batchSize, _ = keyProjection.size()
            assert self.keyMemoryBiases is not None, (
                f"Ensure `self.keyMemoryBiases` is a tensor, received {type(self.keyMemoryBiases)}"
            )
            assert self.valueMemoryBiases is not None, (
                f"Ensure `self.valueMemoryBiases` is a tensor, received {type(self.keyMemoryBiases)}"
            )
            assert keyProjection is not None, (
                f"To add `keyMemoryBiases` the `keyProjection` the input must be a `tensor`, received {type(keyProjection)}"
            )
            assert valueProjection is not None, (
                f"To add `valueProjection` the `valueProjection` the input must be a `tensor`, received {type(valueProjection)}"
            )
            repeatedKeyMemoryBiases = self.keyMemoryBiases.repeat(1, batchSize, 1)
            keyProjection = torch.cat([keyProjection, repeatedKeyMemoryBiases])

            repeatedValueMemoryBiases = self.valueMemoryBiases.repeat(1, batchSize, 1)
            valueProjection = torch.cat([valueProjection, repeatedValueMemoryBiases])

            if attentionMask is not None:
                assert attentionMask.ndimension() == 2, (
                    f"Ensure the `attentionMask` is a matrix, received tensor with {attentionMask.ndimension()} with a shape of {attentionMask.size()}"
                )
                attentionMaskDim0 = attentionMask.size(0)
                attentionMaskPadding = attentionMask.new_zeros(attentionMaskDim0, 1)
                attentionMask = torch.cat([attentionMask, attentionMaskPadding], dim=1)

            if keyPaddingMask is not None:
                assert keyPaddingMask.ndimension() == 2, (
                    f"Ensure the `keyPaddingMask` is a matrix, received tensor with {keyPaddingMask.ndimension()} with a shpae of {keyPaddingMask.size()}"
                )
                keyPaddingMaskDim0 = keyPaddingMask.size(0)
                keyPaddingMaskPadding = keyPaddingMask.new_zeros(keyPaddingMaskDim0, 1)
                keyPaddingMask = L.cat([keyPaddingMask, keyPaddingMaskPadding], dim=1)

        return keyProjection, valueProjection, attentionMask, keyPaddingMask

    def _splitQueryKeyValueProjectionsIntoMultipleHeads(
        self,
        queryProjection: Tensor,
        keyProjection: Optional[Tensor] = None,
        valueProjection: Optional[Tensor] = None,
    ):
        """
        Splits the `queryProjection`, `keyProjection`, `valueProjection` into along the
        feature dimension into `numHeads` where each head has a dimension of `headDim`

        Args:
            Query projection: [sequenceLength, batchSize, topk, qkvHiddenDim]
            Key projection: [sequenceLength, batchSize, qkvHiddenDim]
            Value projection: [sequenceLength, batchSize, qkvHiddenDim]

        Returns:
            Reshaped query projection: [batchSize, topK, numHeads, sequenceLength, headDim]
            Reshaped key projection: [batchSize, numHeads, sequenceLength, headDim]
            Reshaped value projection: [batchSize, numHeads, sequenceLength, headDim]
        """

        assert queryProjection is not None, (
            f"Ensure the `queryProjection` is a `Tensor`, received {type(self.queryProjection)}"
        )

        sequenceLength, batchSize, _, _ = queryProjection.size()
        queryProjection = queryProjection.reshape(
            sequenceLength, batchSize, self.topK, self.numHeads, self.headDim
        )
        queryProjection = queryProjection.permute(1, 2, 3, 0, 4)

        if keyProjection is not None:
            keyProjection = keyProjection.reshape(
                -1, batchSize, self.numHeads, self.headDim
            )
            keyProjection = keyProjection.permute(1, 2, 0, 3)

        if valueProjection is not None:
            valueProjection = valueProjection.reshape(
                -1, batchSize, self.numHeads, self.headDim
            )
            valueProjection = valueProjection.permute(1, 2, 0, 3)

        return queryProjection, keyProjection, valueProjection

    def _updateKeyValueProjectionsUsingLayerSavedState(
        self,
        keyMultiHeadProjection: Optional[Tensor],
        valueMultiHeadProjection: Optional[Tensor],
        keyPaddingMask: Optional[Tensor],
        layerIdx: int,
        incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        savedState: Optional[Dict[str, Optional[Tensor]]] = None,
    ):
        """
        Important to understnad that is computed in `transformer_encoder` and `transformer_decoder`
        """
        """
        Method retrieves the `keyMultiHeadProjection`, `valueMultiHeadProjection`, `previousKeyPaddingMask` 
        from the `savedState` and concatenates it to `keyMultiHeadProjection`, 
        `valueMultiHeadProjection` and `keyPaddingMask` along the `sequenceLength` dimension

        Args:
            `keyMultiHeadProjection`: [batchSize, numHeads, sequenceLength, headDim]
            `valueMultiHeadProjection`: [batchSize, numHeads, sequenceLength, headDim]
            `keyPaddingMask`: [batchSize, sequenceLength]
            `layerIdx`: Integer represention the `id` of the current layer
            `incrementalState`: Dictionary containing the `buffers` or `savedState` for all layers
            `savedState`: Dcitionary containing accumulated `keyMultiHeadProjection`, 
                `valueMultiHeadProjection` and `keyPaddingMask` from previous input batches

        Returns:
            If the `savedState` exists:
                - Key multi-head projection with `savedState`:
                    [batchSize, numHeads, sequenceLength + keyMultiHeadProjectionSequenceLength, headDim]
                - Value multi-head projection with `savedState` buffer:
                    [batchSize, numHeads, sequenceLength + keyMultiHeadProjectionSequenceLength, headDim]
                - Key padding mask:
                    [batchSize, sequenceLength + keyMultiHeadProjectionSequenceLength]
            else the inputs are returned
        """
        if savedState is not None:
            assert (
                keyMultiHeadProjection is not None
                and valueMultiHeadProjection is not None
            ), (
                "Ensure `keyMultiHeadProjection` or `valueMultiHeadProjection` are not `None`"
            )

            if "previousKeyMultiHeadProjection" in savedState:
                keyMultiHeadProjection = self._retrieveProjectionFromSavedSate(
                    keyMultiHeadProjection, "previousKeyMultiHeadProjection", savedState
                )
            if "previousValueMultiHeadProjection" in savedState:
                valueMultiHeadProjection = self._retrieveProjectionFromSavedSate(
                    valueMultiHeadProjection,
                    "previousValueMultiHeadProjection",
                    savedState,
                )

            previousKeyPaddingMask: Optional[Tensor] = None
            if "previousKeyPaddingMask" in savedState:
                previousKeyPaddingMask = savedState["previousKeyPaddingMask"]

            keySequenceLength = keyMultiHeadProjection.size(2)
            keyPaddingMask = self._appendPreviousKeyPaddingMask(
                keyPaddingMask=keyPaddingMask,
                previousKeyPaddingMask=previousKeyPaddingMask,
                batchSize=self.cfg.batchSize,
                sourceLength=keySequenceLength,
                staticKeyValueFlag=self.staticKeyValueFlag,
            )

            incrementalState = self._updateIncrementalState(
                keyMultiHeadProjection,
                valueMultiHeadProjection,
                keyPaddingMask,
                incrementalState,
                savedState,
                layerIdx,
            )

        return (
            keyMultiHeadProjection,
            valueMultiHeadProjection,
            keyPaddingMask,
            incrementalState,
        )

    def _retrieveProjectionFromSavedSate(
        self,
        projection: Optional[Tensor],
        projectionKey: str,
        savedState: Dict[str, Optional[Tensor]],
    ):
        _, batchSize, _ = self.inputShape

        _previousProjection = savedState[projectionKey]
        assert _previousProjection is not None, (
            f"Ensure that `_previousProjection` is not `None` in `_retrieveProjectionFromSavedSate`"
        )
        previousProjection = _previousProjection.view(
            batchSize, self.numHeads, -1, self.headDim
        )
        if self.staticKeyValueFlag:
            projection = previousProjection
        else:
            assert projection is not None, (
                f"Ensure that `projection` is not `None` in `_retrieveProjectionFromSavedSate`"
            )
            projection = L.cat([previousProjection, projection], dim=2)

        return projection

    def _updateIncrementalState(
        self,
        keyMultHeadProjection: Tensor,
        valueMultiHeadProjection: Tensor,
        keyPaddingMask,
        incrementalState: Dict[str, Dict[str, Optional[Tensor]]],
        savedState: Dict[str, Optional[Tensor]],
        layerIdx: Union[str, int],
    ):
        batchSize, _, _, _ = keyMultHeadProjection.size()
        kvProjectionShape = [batchSize, self.numHeads, -1, self.headDim]
        savedState["previousKeyMultiHeadProjection"] = keyMultHeadProjection.view(
            kvProjectionShape
        )
        savedState["previousValueMultiHeadProjection"] = valueMultiHeadProjection.view(
            kvProjectionShape
        )
        savedState["previousKeyPaddingMask"] = keyPaddingMask

        assert incrementalState is not None

        layerId = "attn_state_%d" % layerIdx
        return self.incrementalStateModule.setIncrementalState(
            incrementalState, layerId, savedState
        )

    # @staticmethod
    def _appendPreviousKeyPaddingMask(
        self,
        keyPaddingMask: Optional[Tensor],
        previousKeyPaddingMask: Optional[Tensor],
        batchSize: int,
        sourceLength: int,
        staticKeyValueFlag: bool,
    ):
        if previousKeyPaddingMask is not None and staticKeyValueFlag:
            newKeyPaddingMask = previousKeyPaddingMask
        elif previousKeyPaddingMask is not None and keyPaddingMask is not None:
            newKeyPaddingMask = torch.cat(
                [previousKeyPaddingMask.float(), keyPaddingMask.float()], dim=1
            )
        elif previousKeyPaddingMask is not None:
            if sourceLength > previousKeyPaddingMask.size(1):
                filler = torch.zeros(
                    (batchSize, sourceLength - previousKeyPaddingMask.size(1)),
                    device=previousKeyPaddingMask.device,
                )
                newKeyPaddingMask = torch.cat(
                    [previousKeyPaddingMask.float(), filler.float()], dim=1
                )
            else:
                newKeyPaddingMask = previousKeyPaddingMask.float()

        elif keyPaddingMask is not None:
            if sourceLength > keyPaddingMask.size(1):
                filler = torch.zeros(
                    (batchSize, sourceLength - keyPaddingMask.size(1)),
                    device=keyPaddingMask.device,
                )
                newKeyPaddingMask = torch.cat(
                    [filler.float(), keyPaddingMask.float()], dim=1
                )
            else:
                newKeyPaddingMask = keyPaddingMask.float()
        else:
            newKeyPaddingMask = previousKeyPaddingMask

        return newKeyPaddingMask

    def _addZeroAttention(
        self,
        keyProjection: Tensor,
        valueProjection: Tensor,
        attentionMask: Optional[Tensor],
        keyPaddingMask: Optional[Tensor],
        printStepShapes: bool = False,
    ):
        # TODO: find out exaclty what this zero attention padding things is used for
        """
        From chat GPT:

        Adding a row of zeros to valueProjection can serve different purposes, depending on the context:

        1. Handling Relative Position Bias (Common in Transformer Models)
        Some models use relative positional encodings where an extra token is
        added to align key and value matrices.
        If keys or values are padded but queries are not, an extra zero
        row ensures proper alignment.

        2. Masking a Special Token (e.g., CLS, SEP, or Padding)
        If a transformer processes a sequence with a special token (like [CLS]
        or [SEP]), zero padding ensures that the special token doesn't contribute
        meaningfully to attention.

        3. Preventing Certain Tokens from Affecting Attention
        Zeroing out a row in valueProjection ensures that no
        meaningful values are passed when computing the attention output.
        This can be useful in structured attention models where certain
        tokens (e.g., padding tokens) should contribute nothing to the
        final representation.

        """

        if self.addZeroAttentionFlag:
            assert keyProjection is not None, (
                f"To add `addZeroAttention` the `keyProjection` the input must be a `tensor`, received {type(keyProjection)}"
            )
            assert valueProjection is not None, (
                f"To add `addZeroAttention` the `valueProjection` the input must be a `tensor`, received {type(valueProjection)}"
            )
            sourceLength = keyProjection.size(2)
            sourceLength += 1
            if printStepShapes:
                print(
                    "# `sourceLength` that is the second timension of `keyProjection`",
                    sourceLength,
                )

            keyProjection = F.pad(keyProjection, (0, 0, 0, 1))
            if printStepShapes:
                print("# Added zero padding to the `keyProjection`", sourceLength)
            valueProjection = F.pad(valueProjection, (0, 0, 0, 1))
            if printStepShapes:
                print("# Added zero padding to the `valueProjection`", sourceLength)
            if attentionMask is not None:
                attentionMask = F.pad(attentionMask, (0, 1))
            if keyPaddingMask is not None:
                keyPaddingMask = F.pad(keyPaddingMask, (0, 1))

        return keyProjection, valueProjection, attentionMask, keyPaddingMask

    def _computeAttentionWeights(
        self,
        queryProjection: Tensor,
        keyProjection: Tensor,
        attentionMask: Optional[Tensor],
        incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        totalBatchSize: int,
        targetLength: int,
        sourceLength: int,
    ):
        assert queryProjection is not None, (
            f"When computing `attentionWeights` the `queryProjection` needs to be a `tensor`, received {type(queryProjection)}"
        )
        assert keyProjection is not None, (
            f"When computing `attentionWeights` the `keyProjection` needs to be a `tensor`, received {type(keyProjection)}"
        )

        attentionWeights = L.einsum("bkhie,bhje->bkhij", queryProjection, keyProjection)

        attentionWeights = attentionWeights.reshape(
            totalBatchSize, targetLength, sourceLength
        )

        if self.selfAttentionFlag:
            isIncrementalState = incrementalState is not None
            relativePositionLogits = self._relativePositionLogits(
                queryProjection, sourceLength, last=isIncrementalState
            )

            relativePositionLogits = relativePositionLogits.view(
                totalBatchSize, targetLength, sourceLength
            )

            attentionWeights = attentionWeights + relativePositionLogits

        if attentionMask is not None:
            # TODO: temporary change to adapt the attention mask to be the same shape as
            # `attentionWeights` this is done because you need the shape of the `attentionMask`
            # and `attentionWeights` need to be the same shape. Another thing is that the
            # padding is a tensor of ones, this is because i wanted to take the "stored key"
            # from the previous steps to be taken into account to make the next prediction
            #
            # - It sems that `relativePositionLogits` has is made primarly makde to work for
            # `self attention` at the and without using tensors stored in `savedState`
            # some changes need to be made to the to make sure that relativePositionLogits
            # can be applised to tensors that are not square
            maskPaddingDim0 = attentionWeights.size(1)
            maskPaddingDim1 = attentionWeights.size(2) - attentionMask.size(1)
            maskPadding = torch.zeros(maskPaddingDim0, maskPaddingDim1)

            attentionMask = torch.cat([maskPadding, attentionMask], dim=1)

            attentionMask = attentionMask.unsqueeze(0)
            # TODO: find out if you neeed to use `onnx_trace`
            # if self.onnx_trace:
            #     attentionMask = attentionMask.repeat(attentionWeights.size(0), 1, 1)

            attentionWeights += attentionMask

        return attentionWeights, attentionMask

    def _relativePositionLogits(self, query, length: int, last: bool = False):
        device = query.device
        idx = torch.arange(length, dtype=torch.long, device=device)
        if not last:
            indexGrid = idx[None, :] - idx[:, None]
        else:
            indexGrid = idx[None, :] - (length - 1)
        indexGridMin = 1 - self.maxPositions
        indexGridMax = self.maxPositions - 1
        indexGrid = torch.clamp(indexGrid, min=indexGridMin, max=indexGridMax)
        indexGrid += self.maxPositions
        # TODO: the problem here is is that this seems to only work when
        # the `attentionWeights` is a square tensor them problem is that
        # relative positonal encoding does not work when you add past keyProjection
        # to the input a better understanding of `relativePositionEmbedding` is needed
        # to see what exaclty is it it does
        #

        print("query shape", query.shape)
        print("idx", idx.shape)
        print("difference", (idx[None, :] - idx[:, None]).shape)
        print("indexGrid", indexGrid)
        print("indexGrid shape", indexGrid.shape)
        print("relative postional embedding", self.relativePositionEmbedding.shape)
        print("length", length)

        logits = L.einsum("bkhid,hdj->bkhij", [query, self.relativePositionEmbedding])
        print("length", logits.shape)
        print("-" * 10)

        batchSize, topk, headDim, _, _ = logits.size()
        indexGridReshaped = indexGrid[None, None, None, :, :]
        print("indexGridReshaped ", indexGridReshaped.shape)
        indexGridReshaped = indexGridReshaped.expand(batchSize, topk, headDim, -1, -1)
        print("indexGridReshaped", indexGridReshaped.shape)
        output = logits.gather(-1, indexGridReshaped)
        print("output", output.shape)

        return output

    def _maskAttentionWeightsUsingKeyPaddingMask(
        self,
        attentionWeights: Tensor,
        keyPaddingMask: Optional[Tensor],
        totalBatchSize: int,
        targetLength: int,
        sourceLength: int,
        printStepShapes: bool = False,
    ):
        _, batchSize, _ = self.inputShape
        if keyPaddingMask is not None:
            attentionWeights = attentionWeights.view(
                batchSize, self.topK, self.numHeads, targetLength, sourceLength
            )

            if printStepShapes:
                print(
                    "# `attentionWeights` reshaped to be masked by `keyPaddingMask`: ",
                    attentionWeights.shape,
                )

            keyPaddingMaskReshape = keyPaddingMask[:, None, None, None, :].to(
                torch.bool
            )

            if printStepShapes:
                print(
                    "# `keyPaddingMask` reshaped to mask `attentionWeights`: ",
                    keyPaddingMaskReshape.shape,
                )

            attentionWeights = attentionWeights.masked_fill(
                keyPaddingMaskReshape,
                float("-inf"),
            )

            if printStepShapes:
                print(
                    "# `attentionWeights` masked by `keyPaddingMask`: ",
                    attentionWeights.shape,
                )

            attentionWeights = attentionWeights.view(
                totalBatchSize, targetLength, sourceLength
            )

            if printStepShapes:
                print(
                    "# `attentionWeights` reshaped back into : ",
                    attentionWeights.shape,
                )

        return attentionWeights

    def _computeSoftmaxAttentionWeights(
        self,
        attentionWeights: Tensor,
    ):
        assert attentionWeights is not None, (
            f"`attentionWeights` need to be a `tensor` when computing softmax probabilities attention weights, received {type(attentionWeights)}"
        )
        attentionWeightsFloat = L.softmax(attentionWeights)

        attentionWeights = attentionWeightsFloat.type_as(attentionWeights)
        attentionProbabilities = self.dropoutModule(attentionWeights)
        return attentionProbabilities

    def _computeWeightedValueProjections(
        self,
        attentionProbabilities: Tensor,
        valueProjection: Tensor,
        printStepShapes: bool = False,
    ):
        assert attentionProbabilities is not None, (
            f"To compute weighted ValueProjections `attentionProbabilities` need to be a `tensor`, received {type(attentionProbabilities)}"
        )
        assert valueProjection is not None, (
            f"To compute weighted ValueProjections `valueProjection` need to be a `tensor`, received {type(valueProjection)}"
        )

        _, batchSize, _ = self.inputShape
        targetLength = attentionProbabilities.size(1)
        sourceLength = attentionProbabilities.size(2)
        attentionProbabilitiesReshaped = attentionProbabilities.view(
            batchSize, self.topK, self.numHeads, targetLength, sourceLength
        )
        if printStepShapes:
            print(
                "# `attentionWeights` reshaped to weight `valueProjection` vectors: ",
                attentionProbabilitiesReshaped.shape,
            )

        weightedValueProjections = L.einsum(
            "bkhij,bhje->bkhie", [attentionProbabilitiesReshaped, valueProjection]
        )
        if printStepShapes:
            print("# `weightedValueProjections`: ", weightedValueProjections.shape)

        return weightedValueProjections, targetLength

    def _computeAttentionOutputProjection(
        self,
        weightedValueProjections: Tensor,
        targetLength: int,
        printStepShapes: bool = False,
    ):
        _, batchSize, _ = self.inputShape

        expectedWeightedValueProjectionsShape = [
            batchSize,
            self.topK,
            self.numHeads,
            targetLength,
            self.headDim,
        ]
        assert (
            list(weightedValueProjections.size())
            == expectedWeightedValueProjectionsShape
        ), (
            f"The `weightedValueProjections` needs to have a shape of {expectedWeightedValueProjectionsShape}"
        )

        attentionReshaped = weightedValueProjections.permute(3, 0, 1, 2, 4)
        attentionReshaped = attentionReshaped.reshape(
            targetLength, batchSize, self.topK, self.qkvHiddenDim
        )

        if printStepShapes:
            print(
                "# `weightedValueProjections` reshaped to prepare it for attention `output projection`: ",
                attentionReshaped.shape,
            )

        attentionOutputProjection = self.queryProjectionModel.conputeOutputProjection(
            attentionReshaped
        )

        if printStepShapes:
            print(
                "# `weightedValueProjections` reshaped to prepare it for attention `output projection`: ",
                attentionOutputProjection.shape,
            )

        return attentionOutputProjection


class IncrementalState(object):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.initIncrementaiState()

    def initIncrementaiState(self):
        self.incrementalStateId = str(uuid.uuid4())

    def _getFullIncrementalStateKey(self, key: str) -> str:
        return "{}.{}".format(self.incrementalStateId, key)

    def getIncrementalState(
        self,
        incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
    ) -> Optional[Dict[str, Optional[Tensor]]]:
        """Helper for getting incremental state for an nn.Module"""
        fullKey = self._getFullIncrementalStateKey(key)
        if incrementalState is None or fullKey not in incrementalState:
            return None
        return incrementalState[fullKey]

    def setIncrementalState(
        self,
        incrementalState: Dict[str, Dict[str, Optional[Tensor]]],
        key: str,
        value: Dict[str, Optional[Tensor]],
    ) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
        """Helper for setting incremental state for an nn.Module."""
        if incrementalState is not None:
            fullKey = self._getFullIncrementalStateKey(key)
            incrementalState[fullKey] = value
        return incrementalState
