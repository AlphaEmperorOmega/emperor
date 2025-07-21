import unittest
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from Emperor.components.attention import Attention
from Emperor.components.parameter_generators.utils.losses import AuxiliaryLosses
from Emperor.config import ModelConfig


class GeneralAttentionTestMethods(unittest.TestCase):
    def _setUp(self):
        cfg = ModelConfig()
        self.cfg = cfg
        self.cfg.qkvHiddenDim = 4
        self.cfg.headDim = 2
        self.cfg.auxiliaryLosses = AuxiliaryLosses(self.cfg)
        self.cfg.moeAuxiliaryLosses = AuxiliaryLosses(self.cfg)

        self.model = Attention(
            cfg=self.cfg,
            embeddingDim=self.cfg.embeddingDim,
            qkvHiddenDim=self.cfg.qkvHiddenDim,
            attentionOutputDim=self.cfg.attentionOutputDim,
            numExperts=self.cfg.numExperts,
            topK=self.cfg.topK,
            headDim=self.cfg.headDim,
        )

    def _queryInput(self) -> Tensor:
        return torch.randn(
            self.cfg.sequenceLength, self.cfg.batchSize, self.model.queryInputDim
        )

    def _valueInput(self) -> Tensor:
        return torch.randn(
            self.cfg.sequenceLength, self.cfg.batchSize, self.model.valueInputDim
        )

    def _keyInput(self) -> Tensor:
        return torch.randn(
            self.cfg.sequenceLength, self.cfg.batchSize, self.model.valueInputDim
        )

    def _keyPaddingMask(self) -> Tensor:
        """
        Example of key padding maks
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
        """
        mask = torch.zeros(
            (self.cfg.batchSize, self.cfg.sequenceLength), dtype=torch.bool
        )

        # Generate random sequence lengths for each batch
        paddingStart = self.cfg.sequenceLength // 2
        paddingEnd = self.cfg.sequenceLength
        randomSeqenceLengths = torch.randint(
            paddingStart, paddingEnd, (self.cfg.batchSize,)
        )

        for batchIdx, sequenceLength in enumerate(randomSeqenceLengths):
            # Set valid tokens to True
            mask[batchIdx, :sequenceLength] = 1

        return mask

    def _layerIdx(self) -> int:
        return 1

    def _skipMask(self) -> Tensor:
        return torch.randint(0, 2, (self.cfg.batchSize, self.cfg.sequenceLength))

    def _incrementalState(self) -> Dict[str, Dict[str, Optional[Tensor]]]:
        bs = self.cfg.batchSize
        nh = self.model.numHeads
        sl = self.cfg.sequenceLength
        hd = self.cfg.headDim
        layerIdx1 = 1
        layerIdx2 = 2

        kvProjection = torch.randn(bs, nh, sl, hd)
        savedStateLayer1 = {
            "previousKeyMultiHeadProjection": kvProjection,
            "previousValueMultiHeadProjection": kvProjection,
            "previousKeyPaddingMask": self._keyPaddingMask(),
        }
        savedStateLayer2 = {
            "previousKeyMultiHeadProjection": kvProjection,
            "previousValueMultiHeadProjection": kvProjection,
            "previousKeyPaddingMask": self._keyPaddingMask(),
        }

        incrementalState = {}
        layerId = "attn_state_%d" % layerIdx1
        self.model.incrementalStateModule.setIncrementalState(
            incrementalState,
            layerId,
            savedStateLayer1,
        )

        layerId = "attn_state_%d" % layerIdx2
        self.model.incrementalStateModule.setIncrementalState(
            incrementalState,
            layerId,
            savedStateLayer2,
        )
        return incrementalState


class TestGetLayerSavedStateFromIncrementalState(GeneralAttentionTestMethods):
    def setUp(self) -> None:
        self._setUp()

    def testInputs_none(self):
        key = None
        value = None
        layerIdx = None
        incrementalState = None

        keyInput, valueInput, savedState = (
            self.model._getLayerSavedStateFromIncrementalState(
                key, value, layerIdx, incrementalState
            )
        )

        self.assertIsNone(keyInput)
        self.assertIsNone(valueInput)
        self.assertIsNone(savedState)

    def testInputs_key_value(self):
        key = self._keyInput()
        value = self._valueInput()
        layerIdx = None
        incrementalState = None

        keyInput, valueInput, savedState = (
            self.model._getLayerSavedStateFromIncrementalState(
                key, value, layerIdx, incrementalState
            )
        )

        self.assertIs(keyInput, key)
        self.assertIs(valueInput, value)
        self.assertIsNone(savedState)

    def testInputs_key_value_incrementalState(
        self,
    ):
        key = self._keyInput()
        value = self._valueInput()
        layerIdx = None
        incrementalState = self._incrementalState()

        keyInput, valueInput, savedState = (
            self.model._getLayerSavedStateFromIncrementalState(
                key, value, layerIdx, incrementalState
            )
        )

        self.assertIs(keyInput, key)
        self.assertIs(valueInput, value)
        self.assertIsNone(savedState)

    def testInputs_key_value_layerIdx_incrementalState(
        self,
    ):
        key = self._keyInput()
        value = self._valueInput()
        layerIdx = self._layerIdx()
        incrementalState = self._incrementalState()

        keyInput, valueInput, savedState = (
            self.model._getLayerSavedStateFromIncrementalState(
                key, value, layerIdx, incrementalState
            )
        )

        self.assertIs(keyInput, key)
        self.assertIs(valueInput, value)
        self.assertIsInstance(savedState, Dict)
        self.assertIsInstance(
            savedState["previousKeyMultiHeadProjection"], torch.Tensor
        )

    def testInputs_key_value_layerIdx_incrementalState_statickv(
        self,
    ):
        key = self._keyInput()
        value = self._valueInput()
        layerIdx = self._layerIdx()
        incrementalState = self._incrementalState()
        self.model.staticKeyValueFlag = True

        keyInput, valueInput, savedState = (
            self.model._getLayerSavedStateFromIncrementalState(
                key, value, layerIdx, incrementalState
            )
        )
        self.model.staticKeyValueFlag = False

        self.assertIsNone(keyInput)
        self.assertIsNone(valueInput)
        self.assertIsInstance(savedState, Dict)
        self.assertIsInstance(
            savedState["previousKeyMultiHeadProjection"], torch.Tensor
        )


class TestGetSavedState(GeneralAttentionTestMethods):
    def setUp(self) -> None:
        self._setUp()

    def _layerIdx1(self) -> int:
        return 1

    def _layerIdx2(self) -> int:
        return 2

    def testInputs_none(self):
        layerIdx = None
        incrementalState = None
        savedState = self.model._getSavedState(incrementalState, layerIdx)
        self.assertIsNone(savedState)

    def testInputs_layerIdx(self):
        layerIdx = self._layerIdx1()
        incrementalState = None
        savedState = self.model._getSavedState(incrementalState, layerIdx)
        self.assertIsNone(savedState)

        layerIdx = self._layerIdx2()
        incrementalState = None
        savedState = self.model._getSavedState(incrementalState, layerIdx)
        self.assertIsNone(savedState)

    def testInputs_incrementalState(self):
        layerIdx = self._layerIdx1()
        incrementalState = None
        savedState = self.model._getSavedState(incrementalState, layerIdx)
        self.assertIsNone(savedState)

    def testInputs_incrementalState_layerIdx(self):
        layerIdx = self._layerIdx1()
        incrementalState = self._incrementalState()
        savedState = self.model._getSavedState(incrementalState, layerIdx)
        self.assertIsInstance(savedState, Dict)
        self.assertTrue("previousKeyMultiHeadProjection" in savedState.keys())
        self.assertTrue("previousValueMultiHeadProjection" in savedState.keys())
        self.assertTrue("previousKeyPaddingMask" in savedState.keys())

    def testInputs_createEmptyFlag(self):
        layerIdx = None
        incrementalState = None
        self.model.createEmptyFlag = True
        savedState = self.model._getSavedState(incrementalState, layerIdx)
        self.model.createEmptyFlag = False
        self.assertIsInstance(savedState, Dict)
        self.assertTrue(len(savedState) == 0, Dict)

    def testInputs_layerIdx_createEmptyFlag(self):
        layerIdx = self._layerIdx1()
        incrementalState = None
        self.model.createEmptyFlag = True
        savedState = self.model._getSavedState(incrementalState, layerIdx)
        self.model.createEmptyFlag = False
        self.assertIsInstance(savedState, Dict)
        self.assertTrue(len(savedState) == 0, Dict)

    def testInputs_incrementalState_createEmptyFlag(self):
        layerIdx = None
        incrementalState = self._incrementalState()
        self.model.createEmptyFlag = True
        savedState = self.model._getSavedState(incrementalState, layerIdx)
        self.model.createEmptyFlag = False
        self.assertIsInstance(savedState, Dict)
        self.assertTrue(len(savedState) == 0, Dict)


class TestComputeQueryKeyValueProjections(GeneralAttentionTestMethods):
    def setUp(self) -> None:
        self._setUp()

    def testInputs_query_key_value(self):
        query = self._queryInput()
        key = self._keyInput()
        value = self._valueInput()
        skipMask = None

        queryProjection, keyProjection, valueProjection = (
            self.model._computeQueryKeyValueProjections(
                query,
                key,
                value,
                skipMask,
            )
        )

        expectedQueryShape = [
            self.cfg.sequenceLength,
            self.cfg.batchSize,
            self.cfg.topK,
            self.cfg.qkvHiddenDim,
        ]
        expectedKeyShape = [
            self.cfg.sequenceLength,
            self.cfg.batchSize,
            self.cfg.qkvHiddenDim,
        ]
        expectedValueShape = [
            self.cfg.sequenceLength,
            self.cfg.batchSize,
            self.cfg.qkvHiddenDim,
        ]

        queryProjectionShape = list(queryProjection.size())
        keyProjectionShape = list(keyProjection.size())
        valueProjectionShape = list(valueProjection.size())

        self.assertIsInstance(queryProjection, Tensor)
        self.assertEqual(queryProjectionShape, expectedQueryShape)
        self.assertIsInstance(keyProjection, Tensor)
        self.assertEqual(keyProjectionShape, expectedKeyShape)
        self.assertIsInstance(valueProjection, Tensor)
        self.assertEqual(valueProjectionShape, expectedValueShape)

    def testInputs_query_key_value_skipMask(self):
        query = self._queryInput()
        key = self._keyInput()
        value = self._valueInput()
        skipMask = self._skipMask()

        queryProjection, keyProjection, valueProjection = (
            self.model._computeQueryKeyValueProjections(
                query,
                key,
                value,
                skipMask,
            )
        )

        expectedQueryShape = [
            self.cfg.sequenceLength,
            self.cfg.batchSize,
            self.cfg.topK,
            self.cfg.qkvHiddenDim,
        ]
        expectedKeyShape = [
            self.cfg.sequenceLength,
            self.cfg.batchSize,
            self.cfg.qkvHiddenDim,
        ]
        expectedValueShape = [
            self.cfg.sequenceLength,
            self.cfg.batchSize,
            self.cfg.qkvHiddenDim,
        ]

        queryProjectionShape = list(queryProjection.size())
        keyProjectionShape = list(keyProjection.size())
        valueProjectionShape = list(valueProjection.size())

        self.assertIsInstance(queryProjection, Tensor)
        self.assertEqual(queryProjectionShape, expectedQueryShape)
        self.assertIsInstance(keyProjection, Tensor)
        self.assertEqual(keyProjectionShape, expectedKeyShape)
        self.assertIsInstance(valueProjection, Tensor)
        self.assertEqual(valueProjectionShape, expectedValueShape)

    def testInputs_query_encoderDecorderAttentionFlag(self):
        query = self._queryInput()
        key = None
        value = None
        skipMask = None

        self.model.encoderDecorderAttentionFlag = True
        queryProjection, keyProjection, valueProjection = (
            self.model._computeQueryKeyValueProjections(
                query,
                key,
                value,
                skipMask,
            )
        )
        self.model.encoderDecorderAttentionFlag = False

        expectedQueryShape = [
            self.cfg.sequenceLength,
            self.cfg.batchSize,
            self.cfg.topK,
            self.cfg.qkvHiddenDim,
        ]

        queryProjectionShape = list(queryProjection.size())

        self.assertIsInstance(queryProjection, Tensor)
        self.assertEqual(queryProjectionShape, expectedQueryShape)
        self.assertIsNone(keyProjection)
        self.assertIsNone(valueProjection)

    def testInputs_query_key_value_encoderDecorderAttentionFlag(self):
        query = self._queryInput()
        key = self._valueInput()
        value = self._valueInput()
        skipMask = None

        self.model.encoderDecorderAttentionFlag = True
        queryProjection, keyProjection, valueProjection = (
            self.model._computeQueryKeyValueProjections(
                query,
                key,
                value,
                skipMask,
            )
        )
        self.model.encoderDecorderAttentionFlag = False

        expectedQueryShape = [
            self.cfg.sequenceLength,
            self.cfg.batchSize,
            self.cfg.topK,
            self.cfg.qkvHiddenDim,
        ]

        expectedKeyShape = [
            self.cfg.sequenceLength,
            self.cfg.batchSize,
            self.cfg.qkvHiddenDim,
        ]
        expectedValueShape = [
            self.cfg.sequenceLength,
            self.cfg.batchSize,
            self.cfg.qkvHiddenDim,
        ]

        queryProjectionShape = list(queryProjection.size())
        keyProjectionShape = list(keyProjection.size())
        valueProjectionShape = list(valueProjection.size())

        self.assertIsInstance(queryProjection, Tensor)
        self.assertEqual(queryProjectionShape, expectedQueryShape)
        self.assertIsInstance(keyProjection, Tensor)
        self.assertEqual(keyProjectionShape, expectedKeyShape)
        self.assertIsInstance(valueProjection, Tensor)
        self.assertEqual(valueProjectionShape, expectedValueShape)

    def testInputs_query_key_value_skipMask_encoderDecorderAttentionFlag(self):
        query = self._queryInput()
        key = self._valueInput()
        value = self._valueInput()
        skipMask = self._skipMask()

        self.model.encoderDecorderAttentionFlag = True
        queryProjection, keyProjection, valueProjection = (
            self.model._computeQueryKeyValueProjections(
                query,
                key,
                value,
                skipMask,
            )
        )
        self.model.encoderDecorderAttentionFlag = False

        expectedQueryShape = [
            self.cfg.sequenceLength,
            self.cfg.batchSize,
            self.cfg.topK,
            self.cfg.qkvHiddenDim,
        ]

        expectedKeyShape = [
            self.cfg.sequenceLength,
            self.cfg.batchSize,
            self.cfg.qkvHiddenDim,
        ]
        expectedValueShape = [
            self.cfg.sequenceLength,
            self.cfg.batchSize,
            self.cfg.qkvHiddenDim,
        ]

        queryProjectionShape = list(queryProjection.size())
        keyProjectionShape = list(keyProjection.size())
        valueProjectionShape = list(valueProjection.size())

        self.assertIsInstance(queryProjection, Tensor)
        self.assertEqual(queryProjectionShape, expectedQueryShape)
        self.assertIsInstance(keyProjection, Tensor)
        self.assertEqual(keyProjectionShape, expectedKeyShape)
        self.assertIsInstance(valueProjection, Tensor)
        self.assertEqual(valueProjectionShape, expectedValueShape)


class TestAttachMemoryBiasesToKeyValueProjections(GeneralAttentionTestMethods):
    def setUp(self) -> None:
        self._setUp()

        self.model = Attention(
            cfg=self.cfg,
            embeddingDim=self.cfg.embeddingDim,
            qkvHiddenDim=self.cfg.qkvHiddenDim,
            attentionOutputDim=self.cfg.attentionOutputDim,
            numExperts=self.cfg.numExperts,
            topK=self.cfg.topK,
            headDim=self.cfg.headDim,
            addMemoryBiasKeyValuesFlag=True,
        )

    def _keyProjectionInput(self):
        return torch.randn(
            self.cfg.sequenceLength,
            self.cfg.batchSize,
            self.cfg.qkvHiddenDim,
        )

    def _valueProjectionInput(self):
        return torch.randn(
            self.cfg.sequenceLength,
            self.cfg.batchSize,
            self.cfg.qkvHiddenDim,
        )

    def _attentionMaks(self):
        attentionMaskShape = (self.cfg.sequenceLength, self.cfg.sequenceLength)
        attentionMaskPlaceholder = torch.full(attentionMaskShape, float("-inf"))
        return torch.triu(attentionMaskPlaceholder, 1)

    def testInputs_keyProjection_valueProjection(self):
        """The followin flag must be `True`: `model.addMemoryBiasKeyValuesFlag`"""
        keyProjection = self._keyProjectionInput()
        valueProjection = self._valueProjectionInput()
        attentionMask = None
        keyPaddingMask = None

        keyProjection, valueProjection, attentionMask, keyPaddingMask = (
            self.model._attachMemoryBiasesToKeyValueProjections(
                keyProjection,
                valueProjection,
                attentionMask,
                keyPaddingMask,
            )
        )

        keySequenceWithMemoryToken = self.cfg.sequenceLength + 1
        expectedKeyShape = [
            keySequenceWithMemoryToken,
            self.cfg.batchSize,
            self.cfg.qkvHiddenDim,
        ]
        valueSequenceWithMemoryToken = self.cfg.sequenceLength + 1
        expectedValueShape = [
            valueSequenceWithMemoryToken,
            self.cfg.batchSize,
            self.cfg.qkvHiddenDim,
        ]

        keyOutputShape = list(keyProjection.size())
        valueOutputShape = list(valueProjection.size())

        self.assertIsInstance(keyProjection, Tensor)
        self.assertEqual(keyOutputShape, expectedKeyShape)
        self.assertIsInstance(valueProjection, Tensor)
        self.assertEqual(valueOutputShape, expectedValueShape)
        self.assertIsNone(attentionMask)
        self.assertIsNone(keyPaddingMask)

    def testInputs_keyProjection_valueProjection_attentionMaks(self):
        """The followin flag must be `True`: `model.addMemoryBiasKeyValuesFlag`"""
        keyProjection = self._keyProjectionInput()
        valueProjection = self._valueProjectionInput()
        attentionMask = self._attentionMaks()
        keyPaddingMask = None

        keyProjection, valueProjection, attentionMask, keyPaddingMask = (
            self.model._attachMemoryBiasesToKeyValueProjections(
                keyProjection,
                valueProjection,
                attentionMask,
                keyPaddingMask,
            )
        )

        keySequenceWithMemoryToken = self.cfg.sequenceLength + 1
        expectedKeyShape = [
            keySequenceWithMemoryToken,
            self.cfg.batchSize,
            self.cfg.qkvHiddenDim,
        ]
        valueSequenceWithMemoryToken = self.cfg.sequenceLength + 1
        expectedValueShape = [
            valueSequenceWithMemoryToken,
            self.cfg.batchSize,
            self.cfg.qkvHiddenDim,
        ]

        attentionMaskWithMemoryBiasPadding = self.cfg.sequenceLength + 1
        expectedAttentionMaskShape = [
            self.cfg.sequenceLength,
            attentionMaskWithMemoryBiasPadding,
        ]

        keyOutputShape = list(keyProjection.size())
        valueOutputShape = list(keyProjection.size())
        attentionMaskOutputShape = list(attentionMask.size())

        self.assertIsInstance(keyProjection, Tensor)
        self.assertEqual(keyOutputShape, expectedKeyShape)
        self.assertIsInstance(valueProjection, Tensor)
        self.assertEqual(valueOutputShape, expectedValueShape)
        self.assertIsInstance(attentionMask, Tensor)
        self.assertEqual(attentionMaskOutputShape, expectedAttentionMaskShape)
        self.assertIsNone(keyPaddingMask)

    def testInputs_keyProjection_valueProjection_attentionMaks_keyPaddingMask(self):
        """The followin flag must be `True`: `model.addMemoryBiasKeyValuesFlag`"""
        keyProjection = self._keyProjectionInput()
        valueProjection = self._valueProjectionInput()
        attentionMask = self._attentionMaks()
        keyPaddingMask = self._keyPaddingMask()

        keyProjection, valueProjection, attentionMask, keyPaddingMask = (
            self.model._attachMemoryBiasesToKeyValueProjections(
                keyProjection,
                valueProjection,
                attentionMask,
                keyPaddingMask,
            )
        )

        keySequenceWithMemoryToken = self.cfg.sequenceLength + 1
        expectedKeyShape = [
            keySequenceWithMemoryToken,
            self.cfg.batchSize,
            self.cfg.qkvHiddenDim,
        ]
        valueSequenceWithMemoryToken = self.cfg.sequenceLength + 1
        expectedValueShape = [
            valueSequenceWithMemoryToken,
            self.cfg.batchSize,
            self.cfg.qkvHiddenDim,
        ]

        attentionMaskWithMemoryBiasPadding = self.cfg.sequenceLength + 1
        expectedAttentionMaskShape = [
            self.cfg.sequenceLength,
            attentionMaskWithMemoryBiasPadding,
        ]

        keyPaddingMaskBiasPadding = self.cfg.sequenceLength + 1
        expectedKeyPaddingMaskShape = [
            self.cfg.batchSize,
            keyPaddingMaskBiasPadding,
        ]

        keyOutputShape = list(keyProjection.size())
        valueOutputShape = list(valueProjection.size())
        attentionMaskOutputShape = list(attentionMask.size())
        keyPaddingMaskOutputShape = list(keyPaddingMask.size())

        self.assertIsInstance(keyProjection, Tensor)
        self.assertEqual(keyOutputShape, expectedKeyShape)
        self.assertIsInstance(valueProjection, Tensor)
        self.assertEqual(valueOutputShape, expectedValueShape)
        self.assertIsInstance(attentionMask, Tensor)
        self.assertEqual(attentionMaskOutputShape, expectedAttentionMaskShape)
        self.assertIsInstance(keyPaddingMask, Tensor)
        self.assertEqual(expectedKeyPaddingMaskShape, keyPaddingMaskOutputShape)


class TestSplitQueryKeyValueProjectionsIntoMultipleHeads(GeneralAttentionTestMethods):
    def setUp(self) -> None:
        self._setUp()

    def _queryProjectionInput(self):
        return torch.randn(
            self.cfg.sequenceLength,
            self.cfg.batchSize,
            self.cfg.topK,
            self.cfg.qkvHiddenDim,
        )

    def _keyProjectionInput(self):
        return torch.randn(
            self.cfg.sequenceLength,
            self.cfg.batchSize,
            self.cfg.qkvHiddenDim,
        )

    def _valueProjectionInput(self):
        return torch.randn(
            self.cfg.sequenceLength,
            self.cfg.batchSize,
            self.cfg.qkvHiddenDim,
        )

    def testInputs_queryProjection(self):
        queryProjection = self._queryProjectionInput()
        keyProjection = None
        valueProjection = None

        queryProjection, keyProjection, valueProjection = (
            self.model._splitQueryKeyValueProjectionsIntoMultipleHeads(
                queryProjection, keyProjection, valueProjection
            )
        )

        expectedQueryShape = [
            self.cfg.batchSize,
            self.cfg.topK,
            self.model.numHeads,
            self.cfg.sequenceLength,
            self.cfg.headDim,
        ]

        queryProjectionOutputShape = list(queryProjection.size())

        self.assertIsInstance(queryProjection, Tensor)
        self.assertEqual(queryProjectionOutputShape, expectedQueryShape)
        self.assertIsNone(keyProjection)
        self.assertIsNone(valueProjection)

    def testInputs_queryProjection_keyProjection(self):
        queryProjection = self._queryProjectionInput()
        keyProjection = self._keyProjectionInput()
        valueProjection = None

        queryProjection, keyProjection, valueProjection = (
            self.model._splitQueryKeyValueProjectionsIntoMultipleHeads(
                queryProjection, keyProjection, valueProjection
            )
        )

        expectedQueryShape = [
            self.cfg.batchSize,
            self.cfg.topK,
            self.model.numHeads,
            self.cfg.sequenceLength,
            self.cfg.headDim,
        ]

        expectedValueKeyShape = [
            self.cfg.batchSize,
            self.model.numHeads,
            self.cfg.sequenceLength,
            self.cfg.headDim,
        ]

        queryProjectionOutputShape = list(queryProjection.size())
        keyProjectionOutputShape = list(keyProjection.size())

        self.assertIsInstance(queryProjection, Tensor)
        self.assertEqual(queryProjectionOutputShape, expectedQueryShape)
        self.assertIsInstance(keyProjection, Tensor)
        self.assertEqual(keyProjectionOutputShape, expectedValueKeyShape)
        self.assertIsNone(valueProjection)

    def testInputs_queryProjection_keyProjection_valueProjections(self):
        queryProjection = self._queryProjectionInput()
        keyProjection = self._keyProjectionInput()
        valueProjection = self._valueProjectionInput()

        queryProjection, keyProjection, valueProjection = (
            self.model._splitQueryKeyValueProjectionsIntoMultipleHeads(
                queryProjection, keyProjection, valueProjection
            )
        )

        expectedQueryShape = [
            self.cfg.batchSize,
            self.cfg.topK,
            self.model.numHeads,
            self.cfg.sequenceLength,
            self.cfg.headDim,
        ]

        expectedValueKeyShape = [
            self.cfg.batchSize,
            self.model.numHeads,
            self.cfg.sequenceLength,
            self.cfg.headDim,
        ]

        queryProjectionOutputShape = list(queryProjection.size())
        keyProjectionOutputShape = list(keyProjection.size())
        valueProjectionOutputShape = list(valueProjection.size())

        self.assertIsInstance(queryProjection, Tensor)
        self.assertEqual(queryProjectionOutputShape, expectedQueryShape)
        self.assertIsInstance(keyProjection, Tensor)
        self.assertEqual(keyProjectionOutputShape, expectedValueKeyShape)
        self.assertIsInstance(valueProjection, Tensor)
        self.assertEqual(valueProjectionOutputShape, expectedValueKeyShape)


class TestUpdateKeyValueProjectionsUsingLayerSavedState(GeneralAttentionTestMethods):
    def setUp(self) -> None:
        self._setUp()

    def _keyMultiHeadProjectionInput(self):
        return torch.randn(
            self.cfg.batchSize,
            self.model.numHeads,
            self.cfg.sequenceLength,
            self.cfg.headDim,
        )

    def _valueMultiHeadProjectionInput(self):
        return torch.randn(
            self.cfg.batchSize,
            self.model.numHeads,
            self.cfg.sequenceLength,
            self.cfg.headDim,
        )

    def testInputs_none(self):
        keyMultiHeadProjection = None
        valueMultiHeadProjection = None
        keyPaddingMask = None
        layerIdx = None
        incrementalState = None
        savedState = None

        (
            keyMultiHeadProjection,
            valueMultiHeadProjection,
            keyPaddingMask,
            incrementalState,
        ) = self.model._updateKeyValueProjectionsUsingLayerSavedState(
            keyMultiHeadProjection,
            valueMultiHeadProjection,
            keyPaddingMask,
            layerIdx,
            incrementalState,
            savedState,
        )

        # expectedQueryShape = [
        #     self.cfg.batchSize,
        #     self.cfg.topK,
        #     self.model.numHeads,
        #     self.cfg.sequenceLength,
        #     self.cfg.headDim,
        # ]
        #
        # queryProjectionOutputShape = list(queryProjection.size())
        #
        # self.assertIsInstance(queryProjection, Tensor)
        # self.assertEqual(queryProjectionOutputShape, expectedQueryShape)
        # self.assertIsNone(keyProjection)
        # self.assertIsNone(valueProjection)
