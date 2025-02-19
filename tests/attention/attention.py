from typing import Dict, Optional
import unittest
import torch
import torch.nn as nn
from torch import Tensor
from Emperor.components.attention import Attention
from Emperor.components.parameter_generators.utils.losses import AuxiliaryLosses
from Emperor.config import ModelConfig


class TestGetLayerSavedStateFromIncrementalState(unittest.TestCase):
    def setUp(self) -> None:
        cfg = ModelConfig()
        self.cfg = cfg
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

    def _valueInput(self) -> Tensor:
        bs = self.cfg.batchSize
        sl = self.cfg.sequenceLength
        vd = self.model.valueInputDim
        return torch.arange(sl * bs * vd).reshape(sl, bs, vd).float()

    def _keyInput(self) -> Tensor:
        bs = self.cfg.batchSize
        sl = self.cfg.sequenceLength
        kd = self.model.keyInputDim
        return torch.arange(sl * bs * kd).reshape(sl, bs, kd).float()

    def _keyPaddingMask(self) -> Tensor:
        return torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0],
            ]
        )

    def _layerIdx(self) -> int:
        return 1

    def _incrementalState(self) -> Dict[str, Dict[str, Optional[Tensor]]]:
        bs = self.cfg.batchSize
        sl = self.cfg.sequenceLength
        hd = self.cfg.headDim
        nh = self.model.numHeads
        layerIdx1 = 1
        layerIdx2 = 2

        numElements = bs * nh * hd * sl
        vkShape = (bs, nh, hd, sl)

        kvProjection = torch.arange(numElements).reshape(*vkShape).float()

        savedStateLayer1 = {
            "previousKeyProjection": kvProjection,
            "previousValueProjection": kvProjection,
            "previousKeyPaddingMask": self._keyPaddingMask(),
        }
        savedStateLayer2 = {
            "previousKeyProjection": kvProjection,
            "previousValueProjection": kvProjection,
            "previousKeyPaddingMask": self._keyPaddingMask(),
        }

        incrementalState = {}
        self.model._updateIncrementalState(
            incrementalState,
            savedStateLayer1,
            layerIdx1,
        )
        self.model._updateIncrementalState(
            incrementalState,
            savedStateLayer2,
            layerIdx2,
        )
        return incrementalState

    def test_no_inputs(self):
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

    def test_key_value_inputs(self):
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

    def test_key_value_incrementalState_inputs(
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

    def test_key_value_layerIdx_incrementalState_inputs(
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
        self.assertIsInstance(savedState["previousKeyProjection"], torch.Tensor)

    def test_key_value_layerIdx_incrementalState_statickv_inputs(
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
        self.assertIsInstance(savedState["previousKeyProjection"], torch.Tensor)


class TestComputeQueryKeyValueProjections(unittest.TestCase):
    def setUp(self) -> None:
        cfg = ModelConfig()
        self.cfg = cfg
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

    def _valueInput(self) -> Tensor:
        bs = self.cfg.batchSize
        sl = self.cfg.sequenceLength
        vd = self.model.valueInputDim
        return torch.arange(sl * bs * vd).reshape(sl, bs, vd).float()

    def _keyInput(self) -> Tensor:
        bs = self.cfg.batchSize
        sl = self.cfg.sequenceLength
        kd = self.model.keyInputDim
        return torch.arange(sl * bs * kd).reshape(sl, bs, kd).float()

    def _keyPaddingMask(self) -> Tensor:
        return torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0],
            ]
        )

    def _layerIdx(self) -> int:
        return 1

    def _incrementalState(self) -> Dict[str, Dict[str, Optional[Tensor]]]:
        bs = self.cfg.batchSize
        sl = self.cfg.sequenceLength
        hd = self.cfg.headDim
        nh = self.model.numHeads
        layerIdx1 = 1
        layerIdx2 = 2

        numElements = bs * nh * hd * sl
        vkShape = (bs, nh, hd, sl)

        kvProjection = torch.arange(numElements).reshape(*vkShape).float()

        savedStateLayer1 = {
            "previousKeyProjection": kvProjection,
            "previousValueProjection": kvProjection,
            "previousKeyPaddingMask": self._keyPaddingMask(),
        }
        savedStateLayer2 = {
            "previousKeyProjection": kvProjection,
            "previousValueProjection": kvProjection,
            "previousKeyPaddingMask": self._keyPaddingMask(),
        }

        incrementalState = {}
        self.model._updateIncrementalState(
            incrementalState,
            savedStateLayer1,
            layerIdx1,
        )
        self.model._updateIncrementalState(
            incrementalState,
            savedStateLayer2,
            layerIdx2,
        )
        return incrementalState

    def test_no_inputs(self):
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

    def test_key_value_inputs(self):
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

    def test_key_value_incrementalState_inputs(
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

    def test_key_value_layerIdx_incrementalState_inputs(
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
        self.assertIsInstance(savedState["previousKeyProjection"], torch.Tensor)

    def test_key_value_layerIdx_incrementalState_statickv_inputs(
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
        self.assertIsInstance(savedState["previousKeyProjection"], torch.Tensor)
