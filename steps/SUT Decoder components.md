# Sparse Universal Transformer Decoder Components

-

## 1. Requirements

### 1.1. Import necessary classes

```{python}
import torch
from torch import Tensor
from typing import Optional, List, Dict

from Emperor.library.choice import Library as L
from Emperor.components.sut_layer import TransformerDecorderLayerBase
from Emperor.components.parameter_generators.utils.losses import AuxiliaryLosses
from Emperor.config import ModelConfig

def printData(name, elm):
  if isinstance(elm, torch.Tensor):
    elm = elm.shape
  print(f'# {name}: {elm}')
```

### 1.2. Setup configuration

```{python}
attentionInputOutput = 8
cfg = ModelConfig(
  inputDim=attentionInputOutput,
  outputDim=attentionInputOutput,
  embeddingDim=attentionInputOutput,
  qkvHiddenDim=6,
  sequenceLength=7,
  headDim=2,
  numExperts=7,
  topK=2,
)
cfg.auxiliaryLosses = AuxiliaryLosses(cfg)
cfg.moeAuxiliaryLosses = AuxiliaryLosses(cfg)
print('# Current Parameter Generator: \n', cfg.parameterGeneartorType.name)
```

### 1.3. Initialize model

```{python}
print('# embeddingDim: ', cfg.embeddingDim)
print('# qkvHiddenDim: ', cfg.qkvHiddenDim)
print('# numExperts: ', cfg.numExperts)
print('# topK: ', cfg.topK)
print('# headDim: ', cfg.headDim)
```

```{python}
model = TransformerDecorderLayerBase(
  cfg=cfg,
  crossAttentionFlag=True
)

print('### Transformer Decoder Attention Model: \n', model.selfAttnModel )
print('### Transformer Decoder CorssAttention Model: \n', model.crossAttnModel)
print('### Transformer Decoder Feed Forward Model: \n', model.ffnModel)
```

## 2. Expected Inputs

### 2.1 Input batch

```{python}
inputBatch = torch.randn(cfg.sequenceLength, cfg.batchSize, model.selfAttnModel.queryInputDim)
print(inputBatch.size())
```

### 2.2 Incremental State

```{python}
layerIdx1 = 1
layerIdx2 = 2
keyPaddingMask = torch.tensor(
[[1, 1, 1, 0, 0, 0, 0],
 [1, 1, 1, 1, 1, 0, 0],
 [1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 0, 0, 0],
 [1, 1, 0, 0, 0, 0, 0]])

bs = cfg.batchSize
nh = model.numHeads
hd = model.headDim
sl = cfg.sequenceLength

numElements = bs * nh * hd * sl
savedStateLayer1 = {
  "previousKeyMultiHeadProjection": torch.randn(bs, nh, sl, hd),
  "previousValueMultiHeadProjection": torch.randn(bs, nh, sl, hd),
  "previousKeyPaddingMask": keyPaddingMask
}
savedStateLayer2 = {
  "previousKeyMultiHeadProjection": torch.randn(bs, nh, sl, hd),
  "previousValueMultiHeadProjection": torch.randn(bs, nh, sl, hd),
  "previousKeyPaddingMask": keyPaddingMask
}

incrementalState = {}
layerId = "attn_state_%d" % layerIdx1
model.selfAttnModel.incrementalStateModule.setIncrementalState (
  incrementalState,
  layerId,
  savedStateLayer1,
)

layerId = "attn_state_%d" % layerIdx2
model.selfAttnModel.incrementalStateModule.setIncrementalState (
  incrementalState,
  layerId,
  savedStateLayer2,
)

layerCount = 1
for key, dictElm in incrementalState.items():
  print(f'# Saved state for `layer{layerCount}` with id: ', key)
  for innerKey, dictElm in dictElm.items():
    print('- ',innerKey, ' : ',dictElm.shape)
  layerCount += 1
  print()
```

### 2.2 Incremental State

```{python}
previousSavedState = {
  "previousKeyMultiHeadProjection": torch.arange(bs * nh * sl * hd).reshape(bs, nh, sl, hd).float(),
  "previousValueMultiHeadProjection": torch.arange(bs * nh * sl * hd).reshape(bs, nh, sl, hd).float(),
  "previousKeyPaddingMask": keyPaddingMask
}

previousSavedStateTest = [
  torch.arange(sl * hd).reshape(sl, hd).float(),
  torch.arange(sl * hd).reshape(sl, hd).float(),
  keyPaddingMask
]

print(previousSavedState["previousKeyMultiHeadProjection"].shape)
print(previousSavedState["previousValueMultiHeadProjection"].shape)
print(previousSavedState["previousKeyMultiHeadProjection"][0])
print(previousSavedState["previousValueMultiHeadProjection"][0])
```

## 3. Test methods

### 3.1. `_updateLayerIncrementalState` method

#### 3.1.1 Option tests

```{python}
def test_updateLayerIncrementalState(
  layerIdx: Optional[int]=None,
  previousSavedState: Optional[List[Tensor]] = None,
  incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
):

  printData('Input `layerIdx`', layerIdx)
  printData('Input `previousSavedState`', len(previousSavedState) if previousSavedState is not None else None)
  printData('Input `incrementalState`', incrementalState.keys() if incrementalState is not None else None)
  print()

  if incrementalState is not None:
    layerCount = 1
    for key, dictElm in incrementalState.items():
      print(f'# Saved state for `layer{layerCount}` with id: ', key)
      for innerKey, dictElm in dictElm.items():
        print('- ',innerKey, ' : ',dictElm.shape)
      layerCount += 1
      print()

      print('-'*20)

  model._updateCurrentLayerIncrementalState(
    layerIdx,
    previousSavedState,
    incrementalState
  )

  if incrementalState is not None:
    layerCount = 1
    for key, dictElm in incrementalState.items():
      print(f'# Saved state for `layer{layerCount}` with id: ', key)
      for innerKey, dictElm in dictElm.items():
        print('- ',innerKey, ' : ',dictElm.shape)
      layerCount += 1
      print()

      print('-'*20)

layerIdx1 = 1
layerIdx2 = 2

test_updateLayerIncrementalState()

test_updateLayerIncrementalState(
  layerIdx=layerIdx1,
  previousSavedState=previousSavedStateTest ,
  incrementalState=incrementalState
)
```
