# Attention components

- `_getLayerSavedState`: This method receives as input `incrementalState` a dictionary containing a `savedState` with an id of `layerIdx` for every single `transformer layer` that contains `previousKeyProjection`, `previousValueProjection` and `previousKeyPaddingMask` tensors from previous time steps. This method uses `layerIdx` retrieve the `savedState` for the current layer. If `staticKeyValueFlag` is `True` the `keys` and `values` are set to `None`
- `_computeQueryKeyValueProjections`: computes `queryProjection`, `keyProjection` and `valueProjection` that will be used to compute `key` and `query` attention weights that will be used to store a calculate a weights sum of `valueProjection`.
- `_attachMemoryBiasesToKeyValueProjections`: This method concatenates `keyMemoryBiases` vector to `keyProjection` and `valueMemoryBiases` vector to `valueProjection` generated in `computeQueryKeyValueProjections`. It also adds `zero` padding to the `attentionMask` and `keyPaddingMask` to compensate with the added biases to `projections`
- `_reshapeQueryKeyValueProjections`: Method splits the `queryProjection`, `keyProjection` and `valueProjection` into multiple heads to perform `multiheadAttention`, the shape of `keyProjection` and `valueProjection` will be `[numHeads,  batchSize, sequenceLength, embeddingDim]` and the shape of `queryProjection` will be `[numHeads,  batchSize, sequenceLength, embeddingDim]`
- `_updateKeyValueProjectionsUsingLayerSavedState`: Method retrieves the `keyMultiHeadProjection`, `valueMultiHeadProjection`, `previousKeyPaddingMask` from the `savedState` and concatenates it to `keyMultiHeadProjection`, `valueMultiHeadProjection` and `keyPaddingMask` along the `sequenceLength` dimension
- `_addZeroAttention`: adds `zeroAttention` to `keyProjection` and `valueProjection`, by adding a token of zeros at the end of the sequence two projections. `attentionMask` and `keyPaddingMask` are also padded with `zeros` to compensate with the change of projections
- `_computeAttentionWeights`: `attentionWeights` are computed by multiplying the multihead `queryProjection` and `keyProjection`. To the resulting `attentionWeights` optionally `relativePositionEmbedding` can be added and `attentionMask` also known as the `causal mask` used to force the `transfomer` to learn to make predictions without allowing it to have access to future tokens
- `_maskAttentionWeightsUsingKeyPaddingMask`: This method adds `keyPaddingMask` to the `attentionWeights` generated in `computeAttentionWeights` setting them to `-inf` to make sure that padding tokens are not do not have an influence when making predictions.
- `_computeSoftmaxAttentionWeights`: this methods converts `attentionWeights` into `attentionProbabilities` by performing a `softmax` converting it to a `probability distribution` on the `attentionWeights`. The resulting `attentionProbabilities` are also passed through a `dropoutLayer` that randomly drops elements of the distribution
- `_computeWeightedValueProjections`: this method uses `attentionProbabilities` and performs a weighted sum of `valueProjection` resulting in `weightedValueProjections`. This means each `token` in the `valueProjection` is added up with all other `tokens` according to their `probability distribution` that stands for how relevant the current token is with respect to every other token in the sequence
- `_computeAttentionOutputProjection`: This method takes the `weightedValueProjections` and through the output projection model of the attention mechanism to compute `attentionOutputProjection`
- `_forward`: method that is contains all steps above that computes the full forward pass of the `Attention` mechanism

## 1. Requirements

### 1.1. Import necessary classes

```{python}
import torch
from copy import deepcopy
from torch import Tensor
from typing import Dict, Optional

from Emperor.library.choice import Library as L
from Emperor.components.attention import Attention
from Emperor.components.parameter_generators.utils.losses import AuxiliaryLosses
from Emperor.config import ModelConfig
```

### 1.2. Setup configuration

```{python}
cfg = ModelConfig(
  embeddingDim=4,
  qkvHiddenDim=6,
  sequenceLength=7,
  attentionOutputDim=8,
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
print('# attentionOutputDim: ', cfg.attentionOutputDim)
print('# numExperts: ', cfg.numExperts)
print('# topK: ', cfg.topK)
print('# headDim: ', cfg.headDim)
```

```{python}
model = Attention(
  cfg=cfg,
  embeddingDim=cfg.embeddingDim,
  qkvHiddenDim=cfg.qkvHiddenDim,
  numExperts=cfg.numExperts,
  topK=cfg.topK,
  headDim=cfg.headDim,
)
```

## 2. Expected Inputs

### 2.1 Query, Key and Value inputs

```{python}
bs = cfg.batchSize
sl = cfg.sequenceLength
hd = cfg.headDim
queryInput = torch.randn(sl, bs, model.queryInputDim)
keyInput = torch.randn(sl, bs, model.keyInputDim)
valueInput = torch.randn(sl, bs, model.valueInputDim)
model.inputShape = queryInput.size()

print('# Query input shape: ', queryInput.shape)
print('# Key input shape: ', keyInput.shape)
print('# Value input shape: ', valueInput.shape)
```

### 2.2 Skip mask

```{python}
skipMask = L.randint(0, 2, (cfg.batchSize, cfg.sequenceLength))

print(skipMask.shape)
print(skipMask)
```

### 2.3 Incremental State

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
model.incrementalStateModule.setIncrementalState (
  incrementalState,
  layerId,
  savedStateLayer1,
)

layerId = "attn_state_%d" % layerIdx2
model.incrementalStateModule.setIncrementalState (
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

### 2.4 Attention mask

```{python}
# TODO: find out what the shape of the attention mask is especially when the `key` and `value` are appended from the `incrementalState`
import torch

bs = cfg.batchSize
sl = cfg.sequenceLength
attentionMaskPlaceholder = torch.full((sl, sl), float("-inf"))
attentionMask = torch.triu(attentionMaskPlaceholder, 1)
print('# Attention mask shape: ', attentionMask.shape)
print('# Attention mask: \n', attentionMask)
```

### To be removed later

```{python}
skip_embed_dim_check = False
tgt_len, bsz, embed_dim = queryInput.size()
src_len = tgt_len
if not skip_embed_dim_check:
    assert (
        embed_dim == model.queryInputDim
    ), f"query dim {embed_dim} != {model.queryInputDim}"
assert list(queryInput.size()) == [tgt_len, bsz, embed_dim]
if keyInput is not None:
    src_len, key_bsz, _ = keyInput.size()
```

## 3. Test methods

### 3.1. `_getLayerSavedState` method

#### 3.1.1 Option tests

```{python}
def printData(name, elm):
  if isinstance(elm, torch.Tensor):
    elm = elm.shape
  print(f'# {name}: {elm}')

def test_getLayerSavedStateFromIncrementalState(
  keyInput=None,
  valueInput=None,
  layerIdx1=None,
  incrementalState=None,
  staticKeyValueFlag=False,
  returnResult: bool = False
):
  printData('Input `keyInput`', keyInput)
  printData('Input `valueInput`', valueInput)
  printData('Input `layerIdx1`', layerIdx1)
  print()

  model.staticKeyValueFlag = staticKeyValueFlag
  key, value, savedState = model._getLayerSavedStateFromIncrementalState(
    keyInput,
    valueInput,
    layerIdx1,
    incrementalState
  )
  printData('key', key)
  printData('value', value)
  if savedState is not None:
    print('saved data keys: ', savedState.keys())
    print('Saved state elements: ')
    for dictKey, dictElm in savedState.items():
      printData(dictKey, dictElm)

  print('-'*20)

  if returnResult:
    return key, value, savedState

test_getLayerSavedStateFromIncrementalState(
  keyInput,
  valueInput,
  layerIdx1,
  incrementalState,
  staticKeyValueFlag=True
)

test_getLayerSavedStateFromIncrementalState(
  keyInput,
  valueInput,
  layerIdx1,
  incrementalState
)

test_getLayerSavedStateFromIncrementalState(
  keyInput,
  valueInput,
  layerIdx1,
)

test_getLayerSavedStateFromIncrementalState(
  keyInput,
  valueInput,
)

test_getLayerSavedStateFromIncrementalState(
  keyInput,
)

test_getLayerSavedStateFromIncrementalState()
```

#### 3.1.2 Outputs required for next step

```{python}
keyInput, valueInput, savedState = test_getLayerSavedStateFromIncrementalState(
  keyInput,
  valueInput,
  layerIdx1,
  incrementalState,
  returnResult=True
)
```

### 3.2. `_computeQueryKeyValueProjections` method

#### 3.2.1. Option tests

```{python}
def test_computeQueryKeyValueProjections(
  queryInput: Tensor,
  keyInput: Optional[Tensor] = None,
  valueInput: Optional[Tensor] = None,
  skipMask: Optional[Tensor] = None,
  encoderDecorderAttentionFlag: bool = False,
  returnResult=False,
):
  printData('Input `queryInput` shape', queryInput)
  printData('Input `keyInput` shape', keyInput)
  printData('Input `valueInput` shape', valueInput)
  printData('Input `skipMask` shape', skipMask)
  print()
  model.encoderDecorderAttentionFlag = encoderDecorderAttentionFlag
  queryProjection, keyProjection, valueProjection = model._computeQueryKeyValueProjections(
    queryInput,
    keyInput,
    valueInput,
    skipMask
  )

  printData('Output `queryProjection` shape: ', queryProjection)
  printData('Output `keyProjection` shape: ', keyProjection)
  printData('Output `valueProjection` shape: ', valueProjection)
  print('-'*20)

  if returnResult:
    return queryProjection, keyProjection, valueProjection

test_computeQueryKeyValueProjections(
  queryInput,
  keyInput,
  valueInput,
  skipMask,
  encoderDecorderAttentionFlag=True
)

test_computeQueryKeyValueProjections(
  queryInput,
  keyInput,
  valueInput,
  encoderDecorderAttentionFlag=True
)

test_computeQueryKeyValueProjections(
  queryInput,
  encoderDecorderAttentionFlag=True
)

test_computeQueryKeyValueProjections(
  queryInput,
  keyInput,
  valueInput,
  skipMask
)

test_computeQueryKeyValueProjections(
  queryInput,
  keyInput,
  valueInput,
)
```

#### 3.2.2. Outputs required for next step

```{python}
queryProjection, keyProjection, valueProjection = test_computeQueryKeyValueProjections(
  queryInput,
  keyInput,
  valueInput,
  skipMask,
  returnResult=True
)
```

### 3.3. `_attachMemoryBiasesToKeyValueProjections` method

#### 3.3.1. Option tests

```{python}
def test_attachMemoryBiasesToKeyValueProjections(
  keyProjection: Optional[Tensor] = None,
  valueProjection: Optional[Tensor] = None,
  attentionMask: Optional[Tensor] = None,
  keyPaddingMask: Optional[Tensor] = None,
  returnResult=False
):

  printData('Input `keyProjection` shape', keyProjection)
  printData('Input `valueProjection` shape', valueProjection)
  printData('Input `attentionMask` shape', attentionMask)
  printData('Input `keyPaddingMask` shape', keyPaddingMask)
  print()

  keyProjection, valueProjection, attentionMask, keyPaddingMask = model._attachMemoryBiasesToKeyValueProjections(
    keyProjection,
    valueProjection,
    attentionMask,
    keyPaddingMask
  )
  printData('Output `keyProjection` shape: ', keyProjection)
  printData('Output `valueProjection` shape: ', valueProjection)
  printData('Output `attentionMask` shape: ', attentionMask)
  printData('Output `keyPaddingMask` shape: ', keyPaddingMask)
  print('-'*20)

  if returnResult:
    return keyProjection, valueProjection, attentionMask, keyPaddingMask

test_attachMemoryBiasesToKeyValueProjections(
  keyProjection,
  valueProjection,
  attentionMask,
  keyPaddingMask
)

test_attachMemoryBiasesToKeyValueProjections(
  keyProjection,
  valueProjection,
  attentionMask,
)

test_attachMemoryBiasesToKeyValueProjections(
  keyProjection,
  valueProjection,
)
```

#### 3.3.2 Outputs required for next step

```{python}
keyProjection, valueProjection, attentionMask, keyPaddingMask = test_attachMemoryBiasesToKeyValueProjections(
  keyProjection,
  valueProjection,
  attentionMask,
  keyPaddingMask,
  returnResult=True
)
```

### 3.4. `_reshapeQueryKeyValueProjections` method

#### 3.4.1. Option tests

```{python}
def test_splitQueryKeyValueProjectionsIntoMultipleHeads(
  queryProjection: Tensor,
  keyProjection: Optional[Tensor] = None,
  valueProjection: Optional[Tensor] = None,
  returnResult=False
):
  printData('Input `queryProjection` shape', queryProjection)
  printData('Input `keyProjection` shape', keyProjection)
  printData('Input `valueProjection` shape', valueProjection)
  print()

  queryProjection, keyProjection, valueProjection = model._splitQueryKeyValueProjectionsIntoMultipleHeads(
    queryProjection,
    keyProjection,
    valueProjection
  )

  printData('Output `queryProjection` shape: ', queryProjection)
  printData('Output `keyProjection` shape: ', keyProjection)
  printData('Output `valueProjection` shape: ', valueProjection)
  print('-'*20)

  if returnResult:
    return queryProjection, keyProjection, valueProjection

test_splitQueryKeyValueProjectionsIntoMultipleHeads(
  queryProjection,
  keyProjection,
  valueProjection,
)

test_splitQueryKeyValueProjectionsIntoMultipleHeads(
  queryProjection,
  keyProjection,
)

test_splitQueryKeyValueProjectionsIntoMultipleHeads(
  queryProjection
)
```

#### 3.4.2. Outputs required for next step

```{python}
queryProjection, keyProjection, valueProjection = test_splitQueryKeyValueProjectionsIntoMultipleHeads(
  queryProjection,
  keyProjection,
  valueProjection,
  returnResult=True
)
```

### 3.5. `_updateKeyValueProjectionsUsingLayerSavedState` method

#### 3.5.1. Option tests

```{python}
def test_retrieveAndUpdateProjectionsFromSavedState(
  keyProjection: Optional[Tensor]=None,
  valueProjection: Optional[Tensor]=None,
  keyPaddingMask: Optional[Tensor]=None,
  layerIdx: Optional[int]=None,
  incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]]=None,
  savedState: Optional[Dict[str, Optional[Tensor]]]=None,
  deletePreviousKeyProjection: bool = False,
  deletePreviousValueProjection: bool = False,
  deletePreviousKeyPaddingMask: bool = False,
  returnResult=False
):

  printData('Input `keyProjection` shape', keyProjection)
  printData('Input `valueProjection` shape', valueProjection)
  printData('Input `keyPaddingMask` shape', keyPaddingMask)
  printData('Input `layerIdx` shape', layerIdx)
  if savedState is not None:
    if deletePreviousKeyProjection :
      del savedState["previousKeyMultiHeadProjection"]
    if deletePreviousValueProjection:
      del savedState["previousValueMultiHeadProjection"]
    if deletePreviousKeyPaddingMask:
      del savedState["previousKeyPaddingMask"]

  print('*'*10)
  if incrementalState is not None:
    layerCount = 1
    for key, dictElm in incrementalState.items():
      print(f'# Saved state for `layer{layerCount}` with id: ', key)
      for innerKey, dictElm in dictElm.items():
        if dictElm is not None:
          print('- ',innerKey, ' : ',dictElm.shape)
        else:
          print('- ',innerKey, ' : ',dictElm)

      layerCount += 1
  print('*'*10)
  print()

  keyProjection, valueProjection, keyPaddingMask, incrementalState = model._updateKeyValueProjectionsUsingLayerSavedState(
    keyProjection,
    valueProjection,
    keyPaddingMask,
    layerIdx,
    incrementalState,
    savedState
  )

  printData('Output `keyProjection` shape: ', keyProjection)
  printData('Output `valueProjection` shape: ', valueProjection)
  printData('Output `keyPaddingMask` shape: ', keyPaddingMask)

  if incrementalState is not None:
    layerCount = 1
    print('*'*10)
    for key, dictElm in incrementalState.items():
      print(f'# Saved state for `layer{layerCount}` with id: ', key)
      for innerKey, dictElm in dictElm.items():
        if dictElm is not None:
          print('- ',innerKey, ' : ',dictElm.shape)
        else:
          print('- ',innerKey, ' : ',dictElm)

      layerCount += 1
    print('*'*10)

  print('-'*20)

  if returnResult:
    return keyProjection, valueProjection, keyPaddingMask, incrementalState

# TESTS COMMENDED BECAUSE THEY INTERFERE WITH THE
# GENERATION OF TENSORS FOR THE NEXT STEP
incrementalStateTest = deepcopy(incrementalState)
savedStateTest = deepcopy(savedState)

test_retrieveAndUpdateProjectionsFromSavedState(
  keyProjection=keyProjection,
  valueProjection=valueProjection,
  keyPaddingMask=keyPaddingMask,
  layerIdx=layerIdx1,
  incrementalState=incrementalStateTest,
  savedState=savedStateTest
)

test_retrieveAndUpdateProjectionsFromSavedState(
  keyProjection=keyProjection,
  valueProjection=valueProjection,
  keyPaddingMask=keyPaddingMask,
  layerIdx=layerIdx1,
  incrementalState=incrementalStateTest,
  savedState=savedStateTest,
  deletePreviousKeyProjection=True,
  deletePreviousValueProjection=True,
  deletePreviousKeyPaddingMask=True,
)

test_retrieveAndUpdateProjectionsFromSavedState(
  keyProjection=keyProjection,
  valueProjection=valueProjection,
  keyPaddingMask=keyPaddingMask,
  layerIdx=layerIdx1,
  incrementalState=incrementalStateTest,
  savedState=savedStateTest,
  deletePreviousKeyProjection=True,
  deletePreviousValueProjection=True,
)

test_retrieveAndUpdateProjectionsFromSavedState(
  keyProjection=keyProjection,
  valueProjection=valueProjection,
  keyPaddingMask=keyPaddingMask,
  layerIdx=layerIdx1,
  incrementalState=incrementalStateTest,
  savedState=savedStateTest,
  deletePreviousKeyProjection=True
)

test_retrieveAndUpdateProjectionsFromSavedState(
  keyProjection=keyProjection,
  valueProjection=valueProjection,
  layerIdx=layerIdx1,
  incrementalState=incrementalStateTest,
  savedState=savedStateTest
)

test_retrieveAndUpdateProjectionsFromSavedState()
```

#### 3.5.2. Outputs required for next step

```{python}
keyProjection, valueProjection, keyPaddingMask, incrementalState = test_retrieveAndUpdateProjectionsFromSavedState(
  keyProjection=keyProjection,
  valueProjection=valueProjection,
  keyPaddingMask=keyPaddingMask,
  layerIdx=layerIdx1,
  incrementalState=incrementalState,
  savedState=savedState,
  returnResult=True
)
```

### 3.6. `_addZeroAttention` method

#### 3.6.1. Option tests

```{python}
def test_addZeroAttention(
  keyProjection: Optional[Tensor] = None,
  valueProjection: Optional[Tensor] = None,
  attentionMask: Optional[Tensor] = None,
  keyPaddingMask: Optional[Tensor] = None,
  returnResult: bool = False
):

  printData('Input `keyProjection` shape', keyProjection)
  printData('Input `valueProjection` shape', valueProjection)
  printData('Input `attentionMask` shape', attentionMask)
  printData('Input `keyPaddingMask` shape', keyPaddingMask)
  print()

  keyProjection, valueProjection, attentionMask, keyPaddingMask = model._addZeroAttention(
    keyProjection,
    valueProjection,
    attentionMask,
    keyPaddingMask,
  )

  printData('Output `keyProjection` shape: ', keyProjection)
  printData('Output `valueProjection` shape: ', valueProjection)
  printData('Output `attentionMask` shape: ', attentionMask)
  printData('Output `keyPaddingMask` shape: ', keyPaddingMask)
  print('-'*20)

  if returnResult:
    return keyProjection, valueProjection, attentionMask, keyPaddingMask

test_addZeroAttention(
  keyProjection,
  valueProjection,
  attentionMask,
  keyPaddingMask
)

test_addZeroAttention(
  keyProjection,
  valueProjection,
  attentionMask,
)

test_addZeroAttention(
  keyProjection,
  valueProjection
)
```

#### 3.6.2. Outputs required for next step

```{python}
keyProjection, valueProjection, attentionMask, keyPaddingMask = test_addZeroAttention(
  keyProjection,
  valueProjection,
  attentionMask,
  keyPaddingMask,
  returnResult=True
)
```

### 3.7. `_computeAttentionWeights` method

#### 3.7.1. Option tests

```{python}
def test_computeAttentionWeights(
  queryProjection: Optional[Tensor]=None,
  keyProjection: Optional[Tensor]=None,
  attentionMask: Optional[Tensor]=None,
  incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]]=None,
  selfAttentionFlag: bool = False,
  returnResult: bool = False
):

  totalBatchSize = model.cfg.batchSize * model.topK * model.numHeads
  model.selfAttentionFlag = selfAttentionFlag
  targetLength = queryProjection.size(3) if queryProjection is not None else None
  sourceLength = keyProjection.size(2) if keyProjection is not None else None

  printData('Input `queryProjection` shape', queryProjection)
  printData('Input `keyProjection` shape', keyProjection)
  printData('Input `attentionMask` shape', attentionMask)
  printData('Input `incrementalState` shape', incrementalState.keys() if incrementalState is not None else incrementalState)
  printData('Input `totalBatchSize`,`targetLength`,`sourceLength` shape', (totalBatchSize, targetLength, sourceLength))
  print()

  attentionWeights, attentionMask = model._computeAttentionWeights(
    queryProjection,
    keyProjection,
    attentionMask,
    incrementalState,
    totalBatchSize,
    targetLength,
    sourceLength,
  )

  printData('Output `attentionWeightsTest` shape: ', attentionWeights)
  printData('Output `attentionMask` shape: ', attentionMask)
  print('-'*20)

  if returnResult:
    return attentionWeights, attentionMask

# TODO: make sure that you implement ability for `relativePositionEmbedding`
# to be applied to tensors that add past keyProjection stored in `savedState`
test_computeAttentionWeights(
  queryProjection,
  keyProjection,
  attentionMask,
  incrementalState,
  # selfAttentionFlag=True
)

test_computeAttentionWeights(
  queryProjection,
  keyProjection,
  attentionMask
)

test_computeAttentionWeights(
  queryProjection,
  keyProjection
)
```

#### 3.7.2. Outputs required for next step

```{python}
attentionWeights, attentionMask = test_computeAttentionWeights(
  queryProjection,
  keyProjection,
  attentionMask,
  returnResult=True
)

print(attentionWeights)
print(attentionMask)
```

### 3.8. `_maskAttentionWeightsUsingKeyPaddingMask` method

#### 3.8.1. Option tests

```{python}
def test_maskAttentionWeightsUsingKeyPaddingMask(
  attentionWeights: Optional[Tensor]=None,
  keyPaddingMask: Optional[Tensor]=None,
  returnResult: bool = False,
  printStepShapes: bool = False,
):

  totalBatchSize = model.cfg.batchSize * model.topK * model.numHeads
  targetLength = None
  if attentionWeights is not None:
    targetLength = queryProjection.size(3)

  sourceLength = None
  if attentionWeights is not None:
    sourceLength = keyProjection.size(2)

  printData('Input `attentionWeights` shape', attentionWeights)
  printData('Input `keyPaddingMask` shape', keyPaddingMask)
  printData('Input `incrementalState` shape', incrementalState.keys() if incrementalState is not None else incrementalState)
  printData('Input `totalBatchSize`,`targetLength`,`sourceLength` shape', (totalBatchSize,
                                                                           targetLength,
                                                                           sourceLength))
  print()

  attentionWeights = model._maskAttentionWeightsUsingKeyPaddingMask(
    attentionWeights,
    keyPaddingMask,
    totalBatchSize,
    targetLength,
    sourceLength,
    printStepShapes
  )

  printData('Output `attentionWeights` shape: ', attentionWeights)
  print('-'*20)
  if returnResult:
    return attentionWeights

print(keyPaddingMask)

test_maskAttentionWeightsUsingKeyPaddingMask(
  attentionWeights,
  keyPaddingMask,
  printStepShapes = True,
)

test_maskAttentionWeightsUsingKeyPaddingMask(
  attentionWeights,
  printStepShapes = True,
)
test_maskAttentionWeightsUsingKeyPaddingMask()
```

#### 3.8.2. Outputs required for next step

```{python}
attentionWeights = test_maskAttentionWeightsUsingKeyPaddingMask(
  attentionWeights,
  keyPaddingMask,
  printStepShapes=True,
  returnResult=True
)
```

### 3.9. `_computeSoftmaxAttentionWeights` method

#### 3.9.1. Option tests

```{python}
def test_computeSoftmaxAttentionWeights(
  attentionWeights: Optional[Tensor]=None,
  returnResult: bool = False,
):
  printData('Input `attentionWeights` shape', attentionWeights)

  attentionProbabilities = model._computeSoftmaxAttentionWeights(
    attentionWeights,
  )
  printData('Output `attentionWeights` shape: ', attentionProbabilities)

  if returnResult:
    return attentionWeights

test_computeSoftmaxAttentionWeights(attentionWeights)
```

#### 3.9.2. Outputs required for next step

```{python}
attentionProbabilities = test_computeSoftmaxAttentionWeights(
  attentionWeights,
  returnResult=True
)
```

### 3.10. `_computeWeightedValueProjections` method

#### 3.10.1. Option tests

```{python}
def test_computeAttentionOnValueProjections(
  attentionProbabilities: Optional[Tensor]=None,
  valueProjection: Optional[Tensor]=None,
  printStepShapes: bool = False,
  returnResult: bool = False,
):
  printData('Input `attentionProbabilities` shape', attentionProbabilities)
  printData('Input `valueProjection` shape', valueProjection)
  print()

  weightedValueProjections, targetLength = model._computeWeightedValueProjections(
    attentionProbabilities,
    valueProjection,
    printStepShapes
  )

  printData('Output `weightedValueProjections` shape: ', weightedValueProjections)
  printData('Output `targetLength` shape: ', targetLength)
  print('-'*20)

  if returnResult:
    return weightedValueProjections, targetLength

test_computeAttentionOnValueProjections(
  attentionProbabilities,
  valueProjection,
  printStepShapes = True
)
```

#### 3.10.2. Outputs required for next step

```{python}
weightedValueProjections, targetLength = test_computeAttentionOnValueProjections(
  attentionProbabilities,
  valueProjection,
  returnResult=True
)
```

### 3.11. `_computeAttentionOutputProjection` method

#### 3.11.1. Option tests

```{python}
def test_computeAttentionOutputProjection(
  weightedValueProjections: Optional[Tensor] = None,
  targetLength: Optional[int] = None,
  printStepShapes: bool = False,
  returnResult: bool = False,
):
  printData('Input `weightedValueProjections` shape', weightedValueProjections)
  printData('Input `targetLength` shape', targetLength)
  print()

  attentionOutputProjection = model._computeAttentionOutputProjection(
    weightedValueProjections,
    targetLength,
    printStepShapes
  )

  printData('Output `attentionOutputProjection` shape: ', attentionOutputProjection)
  print('-'*20)

  if returnResult:
    return attentionOutputProjection

test_computeAttentionOutputProjection(
  weightedValueProjections,
  targetLength
)
```

#### 3.11.2. Outputs required for next step

```{python}
attentionOutputProjection = test_computeAttentionOutputProjection(
  weightedValueProjections,
  targetLength
)
```

### 3.10 `_forward` method

```{python}

from torch import Tensor
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union
bs = cfg.batchSize
sl = cfg.sequenceLength
hd = cfg.headDim
bs = cfg.batchSize
nh = model.numHeads
hd = model.headDim
sl = cfg.sequenceLength
queryInput = L.arange(sl * bs * model.queryInputDim).reshape(sl, bs, model.queryInputDim).float()
keyInput = L.arange(sl * bs * model.keyInputDim).reshape(sl, bs, model.keyInputDim).float()
valueInput = L.arange(sl * bs * model.valueInputDim).reshape(sl, bs, model.valueInputDim).float()
skipMask = L.randint(0, 2, (cfg.batchSize, cfg.sequenceLength))
keyPaddingMask = torch.tensor(
[[1, 1, 1, 0, 0, 0, 0],
 [1, 1, 1, 1, 1, 0, 0],
 [1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 0, 0, 0],
 [1, 1, 0, 0, 0, 0, 0]])

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
model.incrementalStateModule.setIncrementalState (
  incrementalState,
  layerId,
  savedStateLayer1,
)

layerId = "attn_state_%d" % layerIdx2
model.incrementalStateModule.setIncrementalState (
  incrementalState,
  layerId,
  savedStateLayer2,
)

bs = cfg.batchSize
sl = cfg.sequenceLength
attentionMaskPlaceholder = torch.ones(sl, sl)
attentionMask = torch.tril(attentionMaskPlaceholder)

def test_forward(
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

  attentionOutputProjection, attentionWeights = model(
    key=query,
    query=key,
    value=value,
    layerIdx=layerIdx,
    keyPaddingMask=keyPaddingMask,
    incrementalState=incrementalState,
    skipMask=skipMask,
    attentionMask=attentionMask
  )

  print('# Attention Projecection shape: ', attentionOutputProjection.shape)
  print('# Attention weights shape: ', attentionWeights.shape if attentionWeights is not None else None)
  print('-'*10)


test_forward(
  query=keyInput,
  key=queryInput,
  value=valueInput,
  layerIdx=layerIdx1,
  keyPaddingMask=keyPaddingMask,
  incrementalState=incrementalState,
  skipMask=skipMask,
  attentionMask=attentionMask
)

test_forward(
  query=keyInput,
  key=queryInput,
  value=valueInput,
  layerIdx=layerIdx1,
  keyPaddingMask=keyPaddingMask,
  incrementalState=incrementalState,
  skipMask=skipMask
)

test_forward(
  query=keyInput,
  key=queryInput,
  value=valueInput,
  layerIdx=layerIdx1,
  keyPaddingMask=keyPaddingMask,
  incrementalState=incrementalState,
)

test_forward(
  query=keyInput,
  key=queryInput,
  value=valueInput,
  layerIdx=layerIdx1,
  keyPaddingMask=keyPaddingMask,
)

test_forward(
  query=queryInput,
  key=keyInput,
  value=valueInput,
  layerIdx=layerIdx1,
)

test_forward(
  query=queryInput,
  key=keyInput,
  value=valueInput
)
```

## 4. TEST MODELS

### 4.1. Single layer of `Attention` on `FashionMNIST`

```{python}
def testAttentionSingleLayer():
  from Emperor.base.utils import Trainer
  from Emperor.base.datasets import FashionMNIST
  from Emperor.experiments import AttentionSingleLayerModel
  from Emperor.config import ModelConfig, ParameterGeneratorOptions

  cfg = ModelConfig(
    batchSize=16,
    inputDim=16,
    hiddenDim=64,
    outputDim=10,
    depthDim=64,

    embeddingDim=16,
    qkvHiddenDim=64,
    attentionOutputDim=10,
    numExperts=6,
    headDim=32,
    topK=2,
    parameterGeneartorType=ParameterGeneratorOptions.matrix_choice_sparse
  )

  data = FashionMNIST(
    batch_size=cfg.batchSize,
    testDatasetFalg=True,
    testDatasetNumSamples=16
  )

  model = AttentionSingleLayerModel(learningRate=0.1, cfg=cfg)
  trainer = Trainer(max_epochs=100)
  trainer.fit(model, data, printLossFlag=True)

testAttentionSingleLayer()
```

## TEST: PatchEmbedding

[Demystifying Visual Transformers with PyTorch: Understanding Patch Embeddings](https://medium.com/@fernandopalominocobo/demystifying-visual-transformers-with-pytorch-understanding-patch-embeddings-part-1-3-ba380f2aa37f)

```{python}
def testPatchEmbedding():

  import torch
  import torch.nn as nn

  batchSize = 512
  numChanels = 1
  imageWidth = 28
  imageHeight = 28
  inputBatch = torch.randn(batchSize, numChanels, imageWidth, imageHeight)
  print('# `inputBatch` shape: ', inputBatch.shape)

  inputChannels = 1
  embeddingDim = 16
  kernelSize=4
  stride=4
  numPatches=49
  patchModel = nn.Conv2d(
    in_channels=inputChannels,
    out_channels=embeddingDim,
    kernel_size=kernelSize,
    stride=stride,
  )
  flatten = nn.Flatten(2)

  classToekns = nn.Parameter(torch.randn((1, 1, embeddingDim)))
  print('# `classToekns` shape: ', classToekns.shape)
  positionEmbeddings = nn.Parameter(torch.randn((1, numPatches + 1, embeddingDim)))
  print('# `positionEmbeddings` shape: ', classToekns.shape)
  print()

  patches = patchModel(inputBatch)
  print('# `patches` shape: ', patches.shape)

  flattenedPatches = flatten(patches)
  print('# `flattenedPatches` shape: ', flattenedPatches.shape)

  convertingToFeatures = flattenedPatches.permute(0, 2, 1)
  print('# `convertingToFeatures` shape: ', convertingToFeatures.shape)

  updatedClassToekns = classToekns.expand(batchSize, -1, -1)
  print('# `updatedClassToekns` shape: ', updatedClassToekns.shape)

  addedClassToken = torch.cat([updatedClassToekns, convertingToFeatures], dim=1)
  print('# `addedClassToken` shape: ', addedClassToken.shape)

  addedPositionalEncoding = positionEmbeddings + addedClassToken
  print('# `addedPositionalEncoding` shape: ', addedPositionalEncoding.shape)


testPatchEmbedding()
```
