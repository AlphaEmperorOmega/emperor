# Sparse Universal Transformer Decoder Components

-

## 1. Requirements

### 1.1. Import necessary classes

```{python}
import torch
from torch import Tensor
from typing import Optional, List, Dict

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
  crossSelfAttentionFlag=True,
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

incrementalStateSelfAttention = {}
layerId = "attn_state_%d" % layerIdx1
model.selfAttnModel.incrementalStateModule.setIncrementalState (
  incrementalStateSelfAttention,
  layerId,
  savedStateLayer1,
)

layerId = "attn_state_%d" % layerIdx2
model.selfAttnModel.incrementalStateModule.setIncrementalState (
  incrementalStateSelfAttention,
  layerId,
  savedStateLayer2,
)

incrementalStateCrossAttention = {}
layerId = "attn_state_%d" % layerIdx1
model.crossAttnModel.incrementalStateModule.setIncrementalState (
  incrementalStateCrossAttention,
  layerId,
  savedStateLayer1,
)

layerId = "attn_state_%d" % layerIdx2
model.crossAttnModel.incrementalStateModule.setIncrementalState (
  incrementalStateCrossAttention,
  layerId,
  savedStateLayer2,
)

incrementalStateTest = {}
layerId = "attn_state_%d" % layerIdx1
model.selfAttnModel.incrementalStateModule.setIncrementalState (
  incrementalStateTest,
  layerId,
  savedStateLayer1,
)

layerId = "attn_state_%d" % layerIdx2
model.selfAttnModel.incrementalStateModule.setIncrementalState (
  incrementalStateTest,
  layerId,
  savedStateLayer2,
)

layerCount = 1
for key, dictElm in incrementalStateSelfAttention.items():
  print(f'# Saved state for `layer{layerCount}` with id: ', key)
  for innerKey, dictElm in dictElm.items():
    print('- ',innerKey, ' : ',dictElm.shape)
  layerCount += 1
  print()
```

### 2.2 Previous saved state

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

### 2.3 Halt mask

```{python}
haltMask = torch.randint(0, 2, (cfg.batchSize, cfg.sequenceLength))

print(haltMask.shape)
print(haltMask)
```

### 2.4 Encoder output

```{python}
encoderOutput = torch.randn(cfg.sequenceLength, cfg.batchSize, model.selfAttnModel.queryInputDim)
print(inputBatch.size())
```

### 2.4 Encoder padding mask

```{python}
mask = torch.zeros(
    (cfg.batchSize, cfg.sequenceLength), dtype=torch.bool
)

paddingStart = cfg.sequenceLength // 2
paddingEnd = cfg.sequenceLength
randomSeqenceLengths = torch.randint(
    paddingStart, paddingEnd, (cfg.batchSize,)
)

for batchIdx, sequenceLength in enumerate(randomSeqenceLengths):
    mask[batchIdx, :sequenceLength] = 1

encoderPaddingMask = mask

print(encoderPaddingMask.shape)
print(encoderPaddingMask)
```

### 2.5 Attention mask

```{python}
bs = cfg.batchSize
sl = cfg.sequenceLength
attentionMaskPlaceholder = torch.ones((sl, sl))
attentionMask = torch.triu(attentionMaskPlaceholder, 1)

print()
print('# Attention mask shape: ', attentionMask.shape)
print('# Attention mask: \n', attentionMask)
```

### 2.6 Padding mask

```{python}
mask = torch.zeros(
    (cfg.batchSize, cfg.sequenceLength), dtype=torch.bool
)

paddingStart = cfg.sequenceLength // 2
paddingEnd = cfg.sequenceLength
randomSeqenceLengths = torch.randint(
    paddingStart, paddingEnd, (cfg.batchSize,)
)

for batchIdx, sequenceLength in enumerate(randomSeqenceLengths):
    mask[batchIdx, :sequenceLength] = 1

selfAttentionPaddingMask = mask

print(selfAttentionPaddingMask.shape)
print(selfAttentionPaddingMask)
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


test_updateLayerIncrementalState()

test_updateLayerIncrementalState(
  layerIdx=layerIdx1,
  previousSavedState=previousSavedStateTest ,
  incrementalState=incrementalStateTest
)
```

### 3.2. `_computeSelfAttention` method

#### 3.2.1 Option tests with method with `crossSelfAttentionFlag=False`

```{python}
def test_computeSelfAttention_corssAttnTrue(
  inputBatch: Optional[Tensor] = None,
  selfAttentionInput: Optional[Tensor] = None,
  haltMask: Optional[Tensor] = None,
  layerIdx: Optional[int] = None,
  incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
  selfAttentionMask: Optional[Tensor] = None,
  selfAttentionPaddingMask: Optional[Tensor] = None,
  encoderOutput: Optional[Tensor] = None,
  encoderPaddingMask: Optional[Tensor] = None,
  checkIfIncrementalStateDoesExistFlag: Optional[bool] = None,
  modelId: Optional[str] = None
):
  oldModelId = modelId
  model = TransformerDecorderLayerBase(
    cfg=cfg,
    crossSelfAttentionFlag=False
  )
  model.selfAttnModel.incrementalStateModule.incrementalStateId = oldModelId
  print(model.crossSelfAttentionFlag)

  if encoderOutput is not None:
    selfAttentionMask, selfAttentionPaddingMask = model._updateSelfAttentionPaddingMasks(
      inputBatch,
      encoderOutput,
      encoderPaddingMask,
      selfAttentionMask,
      selfAttentionPaddingMask,
      checkIfIncrementalStateDoesExistFlag
    )

  printData('Input `inputBatch`', inputBatch if inputBatch is not None else None)
  printData('Input `selfAttentionInput`', selfAttentionInput if selfAttentionInput is not None else None)
  printData('Input `haltMask`', haltMask if haltMask is not None else None)
  printData('Input `layerIdx`', layerIdx if layerIdx is not None else None)
  printData('Input `incrementalState`', incrementalState.keys() if incrementalState is not None else None)
  printData('Input `selfAttentionMask`', selfAttentionMask if selfAttentionMask is not None else None)
  printData('Input `selfAttentionPaddingMask`', selfAttentionPaddingMask if selfAttentionPaddingMask is not None else None)
  printData('Input `encoderOutput`', encoderOutput if encoderOutput is not None else None)
  printData('Input `checkIfIncrementalStateDoesExistFlag`', checkIfIncrementalStateDoesExistFlag if checkIfIncrementalStateDoesExistFlag is not None else None)

  print()

  attnOutput = model._computeSelfAttention(
    inputBatch,
    selfAttentionInput,
    haltMask,
    layerIdx,
    incrementalState,
    selfAttentionMask,
    selfAttentionPaddingMask,
    encoderOutput,
    checkIfIncrementalStateDoesExistFlag,
  )

  printData('Output `attnOutput`', attnOutput)

  print('-'*20)

modelId = model.selfAttnModel.incrementalStateModule.incrementalStateId

test_computeSelfAttention_corssAttnTrue(
  inputBatch=inputBatch,
  selfAttentionInput=inputBatch,
  haltMask=haltMask,
  layerIdx=layerIdx1,
  incrementalState=incrementalStateSelfAttention,
  selfAttentionMask=attentionMask,
  selfAttentionPaddingMask=selfAttentionPaddingMask,
  encoderOutput=encoderOutput,
  checkIfIncrementalStateDoesExistFlag=False,
  modelId = modelId
)

test_computeSelfAttention_corssAttnTrue(
  inputBatch=inputBatch,
  selfAttentionInput=inputBatch,
  haltMask=haltMask,
  layerIdx=layerIdx1,
  incrementalState=incrementalStateSelfAttention,
  selfAttentionMask=attentionMask,
  selfAttentionPaddingMask=selfAttentionPaddingMask,
  encoderOutput=encoderOutput,
  encoderPaddingMask=encoderPaddingMask,
  modelId = modelId
)

test_computeSelfAttention_corssAttnTrue(
  inputBatch=inputBatch,
  selfAttentionInput=inputBatch,
  haltMask=haltMask,
  layerIdx=layerIdx1,
  incrementalState=incrementalStateSelfAttention,
  selfAttentionMask=attentionMask,
  selfAttentionPaddingMask=selfAttentionPaddingMask,
  modelId = modelId
)

# When using self attention relative positional embedding is added
# see `_relativePositionLogits` in `Attention` which is why
# it has been disabled this should be fixed in the future
test_computeSelfAttention_corssAttnTrue(
  inputBatch=inputBatch,
  selfAttentionInput=inputBatch,
  haltMask=haltMask,
  layerIdx=layerIdx1,
  incrementalState=incrementalStateSelfAttention,
  selfAttentionMask=attentionMask,
  modelId = modelId
)

test_computeSelfAttention_corssAttnTrue(
  inputBatch=inputBatch,
  selfAttentionInput=inputBatch,
  haltMask=haltMask,
  layerIdx=layerIdx1,
  incrementalState=incrementalStateSelfAttention,
  modelId = modelId
)

test_computeSelfAttention_corssAttnTrue(
  inputBatch=inputBatch,
  selfAttentionInput=inputBatch,
  haltMask=haltMask,
  layerIdx=layerIdx1,
  modelId = modelId
)

test_computeSelfAttention_corssAttnTrue(
  inputBatch=inputBatch,
  selfAttentionInput=inputBatch,
  haltMask=haltMask,
  modelId = modelId
)

test_computeSelfAttention_corssAttnTrue(
  inputBatch=inputBatch,
  selfAttentionInput=inputBatch,
  modelId = modelId
)
```

#### 3.2.2 Option tests with method with `crossSelfAttentionFlag=True`

IMPORTANT: when `crossSelfAttentionFlag=True` the `encoderOutput` and `encoderPaddingMask` are required inputs

```{python}
def test_computeSelfAttention_corssAttnTrue(
  inputBatch: Optional[Tensor] = None,
  selfAttentionInput: Optional[Tensor] = None,
  haltMask: Optional[Tensor] = None,
  layerIdx: Optional[int] = None,
  incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
  selfAttentionMask: Optional[Tensor] = None,
  selfAttentionPaddingMask: Optional[Tensor] = None,
  encoderOutput: Optional[Tensor] = None,
  encoderPaddingMask: Optional[Tensor] = None,
  checkIfIncrementalStateDoesExistFlag: Optional[bool] = None,
  returnResult=False,
):

  if encoderOutput is not None:
    selfAttentionMask, selfAttentionPaddingMask = model._updateSelfAttentionPaddingMasks(
      inputBatch,
      encoderOutput,
      encoderPaddingMask,
      selfAttentionMask,
      selfAttentionPaddingMask,
      checkIfIncrementalStateDoesExistFlag
    )

  printData('Input `inputBatch`', inputBatch if inputBatch is not None else None)
  printData('Input `selfAttentionInput`', selfAttentionInput if selfAttentionInput is not None else None)
  printData('Input `haltMask`', haltMask if haltMask is not None else None)
  printData('Input `layerIdx`', layerIdx if layerIdx is not None else None)
  printData('Input `incrementalState`', incrementalState.keys() if incrementalState is not None else None)
  printData('Input `selfAttentionMask`', selfAttentionMask if selfAttentionMask is not None else None)
  printData('Input `selfAttentionPaddingMask`', selfAttentionPaddingMask if selfAttentionPaddingMask is not None else None)
  printData('Input `encoderOutput`', encoderOutput if encoderOutput is not None else None)
  printData('Input `checkIfIncrementalStateDoesExistFlag`', checkIfIncrementalStateDoesExistFlag if checkIfIncrementalStateDoesExistFlag is not None else None)

  print()

  attnOutput = model._computeSelfAttention(
    inputBatch,
    selfAttentionInput,
    haltMask,
    layerIdx,
    incrementalState,
    selfAttentionMask,
    selfAttentionPaddingMask,
    encoderOutput,
    checkIfIncrementalStateDoesExistFlag,
  )

  printData('Output `attnOutput`', attnOutput)

  print('-'*20)

  if returnResult:
    return attnOutput


test_computeSelfAttention_corssAttnTrue(
  inputBatch=inputBatch,
  selfAttentionInput=inputBatch,
  haltMask=haltMask,
  layerIdx=layerIdx1,
  incrementalState=incrementalStateSelfAttention,
  selfAttentionMask=attentionMask,
  selfAttentionPaddingMask=selfAttentionPaddingMask,
  encoderOutput=encoderOutput,
  encoderPaddingMask=encoderPaddingMask
)

# # When using self attention relative positional embedding is added
# # see `_relativePositionLogits` in `Attention` which is why
# # it has been disabled this should be fixed in the future
test_computeSelfAttention_corssAttnTrue(
  inputBatch=inputBatch,
  selfAttentionInput=inputBatch,
  haltMask=haltMask,
  layerIdx=layerIdx1,
  incrementalState=incrementalStateSelfAttention,
  selfAttentionMask=attentionMask,
  encoderOutput=encoderOutput,
  encoderPaddingMask=encoderPaddingMask
)

test_computeSelfAttention_corssAttnTrue(
  inputBatch=inputBatch,
  selfAttentionInput=inputBatch,
  haltMask=haltMask,
  layerIdx=layerIdx1,
  incrementalState=incrementalStateSelfAttention,
  encoderOutput=encoderOutput,
  encoderPaddingMask=encoderPaddingMask
)

test_computeSelfAttention_corssAttnTrue(
  inputBatch=inputBatch,
  selfAttentionInput=inputBatch,
  haltMask=haltMask,
  layerIdx=layerIdx1,
  encoderOutput=encoderOutput,
  encoderPaddingMask=encoderPaddingMask
)

test_computeSelfAttention_corssAttnTrue(
  inputBatch=inputBatch,
  selfAttentionInput=inputBatch,
  haltMask=haltMask,
  encoderOutput=encoderOutput,
  encoderPaddingMask=encoderPaddingMask
)

test_computeSelfAttention_corssAttnTrue(
  inputBatch=inputBatch,
  selfAttentionInput=inputBatch,
  encoderOutput=encoderOutput,
  encoderPaddingMask=encoderPaddingMask
)
```

#### 3.2.3 Outputs required for next step

```{python}
attnOutput = test_computeSelfAttention_corssAttnTrue(
  inputBatch=inputBatch,
  selfAttentionInput=inputBatch,
  haltMask=haltMask,
  layerIdx=layerIdx1,
  incrementalState=incrementalStateSelfAttention,
  selfAttentionMask=attentionMask,
  selfAttentionPaddingMask=selfAttentionPaddingMask,
  encoderOutput=encoderOutput,
  encoderPaddingMask=encoderPaddingMask,
  returnResult=True
)
```

### 3.3. `_computeCrossAttention` method

#### 3.3.1 Option tests with method

```{python}
def test_computeCrossAttention(
  selfAttnOutput: Optional[Tensor]=None,
  encoderOutput: Optional[Tensor]=None,
  encoderPaddingMask: Optional[Tensor]=None,
  haltMask: Optional[Tensor]=None,
  layerIdx: Optional[int]=None,
  incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]]=None,
  previousCrossAttnState: Optional[List[Tensor]]=None,
  selfAttentionPaddingMask: Optional[Tensor]=None,

  returnResult=False,
):
  printData('Input `selfAttnOutput`', selfAttnOutput if selfAttnOutput is not None else None)
  printData('Input `encoderOutput`', encoderOutput if encoderOutput is not None else None)
  printData('Input `encoderPaddingMask`', encoderPaddingMask if encoderPaddingMask is not None else None)
  printData('Input `haltMask`', haltMask if haltMask is not None else None)
  printData('Input `layerIdx`', layerIdx if layerIdx is not None else None)
  printData('Input `incrementalState`', incrementalState.keys() if incrementalState is not None else None)
  printData('Input `previousCrossAttnState`', len(previousCrossAttnState) if previousCrossAttnState is not None else None)
  printData('Input `selfAttentionPaddingMask`', selfAttentionPaddingMask if selfAttentionPaddingMask is not None else None)
  print()

  attnOutput, attentionWeights = model._computeCrossAttention(
    selfAttnOutput,
    encoderOutput,
    encoderPaddingMask,
    haltMask,
    layerIdx,
    incrementalState,
    previousCrossAttnState,
    selfAttentionPaddingMask,
  )

  printData('Output `attnOutput`', attnOutput)
  printData('Output `attentionWeights`', attentionWeights)

  print('-'*20)

  if returnResult:
    return attnOutput, attentionWeights

test_computeCrossAttention(
  selfAttnOutput=attnOutput,
  encoderOutput=encoderOutput,
  encoderPaddingMask=encoderPaddingMask,
  haltMask=haltMask,
  layerIdx=layerIdx1,
  incrementalState=incrementalStateCrossAttention,
  previousCrossAttnState=previousSavedStateTest,
  selfAttentionPaddingMask=selfAttentionPaddingMask
)

test_computeCrossAttention(
  selfAttnOutput=attnOutput,
  encoderOutput=encoderOutput,
  encoderPaddingMask=encoderPaddingMask,
  haltMask=haltMask,
  layerIdx=layerIdx1,
  incrementalState=incrementalStateCrossAttention,
  previousCrossAttnState=previousSavedStateTest
)

test_computeCrossAttention(
  selfAttnOutput=attnOutput,
  encoderOutput=encoderOutput,
  encoderPaddingMask=encoderPaddingMask,
  haltMask=haltMask,
  layerIdx=layerIdx1,
  incrementalState=incrementalStateCrossAttention
)

test_computeCrossAttention(
  selfAttnOutput=attnOutput,
  encoderOutput=encoderOutput,
  encoderPaddingMask=encoderPaddingMask,
  haltMask=haltMask,
)

test_computeCrossAttention(
  selfAttnOutput=attnOutput,
  encoderOutput=encoderOutput,
  encoderPaddingMask=encoderPaddingMask,
)

test_computeCrossAttention(
  selfAttnOutput=attnOutput,
  encoderOutput=encoderOutput,
)
```

#### 3.3.2 Outputs required for next step

```{python}
crossAttnOutput, attentionWeights = test_computeCrossAttention(
  selfAttnOutput=attnOutput,
  encoderOutput=encoderOutput,
  encoderPaddingMask=encoderPaddingMask,
  haltMask=haltMask,
  layerIdx=layerIdx1,
  incrementalState=incrementalStateCrossAttention,
  previousCrossAttnState=previousSavedStateTest,
  selfAttentionPaddingMask=selfAttentionPaddingMask,

  returnResult=True,
)
```

### 3.4. `_computeFeedForward` method

#### 3.4.1 Option tests with method

```{python}
def test_computeFeedForward(
  attnOutput: Optional[Tensor]=None,
  haltMask: Optional[Tensor]=None,

  returnResult=False,
):
  printData('Input `attnOutput`', attnOutput if attnOutput is not None else None)
  printData('Input `haltMask`', haltMask if haltMask is not None else None)
  print()

  ffnOutput = model._computeFeedForward(
    attnOutput,
    haltMask,
  )

  printData('Output `attnOutput`', ffnOutput)

  print('-'*20)

  if returnResult:
    return ffnOutput

test_computeFeedForward(
  attnOutput=crossAttnOutput,
  haltMask=haltMask
)

test_computeFeedForward(
  attnOutput=crossAttnOutput
)
```

#### 3.4.2 Outputs required for next step

```{python}
ffnOutput = test_computeFeedForward(
  attnOutput=crossAttnOutput,
  haltMask=haltMask,

  returnResult=True,
)
```
