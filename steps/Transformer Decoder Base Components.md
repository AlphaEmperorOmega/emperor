# Transformer Encoder Base Components

## 1. Requirements

### 1.1. Import necessary classes

```{python}
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional, List, Tuple

from Emperor.components.transformer_decoder import TransformerDecoderBase
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
  inputEmbeddingDim=attentionInputOutput,
  outputEmbeddingDim=attentionInputOutput,
  qkvHiddenDim=6,
  sequenceLength=7,
  headDim=2,
  numExperts=7,
  topK=2,
  addPositionalEmbeddingFlag=True,
  tokenEmbeddingLayerNormFlag=True,
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
print([i for i in range(10)])
```

```{python}
model = TransformerDecoderBase(
  cfg=cfg,
  dictionary=[i for i in range(10)],
  tokenEmbeddingModule = nn.Embedding(
    num_embeddings=20,
    embedding_dim=cfg.embeddingDim,
    padding_idx=1
  )
)

print('# Transformer Encoder Base Model: \n', model)
```

## 2. Expected Inputs

### 2.1 Source Tokens

```{python}
targetTokens = torch.randint(1, 10, (cfg.batchSize, cfg.sequenceLength))
print(targetTokens.size())
```

### 2.2 Input Token Embeddings

```{python}
inputTokenEmbeddings = torch.randn(cfg.batchSize, cfg.sequenceLength, cfg.embeddingDim)
print(inputTokenEmbeddings.size())
```

### 2.3 Token Embeddings

```{python}
inputTokenEmbeddings = torch.randn(cfg.batchSize, cfg.sequenceLength, cfg.embeddingDim)
print(inputTokenEmbeddings.size())
```

### 2.4 Encoder output dictionary

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

```{python}
layersOutput = torch.randn(cfg.sequenceLength, cfg.batchSize, cfg.embeddingDim)
rawTokenEmbedding = torch.randn(cfg.batchSize, cfg.sequenceLength, cfg.embeddingDim)

encoderOutput = {
    "encoderOutput": [layersOutput],  # T x B x C
    "encoderPaddingMask": [mask],  # B x T
    "encoderRawEmbeddings": [rawTokenEmbedding],  # B x T x C
    "encoderStates": [],  # List[T x B x C]
    "ffnRawOutputList": [],  # List[T x B x C]
    "sourceTokens": [],
    "sourceSequenceLengths": [None],
    "encoderLoss": [0.0],
    "encoderHaltLoss": 0.0,
}
```

### 2.5 Halt mask

```{python}
haltMask = torch.randint(0, 2, (cfg.batchSize, cfg.sequenceLength))

print(haltMask.shape)
print(haltMask)
```

```{python}
inputTokenEmbeddings = torch.randn(cfg.batchSize, cfg.sequenceLength, cfg.embeddingDim)
print(inputTokenEmbeddings.size())
```

### 2.6 Incremental State

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
nh = model.decoderModel.model.numHeads
hd = model.decoderModel.model.headDim
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
model.decoderModel.model.selfAttnModel.incrementalStateModule.setIncrementalState (
  incrementalState,
  layerId,
  savedStateLayer1,
)

layerId = "attn_state_%d" % layerIdx2
model.decoderModel.model.selfAttnModel.incrementalStateModule.setIncrementalState (
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

## 3. Test methods

### 3.1. `_computeTokenEmbeddings` method

#### 3.1.1 Option tests

```{python}
def test_getEncoderOutputAndPaddingMask(
  targetTokens: Optional[Tensor] = None,
  tokenEmbeddings: Optional[Tensor] = None,
  encoderOutput: Optional[Dict[str, List[Tensor]]] = None,
  returnResult: bool = False
):

  encoderOutputLength = len(encoderOutput) if encoderOutput is not None else None
  printData('Input `targetTokens`', targetTokens)
  printData('Input `tokenEmbeddings`', tokenEmbeddings)
  printData('Input `encoderOutput`', encoderOutputLength)
  print()

  (
    encoderOutputTensor,
    encoderPaddingMask
  ) = model._getEncoderOutputAndPaddingMask(
    targetTokens=targetTokens,
    tokenEmbeddings=tokenEmbeddings,
    encoderOutput=encoderOutput
  )

  printData('encoderOutputTensor', encoderOutputTensor)
  printData('encoderPaddingMask', encoderPaddingMask)

  print('-'*20)

  if returnResult:
    return (
      encoderOutputTensor,
      encoderPaddingMask
    )

test_getEncoderOutputAndPaddingMask(
  targetTokens=targetTokens,
)

test_getEncoderOutputAndPaddingMask(
  tokenEmbeddings=inputTokenEmbeddings,
)

test_getEncoderOutputAndPaddingMask(
  tokenEmbeddings=inputTokenEmbeddings,
  encoderOutput=encoderOutput
)
```

#### 3.1.2 Outputs required for next step

```{python}
(
    encoderOutputTensor,
    encoderPaddingMask
) = test_getEncoderOutputAndPaddingMask(
  targetTokens=targetTokens,
  encoderOutput=encoderOutput,
  returnResult=True
)
```

### 3.2. `_computeTokenEmbeddings` method

#### 3.2.1 Option tests

```{python}
def test_computeTokenEmbeddings(
  targetTokens: Optional[Tensor] = None,
  tokenEmbeddings: Optional[torch.Tensor] = None,
  incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
  returnResult: bool = False
):

  printData('Input `targetTokens`', targetTokens)
  printData('Input `tokenEmbeddings`', tokenEmbeddings)
  printData('Input `incrementalState`', type(incrementalState))
  print()

  (
    tokenEmbeddings
  ) = model._computeTokenEmbeddings(
    targetTokens=targetTokens,
    tokenEmbeddings=tokenEmbeddings,
    incrementalState=incrementalState
  )

  printData('tokenEmbeddings', tokenEmbeddings)

  print('-'*20)

  if returnResult:
    return (
      tokenEmbeddings
    )

test_computeTokenEmbeddings(
  targetTokens=targetTokens,
)

test_computeTokenEmbeddings(
  tokenEmbeddings=inputTokenEmbeddings
)
```

#### 3.2.2 Outputs required for next step

```{python}
tokenEmbeddings = test_computeTokenEmbeddings(
  tokenEmbeddings=inputTokenEmbeddings,
  returnResult=True
)
```

### 3.3. `_computeLayerOutput` method

#### 3.3.1 Option tests

```{python}
def test_computeLayerOutput(
    tokenEmbeddings: Tensor,
    selfAttentionInput: Optional[Tensor] = None,
    haltMask: Optional[Tensor] = None,
    layerIdx: Optional[int] = None,
    encoderOutputTensor: Optional[Tensor] = None,
    encoderPaddingMask: Optional[Tensor] = None,
    selfAttentionPaddingMask: Optional[Tensor] = None,
    incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    adaptiveComputationTimeState: Optional[Tuple] = None,
    fullContextAlignmentFlag: bool = False,
    returnResult: bool = False
):

  printData('Input `tokenEmbeddings`', tokenEmbeddings)
  printData('Input `selfAttentionInput`', selfAttentionInput)
  printData('Input `haltMask`', haltMask)
  printData('Input `layerIdx`', layerIdx)
  printData('Input `encoderOutputTensor`', encoderOutputTensor)
  printData('Input `encoderPaddingMask`', encoderPaddingMask)
  printData('Input `selfAttentionPaddingMask`', selfAttentionPaddingMask)
  printData('Input `incrementalState`', incrementalState)
  printData('Input `adaptiveComputationTimeState`', adaptiveComputationTimeState)
  printData('Input `fullContextAlignmentFlag`', fullContextAlignmentFlag)
  print()

  (
    layerOutput,
    selfAttentionInput,
    adaptiveComputationTimeState,
    adaptiveComputationTimeLoss,
  ) = model._computeLayerOutput(
      tokenEmbeddings,
      selfAttentionInput,
      haltMask,
      layerIdx,
      encoderOutputTensor,
      encoderPaddingMask,
      selfAttentionPaddingMask,
      incrementalState,
      adaptiveComputationTimeState,
      fullContextAlignmentFlag,
  )

  printData('tokenEmbeddings', tokenEmbeddings)

  print('-'*20)

  if returnResult:
    return (
      layerOutput,
      selfAttentionInput,
      adaptiveComputationTimeState,
      adaptiveComputationTimeLoss,
    )

test_computeLayerOutput(
  tokenEmbeddings,
  tokenEmbeddings,
  haltMask,
  1,
  encoderOutputTensor=encoderOutputTensor,
  encoderPaddingMask=encoderPaddingMask
)

test_computeLayerOutput(
  tokenEmbeddings,
  tokenEmbeddings,
  haltMask,
  encoderOutputTensor=encoderOutputTensor,
  encoderPaddingMask=encoderPaddingMask
)

test_computeLayerOutput(
  tokenEmbeddings,
  tokenEmbeddings,
  encoderOutputTensor=encoderOutputTensor,
  encoderPaddingMask=encoderPaddingMask
)

test_computeLayerOutput(
  tokenEmbeddings,
  encoderOutputTensor=encoderOutputTensor,
  encoderPaddingMask=encoderPaddingMask,
)
```

#### 3.3.2 Outputs required for next step

```{python}
(
  layerOutput,
  selfAttentionInput,
  adaptiveComputationTimeState,
  adaptiveComputationTimeLoss,
) = test_computeLayerOutput(
  tokenEmbeddings,
  tokenEmbeddings,
  haltMask,
  1,
  encoderOutputTensor=encoderOutputTensor,
  encoderPaddingMask=encoderPaddingMask,
  returnResult = True
)
```

### 3.4. `_computeAllLayersOutput` method

#### 3.4.1 Option tests

```{python}
def test_computeAllLayersOutput(
    targetTokens: Tensor,
    tokenEmbeddings: Tensor,
    encoderOutputTensor: Optional[Tensor] = None,
    encoderPaddingMask: Optional[Tensor] = None,
    incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    alignmentLayer: Optional[int] = None,
    fullContextAlignmentFlag: bool = False,
    returnResult: bool = False
):

  printData('Input `targetTokens`', targetTokens)
  printData('Input `tokenEmbeddings`', tokenEmbeddings)
  printData('Input `encoderOutputTensor`', encoderOutputTensor)
  printData('Input `encoderPaddingMask`', encoderPaddingMask)
  printData('Input `incrementalState`', incrementalState)
  printData('Input `alignmentLayer`', alignmentLayer)
  printData('Input `fullContextAlignmentFlag`', fullContextAlignmentFlag)
  print()

  (
    layerOutput,
    attentionWeights,
    adaptieComputationTimeAttention,
    adaptiveComputationTimeLoss,
    layerHiddenStates
  ) = model._computeAllLayersOutput(
      targetTokens,
      tokenEmbeddings,
      encoderOutputTensor,
      encoderPaddingMask,
      incrementalState,
      alignmentLayer,
      fullContextAlignmentFlag,
    )

  printData('layerOutput', layerOutput)
  printData('attentionWeights', attentionWeights)
  printData('selfAttentionInput', selfAttentionInput)
  printData('adaptiveComputationTimeState', adaptiveComputationTimeState)
  printData('adaptiveComputationTimeLoss', adaptiveComputationTimeLoss)

  print('-'*20)

  if returnResult:
    return (
      layerOutput,
      attentionWeights,
      adaptieComputationTimeAttention,
      adaptiveComputationTimeLoss,
      layerHiddenStates
    )


test_computeAllLayersOutput(
  targetTokens=targetTokens,
  tokenEmbeddings=tokenEmbeddings,
  encoderOutputTensor=encoderOutputTensor,
  encoderPaddingMask=encoderPaddingMask,
)
```

#### 3.4.2 Outputs required for next step

```{python}
(
  layerOutput,
  attentionWeights,
  adaptiveComputationTimeAttention,
  adaptiveComputationTimeLoss,
  layerHiddenStates
) = test_computeAllLayersOutput(
  targetTokens=targetTokens,
  tokenEmbeddings=tokenEmbeddings,
  encoderOutputTensor=encoderOutputTensor,
  encoderPaddingMask=encoderPaddingMask,
  returnResult=True
)
```

### 3.4. `_prepareOutput` method

#### 3.4.1 Option tests

```{python}
def test_prepareOutput(
    layersOutput: Optional[Tensor] = None,
    attentionWeights: Optional[Tensor] = None,
    adaptiveComputationTimeAttention: Optional[Tensor] = None,
    adaptiveComputationTimeLoss: Optional[Tensor] = None,
    layerHiddenStates: Optional[Tensor] = None,
    encoderOutputDict: Optional[Dict[str, List[Tensor]]] = None,
    alignmentHeads: Optional[int] = None,
    returnResult: bool = False
):

  printData('Input `layersOutput`', layersOutput)
  printData('Input `attentionWeights`', attentionWeights)
  printData('Input `adaptiveComputationTimeLoss`', adaptiveComputationTimeLoss)
  printData('Input `layerHiddenStates`', layerHiddenStates)
  printData('Input `encoderOutputDict`', encoderOutputDict)
  printData('Input `alignmentHeads`', alignmentHeads)
  print()

  (
    layersOutput,
    results
  ) = model._prepareOutput(
      layersOutput,
      attentionWeights,
      adaptiveComputationTimeAttention,
      adaptiveComputationTimeLoss,
      layerHiddenStates,
      encoderOutputDict,
      alignmentHeads,
    )

  printData('layerOutput', layerOutput)
  printData('results', len(results))

  print('-'*20)

  if returnResult:
    return (
      layersOutput,
      results
    )

test_prepareOutput(
  layersOutput,
  attentionWeights,
  adaptiveComputationTimeAttention,
  adaptiveComputationTimeLoss,
  encoderOutputDict=encoderOutput
)
```

### 3.5. `forward` method

#### 3.5.1 Option tests

```{python}
def test_forward(
    targetTokens: Optional[Tensor] = None,
    tokenEmbeddings: Optional[Tensor] = None,
    encoderOutputDict: Optional[Dict[str, List[Tensor]]] = None,
    incrementalState: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    featuresOnlyFlag: bool = False,
    fullContextAlignmentFlag: bool = False,
    alignmentLayer: Optional[int] = None,
    alignmentHeads: Optional[int] = None,
    returnResult: bool = False
):

  printData('Input `targetTokens`', targetTokens)
  printData('Input `tokenEmbeddings`', tokenEmbeddings)
  printData('Input `encoderOutputDict`', encoderOutputDict)
  printData('Input `incrementalState`', incrementalState)
  printData('Input `featuresOnlyFlag`', featuresOnlyFlag)
  printData('Input `fullContextAlignmentFlag`', fullContextAlignmentFlag)
  printData('Input `alignmentLayer`', alignmentLayer)
  printData('Input `alignmentHeads`', alignmentHeads)
  print()

  (
    layersOutput,
    results
  ) = model.forward(
      targetTokens,
      tokenEmbeddings,
      encoderOutputDict,
      incrementalState,
      featuresOnlyFlag,
      fullContextAlignmentFlag,
      alignmentLayer,
      alignmentHeads,
    )

  printData('layerOutput', layerOutput)
  printData('results', len(results) if results is not None else None)
  print('-'*20)

  if returnResult:
    return (
      layersOutput,
      results
    )

print(tokenEmbeddings.shape)
print(encoderPaddingMask.shape)

test_forward(
  tokenEmbeddings=inputTokenEmbeddings,
  # encoderOutputDict=encoderOutput
)
#
# test_forward(
#   targetTokens,
#   inputTokenEmbeddings,
#   encoderOutputDict=encoderOutput
# )
#
# test_forward(
#   targetTokens,
# )
```

## 4. TEST MODELS

### 4.1. Single layer of `TransformerDecoderLayerBase` on `FashionMNIST`

```{python}
def testTransformerDecoderLayerBaseSingleLayerModel():
  from Emperor.base.utils import Trainer
  from Emperor.base.datasets import FashionMNIST
  from Emperor.experiments import TransformerDecoderBaseSingleLayerModel
  from Emperor.config import ModelConfig, ParameterGeneratorOptions

  cfg = ModelConfig(
    batchSize=16,
    inputDim=16,
    depthDim=64,
    sequenceLength=50,
    embeddingDim=16,
    qkvHiddenDim=64,
    numExperts=16,
    headDim=32,
    topK=2,
    parameterGeneartorType=ParameterGeneratorOptions.matrix_choice_sparse,
    inputEmbeddingDim=16
  )

  data = FashionMNIST(
    batch_size=cfg.batchSize,
    testDatasetFalg=True,
    testDatasetNumSamples=16
  )

  model = TransformerDecoderBaseSingleLayerModel(learningRate=0.001, cfg=cfg)
  trainer = Trainer(max_epochs=100)
  trainer.fit(model, data, printLossFlag=True)

testTransformerDecoderLayerBaseSingleLayerModel()
```
