# Transformer Encoder Base Components

## 1. Requirements

### 1.1. Import necessary classes

```{python}
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from Emperor.library.choice import Library as L
from Emperor.components.transformer_encoder import TransformerEncoderBase
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
```

```{python}
model = TransformerEncoderBase(
  cfg=cfg,
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
sourceTokens = torch.randint(1, 10, (cfg.batchSize, cfg.sequenceLength))
print(sourceTokens.size())
```

### 2.2 Input Token Embeddings

```{python}
inputTokenEmbeddings = torch.randn(cfg.batchSize, cfg.sequenceLength, cfg.embeddingDim)
print(inputTokenEmbeddings.size())
```

## 3. Test methods

### 3.1. `_computeTokenEmbeddings` method

#### 3.1.1 Option tests

```{python}
def test_computeTokenEmbeddings(
  sourceTokens: Optional[Tensor] = None,
  tokenEmbeddings: Optional[torch.Tensor] = None,
  returnResult: bool = False,
):

  printData('Input `sourceTokens`', sourceTokens)
  printData('Input `tokenEmbeddings`', tokenEmbeddings)
  print()

  (
    tokenEmbeddings,
    paddingMask,
    rawTokenEmbedding,
    hasPaddingMask
  ) = model._computeTokenEmbeddings(
    sourceTokens = sourceTokens,
    tokenEmbeddings = tokenEmbeddings,
  )

  printData('tokenEmbeddings', tokenEmbeddings)
  printData('paddingMask', paddingMask)
  printData('rawTokenEmbedding', rawTokenEmbedding)
  printData('hasPaddingMask', hasPaddingMask)

  print('-'*20)

  if returnResult:
    return (
      tokenEmbeddings,
      paddingMask,
      rawTokenEmbedding,
      hasPaddingMask
    )

test_computeTokenEmbeddings(
  sourceTokens,
  inputTokenEmbeddings
)

test_computeTokenEmbeddings(
  sourceTokens
)
```

#### 3.1.2 Outputs required for next step

```{python}
(
  tokenEmbeddings,
  paddingMask,
  rawTokenEmbedding,
  hasPaddingMask
) = test_computeTokenEmbeddings(
  sourceTokens,
  returnResult=True
)
```

### 3.2. `_retrieveTokenEmbedding` method

#### 3.2.1 Option tests

```{python}
def test_retrieveTokenEmbedding(
  sourceTokens: Optional[Tensor] = None,
  tokenEmbeddings: Optional[torch.Tensor] = None,
  returnResult: bool = False,
):
  printData('Input `sourceTokens`', sourceTokens)
  printData('Input `tokenEmbeddings`', tokenEmbeddings)
  print()

  (
    tokenEmbeddings,
    rawTokenEmbedding,
    positionalEmbedding
  ) = model._retrieveTokenEmbedding(
    sourceTokens = sourceTokens,
    tokenEmbeddings = tokenEmbeddings,
  )

  printData('tokenEmbeddings', tokenEmbeddings)
  printData('rawTokenEmbedding', rawTokenEmbedding)
  printData('positionalEmbedding', positionalEmbedding)

  print('-'*20)

  if returnResult:
    return (
      tokenEmbeddings,
      rawTokenEmbedding,
      positionalEmbedding
    )

test_retrieveTokenEmbedding(
  sourceTokens,
  inputTokenEmbeddings
)

test_retrieveTokenEmbedding(
  sourceTokens
)
```

### 3.3. `_computeAllEncoderLayersOutput` method

#### 3.3.1 Option tests

```{python}
def test_computeAllEncoderLayersOutput(
  tokenEmbeddings: Optional[Tensor] = None,
  paddingMask: Optional[Tensor] = None,
  hasPaddingMask: Optional[bool] = None,
  returnResult: bool = False,
):
  printData('Input `tokenEmbeddings`', sourceTokens)
  printData('Input `paddingMask`', tokenEmbeddings)
  printData('Input `hasPaddingMask`', tokenEmbeddings)
  print()

  (
    layerOutput,
    softHaltingInput,
    act_state,
    act_loss
  ) = model._computeAllEncoderLayersOutput(
    tokenEmbeddings=tokenEmbeddings,
    paddingMask=paddingMask,
    hasPaddingMask=hasPaddingMask,
  )

  printData('layerOutput', layerOutput)
  printData('softHaltingInput', softHaltingInput)
  printData('act_state', act_state)
  printData('act_loss', act_loss)

  print('-'*20)

  if returnResult:
    return (
      layerOutput,
      softHaltingInput,
      act_state,
      act_loss
    )

test_computeAllEncoderLayersOutput(
  tokenEmbeddings,
  paddingMask,
  hasPaddingMask = True,
)

test_computeAllEncoderLayersOutput(
  tokenEmbeddings,
  paddingMask
)
```

#### 3.3.2 Outputs required for next step

```{python}
(
  layerOutput,
  softHaltingInput,
  act_state,
  act_loss
) = test_computeAllEncoderLayersOutput(
  tokenEmbeddings,
  paddingMask,
  hasPaddingMask,
  returnResult = True
)
```

### 3.4. `_prepareEncoderOutput` method

#### 3.4.1 Option tests

```{python}
def test_prepareEncoderOutput(
  layersOutput: Optional[Tensor] = None,
  sourceTokens: Optional[Tensor] = None,
  softHaltingInput: Optional[Tensor] = None,
  act_loss: Optional[Tensor] = None,
  returnResult: bool = False,
):
  printData('Input `layersOutput`', layersOutput)
  printData('Input `sourceTokens`', sourceTokens)
  printData('Input `softHaltingInput`', softHaltingInput)
  print()

  (
    layersOutput,
    totalEncoderLoss,
    sourceLengths,
  ) = model._prepareEncoderOutput(
    layersOutput=layersOutput,
    sourceTokens=sourceTokens,
    softHaltingInput=softHaltingInput,
    act_loss=act_loss,
  )

  printData('layerOutput', layerOutput)
  printData('totalEncoderLoss', totalEncoderLoss)
  printData('sourceLengths', sourceLengths)
  print('-'*20)

  if returnResult:
    return (
      layersOutput,
      totalEncoderLoss,
      sourceLengths,
    )

test_prepareEncoderOutput(
  layerOutput,
  sourceTokens,
  softHaltingInput
)

test_prepareEncoderOutput(
  layerOutput,
  sourceTokens,
)
```

#### 3.4.2 Outputs required for next step

```{python}
(
  layersOutput,
  totalEncoderLoss,
  sourceLengths,
) = test_prepareEncoderOutput(
  layerOutput,
  sourceTokens,
  softHaltingInput,
  returnResult=True
)
```

### 3.5. `forward` method

#### 3.5.1 Option tests

```{python}
def test_forward(
  sourceTokens: Tensor,
  sourceSequenceLengths: Optional[torch.Tensor] = None,
  tokenEmbeddings: Optional[torch.Tensor] = None,
  returnResult: bool = False,
):
  printData('Input `sourceTokens`', sourceTokens)
  printData('Input `sourceSequenceLengths`', sourceSequenceLengths)
  printData('Input `tokenEmbeddings`', tokenEmbeddings)
  print()

  encoderOutputDictionary = model.forward(
    sourceTokens=sourceTokens,
    sourceSequenceLengths=sourceSequenceLengths,
    tokenEmbeddings=tokenEmbeddings,
  )

  printData('encoderOutputDictionary', encoderOutputDictionary)
  print('-'*20)

  if returnResult:
    return encoderOutputDictionary

test_forward(
  sourceTokens,
  tokenEmbeddings=inputTokenEmbeddings
)

test_forward(
  sourceTokens,
)
```

#### 3.5.2 Outputs required for next step

```{python}
encoderOutputDictionary = test_forward(
  sourceTokens,
  tokenEmbeddings=inputTokenEmbeddings
)
```

## 4. TEST MODELS

### 4.1. `TransformerEncoderBase` on `FashionMNIST`

```{python}
def testTransformerEncoderBaseSingleLayerModel():
  from Emperor.base.utils import Trainer
  from Emperor.base.datasets import FashionMNIST
  from Emperor.experiments import TransformerEncoderBaseSingleLayerModel
  from Emperor.config import ModelConfig, ParameterGeneratorOptions

  cfg = ModelConfig(
    batchSize=16,
    inputDim=16,
    depthDim=64,
    embeddingDim=16,
    qkvHiddenDim=64,
    numExperts=16,
    headDim=16,
    numLayers=3,
    topK=2,
    parameterGeneartorType=ParameterGeneratorOptions.matrix_choice_sparse
  )

  data = FashionMNIST(
    batch_size=cfg.batchSize,
    testDatasetFalg=True,
    testDatasetNumSamples=16
  )

  model = TransformerEncoderBaseSingleLayerModel(learningRate=0.01, cfg=cfg)
  trainer = Trainer(max_epochs=100)
  trainer.fit(model, data, printLossFlag=True)

testTransformerEncoderBaseSingleLayerModel()
```
