# Adaptive Computation Time Components

## 1. Requirements

### 1.1. Import necessary classes

```{python}
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import torch.nn.functional as F

from Emperor.components.sut_layer import TransformerEncoderLayerBase
from Emperor.components.act import AdaptiveComputationTimeWrapper
from Emperor.components.parameter_generators.utils.losses import AuxiliaryLosses
from Emperor.config import ModelConfig

def printData(name, elm):
  if isinstance(elm, torch.Tensor):
    elm = elm.shape
  print(f'# {name}: {elm}')
```

### 1.2. Setup configuration

```{python}
attentionInputOutput = 5
cfg = ModelConfig(
  batchSize=3,
  inputDim=attentionInputOutput,
  outputDim=attentionInputOutput,
  embeddingDim=attentionInputOutput,
  qkvHiddenDim=6,
  sequenceLength=4,
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

### 1.3. Initialize `TransformerEncoderLayerBase` model

```{python}
encoderLayer = TransformerEncoderLayerBase(
  cfg=cfg,
  returnRawFFNOutputFlag = True,
)

print('# Transformer Encoder Attention Model: \n', encoderLayer.attentionModel)
print('# Transformer Encoder Feed Forward Model: ', encoderLayer.ffnModel)
```

### 1.4. Initialize `AdaptiveComputationTimeWrapper` model

```{python}
model = AdaptiveComputationTimeWrapper(
  cfg=cfg,
  model=encoderLayer,
)

print('# Adaptive Computation Time Wrapper: \n', model)
```

## 2. Expected Inputs

### 2.1 Previous Adaptive Computation State

```{python}
previousAdaptiveComputationState = None
```

### 2.2 Layer Input

```{python}
layerInput = torch.randn(cfg.sequenceLength, cfg.batchSize, cfg.embeddingDim)
print(layerInput.size())
```

### 2.3 Self Attention Input

```{python}
selfAttentionInput = None
```

### 2.4 Padding Mask

```{python}
# REMINDER: this is correct this the shape needs to be
# (cfg.sequenceLength, cfg.batchSize) because this is
# the padding mask transposed
mask = torch.zeros(
    (cfg.sequenceLength, cfg.batchSize), dtype=torch.bool
)

paddingStart = cfg.sequenceLength // 2
paddingEnd = cfg.sequenceLength
randomSeqenceLengths = torch.randint(
    paddingStart, paddingEnd, (cfg.sequenceLength,)
)

for batchIdx, sequenceLength in enumerate(randomSeqenceLengths):
    mask[batchIdx, :sequenceLength] = 1

paddingMask = mask

print(paddingMask.shape)
print(paddingMask)
```

```{python}
# REMINDER: this is correct this the shape needs to be
# (cfg.sequenceLength, cfg.batchSize) because this is
# the padding mask transposed
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

print(paddingMask.shape)
print(paddingMask)
```

### 2.4 Layer index

```{python}
layerIdx = 1
```

## 3. Test methods first pass that initializes tensors

### 3.1. `_prepareStateAndHaltingMask` method

#### 3.1.1 Option tests

```{python}
def test_prepareStateAndHaltingMask(
    previousAdaptiveComputationState: Optional[Tuple] = None,
    previousHiddenState: Optional[Tensor] = None,
    paddingMask: Optional[Tensor] = None,
    returnResult: bool = False
):

  previousACT = len(previousAdaptiveComputationState) if previousAdaptiveComputationState is not None else None
  printData('Input: previousAdaptiveComputationState', previousACT)
  printData('Input: previousHiddenState', previousHiddenState)
  printData('Input: paddingMask', paddingMask)
  print()

  updatedAdaptiveComputationState, haltingMask = model._prepareStateAndHaltingMask(
    previousAdaptiveComputationState,
    previousHiddenState,
    paddingMask,
  )

  updatedACT = len(updatedAdaptiveComputationState) if updatedAdaptiveComputationState is not None else None
  printData('Output: updatedACT', updatedACT )
  printData('Output: haltingMask', haltingMask)
  print('-'*20)

  if returnResult:
    return (
      updatedAdaptiveComputationState,
      haltingMask
    )

test_prepareStateAndHaltingMask(
  previousAdaptiveComputationState=previousAdaptiveComputationState,
  previousHiddenState=layerInput,
  paddingMask=paddingMask
)

test_prepareStateAndHaltingMask(
  previousAdaptiveComputationState=previousAdaptiveComputationState,
  previousHiddenState=layerInput,
)
```

#### 3.1.2 Outputs required for next step

```{python}
(
  currentAdaptiveComputationState,
  haltingMask
) = test_prepareStateAndHaltingMask(
  previousAdaptiveComputationState=previousAdaptiveComputationState,
  previousHiddenState=layerInput,
  paddingMask=paddingMask,
  returnResult=True
)
```

### 3.2. Generating encoder layer output

```{python}
encoderModelOutput = model.model.forward(
    inputBatch=layerInput,
    selfAttentionInput=selfAttentionInput,
    haltMask=haltingMask,
    layerIdx=layerIdx,
    encoderPaddingMask=encoderPaddingMask
)
```

### 3.3. `_computeSelfAttentionAndACTLoss` method

#### 3.3.1 Option tests

```{python}
def test_computeSelfAttentionAndACTLoss(
    layerOutput: Optional[Tensor] = None,
    selfAttentionInput: Optional[Tensor] = None,
    previousAdaptiveComputationState: Optional[Tuple] = None,
    currentAdaptiveComputationState: Optional[Tuple] = None,
    haltingMask: Optional[Tensor] = None,
    paddingMask: Optional[Tensor] = None,
    returnResult: bool = False
):

  previousACT = len(previousAdaptiveComputationState) if previousAdaptiveComputationState is not None else None
  currentACT = len(currentAdaptiveComputationState) if currentAdaptiveComputationState is not None else None
  printData('Input: layerOutput', layerOutput[0])
  printData('Input: selfAttentionInput', selfAttentionInput)
  printData('Input: previousAdaptiveComputationState', previousACT)
  printData('Input: currentAdaptiveComputationState', currentACT)
  printData('Input: haltingMask', haltingMask)
  printData('Input: paddingMask', paddingMask)
  print()

  selfAttentionInput, actLoss = model._computeSelfAttentionAndACTLoss(
        layerOutput[0],
        selfAttentionInput,
        previousAdaptiveComputationState,
        currentAdaptiveComputationState,
        haltingMask,
        paddingMask,
  )

  printData('Output: selfAttentionInput', selfAttentionInput)
  printData('Output: actLoss', actLoss)
  print('-'*20)

  if returnResult:
    return (
      selfAttentionInput,
      actLoss
    )

test_computeSelfAttentionAndACTLoss(
  encoderModelOutput,
  selfAttentionInput,
  previousAdaptiveComputationState,
  currentAdaptiveComputationState,
  paddingMask,
  haltingMask
)

test_computeSelfAttentionAndACTLoss(
  encoderModelOutput,
  selfAttentionInput,
  previousAdaptiveComputationState,
  currentAdaptiveComputationState,
  haltingMask,
)

test_computeSelfAttentionAndACTLoss(
  encoderModelOutput,
  selfAttentionInput,
  previousAdaptiveComputationState,
  currentAdaptiveComputationState
)

test_computeSelfAttentionAndACTLoss(
  encoderModelOutput,
  selfAttentionInput,
  previousAdaptiveComputationState
)

test_computeSelfAttentionAndACTLoss(
  encoderModelOutput,
  selfAttentionInput,
)

test_computeSelfAttentionAndACTLoss(
  encoderModelOutput
)
```

#### 3.3.2 Outputs required for next step

```{python}
(
  selfAttentionInput,
  actLoss
) = test_computeSelfAttentionAndACTLoss(
  encoderModelOutput,
  selfAttentionInput,
  previousAdaptiveComputationState,
  currentAdaptiveComputationState,
  paddingMask,
  haltingMask,
  returnResult= True
)
```

### 4.1. `forward` method

#### 4.1.1 Option tests

```{python}
(
  currentAdaptiveComputationState,
  modelOutput,
  selfAttentionInput,
  actLoss,
) = model.forward(
  previousAdaptiveComputationState=currentAdaptiveComputationState,
  previousHiddenState=encoderModelOutput[0],
  selfAttentionInput=selfAttentionInput,
  paddingMask=paddingMask,
  layerIdx=layerIdx,
  encoderPaddingMask=encoderPaddingMask,
)

print(modelOutput.shape)
print(selfAttentionInput.shape)
print(actLoss)
print(currentAdaptiveComputationState)
```

## 4. TEST MODELS

### 4.1. `TransformerEncoderBase` using `AdaptiveComputationTimeWrapper` on `FashionMNIST`

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

  model = TransformerEncoderBaseSingleLayerModel(
    learningRate=0.01,
    cfg=cfg,
    encoderHaltingFlag=True
  )

  trainer = Trainer(max_epochs=100)
  trainer.fit(model, data, printLossFlag=True)

testTransformerEncoderBaseSingleLayerModel()
```

## 4. Step by step implementation

```{python}
previousAdaptiveComputationState = currentAdaptiveComputationState
```

- unpack the `previousAdaptiveComputationState`

```{python}
index, log_never_halt, acc_h, acc_expect_depth = (
    previousAdaptiveComputationState
)
print(index)
print(log_never_halt.shape)
print(acc_h.shape)
print(acc_expect_depth.shape)
print(log_never_halt)
print(acc_h)
print(acc_expect_depth)
```

### `_computeGatingLogits` - method begin

- generate scores based on the output of the previous iteration
  `layerOutput`, of shape `(sequenceLength, batchSize, 2)`, each
  in the matrices of shape `(batchSize, 2)` will be later split
  to return two types of score

```{python}
gatingModel = nn.Sequential(
    nn.Linear(cfg.embeddingDim, cfg.embeddingDim),
    nn.GELU(),
    nn.Dropout(0.0),
    nn.Linear(cfg.embeddingDim, 2, bias=False),
)
gatingLogits = gatingModel(layerOutput)
print(gatingLogits.shape)
print(gatingLogits)
```

- perform `log_softmax` on the scores generated above, here are reasons
  for using log:

1. Numerical Stability

   - Softmax alone can cause very small probabilities, leading to
     underflow in floating-point arithmetic.
   - Logarithm of a very small number (log(softmax(x))) can lead
     to numerical instability (log(0) is undefined).
   - log_softmax(x) avoids this by using the log-sum-exp trick,
     which improves stability.

2. Better for `NLLLoss` (Negative Log-Likelihood Loss)

   - In PyTorch, NLLLoss expects log probabilities, not softmax probabilities.
   - Instead of applying softmax first and then log, we directly use log_softmax.

3. More Efficient Computation
   - Computing log(softmax(x)) separately means computing softmax first,
     then taking log.
   - log_softmax(x) combines these steps in a single operation,
     reducing redundant calculations.

```{python}
gatingLogSoftmaxScores = F.log_softmax(gatingLogits, dim=-1)
print(gatingLogSoftmaxScores.shape)
print(gatingLogSoftmaxScores)
```

### `_updateHalting`

- this converts the `(sequenceLength, batchSize)` tensor to a shape
  `(sequenceLength, batchSize, 1)` that will be added to `gatingLogSoftmaxScores`

```{python}
print(log_never_halt.shape)
print(log_never_halt.unsqueeze(-1).shape)
print(log_never_halt[..., None].shape)
print(log_never_halt[..., None])
```

```{python}
print(gatingLogSoftmaxScores.shape)
print(gatingLogSoftmaxScores)
```

- the values of the previous layer held in `log_never_halt` are
  added elementwise addition. The question is why ? What purpose
  does this have ?

```{python}
log_halt = log_never_halt[..., None] + gatingLogSoftmaxScores
print(log_halt)
```

- the columns of `log_halt` into two `(sequenceLength, batchSize, 1)` tensors
  `log_never_halt` and `p`

```{python}
print(log_halt.shape)
log_never_halt = log_halt[..., 0]
print(log_never_halt.shape)
print(log_never_halt)
```

```{python}
p = torch.exp(log_halt[..., 1])
print(p.shape)
print(p)
```

### continued

```{python}
print(layerInput.shape)
print(p[..., None].shape)
print(p[..., None])
```

- i think `acc_h` stands for accumulated halting scores
- the output of the `layerInput` that is the output of the
  previous iteration is multiplied by `p` where
  each `token` in the input is multiplied by a different score
  and accumulated into `acc_h`

```{python}
acc_h = acc_h + p[..., None] * layerInput
print(acc_h.shape)
print(p[..., None].shape)
print(layerInput.shape)
```

- the current `index` that is at the moment 0 is multiplied by `p`
  and the scores are accumulated `acc_expect_depth`, probably accumulated
  until those scores reach a threshold. For the second layer
  the scores will still be 0 and this will kick in on the third iteration
  meaning all tokens are processed for at least 2 layers and the
  dropping will start at the third

```{python}
acc_expect_depth = acc_expect_depth + index * p
print(acc_expect_depth.shape)
print(acc_expect_depth)
```

- to make sure the tensor is positive `exp` is used
  on `log_never_halt`

```{python}
p_never_halt = log_never_halt.exp()
print(p_never_halt.shape)
print(p_never_halt)
```

```{python}
test = torch.tensor([-5]).exp()
print(test )
```

- any tokens with a score smaller than `0.01`

```{python}
print(p_never_halt < (1 - 0.99))
p_never_halt = (
    p_never_halt.masked_fill((p_never_halt < (1 - 0.99)), 0) * paddingMask
)
print(p_never_halt.shape)
print(p_never_halt)
```

```{python}
p_never_halt = p_never_halt.contiguous()
print(p_never_halt.shape)
print(p_never_halt)
```

```{python}
index = index + 1
print(index)
```

- forward pass

```{python}
layerOutputs = model.forward(
  inputBatch=layerInput,
  selfAttentionInput=None,
  haltMask=hatlingMask,
  layerIdx=layerIdx,
)

curr_h = layerOutputs[0]
print(layerOutput.shape)
```

-

```{python}
thresholdCheck = p_never_halt[..., None] < (1 - 0.99)
print(thresholdCheck.shape)
multiplyLayerOuputByHaltMask = p_never_halt[..., None] * curr_h
print(multiplyLayerOuputByHaltMask.shape )
act = acc_h + multiplyLayerOuputByHaltMask
print(act.shape)
self_attn_input = torch.where(
    thresholdCheck,
    layerOutputs,
    (act)
)
print(layerOutput.shape)
```

```{python}
act_loss = (acc_expect_depth + p_never_halt * i) * pad_mask
print(layerOutput.shape)
```

```{python}
act_loss = act_loss.sum() / pad_mask.sum()
print(layerOutput.shape)
```
