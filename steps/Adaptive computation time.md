# Transformer Encoder Base Components

## 1. Requirements

### 1.1. Import necessary classes

```{python}
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import torch.nn.functional as F

from Emperor.components.sut_layer import TransformerEncoderLayerBase
# from Emperor.components.act import AdaptiveComputationTimeWrapper
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

### 1.3. Initialize Transformer Model

```{python}
print('# embeddingDim: ', cfg.embeddingDim)
print('# qkvHiddenDim: ', cfg.qkvHiddenDim)
print('# numExperts: ', cfg.numExperts)
print('# topK: ', cfg.topK)
print('# headDim: ', cfg.headDim)
```

```{python}
model = TransformerEncoderLayerBase(
  cfg=cfg,
  returnRawFFNOutputFlag = True,
)

print('# Transformer Encoder Attention Model: \n', model.attentionModel)
print('# Transformer Encoder Feed Forward Model: ', model.ffnModel)
```

## 2. Expected Inputs First Pass

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
selfAttentionInput = layerInput
print(selfAttentionInput.size())
```

### 2.4 Padding Mask

```{python}
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

### 2.4 Layer index

```{python}
layerIdx = 1
```

## 3. First pass step by step

- create placeholders that will be used in the next iteration for
  `log_never_halt` and `acc_expect_depth`, i need to come back when i understand
  what those are used for

```{python}
log_never_halt = acc_expect_depth = torch.zeros_like(layerInput[..., 0])
print(log_never_halt.shape)
print(log_never_halt)
```

- this seems to be some kind of mask across every single
  feature in the `layerInput`

```{python}
acc_h = torch.zeros_like(layerInput)
print(acc_h.shape)
print(acc_h)
```

```{python}
index = 0
print(index)
```

```{python}
hatlingMask = paddingMask
print(hatlingMask)
```

- store the defaults created above to be changed in the next iteration

```{python}
currentAdaptiveComputationState = (
  index,
  log_never_halt,
  acc_h,
  acc_expect_depth,
)
print(hatlingMask)
```

```{python}
layerOutputs = model.forward(
  inputBatch=layerInput,
  selfAttentionInput=None,
  haltMask=hatlingMask,
  layerIdx=layerIdx,
)

layerOutput = layerOutputs[0]
print(layerOutput.shape)
```

```{python}
selfAttentionInput = layerOutput
act_loss = 0
```

```{python}
firstPassOutput =(
  currentAdaptiveComputationState,
  layerOutputs,
  selfAttentionInput,
  act_loss,
)
```

## 4. Second pass pass step by step

```{python}
previousAdaptiveComputationState = currentAdaptiveComputationState
```

- unpack the `previousAdaptiveComputationState` in order to update them during
  the current iteration.

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

### `_computeGatingLogits`

- generate scores based on the output of the previous iteration
  `layerOutput`, of shape `(sequenceLength, batchSize, 2)`

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

```{python}
print(log_never_halt.shape)
print(log_never_halt[..., None].shape)
print(log_never_halt[..., None])
```

```{python}
print(gatingLogSoftmaxScores.shape)
print(gatingLogSoftmaxScores)
```

- the first column of each matrix from `log_never_halt` is added to
  the elementwise to each matrix in `gatingLogSoftmaxScores`

```{python}
log_halt = log_never_halt[..., None] + gatingLogSoftmaxScores
print(log_halt)
```

- the columns of `log_halt` into two `sequenceLength, batchSize, 1` tensors
  `log_never_halt` and `p`

```{python}
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

- accumulated probabilities ? i think

```{python}
acc_h = acc_h + p[..., None] * layerInput
print(acc_h.shape)
print(p[..., None].shape)
print(layerInput.shape)
```

```{python}
acc_expect_depth = acc_expect_depth + index * p
print(acc_expect_depth.shape)
print(acc_expect_depth)
```

```{python}
p_never_halt = log_never_halt.exp()
print(p_never_halt.shape)
print(p_never_halt)
```

```{python}
print(paddingMask)
print(p_never_halt.shape)
print(p_never_halt < (1 - 0.99))
p_never_halt = (
    p_never_halt.masked_fill((p_never_halt < (1 - 0.99)), 0) * paddingMask
)
print(p_never_halt.shape)
print(p_never_halt)
```

```{python}
p_never_halt = p_never_halt.contiguous()
print(p.shape)
print(p)
```

```{python}
index = index + 1
print(p.shape)
print(p)
```
