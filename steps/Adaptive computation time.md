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

- the `layerInput` is of shape `(sequenceLength, batchSize, embeddingDim)`
  and `layerInput[..., 0]` retrieves the first feature of every single
  sequence in the batch, and stores them in different columns for each
  batch `(sequenceLength, batchSize)`

```{python}
print(layerInput.shape)
print(layerInput[..., 0].shape)
print(layerInput[..., 0])
print(layerInput)
```

- create `zero tensors` of shape `(sequenceLength, batchSize)` as placeholders
  in the as the next iteration

```{python}
log_never_halt = torch.zeros_like(layerInput[..., 0])
acc_expect_depth = torch.zeros_like(layerInput[..., 0])
print(log_never_halt.shape)
print(log_never_halt)
```

- create zeros tensor of shape `(batchSize, sequenceLength, embeddingDim)` as
  placeholder to be used in the next iteration.

```{python}
acc_h = torch.zeros_like(layerInput)
print(acc_h.shape)
print(acc_h)
```

- default index of the first iteration

```{python}
index = 0
print(index)
```

- The default halting mask starts as the sequence `paddingMask` it will be used
  in next iterations to mask out tokens

```{python}
hatlingMask = paddingMask
print(hatlingMask)
```

- Store default values in `currentAdaptiveComputationState` that will be
  used in the next iteration

```{python}
currentAdaptiveComputationState = (
  index,
  log_never_halt,
  acc_h,
  acc_expect_depth,
)
print(hatlingMask)
```

- forward pass

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
